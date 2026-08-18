from __future__ import annotations

import time

import numpy as np

import tracer_core.highlevel_rl.m5_transport as base


class M7M5LockstepTransport(
    base.M7M5Transport
):
    """
    Evaluation-only synchronous simulator transport.

    Normal transition contract:

        boundary state s_k
        -> latch command seq=k
        -> exactly N physics ticks
        -> boundary state s_{k+1}

    Wall clock is used only for IPC timeout/retry.
    """

    def __init__(
        self,
        *,
        physics_steps: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs
        )

        self.physics_steps = int(
            physics_steps
        )

        if self.physics_steps <= 0:
            raise ValueError(
                "physics_steps must be > 0"
            )

    @staticmethod
    def _is_boundary(
        state,
    ) -> bool:
        return bool(
            state.get(
                "lockstep_eval",
                False,
            )
            and state.get(
                "lockstep_boundary",
                False,
            )
        )

    @staticmethod
    def _boundary_telemetry(
        state,
        *,
        command_seq,
    ):
        return {
            "command_seq":
                command_seq,

            "safety_state":
                str(
                    state.get(
                        "lockstep_safety_state",
                        "normal",
                    )
                ),

            "override_active":
                bool(
                    state.get(
                        "lockstep_override_active",
                        False,
                    )
                ),

            "override_reasons":
                list(
                    state.get(
                        "lockstep_override_reasons",
                        [],
                    )
                ),

            "sim_time_s":
                float(
                    state.get(
                        "lockstep_terminal_end_time_s",
                        state[
                            "sample_time_s"
                        ],
                    )
                ),

            "applied":
                state.get(
                    "lockstep_applied"
                ),

            "lockstep_eval":
                True,
        }

    def wait_initial(
        self,
        *,
        timeout_s: float = 15.0,
    ):
        deadline = (
            time.monotonic()
            + float(timeout_s)
        )

        while (
            time.monotonic()
            < deadline
        ):
            # Drain but deliberately ignore asynchronous
            # M5 telemetry in lockstep mode.
            self._drain_telemetry()

            states = (
                self._drain_state()
            )

            for state in states:
                if not self._is_boundary(
                    state
                ):
                    continue

                completed = state.get(
                    "lockstep_completed_command_seq"
                )

                if completed is not None:
                    continue

                step_num = int(
                    state[
                        "lockstep_step_num"
                    ]
                )

                if step_num != 0:
                    raise RuntimeError(
                        "Initial lockstep boundary "
                        "must be physics step 0; "
                        f"got {step_num}"
                    )

                telemetry = (
                    self._boundary_telemetry(
                        state,
                        command_seq=None,
                    )
                )

                self.latest_state = (
                    state
                )

                self.latest_telemetry = (
                    telemetry
                )

                return (
                    telemetry,
                    state,
                )

            time.sleep(0.001)

        raise TimeoutError(
            "Timed out waiting for initial "
            "lockstep boundary state"
        )

    def step(
        self,
        normalized_action,
        *,
        target_sim_dt: float = 0.20,
        timeout_s: float = 10.0,
    ):
        action = np.asarray(
            normalized_action,
            dtype=np.float64,
        ).reshape(-1)

        if action.shape != (4,):
            raise ValueError(
                "normalized_action must have "
                f"shape (4,), got {action.shape}"
            )

        policy_command = (
            base
            .normalized_action_to_meta_gait(
                action
            )
        )

        seq = self.next_command_seq
        self.next_command_seq += 1

        udp_command = (
            base.make_command(
                seq=seq,
                label=(
                    f"m7_policy_{seq}"
                ),
                vx=policy_command.vx,
                yaw_rate=(
                    policy_command.yaw_rate
                ),
                body_height=(
                    policy_command.body_height
                ),
                swing_clearance=(
                    policy_command
                    .swing_clearance
                ),
                gait_period=(
                    policy_command.gait_period
                ),
                duty_factor=(
                    policy_command.duty_factor
                ),
            )
        )

        raw_command = (
            base.encode_command(
                udp_command
            )
        )

        requested_physical = tuple(
            float(x)
            for x in udp_command.values
        )

        start_state = (
            self.latest_state
        )

        if start_state is None:
            raise RuntimeError(
                "Lockstep step has no "
                "starting boundary state"
            )

        if not self._is_boundary(
            start_state
        ):
            raise RuntimeError(
                "Lockstep step entry is not "
                "a boundary state"
            )

        start_episode = int(
            start_state[
                "episode_index"
            ]
        )

        start_step = int(
            start_state[
                "lockstep_step_num"
            ]
        )

        expected_previous_seq = (
            None
            if seq == 0
            else seq - 1
        )

        actual_previous_seq = (
            start_state.get(
                "lockstep_completed_command_seq"
            )
        )

        if (
            actual_previous_seq
            != expected_previous_seq
        ):
            raise RuntimeError(
                "Lockstep sequence discontinuity "
                "at step entry: "
                f"command={seq} "
                f"expected_previous="
                f"{expected_previous_seq} "
                f"actual_previous="
                f"{actual_previous_seq}"
            )

        deadline = (
            time.monotonic()
            + float(timeout_s)
        )

        next_send = 0.0
        matched_state = None
        reset_boundary_seen = False

        while (
            time.monotonic()
            < deadline
        ):
            now = time.monotonic()

            if now >= next_send:
                self.command_sock.sendto(
                    raw_command,
                    self.command_addr,
                )

                next_send = (
                    now
                    + self.command_repeat_period_s
                )

            # Prevent unused asynchronous telemetry
            # from accumulating in the socket buffer.
            self._drain_telemetry()

            states = (
                self._drain_state()
            )

            for state in states:
                if not self._is_boundary(
                    state
                ):
                    continue

                if int(
                    state[
                        "episode_index"
                    ]
                ) != start_episode:
                    reset_boundary_seen = True
                    continue

                completed = state.get(
                    "lockstep_completed_command_seq"
                )

                if completed is None:
                    continue

                completed = int(
                    completed
                )

                if completed > seq:
                    raise RuntimeError(
                        "Lockstep boundary skipped "
                        "requested command: "
                        f"expected={seq} "
                        f"completed={completed}"
                    )

                if completed == seq:
                    matched_state = state
                    break

            if matched_state is not None:
                break

            if reset_boundary_seen:
                raise RuntimeError(
                    "Lockstep simulator episode "
                    "changed before a matching "
                    f"terminal/boundary packet for seq={seq}"
                )

            time.sleep(0.001)

        if matched_state is None:
            raise TimeoutError(
                "Timed out waiting for exact "
                "lockstep boundary after "
                f"command seq={seq}"
            )

        terminal_boundary = bool(
            matched_state.get(
                "lockstep_terminal",
                False,
            )
        )

        if terminal_boundary:
            end_step = int(
                matched_state[
                    "lockstep_terminal_end_step_num"
                ]
            )

            end_time = float(
                matched_state[
                    "lockstep_terminal_end_time_s"
                ]
            )

        else:
            end_step = int(
                matched_state[
                    "lockstep_step_num"
                ]
            )

            end_time = float(
                matched_state[
                    "sample_time_s"
                ]
            )

        step_delta = (
            end_step
            - start_step
        )

        if terminal_boundary:
            if not (
                0
                < step_delta
                <= self.physics_steps
            ):
                raise RuntimeError(
                    "Terminal lockstep physics-step "
                    "contract violated: "
                    f"start={start_step} "
                    f"end={end_step} "
                    f"delta={step_delta} "
                    f"max={self.physics_steps}"
                )

        elif (
            step_delta
            != self.physics_steps
        ):
            raise RuntimeError(
                "Lockstep physics-step contract "
                "violated: "
                f"start={start_step} "
                f"end={end_step} "
                f"delta={step_delta} "
                f"expected={self.physics_steps}"
            )

        sim_dt = (
            end_time
            - float(
                start_state[
                    "sample_time_s"
                ]
            )
        )

        if terminal_boundary:
            if not (
                0.0
                < sim_dt
                <= float(
                    target_sim_dt
                ) + 1e-9
            ):
                raise RuntimeError(
                    "Terminal lockstep simulator-time "
                    "contract violated: "
                    f"dt={sim_dt:.15f} "
                    f"max="
                    f"{float(target_sim_dt):.15f}"
                )

        elif abs(
            sim_dt
            - float(target_sim_dt)
        ) > 1e-9:
            raise RuntimeError(
                "Lockstep simulator-time "
                "contract violated: "
                f"dt={sim_dt:.15f} "
                f"expected="
                f"{float(target_sim_dt):.15f}"
            )

        telemetry = (
            self._boundary_telemetry(
                matched_state,
                command_seq=seq,
            )
        )

        applied_physical = (
            base._command_tuple(
                telemetry.get(
                    "applied"
                )
            )
        )

        applied_normalized = None

        if applied_physical is not None:
            applied_normalized = tuple(
                float(x)
                for x in (
                    base
                    .physical_action_to_normalized(
                        applied_physical[:4]
                    )
                )
            )

        energy_interval = (
            base._mechanical_energy_interval(
                start_state,
                matched_state,
            )
        )

        # Historical tracer_cost_v3 interval.
        traction_interval = (
            base._traction_cost_interval(
                start_state,
                matched_state,
            )
        )

        # tracer_cost_v4 parallel latched interval.
        latched_traction_interval = (
            base._latched_traction_cost_interval(
                start_state,
                matched_state,
            )
        )

        self.latest_state = (
            matched_state
        )

        self.latest_telemetry = (
            telemetry
        )

        return base.M7TransportSample(
            command_seq=seq,

            requested_normalized=tuple(
                float(x)
                for x in action
            ),

            requested_physical=(
                requested_physical
            ),

            applied_physical=(
                applied_physical
            ),

            applied_normalized=(
                applied_normalized
            ),

            state=matched_state,
            telemetry=telemetry,

            sim_dt_s=float(
                sim_dt
            ),

            energy_interval=(
                energy_interval
            ),

            traction_interval=(
                traction_interval
            ),

            latched_traction_interval=(
                latched_traction_interval
            ),
        )
