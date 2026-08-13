from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

import gymnasium as gym
import numpy as np

from tracer_core.highlevel_rl.contracts import (
    M7Observation,
    build_observation,
    normalized_action_to_meta_gait,
)

from tracer_core.highlevel_rl.m5_transport import (
    COMMAND_KEYS,
    M7M5Transport,
)

from tracer_core.highlevel_rl.reward_v1 import (
    compute_fixed_additive_baseline,
    compute_tracer_uniform_reward,
)

from tracer_core.highlevel_rl.reward_v2 import (
    compute_simplified_tracer_costs,
    m4_unsafe_attitude_limits,
    objective_reward,
    task_feasibility_tail_cost,
)

from tracer_core.highlevel_rl.reward_slr_hl_v1 import (
    compute_slr_hl_reward,
)

from tracer_core.highlevel_rl.reward_slr_hl_v2 import (
    compute_slr_hl_reward_v2,
)



from tracer_core.highlevel_rl.terrain import (
    get_terrain_preset,
)


ROOT = Path(__file__).resolve().parents[2]


REWARD_MODES = (
    "fixed_additive",
    "tracer_uniform",
    "tracer_cost_v2",
    "slr_hl_v1",
    "slr_hl_v2",
)


def _wrap_pi(x: float) -> float:
    return (
        float(x) + math.pi
    ) % (2.0 * math.pi) - math.pi


def _command_tuple(mapping):
    if mapping is None:
        return None

    return tuple(
        float(mapping[k])
        for k in COMMAND_KEYS
    )


def _resolve_episode_status(
    *,
    success: bool,
    native_terminated: bool,
    native_truncated: bool,
    episode_step: int,
    max_episode_steps: int,
    safety_state: str,
    override_active: bool,
    terminate_on_m4_unsafe: bool,
) -> dict:
    """
    Resolve Gymnasium termination semantics independently
    of the physical simulator.

    WATCH is diagnostic only.
    Only UNSAFE or active override counts as M4 intervention.
    """

    m4_unsafe = (
        str(safety_state).strip().lower()
        == "unsafe"
    )

    m4_intervention = bool(
        override_active
        or m4_unsafe
    )

    m4_terminal = bool(
        terminate_on_m4_unsafe
        and m4_intervention
    )

    terminated = bool(
        success
        or native_terminated
        or m4_terminal
    )

    time_limit = (
        int(episode_step)
        >= int(max_episode_steps)
    )

    truncated = bool(
        native_truncated
        or (
            time_limit
            and not terminated
        )
    )

    return {
        "m4_unsafe": m4_unsafe,
        "m4_intervention":
            m4_intervention,
        "m4_terminal": m4_terminal,
        "terminated": terminated,
        "truncated": truncated,
        "time_limit": time_limit,
    }


class PyMPCM7Env(gym.Env):
    """
    Minimal synchronous-facing M7 Gymnasium environment.

    M7-v0:
      - policy action: continuous [-1,1]^4
      - structural gait parameters fixed
      - oracle terrain context given
      - point goal given relative to reset pose
      - frozen M5/M4/PyMPC execution underneath

    reset() owns a fresh PyMPC simulation subprocess.
    This is intentionally simple and semantically real.
    Persistent multi-episode simulation can be optimized later.
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        *,
        terrain: str = "flat",
        terrain_friction: float | None = None,
        rough_height_scale: float = 1.0,
        oracle_context: Sequence[float] | None = None,
        goal_distance_m: float = 0.50,
        success_radius_m: float = 0.15,
        decision_dt_s: float = 0.20,
        max_episode_steps: int = 25,
        terminate_on_m4_unsafe: bool = False,
        reward_mode: str = "fixed_additive",
        host: str = "127.0.0.1",
        command_port: int = 50610,
        telemetry_port: int = 50611,
        state_port: int = 50612,
        command_repeat_hz: float = 20.0,
        telemetry_hz: float = 100.0,
        state_hz: float = 100.0,
        log_dir: str | Path | None = None,
        render_runner: bool = False,
    ):
        super().__init__()

        self.terrain_preset = (
            get_terrain_preset(
                terrain,
                friction_override=(
                    terrain_friction
                ),
            )
        )

        self.terrain_name = (
            self.terrain_preset.name
        )

        self.terrain_scene = (
            self.terrain_preset.scene
        )

        self.terrain_friction = float(
            self.terrain_preset.friction_coeff
        )

        self.rough_height_scale = float(
            rough_height_scale
        )

        if self.rough_height_scale <= 0.0:
            raise ValueError(
                "rough_height_scale must be > 0"
            )

        preset_context = tuple(
            float(x)
            for x in (
                self.terrain_preset
                .oracle_context
            )
        )

        # Backward compatibility:
        # old M7 callers may still explicitly provide
        # [1,0,0] for the flat terrain.
        #
        # Terrain is now the source-of-truth, so an
        # explicitly supplied context must agree with it.
        if oracle_context is None:
            self.oracle_context = (
                preset_context
            )

        else:
            provided_context = tuple(
                float(x)
                for x in oracle_context
            )

            if (
                len(provided_context)
                != len(preset_context)
                or not np.allclose(
                    np.asarray(
                        provided_context,
                        dtype=float,
                    ),
                    np.asarray(
                        preset_context,
                        dtype=float,
                    ),
                    rtol=0.0,
                    atol=1e-8,
                )
            ):
                raise ValueError(
                    "oracle_context disagrees "
                    "with terrain preset: "
                    f"terrain={self.terrain_name!r} "
                    f"preset={preset_context} "
                    f"provided={provided_context}"
                )

            self.oracle_context = (
                provided_context
            )

        self.goal_distance_m = float(
            goal_distance_m
        )

        self.success_radius_m = float(
            success_radius_m
        )

        self.decision_dt_s = float(
            decision_dt_s
        )

        self.max_episode_steps = int(
            max_episode_steps
        )

        self.terminate_on_m4_unsafe = bool(
            terminate_on_m4_unsafe
        )

        reward_mode = str(
            reward_mode
        ).strip().lower()

        if reward_mode not in REWARD_MODES:
            raise ValueError(
                "Unsupported reward_mode: "
                f"{reward_mode!r}; "
                f"expected one of {REWARD_MODES}"
            )

        self.reward_mode = reward_mode

        self.host = str(host)

        self.command_port = int(
            command_port
        )

        self.telemetry_port = int(
            telemetry_port
        )

        self.state_port = int(
            state_port
        )

        self.command_repeat_hz = float(
            command_repeat_hz
        )

        self.telemetry_hz = float(
            telemetry_hz
        )

        self.state_hz = float(
            state_hz
        )

        # Visualization is opt-in. Training and deterministic
        # evaluation remain headless by default.
        self.render_runner = bool(
            render_runner
        )

        self.log_dir = (
            None
            if log_dir is None
            else Path(log_dir)
        )

        if self.log_dir is not None:
            self.log_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        obs_dim = (
            len(self.oracle_context)
            + 18
        )

        self.action_space = gym.spaces.Box(
            low=np.full(
                4,
                -1.0,
                dtype=np.float32,
            ),
            high=np.full(
                4,
                1.0,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=np.full(
                obs_dim,
                -np.inf,
                dtype=np.float32,
            ),
            high=np.full(
                obs_dim,
                np.inf,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.transport = None
        self.runner_process = None
        self.runner_log_handle = None

        self.episode_index = -1
        self.episode_step = 0
        self.episode_done = True

        self.runner_seed = None

        self.goal_world = None

        self.previous_action = np.zeros(
            4,
            dtype=np.float32,
        )

        # Reward-only history for the SLR-HL adapted
        # second-difference action regularizer.
        #
        # This is NOT added to the observation and therefore
        # preserves the frozen 21D M7 observation contract.
        self.previous_previous_action = np.zeros(
            4,
            dtype=np.float32,
        )

        self.previous_goal_distance = None

        self.last_observation = None
        self.last_info = None

    # ------------------------------------------------------------
    # Runner lifecycle
    # ------------------------------------------------------------

    def _stop_runtime(self):
        if self.transport is not None:
            try:
                self.transport.close()
            finally:
                self.transport = None

        if self.runner_process is not None:
            proc = self.runner_process

            if proc.poll() is None:
                proc.terminate()

                try:
                    proc.wait(timeout=5.0)

                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)

            self.runner_process = None

        if self.runner_log_handle is not None:
            self.runner_log_handle.close()
            self.runner_log_handle = None

    def _runner_duration_s(self) -> float:
        # Enough simulation horizon for one Gym episode
        # plus a startup/termination margin.
        return (
            self.max_episode_steps
            * self.decision_dt_s
            + 2.0
        )

    def _start_runtime(
        self,
        *,
        seed: int,
    ):
        self._stop_runtime()

        # Bind receive-side telemetry/state ports BEFORE
        # launching the PyMPC process, so initial packets
        # cannot be missed.
        self.transport = M7M5Transport(
            host=self.host,
            command_port=self.command_port,
            telemetry_port=self.telemetry_port,
            state_port=self.state_port,
            command_repeat_hz=(
                self.command_repeat_hz
            ),
        )

        env = os.environ.copy()

        env["TRACER_M7_STATE_HOST"] = (
            self.host
        )

        env["TRACER_M7_STATE_PORT"] = str(
            self.state_port
        )

        env["TRACER_M7_STATE_HZ"] = str(
            self.state_hz
        )

        runner_script = (
            ROOT
            / "scripts"
            / "icra27"
            / "run_m7_pympc_state_tap_terrain_seed_v0.py"
        )

        command = [
            sys.executable,
            str(runner_script),

            "--terrain",
            self.terrain_name,

            "--terrain-friction",
            str(self.terrain_friction),
            "--rough-height-scale",
            str(self.rough_height_scale),

            "--command-host",
            self.host,

            "--command-port",
            str(self.command_port),

            "--telemetry-host",
            self.host,

            "--telemetry-port",
            str(self.telemetry_port),

            "--telemetry-hz",
            str(self.telemetry_hz),

            "--duration",
            str(self._runner_duration_s()),

            "--seed",
            str(seed),
        ]

        if not self.render_runner:
            command.append(
                "--no-render"
            )

        stdout = subprocess.DEVNULL

        if self.log_dir is not None:
            runner_log = (
                self.log_dir
                / (
                    f"runner_episode_"
                    f"{self.episode_index:04d}.log"
                )
            )

            self.runner_log_handle = open(
                runner_log,
                "w",
            )

            stdout = self.runner_log_handle

        self.runner_process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )

    # ------------------------------------------------------------
    # Goal / observation
    # ------------------------------------------------------------

    def _goal_features(
        self,
        state: dict,
    ):
        pos = state[
            "base_position_world"
        ]

        rpy = state[
            "base_rpy"
        ]

        x = float(pos[0])
        y = float(pos[1])
        yaw = float(rpy[2])

        gx, gy = self.goal_world

        dx_world = float(gx - x)
        dy_world = float(gy - y)

        c = math.cos(yaw)
        s = math.sin(yaw)

        dx_body = (
            c * dx_world
            + s * dy_world
        )

        dy_body = (
            -s * dx_world
            + c * dy_world
        )

        distance = math.hypot(
            dx_world,
            dy_world,
        )

        heading_error = _wrap_pi(
            math.atan2(
                dy_body,
                dx_body,
            )
        )

        return (
            float(dx_body),
            float(dy_body),
            float(distance),
            float(heading_error),
        )

    def _build_obs(
        self,
        *,
        state: dict,
        applied_physical,
        previous_action,
    ):
        goal = self._goal_features(
            state
        )

        vel_body = state[
            "base_linear_velocity_body_yaw"
        ]

        ang_base = state[
            "base_angular_velocity_base"
        ]

        pos = state[
            "base_position_world"
        ]

        rpy = state[
            "base_rpy"
        ]

        obs = M7Observation(
            oracle_context=(
                self.oracle_context
            ),

            goal_dx_body=goal[0],
            goal_dy_body=goal[1],
            goal_distance=goal[2],
            heading_error=goal[3],

            base_vx_body=float(
                vel_body[0]
            ),

            base_vy_body=float(
                vel_body[1]
            ),

            yaw_rate=float(
                ang_base[2]
            ),

            base_z=float(
                pos[2]
            ),

            roll=float(
                rpy[0]
            ),

            pitch=float(
                rpy[1]
            ),

            applied_vx=float(
                applied_physical[0]
            ),

            applied_yaw=float(
                applied_physical[1]
            ),

            applied_height=float(
                applied_physical[2]
            ),

            applied_clearance=float(
                applied_physical[3]
            ),

            previous_action=tuple(
                float(x)
                for x in previous_action
            ),
        )

        return (
            build_observation(obs),
            goal,
        )

    # ------------------------------------------------------------
    # Episode log
    # ------------------------------------------------------------

    def _log_row(
        self,
        row: dict,
    ):
        if self.log_dir is None:
            return

        path = (
            self.log_dir
            / (
                f"episode_"
                f"{self.episode_index:04d}.jsonl"
            )
        )

        with open(
            path,
            "a",
        ) as f:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )

    # ------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        del options

        if seed is None:
            seed = 0

        seed = int(seed)

        self.episode_index += 1
        self.episode_step = 0
        self.episode_done = False
        self.runner_seed = seed

        self.previous_action = np.zeros(
            4,
            dtype=np.float32,
        )

        # Reward-only history for the SLR-HL adapted
        # second-difference action regularizer.
        #
        # This is NOT added to the observation and therefore
        # preserves the frozen 21D M7 observation contract.
        self.previous_previous_action = np.zeros(
            4,
            dtype=np.float32,
        )

        self.previous_goal_distance = None

        self._start_runtime(
            seed=seed
        )

        try:
            telemetry, state = (
                self.transport.wait_initial(
                    timeout_s=60.0
                )
            )

        except Exception:
            self._stop_runtime()
            raise

        if (
            self.runner_process is None
            or self.runner_process.poll()
            is not None
        ):
            raise RuntimeError(
                "PyMPC runner exited during reset"
            )

        applied = _command_tuple(
            telemetry.get("applied")
        )

        if applied is None:
            # Initial transport state is expected to
            # be nominal. Use the exact M7 nominal
            # contract rather than inventing values.
            nominal = (
                normalized_action_to_meta_gait(
                    [0.0, 0.0, 0.0, 0.0]
                )
            )

            applied = (
                float(nominal.vx),
                float(nominal.yaw_rate),
                float(nominal.body_height),
                float(nominal.swing_clearance),
                float(nominal.gait_period),
                float(nominal.duty_factor),
            )

        pos = state[
            "base_position_world"
        ]

        yaw = float(
            state["base_rpy"][2]
        )

        self.goal_world = (
            float(pos[0])
            + self.goal_distance_m
            * math.cos(yaw),

            float(pos[1])
            + self.goal_distance_m
            * math.sin(yaw),
        )

        observation, goal = self._build_obs(
            state=state,
            applied_physical=applied,
            previous_action=(
                self.previous_action
            ),
        )

        self.previous_goal_distance = (
            goal[2]
        )

        info = {
            "episode_index":
                self.episode_index,

            "seed":
                seed,

            "runner_pid":
                self.runner_process.pid,

            "oracle_context":
                list(self.oracle_context),

            "goal_world":
                list(self.goal_world),

            "goal_distance":
                float(goal[2]),

            "safety_state":
                str(
                    telemetry[
                        "safety_state"
                    ]
                ),

            "override_active":
                bool(
                    telemetry[
                        "override_active"
                    ]
                ),

            "initial_state_sim_time_s":
                float(
                    state[
                        "sample_time_s"
                    ]
                ),
        }

        self.last_observation = (
            observation.copy()
        )

        self.last_info = dict(info)

        self._log_row(
            {
                "event": "reset",
                **info,
                "observation":
                    observation.tolist(),
            }
        )

        return (
            observation,
            info,
        )

    def step(
        self,
        action,
    ):
        if self.episode_done:
            raise RuntimeError(
                "Episode is done; call reset() "
                "before step()."
            )

        if (
            self.runner_process is None
            or self.runner_process.poll()
            is not None
        ):
            raise RuntimeError(
                "PyMPC runner is not alive"
            )

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape != (4,):
            raise ValueError(
                "action must have shape (4,), "
                f"got {action.shape}"
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        sample = self.transport.step(
            action,
            target_sim_dt=(
                self.decision_dt_s
            ),
            timeout_s=10.0,
        )

        if sample.applied_physical is None:
            raise RuntimeError(
                "M5 telemetry did not expose "
                "an applied command"
            )

        self.episode_step += 1

        observation, goal = self._build_obs(
            state=sample.state,
            applied_physical=(
                sample.applied_physical
            ),
            previous_action=action,
        )

        success = (
            goal[2]
            <= self.success_radius_m
        )

        native_terminated = bool(
            sample.terminated
        )

        native_truncated = bool(
            sample.truncated
        )

        episode_status = (
            _resolve_episode_status(
                success=success,
                native_terminated=(
                    native_terminated
                ),
                native_truncated=(
                    native_truncated
                ),
                episode_step=(
                    self.episode_step
                ),
                max_episode_steps=(
                    self.max_episode_steps
                ),
                safety_state=(
                    sample.safety_state
                ),
                override_active=(
                    sample.override_active
                ),
                terminate_on_m4_unsafe=(
                    self.terminate_on_m4_unsafe
                ),
            )
        )

        m4_unsafe = (
            episode_status[
                "m4_unsafe"
            ]
        )

        m4_intervention = (
            episode_status[
                "m4_intervention"
            ]
        )

        m4_terminal = (
            episode_status[
                "m4_terminal"
            ]
        )

        terminated = (
            episode_status[
                "terminated"
            ]
        )

        truncated = (
            episode_status[
                "truncated"
            ]
        )

        time_limit = (
            episode_status[
                "time_limit"
            ]
        )

        reward_kwargs = {
            "previous_goal_distance":
                self.previous_goal_distance,

            "goal_distance":
                goal[2],

            "heading_error":
                goal[3],

            "roll":
                float(
                    sample.state[
                        "base_rpy"
                    ][0]
                ),

            "pitch":
                float(
                    sample.state[
                        "base_rpy"
                    ][1]
                ),

            "normalized_action":
                action,

            "previous_normalized_action":
                self.previous_action,

            "decision_dt":
                sample.sim_dt_s,

            "override_active":
                sample.override_active,

            "safety_state":
                sample.safety_state,

            "success":
                success,

            "terminated":
                terminated,

            "truncated":
                truncated,
        }

        if self.reward_mode == "fixed_additive":
            reward_fn = (
                compute_fixed_additive_baseline
            )

            reward, reward_components = (
                reward_fn(
                    **reward_kwargs
                )
            )

        elif self.reward_mode == "tracer_uniform":
            reward_fn = (
                compute_tracer_uniform_reward
            )

            reward, reward_components = (
                reward_fn(
                    **reward_kwargs
                )
            )

        elif self.reward_mode == "tracer_cost_v2":
            energy_interval = (
                sample.energy_interval
            )

            if energy_interval is None:
                raise RuntimeError(
                    "tracer_cost_v2 requires "
                    "matched mechanical-energy interval"
                )

            (
                roll_unsafe_rad,
                pitch_unsafe_rad,
            ) = m4_unsafe_attitude_limits()

            tracer_costs = (
                compute_simplified_tracer_costs(
                    previous_goal_distance=(
                        self.previous_goal_distance
                    ),

                    goal_distance=goal[2],

                    decision_dt=(
                        sample.sim_dt_s
                    ),

                    roll=float(
                        sample.state[
                            "base_rpy"
                        ][0]
                    ),

                    pitch=float(
                        sample.state[
                            "base_rpy"
                        ][1]
                    ),

                    roll_unsafe_rad=(
                        roll_unsafe_rad
                    ),

                    pitch_unsafe_rad=(
                        pitch_unsafe_rad
                    ),

                    applied_abs_energy_j=(
                        energy_interval
                        .applied_abs_j
                    ),

                    energy_dt_s=(
                        energy_interval.dt_s
                    ),
                )
            )

            tracer_beta = (
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            )

            objective_reward_value = (
                objective_reward(
                    costs=tracer_costs,
                    beta=tracer_beta,
                )
            )

            (
                task_feasibility_cost,
                task_failure_terminal,
                task_remaining_steps,
            ) = task_feasibility_tail_cost(
                episode_step=(
                    self.episode_step
                ),
                max_episode_steps=(
                    self.max_episode_steps
                ),
                success=success,
                m4_terminal=m4_terminal,
                native_terminated=(
                    native_terminated
                ),
                native_truncated=(
                    native_truncated
                ),
            )

            reward = (
                float(
                    objective_reward_value
                )
                - float(
                    task_feasibility_cost
                )
            )

            reward_components = {
                "reward_schema":
                    "icra27_simplified_tracer_cost_v2",

                # Task/safety diagnostics only.
                #
                # These values do NOT participate in
                # beta^T C or objective_reward.
                "m4_unsafe":
                    bool(m4_unsafe),

                "m4_intervention":
                    float(
                        bool(m4_intervention)
                    ),

                "m4_terminal":
                    bool(m4_terminal),

                "success":
                    bool(success),

                "time_limit":
                    bool(time_limit),

                "beta_motion":
                    tracer_beta[0],

                "beta_stability":
                    tracer_beta[1],

                "beta_energy":
                    tracer_beta[2],

                "objective_cost":
                    -float(
                        objective_reward_value
                    ),

                "objective_reward":
                    float(
                        objective_reward_value
                    ),

                "task_feasibility_cost":
                    float(
                        task_feasibility_cost
                    ),

                "task_failure_terminal":
                    bool(
                        task_failure_terminal
                    ),

                "task_remaining_steps":
                    int(
                        task_remaining_steps
                    ),

                "total_reward":
                    float(reward),

                **tracer_costs.as_dict(),

                "energy_abs_j":
                    float(
                        energy_interval
                        .applied_abs_j
                    ),

                "energy_dt_s":
                    float(
                        energy_interval.dt_s
                    ),

                "m4_roll_unsafe_rad":
                    float(
                        roll_unsafe_rad
                    ),

                "m4_pitch_unsafe_rad":
                    float(
                        pitch_unsafe_rad
                    ),

                "task_feasibility_shaping":
                    "absorbing_failure_tail_v1",
            }

        elif self.reward_mode == "slr_hl_v1":
            energy_interval = (
                sample.energy_interval
            )

            if energy_interval is None:
                raise RuntimeError(
                    "slr_hl_v1 requires "
                    "matched mechanical-energy interval"
                )

            vel_body = sample.state[
                "base_linear_velocity_body_yaw"
            ]

            vel_world = sample.state[
                "base_linear_velocity_world"
            ]

            ang_base = sample.state[
                "base_angular_velocity_base"
            ]

            pos = sample.state[
                "base_position_world"
            ]

            rpy = sample.state[
                "base_rpy"
            ]

            slr_hl = compute_slr_hl_reward(
                heading_error=float(
                    goal[3]
                ),

                base_vx_body=float(
                    vel_body[0]
                ),

                base_vy_body=float(
                    vel_body[1]
                ),

                base_vz_world=float(
                    vel_world[2]
                ),

                base_wx=float(
                    ang_base[0]
                ),

                base_wy=float(
                    ang_base[1]
                ),

                base_wz=float(
                    ang_base[2]
                ),

                base_z=float(
                    pos[2]
                ),

                roll=float(
                    rpy[0]
                ),

                pitch=float(
                    rpy[1]
                ),

                applied_abs_energy_j=float(
                    energy_interval.applied_abs_j
                ),

                energy_dt_s=float(
                    energy_interval.dt_s
                ),

                normalized_action=action,

                previous_normalized_action=(
                    self.previous_action
                ),

                previous_previous_normalized_action=(
                    self.previous_previous_action
                ),

                decision_dt_s=float(
                    sample.sim_dt_s
                ),
            )

            (
                task_feasibility_cost,
                task_failure_terminal,
                task_remaining_steps,
            ) = task_feasibility_tail_cost(
                episode_step=(
                    self.episode_step
                ),

                max_episode_steps=(
                    self.max_episode_steps
                ),

                success=success,

                m4_terminal=m4_terminal,

                native_terminated=(
                    native_terminated
                ),

                native_truncated=(
                    native_truncated
                ),
            )

            reward = (
                float(
                    slr_hl.centered_reward
                )
                - float(
                    task_feasibility_cost
                )
            )

            reward_components = {
                **slr_hl.as_dict(),

                # Common task/safety semantics shared with
                # simplified TRACER reward-v2.
                "m4_unsafe":
                    bool(m4_unsafe),

                "m4_intervention":
                    float(
                        bool(m4_intervention)
                    ),

                "m4_terminal":
                    bool(m4_terminal),

                "success":
                    bool(success),

                "time_limit":
                    bool(time_limit),

                "objective_reward":
                    float(
                        slr_hl.centered_reward
                    ),

                "task_feasibility_cost":
                    float(
                        task_feasibility_cost
                    ),

                "task_failure_terminal":
                    bool(
                        task_failure_terminal
                    ),

                "task_remaining_steps":
                    int(
                        task_remaining_steps
                    ),

                "total_reward":
                    float(reward),

                "energy_abs_j":
                    float(
                        energy_interval.applied_abs_j
                    ),

                "energy_dt_s":
                    float(
                        energy_interval.dt_s
                    ),

                "task_feasibility_shaping":
                    "absorbing_failure_tail_v1",

                "comparison_role":
                    "slr_hl_adapted_dense_baseline",
            }

        elif self.reward_mode == "slr_hl_v2":
            energy_interval = (
                sample.energy_interval
            )

            if energy_interval is None:
                raise RuntimeError(
                    "slr_hl_v2 requires "
                    "matched mechanical-energy interval"
                )

            vel_body = sample.state[
                "base_linear_velocity_body_yaw"
            ]

            vel_world = sample.state[
                "base_linear_velocity_world"
            ]

            ang_base = sample.state[
                "base_angular_velocity_base"
            ]

            robot_height_estimate = (
                sample.state.get(
                    "pympc_robot_height_estimate"
                )
            )

            if robot_height_estimate is None:
                raise RuntimeError(
                    "slr_hl_v2 requires "
                    "pympc_robot_height_estimate"
                )

            height_estimate_phase = (
                sample.state.get(
                    "pympc_height_estimate_phase"
                )
            )

            if (
                height_estimate_phase
                !=
                "post_controller_compute_pre_env_step"
            ):
                raise RuntimeError(
                    "slr_hl_v2 unexpected PyMPC "
                    "height-estimate phase: "
                    f"{height_estimate_phase!r}"
                )

            applied_body_height = float(
                sample.applied_physical[2]
            )

            rpy = sample.state[
                "base_rpy"
            ]

            slr_hl = compute_slr_hl_reward_v2(
                heading_error=float(
                    goal[3]
                ),

                base_vx_body=float(
                    vel_body[0]
                ),

                base_vy_body=float(
                    vel_body[1]
                ),

                base_vz_world=float(
                    vel_world[2]
                ),

                base_wx=float(
                    ang_base[0]
                ),

                base_wy=float(
                    ang_base[1]
                ),

                base_wz=float(
                    ang_base[2]
                ),

                pympc_robot_height_estimate=float(
                    robot_height_estimate
                ),

                applied_body_height=float(
                    applied_body_height
                ),

                roll=float(
                    rpy[0]
                ),

                pitch=float(
                    rpy[1]
                ),

                applied_abs_energy_j=float(
                    energy_interval.applied_abs_j
                ),

                energy_dt_s=float(
                    energy_interval.dt_s
                ),

                normalized_action=action,

                previous_normalized_action=(
                    self.previous_action
                ),

                previous_previous_normalized_action=(
                    self.previous_previous_action
                ),

                decision_dt_s=float(
                    sample.sim_dt_s
                ),
            )

            (
                task_feasibility_cost,
                task_failure_terminal,
                task_remaining_steps,
            ) = task_feasibility_tail_cost(
                episode_step=(
                    self.episode_step
                ),

                max_episode_steps=(
                    self.max_episode_steps
                ),

                success=success,

                m4_terminal=m4_terminal,

                native_terminated=(
                    native_terminated
                ),

                native_truncated=(
                    native_truncated
                ),
            )

            reward = (
                float(
                    slr_hl.centered_reward
                )
                - float(
                    task_feasibility_cost
                )
            )

            reward_components = {
                **slr_hl.as_dict(),

                # Common task/safety semantics shared with
                # simplified TRACER reward-v2.
                "m4_unsafe":
                    bool(m4_unsafe),

                "m4_intervention":
                    float(
                        bool(m4_intervention)
                    ),

                "m4_terminal":
                    bool(m4_terminal),

                "success":
                    bool(success),

                "time_limit":
                    bool(time_limit),

                "objective_reward":
                    float(
                        slr_hl.centered_reward
                    ),

                "task_feasibility_cost":
                    float(
                        task_feasibility_cost
                    ),

                "task_failure_terminal":
                    bool(
                        task_failure_terminal
                    ),

                "task_remaining_steps":
                    int(
                        task_remaining_steps
                    ),

                "total_reward":
                    float(reward),

                "energy_abs_j":
                    float(
                        energy_interval.applied_abs_j
                    ),

                "energy_dt_s":
                    float(
                        energy_interval.dt_s
                    ),

                "slr_pympc_robot_height_estimate_m":
                    float(
                        robot_height_estimate
                    ),

                "slr_applied_body_height_target_m":
                    float(
                        applied_body_height
                    ),

                "slr_height_tracking_error_m":
                    float(
                        robot_height_estimate
                        - applied_body_height
                    ),

                "task_feasibility_shaping":
                    "absorbing_failure_tail_v1",

                "comparison_role":
                    "slr_hl_adapted_v2_dense_baseline",
            }

        else:
            raise RuntimeError(
                "Invalid internal reward mode: "
                f"{self.reward_mode!r}"
            )

        info = {
            "episode_index":
                self.episode_index,

            "episode_step":
                self.episode_step,

            "command_seq":
                sample.command_seq,

            "decision_dt_s":
                float(
                    sample.sim_dt_s
                ),

            # Reward/diagnostic-only PyMPC terrain estimator
            # telemetry. These fields are intentionally NOT
            # part of the frozen 21-D observation.
            "pympc_terrain_height_estimate_world":
                (
                    None
                    if sample.state.get(
                        "pympc_terrain_height_estimate_world"
                    ) is None
                    else float(
                        sample.state[
                            "pympc_terrain_height_estimate_world"
                        ]
                    )
                ),

            "pympc_robot_height_estimate":
                (
                    None
                    if sample.state.get(
                        "pympc_robot_height_estimate"
                    ) is None
                    else float(
                        sample.state[
                            "pympc_robot_height_estimate"
                        ]
                    )
                ),

            "pympc_height_estimate_phase":
                sample.state.get(
                    "pympc_height_estimate_phase"
                ),

            "goal_world":
                list(self.goal_world),

            "goal_dx_body":
                float(goal[0]),

            "goal_dy_body":
                float(goal[1]),

            "goal_distance":
                float(goal[2]),

            "heading_error":
                float(goal[3]),

            "requested_normalized":
                list(
                    sample
                    .requested_normalized
                ),

            "requested_physical":
                list(
                    sample
                    .requested_physical
                ),

            "applied_physical":
                list(
                    sample
                    .applied_physical
                ),

            "applied_normalized":
                (
                    None
                    if (
                        sample
                        .applied_normalized
                        is None
                    )
                    else list(
                        sample
                        .applied_normalized
                    )
                ),

            "safety_state":
                sample.safety_state,

            "override_active":
                sample.override_active,

            "terminate_on_m4_unsafe":
                self.terminate_on_m4_unsafe,

            "m4_terminal":
                m4_terminal,

            "override_reasons":
                list(
                    sample.telemetry.get(
                        "override_reasons",
                        [],
                    )
                ),

            "m4_intervention":
                bool(
                    m4_intervention
                ),

            "success":
                bool(success),

            "native_terminated":
                native_terminated,

            "native_truncated":
                native_truncated,

            "time_limit":
                bool(time_limit),

            "reward_mode":
                self.reward_mode,

            "reward_schema":
                reward_components.get(
                    "reward_schema"
                ),

            "reward_components":
                reward_components,

            "state_sim_time_s":
                float(
                    sample.state[
                        "sample_time_s"
                    ]
                ),
        }

        self._log_row(
            {
                "event": "step",
                **info,
                "reward":
                    float(reward),
                "terminated":
                    bool(terminated),
                "truncated":
                    bool(truncated),
                "observation":
                    observation.tolist(),
            }
        )

        self.previous_goal_distance = (
            goal[2]
        )

        self.previous_previous_action = (
            self.previous_action.copy()
        )

        self.previous_action = (
            action.copy()
        )

        self.last_observation = (
            observation.copy()
        )

        self.last_info = dict(info)

        self.episode_done = bool(
            terminated
            or truncated
        )

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self):
        self.episode_done = True
        self._stop_runtime()
