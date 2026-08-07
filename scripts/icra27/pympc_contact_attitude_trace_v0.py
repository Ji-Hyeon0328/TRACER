#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ICRA27_SCRIPTS = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27_SCRIPTS))

import pympc_meta_gait_integration_v0 as base
import pympc_meta_gait_guarded_transition_v0 as guarded

from gym_quadruped.quadruped_env import QuadrupedEnv
from tracer_core.highlevel.meta_gait import MetaGaitCommand


LEGS = tuple(base.LEGS)

ORIGINAL_ENV_STEP = QuadrupedEnv.step
PREVIOUS_COMPUTE_ACTIONS = (
    base.sim.QuadrupedPyMPC_Wrapper.compute_actions
)
ORIGINAL_GUARDED_COMPUTE = (
    guarded.guarded_compute_actions
)


TRACE = {
    "time_s": [],
    "step_num": [],

    "base_pos": [],
    "base_ori_rpy": [],
    "base_ang_vel": [],

    "planned_contact": [],
    "physical_contact": [],
    "grf_world": [],

    "foot_rel_base": [],
    "foot_rel_hip": [],

    "applied_frequency_hz": [],
    "applied_duty_factor": [],
}

TERMINATION_EVENTS = []


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            jsonable(v)
            for v in value
        ]

    return value


def get_arg(
    args,
    kwargs,
    name,
    index,
):
    if name in kwargs:
        return kwargs[name]

    return args[index]


def leg_value(
    obj,
    leg,
    index,
):
    if hasattr(obj, leg):
        return getattr(obj, leg)

    try:
        return obj[leg]
    except (KeyError, TypeError, IndexError):
        pass

    return obj[index]


def leg_vector_array(obj):
    result = []

    for i, leg in enumerate(LEGS):
        value = np.asarray(
            leg_value(obj, leg, i),
            dtype=float,
        ).reshape(-1)

        if value.size < 3:
            padded = np.zeros(3, dtype=float)
            padded[:value.size] = value
            value = padded

        result.append(value[:3])

    return np.asarray(
        result,
        dtype=float,
    )


def contact_array(obj):
    result = []

    for i, leg in enumerate(LEGS):
        value = leg_value(
            obj,
            leg,
            i,
        )

        result.append(
            bool(
                np.asarray(value)
                .reshape(-1)[0]
            )
        )

    return np.asarray(
        result,
        dtype=bool,
    )


def traced_compute_actions(
    self,
    *args,
    **kwargs,
):
    tau = ORIGINAL_GUARDED_COMPUTE(
        self,
        *args,
        **kwargs,
    )

    base_pos = np.asarray(
        get_arg(
            args,
            kwargs,
            "base_pos",
            1,
        ),
        dtype=float,
    ).reshape(-1)

    base_ori = np.asarray(
        get_arg(
            args,
            kwargs,
            "base_ori_euler_xyz",
            3,
        ),
        dtype=float,
    ).reshape(-1)

    base_ang_vel = np.asarray(
        get_arg(
            args,
            kwargs,
            "base_ang_vel",
            4,
        ),
        dtype=float,
    ).reshape(-1)

    feet_pos = leg_vector_array(
        get_arg(
            args,
            kwargs,
            "feet_pos",
            5,
        )
    )

    hip_pos = leg_vector_array(
        get_arg(
            args,
            kwargs,
            "hip_pos",
            6,
        )
    )

    step_num = int(
        get_arg(
            args,
            kwargs,
            "step_num",
            13,
        )
    )

    dt = float(
        get_arg(
            args,
            kwargs,
            "simulation_dt",
            10,
        )
    )

    TRACE["time_s"].append(
        float(step_num * dt)
    )

    TRACE["step_num"].append(
        step_num
    )

    TRACE["base_pos"].append(
        base_pos[:3].copy()
    )

    TRACE["base_ori_rpy"].append(
        base_ori[:3].copy()
    )

    TRACE["base_ang_vel"].append(
        base_ang_vel[:3].copy()
    )

    TRACE["planned_contact"].append(
        contact_array(
            self.wb_interface.current_contact
        )
    )

    TRACE["foot_rel_base"].append(
        feet_pos - base_pos[:3]
    )

    TRACE["foot_rel_hip"].append(
        feet_pos - hip_pos
    )

    if base.LOG["step_freq"]:
        frequency = float(
            base.LOG["step_freq"][-1]
        )
    else:
        frequency = np.nan

    if base.LOG["duty_factor"]:
        duty = float(
            base.LOG["duty_factor"][-1]
        )
    else:
        duty = np.nan

    TRACE[
        "applied_frequency_hz"
    ].append(
        frequency
    )

    TRACE[
        "applied_duty_factor"
    ].append(
        duty
    )

    return tau


def instrumented_env_step(
    self,
    action,
):
    (
        physical_contact,
        _,
        grf,
    ) = self.feet_contact_state(
        ground_reaction_forces=True
    )

    global_sample_index = len(
        TRACE["physical_contact"]
    )

    TRACE["physical_contact"].append(
        contact_array(
            physical_contact
        )
    )

    TRACE["grf_world"].append(
        leg_vector_array(grf)
    )

    result = ORIGINAL_ENV_STEP(
        self,
        action=action,
    )

    (
        state,
        reward,
        terminated,
        truncated,
        info,
    ) = result

    if terminated or truncated:
        invalid_contacts = info.get(
            "invalid_contacts",
            {},
        )

        invalid_names = sorted(
            str(name)
            for name
            in invalid_contacts.keys()
        )

        out_of_bounds = bool(
            self._check_out_of_terrain_bounds()
        )

        if invalid_names:
            reason = (
                "invalid_nonfoot_ground_contact"
            )
        elif out_of_bounds:
            reason = (
                "out_of_terrain_bounds"
            )
        elif truncated:
            reason = "truncated"
        else:
            reason = (
                "terminated_unknown"
            )

        TERMINATION_EVENTS.append({
            "global_sample_index":
                int(global_sample_index),

            "simulation_time_s":
                float(self.simulation_time),

            "reason":
                reason,

            "invalid_contact_names":
                invalid_names,

            "out_of_terrain_bounds":
                out_of_bounds,

            "base_position_m": [
                float(x)
                for x in self.base_pos
            ],
        })

    return result


def edge_indices(
    contact,
    leg_index,
    start,
    stop,
    rising,
):
    result = []

    lo = max(
        1,
        start,
    )

    for i in range(
        lo,
        stop,
    ):
        before = bool(
            contact[i - 1, leg_index]
        )
        after = bool(
            contact[i, leg_index]
        )

        if rising:
            is_edge = (
                (not before)
                and after
            )
        else:
            is_edge = (
                before
                and (not after)
            )

        if is_edge:
            result.append(i)

    return result


def match_edge_times(
    planned_times,
    physical_times,
    max_abs_lag_s,
):
    used = set()
    matches = []

    for planned_time in planned_times:
        candidates = []

        for j, physical_time in enumerate(
            physical_times
        ):
            if j in used:
                continue

            lag = (
                physical_time
                - planned_time
            )

            if (
                abs(lag)
                <= max_abs_lag_s
            ):
                candidates.append(
                    (
                        abs(lag),
                        j,
                        lag,
                        physical_time,
                    )
                )

        if not candidates:
            continue

        candidates.sort()
        _, j, lag, physical_time = (
            candidates[0]
        )

        used.add(j)

        matches.append({
            "planned_time_s":
                float(planned_time),

            "physical_time_s":
                float(physical_time),

            "lag_s":
                float(lag),
        })

    return matches


def lag_summary(matches):
    if not matches:
        return {
            "matched_count": 0,
            "mean_lag_ms": None,
            "median_lag_ms": None,
            "max_abs_lag_ms": None,
        }

    lags = np.asarray(
        [
            item["lag_s"]
            for item in matches
        ],
        dtype=float,
    )

    return {
        "matched_count":
            int(len(matches)),

        "mean_lag_ms":
            float(
                1000.0
                * np.mean(lags)
            ),

        "median_lag_ms":
            float(
                1000.0
                * np.median(lags)
            ),

        "max_abs_lag_ms":
            float(
                1000.0
                * np.max(np.abs(lags))
            ),
    }


def vector_stats(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return None

    return {
        "mean": np.mean(
            values,
            axis=0,
        ).tolist(),

        "std": np.std(
            values,
            axis=0,
        ).tolist(),

        "min": np.min(
            values,
            axis=0,
        ).tolist(),

        "max": np.max(
            values,
            axis=0,
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frequency",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--duty-factor",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--nominal-duration",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--target-duration",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    nominal = MetaGaitCommand(
        vx=0.20,
        yaw_rate=0.0,
        body_height=0.30,
        swing_clearance=0.06,
        gait_period=1.0 / 1.4,
        duty_factor=0.65,
        source_mode="icra27_trace",
        source_reason="nominal",
    )

    target = MetaGaitCommand(
        vx=0.20,
        yaw_rate=0.0,
        body_height=0.30,
        swing_clearance=0.06,
        gait_period=(
            1.0 / args.frequency
        ),
        duty_factor=args.duty_factor,
        source_mode="icra27_trace",
        source_reason="target",
    )

    total_duration = (
        args.nominal_duration
        + args.target_duration
    )

    guarded.SCHEDULE = [
        (
            "nominal",
            0.0,
            args.nominal_duration,
            nominal,
        ),
        (
            "target",
            args.nominal_duration,
            total_duration,
            target,
        ),
    ]

    guarded.DURATION = total_duration

    guarded.clear_logs()

    for value in TRACE.values():
        value.clear()

    TERMINATION_EVENTS.clear()

    base.cfg.simulation_params[
        "gait"
    ] = "trot"

    base.cfg.mpc_params[
        "optimize_step_freq"
    ] = False

    gait_cfg = (
        base.cfg.simulation_params[
            "gait_params"
        ]["trot"]
    )

    gait_cfg["step_freq"] = 1.4
    gait_cfg["duty_factor"] = 0.65

    base.cfg.simulation_params[
        "step_height"
    ] = 0.06

    base.cfg.simulation_params[
        "ref_z"
    ] = 0.30

    (
        base.sim
        .QuadrupedPyMPC_Wrapper
        .compute_actions
    ) = traced_compute_actions

    QuadrupedEnv.step = (
        instrumented_env_step
    )

    print("=" * 72)
    print(
        "ICRA27 M3 CONTACT / ATTITUDE / FOOTHOLD TRACE"
    )
    print("=" * 72)

    print(
        "nominal:"
        " f=1.400 D=0.650"
        f" for {args.nominal_duration:.1f}s"
    )

    print(
        "target :"
        f" f={args.frequency:.3f}"
        f" D={args.duty_factor:.3f}"
        f" for {args.target_duration:.1f}s"
    )

    print("=" * 72)

    try:
        base.sim.run_simulation(
            qpympc_cfg=base.cfg,
            num_episodes=1,
            num_seconds_per_episode=(
                total_duration
            ),
            ref_base_lin_vel=(
                0.20
                / float(
                    base.cfg.hip_height
                )
            ),
            ref_base_ang_vel=0.0,
            friction_coeff=0.8,
            base_vel_command_type=(
                "forward"
            ),
            seed=args.seed,
            render=False,
            recording_path=None,
        )

    finally:
        QuadrupedEnv.step = (
            ORIGINAL_ENV_STEP
        )

        (
            base.sim
            .QuadrupedPyMPC_Wrapper
            .compute_actions
        ) = PREVIOUS_COMPUTE_ACTIONS

    lengths = {
        key: len(value)
        for key, value in TRACE.items()
    }

    print()
    print("TRACE LENGTHS")
    print(lengths)

    n = min(
        lengths.values()
    )

    if n <= 0:
        raise RuntimeError(
            "No aligned trace samples"
        )

    if len(
        set(lengths.values())
    ) != 1:
        print(
            "WARNING: trace lengths differ; "
            f"using first {n} samples"
        )

    time = np.asarray(
        TRACE["time_s"][:n],
        dtype=float,
    )

    base_pos = np.asarray(
        TRACE["base_pos"][:n],
        dtype=float,
    )

    base_ori = np.asarray(
        TRACE["base_ori_rpy"][:n],
        dtype=float,
    )

    base_ang_vel = np.asarray(
        TRACE["base_ang_vel"][:n],
        dtype=float,
    )

    planned = np.asarray(
        TRACE["planned_contact"][:n],
        dtype=bool,
    )

    physical = np.asarray(
        TRACE["physical_contact"][:n],
        dtype=bool,
    )

    grf = np.asarray(
        TRACE["grf_world"][:n],
        dtype=float,
    )

    foot_rel_base = np.asarray(
        TRACE["foot_rel_base"][:n],
        dtype=float,
    )

    foot_rel_hip = np.asarray(
        TRACE["foot_rel_hip"][:n],
        dtype=float,
    )

    applied_frequency = np.asarray(
        TRACE[
            "applied_frequency_hz"
        ][:n],
        dtype=float,
    )

    applied_duty = np.asarray(
        TRACE[
            "applied_duty_factor"
        ][:n],
        dtype=float,
    )

    target_commit = None

    for event in (
        guarded.STRUCTURAL_EVENTS
    ):
        if (
            event.get(
                "target_label"
            )
            == "target"
        ):
            target_commit = event
            break

    if target_commit is None:
        raise RuntimeError(
            "No target structural commit"
        )

    commit_time = float(
        target_commit["time_s"]
    )

    if TERMINATION_EVENTS:
        first_termination = (
            TERMINATION_EVENTS[0]
        )

        analysis_stop = min(
            int(
                first_termination[
                    "global_sample_index"
                ]
            ),
            n,
        )
    else:
        first_termination = None
        analysis_stop = n

    first_segment_time = (
        time[:analysis_stop]
    )

    analysis_start = int(
        np.searchsorted(
            first_segment_time,
            commit_time,
            side="left",
        )
    )

    if (
        analysis_stop
        <= analysis_start
    ):
        raise RuntimeError(
            "No post-commit first-rollout "
            "analysis window"
        )

    sl = slice(
        analysis_start,
        analysis_stop,
    )

    roll = base_ori[sl, 0]
    pitch = base_ori[sl, 1]

    wx = base_ang_vel[sl, 0]
    wy = base_ang_vel[sl, 1]

    z = base_pos[sl, 2]

    y0 = float(
        base_pos[
            analysis_start,
            1,
        ]
    )

    lateral = (
        base_pos[sl, 1]
        - y0
    )

    physical_count = np.sum(
        physical[sl],
        axis=1,
    )

    attitude_summary = {
        "analysis_start_s":
            float(commit_time),

        "analysis_end_s":
            float(
                time[
                    analysis_stop - 1
                ]
            ),

        "duration_s":
            float(
                time[
                    analysis_stop - 1
                ]
                - commit_time
            ),

        "max_abs_roll_deg":
            float(
                np.degrees(
                    np.max(
                        np.abs(roll)
                    )
                )
            ),

        "max_abs_pitch_deg":
            float(
                np.degrees(
                    np.max(
                        np.abs(pitch)
                    )
                )
            ),

        "rms_roll_deg":
            float(
                np.degrees(
                    np.sqrt(
                        np.mean(
                            roll ** 2
                        )
                    )
                )
            ),

        "rms_pitch_deg":
            float(
                np.degrees(
                    np.sqrt(
                        np.mean(
                            pitch ** 2
                        )
                    )
                )
            ),

        "max_abs_wx_rad_s":
            float(
                np.max(
                    np.abs(wx)
                )
            ),

        "max_abs_wy_rad_s":
            float(
                np.max(
                    np.abs(wy)
                )
            ),

        "min_base_height_m":
            float(
                np.min(z)
            ),

        "max_base_height_m":
            float(
                np.max(z)
            ),

        "max_abs_lateral_drift_m":
            float(
                np.max(
                    np.abs(lateral)
                )
            ),

        "physical_contact_count_fraction": {
            str(k): float(
                np.mean(
                    physical_count == k
                )
            )
            for k in range(5)
        },
    }

    max_match_lag = (
        0.40
        / float(
            args.frequency
        )
    )

    per_leg = {}

    for leg_index, leg in enumerate(
        LEGS
    ):
        planned_td_idx = edge_indices(
            planned,
            leg_index,
            analysis_start,
            analysis_stop,
            rising=True,
        )

        physical_td_idx = edge_indices(
            physical,
            leg_index,
            analysis_start,
            analysis_stop,
            rising=True,
        )

        planned_lo_idx = edge_indices(
            planned,
            leg_index,
            analysis_start,
            analysis_stop,
            rising=False,
        )

        physical_lo_idx = edge_indices(
            physical,
            leg_index,
            analysis_start,
            analysis_stop,
            rising=False,
        )

        planned_td_times = [
            time[i]
            for i in planned_td_idx
        ]

        physical_td_times = [
            time[i]
            for i in physical_td_idx
        ]

        planned_lo_times = [
            time[i]
            for i in planned_lo_idx
        ]

        physical_lo_times = [
            time[i]
            for i in physical_lo_idx
        ]

        td_matches = match_edge_times(
            planned_td_times,
            physical_td_times,
            max_match_lag,
        )

        lo_matches = match_edge_times(
            planned_lo_times,
            physical_lo_times,
            max_match_lag,
        )

        touchdown_rel_base = [
            foot_rel_base[
                i,
                leg_index,
                :,
            ]
            for i
            in physical_td_idx
        ]

        touchdown_rel_hip = [
            foot_rel_hip[
                i,
                leg_index,
                :,
            ]
            for i
            in physical_td_idx
        ]

        per_leg[leg] = {
            "touchdown": {
                "planned_count":
                    len(
                        planned_td_idx
                    ),

                "physical_count":
                    len(
                        physical_td_idx
                    ),

                **lag_summary(
                    td_matches
                ),

                "matches":
                    td_matches,

                "physical_rel_base_m":
                    vector_stats(
                        touchdown_rel_base
                    ),

                "physical_rel_hip_m":
                    vector_stats(
                        touchdown_rel_hip
                    ),
            },

            "liftoff": {
                "planned_count":
                    len(
                        planned_lo_idx
                    ),

                "physical_count":
                    len(
                        physical_lo_idx
                    ),

                **lag_summary(
                    lo_matches
                ),

                "matches":
                    lo_matches,
            },
        }

    if first_termination is None:
        classification = (
            "survived_target_window"
        )
    else:
        classification = (
            "failed_after_target_commit"
        )

    result = {
        "command": {
            "frequency_hz":
                float(
                    args.frequency
                ),

            "duty_factor":
                float(
                    args.duty_factor
                ),

            "vx_mps":
                0.20,

            "body_height_m":
                0.30,

            "swing_clearance_m":
                0.06,
        },

        "protocol": {
            "nominal_frequency_hz":
                1.4,

            "nominal_duty_factor":
                0.65,

            "nominal_duration_s":
                float(
                    args.nominal_duration
                ),

            "target_duration_s":
                float(
                    args.target_duration
                ),

            "seed":
                int(args.seed),
        },

        "target_commit":
            jsonable(
                target_commit
            ),

        "classification":
            classification,

        "first_termination":
            jsonable(
                first_termination
            ),

        "attitude_and_support":
            attitude_summary,

        "edge_matching": {
            "max_abs_match_lag_s":
                float(
                    max_match_lag
                ),

            "per_leg":
                per_leg,
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    npz_path = (
        args.output.with_suffix(
            ".npz"
        )
    )

    np.savez_compressed(
        npz_path,

        time_s=time,
        base_pos=base_pos,
        base_ori_rpy=base_ori,
        base_ang_vel=base_ang_vel,

        planned_contact=(
            planned.astype(np.int8)
        ),

        physical_contact=(
            physical.astype(np.int8)
        ),

        grf_world=grf,

        foot_rel_base=(
            foot_rel_base
        ),

        foot_rel_hip=(
            foot_rel_hip
        ),

        applied_frequency_hz=(
            applied_frequency
        ),

        applied_duty_factor=(
            applied_duty
        ),
    )

    print()
    print("RESULT")
    print("-" * 72)

    print(
        "classification:",
        classification,
    )

    print(
        "commit time:",
        f"{commit_time:.3f}s",
    )

    if first_termination is None:
        print(
            "first termination: NONE"
        )
    else:
        print(
            "first termination:",
            f"{first_termination['simulation_time_s']:.3f}s",
            first_termination["reason"],
            first_termination[
                "invalid_contact_names"
            ],
        )

    print()
    print("ATTITUDE / SUPPORT")
    print(
        json.dumps(
            attitude_summary,
            indent=2,
        )
    )

    print()
    print("CONTACT EDGE TIMING")
    print("-" * 72)

    for leg in LEGS:
        item = per_leg[leg]

        td = item["touchdown"]
        lo = item["liftoff"]

        print(
            f"{leg}: "
            f"TD {td['matched_count']}/"
            f"{td['planned_count']} "
            f"mean={td['mean_lag_ms']} ms "
            f"maxabs={td['max_abs_lag_ms']} ms | "
            f"LO {lo['matched_count']}/"
            f"{lo['planned_count']} "
            f"mean={lo['mean_lag_ms']} ms "
            f"maxabs={lo['max_abs_lag_ms']} ms"
        )

    print()
    print("saved summary:", args.output)
    print("saved raw trace:", npz_path)


if __name__ == "__main__":
    main()
