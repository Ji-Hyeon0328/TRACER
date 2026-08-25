#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
ICRA27 = ROOT / "scripts" / "icra27"

sys.path.insert(
    0,
    str(ROOT),
)

sys.path.insert(
    0,
    str(ICRA27),
)


K5_PATH = (
    ICRA27
    / "evaluate_os_t4p5k5_lockstep_checkpoint_sweep_v0.py"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5k6h_u10_k11_perfoot_trace_v0"
)

UPDATE = 10
TERRAIN = "low_friction"
SEED = 27200

# After five settling decisions:
#
# k=0  : (1.0, 1.2]
# ...
# k=11 : (3.2, 3.4]
TRACE_TMIN_S = 3.2
TRACE_TMAX_S = 3.4

SELECTED_BETAS = (
    "balanced",
    "stability",
    "energy",
)


def load_module(
    path: Path,
    name: str,
):
    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load module: {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def read_jsonl(
    path: Path,
):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_no}: "
                    "invalid JSON"
                ) from exc

    return rows


def restore_env(
    saved,
):
    for name, old in saved.items():
        if old is None:
            os.environ.pop(
                name,
                None,
            )

        else:
            os.environ[
                name
            ] = old


def foot_summary(
    rows,
    leg,
):
    established = []

    for row in rows:
        foot = (
            row["feet"][leg]
        )

        if not bool(
            foot["established"]
        ):
            continue

        speed = float(
            foot[
                "slip_speed_mps"
            ]
        )

        cost = float(
            foot[
                "traction_cost"
            ]
        )

        established.append(
            (
                speed,
                cost,
                float(
                    row["dt_s"]
                ),
            )
        )

    total_dt = sum(
        dt
        for _speed, _cost, dt
        in established
    )

    def tail_dt(
        threshold,
    ):
        return sum(
            dt
            for speed, _cost, dt
            in established
            if speed >= threshold
        )

    if total_dt > 0.0:
        mean_cost = (
            sum(
                cost * dt
                for _speed, cost, dt
                in established
            )
            / total_dt
        )
    else:
        mean_cost = 0.0

    return {
        "established_dt_s":
            total_dt,

        "mean_traction_cost":
            mean_cost,

        "max_speed_mps":
            (
                max(
                    speed
                    for speed, _cost, _dt
                    in established
                )
                if established
                else 0.0
            ),

        "tail_ge_0p15_s":
            tail_dt(
                0.15
            ),

        "tail_ge_0p30_s":
            tail_dt(
                0.30
            ),
    }


def main():
    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists; "
            "refusing overwrite: "
            f"{OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    k5 = load_module(
        K5_PATH,
        "os_t4p5k6h_k5",
    )

    helper = k5.load_module(
        k5.HELPER_PATH,
        "os_t4p5k6h_helper",
    )

    phase2b = helper.load_module(
        helper.PHASE2B_PATH,
        "os_t4p5k6h_phase2b",
    )

    # --------------------------------------------------------
    # Reward-mode provenance.
    #
    # There are TWO historical globals here:
    #
    #   phase2b.base.EVAL_REWARD_MODE
    #       -> common run_episode()/reward_components() checker
    #
    #   phase2b.EVAL_REWARD_MODE
    #       -> phase2b.make_env() -> PyMPCM7Env(reward_mode=...)
    #
    # k6h uses the tracer_cost_v3 checkpoint/evaluation contract,
    # therefore both must be promoted together.
    # --------------------------------------------------------

    historical_base_mode = (
        phase2b.base.EVAL_REWARD_MODE
    )

    historical_phase2b_mode = (
        phase2b.EVAL_REWARD_MODE
    )

    if historical_base_mode != (
        "tracer_cost_v2"
    ):
        raise RuntimeError(
            "Unexpected historical common-evaluator "
            f"mode: {historical_base_mode!r}"
        )

    if historical_phase2b_mode != (
        "tracer_cost_v2"
    ):
        raise RuntimeError(
            "Unexpected historical phase2b env "
            f"mode: {historical_phase2b_mode!r}"
        )

    phase2b.base.EVAL_REWARD_MODE = (
        "tracer_cost_v3"
    )

    phase2b.EVAL_REWARD_MODE = (
        "tracer_cost_v3"
    )

    if (
        phase2b.base.EVAL_REWARD_MODE
        != "tracer_cost_v3"
        or phase2b.EVAL_REWARD_MODE
        != "tracer_cost_v3"
    ):
        raise RuntimeError(
            "k6h reward-mode promotion failed."
        )

    # --------------------------------------------------------
    # Same additive lockstep override as k5.
    # --------------------------------------------------------

    original_ctor = (
        phase2b.PyMPCM7Env
    )

    def lockstep_ctor(
        *args,
        **kwargs,
    ):
        kwargs[
            "lockstep_eval"
        ] = True

        kwargs[
            "lockstep_physics_steps"
        ] = 100

        env = original_ctor(
            *args,
            **kwargs,
        )

        if not bool(
            env.lockstep_eval
        ):
            raise RuntimeError(
                "k6h environment is not lockstep."
            )

        if int(
            env.lockstep_physics_steps
        ) != 100:
            raise RuntimeError(
                "k6h requires 100 physics ticks."
            )

        return env

    phase2b.PyMPCM7Env = (
        lockstep_ctor
    )

    ckpt = k5.checkpoint_path(
        UPDATE
    )

    policy = (
        phase2b.base.load_policy(
            name="beta_conditioned",
            path=ckpt,
            expected_reward_mode=(
                "tracer_cost_v3"
            ),
        )
    )

    phase2b.validate_checkpoint_beta_bank(
        policy["payload"]
    )

    beta_lookup = {
        str(name):
            tuple(
                float(x)
                for x in beta
            )
        for name, beta
        in phase2b.BETA_BANK
    }

    missing = [
        name
        for name in SELECTED_BETAS
        if name not in beta_lookup
    ]

    if missing:
        raise RuntimeError(
            "Missing beta anchors: "
            + ", ".join(missing)
        )

    base_port = (
        k5.choose_base_port()
    )

    print("=" * 112)
    print(
        "ICRA27 OS-T4.5k6h "
        "u10 LOW-FRICTION k=11 "
        "500 Hz PER-FOOT TRACE"
    )
    print("=" * 112)

    print(
        "checkpoint :",
        ckpt,
    )

    print(
        "terrain    :",
        TERRAIN,
    )

    print(
        "seed       :",
        SEED,
    )

    print(
        "window     : "
        f"({TRACE_TMIN_S:.3f}, "
        f"{TRACE_TMAX_S:.3f}] s"
    )

    print(
        "expected   : "
        "100 physics rows per beta"
    )

    print(
        "base port  :",
        base_port,
    )

    print()

    all_traces = {}
    episode_rows = {}

    env_names = (
        "TRACER_M7_SLIP_TRACE",
        "TRACER_M7_SLIP_TRACE_TAG",
        "TRACER_M7_SLIP_TRACE_TMIN_S",
        "TRACER_M7_SLIP_TRACE_TMAX_S",
    )

    for beta_index, beta_name in enumerate(
        SELECTED_BETAS
    ):
        beta = beta_lookup[
            beta_name
        ]

        trace_path = (
            OUT_DIR
            / f"u10_{beta_name}_"
              "low27200_k11_slip500.jsonl"
        )

        log_dir = (
            OUT_DIR
            / "env_logs"
            / beta_name
        )

        saved = {
            name:
                os.environ.get(name)
            for name in env_names
        }

        os.environ[
            "TRACER_M7_SLIP_TRACE"
        ] = str(
            trace_path.resolve()
        )

        os.environ[
            "TRACER_M7_SLIP_TRACE_TAG"
        ] = (
            f"u10/{beta_name}/"
            "low27200/k11"
        )

        os.environ[
            "TRACER_M7_SLIP_TRACE_TMIN_S"
        ] = str(
            TRACE_TMIN_S
        )

        os.environ[
            "TRACER_M7_SLIP_TRACE_TMAX_S"
        ] = str(
            TRACE_TMAX_S
        )

        command_port = (
            int(base_port)
            + 100 * beta_index
        )

        env = None

        try:
            env = phase2b.make_env(
                terrain=TERRAIN,
                beta=beta,
                command_port=(
                    command_port
                ),
                log_dir=log_dir,
            )

            row = (
                phase2b.base.run_episode(
                    env,
                    policy_name=(
                        "beta_conditioned"
                    ),
                    checkpoint=ckpt,
                    trained_reward_mode=(
                        policy[
                            "trained_reward_mode"
                        ]
                    ),
                    model=(
                        policy["model"]
                    ),
                    group=(
                        "low_friction_"
                        "u10_k11_trace"
                    ),
                    terrain=TERRAIN,
                    seed=SEED,
                )
            )

            episode_rows[
                beta_name
            ] = row

        finally:
            if env is not None:
                env.close()

            restore_env(
                saved
            )

        if not trace_path.exists():
            raise RuntimeError(
                "Trace file was not created: "
                f"{trace_path}"
            )

        trace = read_jsonl(
            trace_path
        )

        if len(trace) != 100:
            raise RuntimeError(
                f"{beta_name}: expected exactly "
                f"100 trace rows, got {len(trace)}"
            )

        for index, item in enumerate(
            trace
        ):
            dt = float(
                item["dt_s"]
            )

            if abs(
                dt - 0.002
            ) > 1e-12:
                raise RuntimeError(
                    f"{beta_name}: "
                    f"row {index} dt={dt}"
                )

            t = float(
                item["time_post_s"]
            )

            if not (
                TRACE_TMIN_S
                < t
                <= TRACE_TMAX_S
                + 1e-12
            ):
                raise RuntimeError(
                    f"{beta_name}: "
                    f"trace time outside window: "
                    f"{t}"
                )

        all_traces[
            beta_name
        ] = trace

        print(
            f"{beta_name:<9} "
            f"status="
            f"{row['status']:<12} "
            f"policy_steps="
            f"{row['policy_steps']:<3} "
            f"trace_rows="
            f"{len(trace)}"
        )

    # --------------------------------------------------------
    # Per-foot burden.
    # --------------------------------------------------------

    summaries = {}

    print()
    print("=" * 112)
    print(
        "PER-FOOT ESTABLISHED-STANCE BURDEN "
        "DURING k=11"
    )
    print("=" * 112)

    for beta_name in SELECTED_BETAS:
        print()
        print(
            beta_name.upper()
        )

        summaries[
            beta_name
        ] = {}

        for leg in (
            "FL",
            "FR",
            "RL",
            "RR",
        ):
            s = foot_summary(
                all_traces[
                    beta_name
                ],
                leg,
            )

            summaries[
                beta_name
            ][leg] = s

            print(
                f"  {leg} "
                f"est="
                f"{1000*s['established_dt_s']:6.1f}ms "
                f"tail>=.15="
                f"{1000*s['tail_ge_0p15_s']:5.1f}ms "
                f">=.30="
                f"{1000*s['tail_ge_0p30_s']:5.1f}ms "
                f"max="
                f"{s['max_speed_mps']:.4f} "
                f"meanCtr="
                f"{s['mean_traction_cost']:.5f}"
            )

    # --------------------------------------------------------
    # Same-tick comparison.
    # --------------------------------------------------------

    by_step = {
        beta_name: {
            int(
                row[
                    "physics_step_num_post"
                ]
            ):
                row
            for row in trace
        }
        for beta_name, trace
        in all_traces.items()
    }

    common_steps = sorted(
        set(
            by_step["balanced"]
        )
        & set(
            by_step["stability"]
        )
        & set(
            by_step["energy"]
        )
    )

    if len(common_steps) != 100:
        raise RuntimeError(
            "Expected 100 aligned physics ticks, "
            f"got {len(common_steps)}"
        )

    events = []

    for step_num in common_steps:
        b = by_step[
            "balanced"
        ][step_num]

        s = by_step[
            "stability"
        ][step_num]

        e = by_step[
            "energy"
        ][step_num]

        for leg in (
            "FL",
            "FR",
            "RL",
            "RR",
        ):
            sf = s["feet"][leg]

            if (
                not sf["established"]
                or sf[
                    "slip_speed_mps"
                ] is None
            ):
                continue

            speed_s = float(
                sf[
                    "slip_speed_mps"
                ]
            )

            if speed_s < 0.15:
                continue

            def speed_or_none(
                row,
            ):
                foot = row[
                    "feet"
                ][leg]

                if (
                    not foot["contact"]
                    or foot[
                        "slip_speed_mps"
                    ] is None
                ):
                    return None

                return float(
                    foot[
                        "slip_speed_mps"
                    ]
                )

            events.append(
                {
                    "step_num":
                        step_num,

                    "time_post_s":
                        float(
                            s[
                                "time_post_s"
                            ]
                        ),

                    "leg":
                        leg,

                    "age_s":
                        float(
                            sf[
                                "contact_age_s"
                            ]
                        ),

                    "balanced_speed":
                        speed_or_none(b),

                    "stability_speed":
                        speed_s,

                    "energy_speed":
                        speed_or_none(e),
                }
            )

    events.sort(
        key=lambda row: (
            row[
                "stability_speed"
            ]
        ),
        reverse=True,
    )

    print()
    print("=" * 112)
    print(
        "TOP SAME-TICK STABILITY HIGH-SLIP EVENTS"
    )
    print("=" * 112)

    for row in events[:20]:
        def fmt(value):
            return (
                "None"
                if value is None
                else f"{value:.4f}"
            )

        print(
            f"t={row['time_post_s']:.3f} "
            f"tick={row['step_num']:>4} "
            f"leg={row['leg']} "
            f"age={row['age_s']:.3f} "
            f"B={fmt(row['balanced_speed'])} "
            f"S={fmt(row['stability_speed'])} "
            f"E={fmt(row['energy_speed'])}"
        )

    manifest = {
        "schema":
            "icra27_os_t4p5k6h_"
            "u10_k11_perfoot_trace_v0",

        "checkpoint":
            str(ckpt),

        "update":
            UPDATE,

        "terrain":
            TERRAIN,

        "seed":
            SEED,

        "trace_window_s":
            {
                "lower_open":
                    TRACE_TMIN_S,

                "upper_closed":
                    TRACE_TMAX_S,
            },

        "physics_dt_s":
            0.002,

        "physics_rows_per_beta":
            100,

        "betas":
            {
                name:
                    list(
                        beta_lookup[name]
                    )
                for name
                in SELECTED_BETAS
            },

        "episode_rows":
            episode_rows,

        "per_foot_summary":
            summaries,

        "top_stability_high_slip_events":
            events[:20],
    }

    with (
        OUT_DIR
        / "manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    print()
    print("=" * 112)
    print(
        "[ICRA27] OS-T4.5k6h "
        "u10 k11 per-foot 500 Hz trace: PASS"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
