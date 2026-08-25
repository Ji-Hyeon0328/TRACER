#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


PHASE2B_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_phase2b_beta_train_archive_v0.py"
)

T4P1D_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_os_t4p1d_established_slip_distribution_train9_v0.py"
)

DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "icra27"
    / "phase1a_beta_conditioned_v2_seed27027"
    / "phase1_beta_identifiable_selected_u20.pt"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p1d3b_marginal_q95_v0"
    / "u0020_train6"
)

MUS = (
    0.152,
    0.151,
)

SEEDS = (
    27200,
    27201,
    27202,
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {path}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def bindable(port):
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    try:
        sock.bind(
            ("127.0.0.1", port)
        )
        return True

    except OSError:
        return False

    finally:
        sock.close()


def choose_base():
    offsets = (
        0, 1, 2,
        10, 11, 12,
    )

    for base in range(
        64000,
        64900,
        100,
    ):
        if all(
            bindable(base + x)
            for x in offsets
        ):
            return base

    raise RuntimeError(
        "No free UDP layout."
    )


def make_env(
    phase2b,
    *,
    beta,
    mu,
    port,
    log_dir,
):
    base_env = phase2b.PyMPCM7Env(
        terrain="low_friction",
        terrain_friction=float(mu),

        reward_mode=(
            phase2b.EVAL_REWARD_MODE
        ),

        tracer_beta=beta,

        terminate_on_m4_unsafe=True,

        goal_distance_m=(
            phase2b.base.GOAL_DISTANCE_M
        ),

        success_radius_m=(
            phase2b.base.SUCCESS_RADIUS_M
        ),

        decision_dt_s=(
            phase2b.base.DECISION_DT_S
        ),

        max_episode_steps=(
            phase2b.base.POLICY_HORIZON
            + phase2b.base.SETTLING_STEPS
        ),

        command_port=port,
        telemetry_port=port + 1,
        state_port=port + 2,

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=log_dir,
    )

    env = (
        phase2b.M7BetaConditioningWrapper(
            phase2b.M7FixedClearanceActionWrapper(
                base_env
            ),
            beta=beta,
        )
    )

    return env


def main():
    if OUT_DIR.exists():
        raise RuntimeError(
            f"{OUT_DIR} already exists."
        )

    OUT_DIR.mkdir(
        parents=True
    )

    phase2b = load_module(
        PHASE2B_PATH,
        "os_t4p1d3b_phase2b",
    )

    dist = load_module(
        T4P1D_PATH,
        "os_t4p1d3b_dist",
    )

    beta_bank = {
        str(name): tuple(
            float(x)
            for x in beta
        )
        for name, beta
        in phase2b.BETA_BANK
    }

    beta = beta_bank["balanced"]

    policy = phase2b.base.load_policy(
        name="beta_conditioned",
        path=DEFAULT_CHECKPOINT,
        expected_reward_mode="tracer_cost_v2",
    )

    phase2b.validate_checkpoint_beta_bank(
        policy["payload"]
    )

    base_port = choose_base()

    all_eps = []

    print("=" * 96)
    print(
        "OS-T4.1d3b MARGINAL FRICTION Q95 REFINEMENT"
    )
    print("=" * 96)

    for mu_index, mu in enumerate(MUS):
        env = make_env(
            phase2b,
            beta=beta,
            mu=mu,
            port=base_port + 10 * mu_index,
            log_dir=(
                OUT_DIR
                / "env_logs"
                / f"mu_{mu:.3f}"
            ),
        )

        print(
            f"[mu={mu:.3f}]"
        )

        try:
            for seed in SEEDS:
                row = phase2b.base.run_episode(
                    env,

                    policy_name="beta_conditioned",

                    checkpoint=policy["path"],

                    trained_reward_mode=(
                        policy[
                            "trained_reward_mode"
                        ]
                    ),

                    model=policy["model"],

                    group=(
                        f"os_t4p1d3b_mu_{mu:.3f}"
                    ),

                    terrain="low_friction",

                    seed=seed,
                )

                established = (
                    dist.extract_established_histogram(
                        env
                    )
                )

                q95 = (
                    dist.conservative_quantile(
                        established["histogram"],
                        0.95,
                    )
                )

                item = {
                    "mu":
                        float(mu),

                    "seed":
                        int(seed),

                    "success":
                        bool(
                            row.get(
                                "success",
                                False,
                            )
                        ),

                    "status":
                        row.get(
                            "status"
                        ),

                    "progress_m":
                        row.get(
                            "progress_m"
                        ),

                    "q95":
                        q95,

                    "established":
                        established,
                }

                all_eps.append(
                    item
                )

                print(
                    f"  seed={seed} "
                    f"success={item['success']} "
                    f"status={item['status']} "
                    f"Q95-bin=["
                    f"{q95['bin_low_mps']:.3f},"
                    f"{q95['bin_high_mps']}) "
                    f"upper="
                    f"{q95['conservative_upper_mps']:.3f}"
                )

        finally:
            env.close()

    summary = {}

    for mu in MUS:
        subset = [
            item
            for item in all_eps
            if abs(
                item["mu"] - mu
            ) < 1e-12
        ]

        pooled = dist.pool_histograms(
            [
                {
                    "established": {
                        "histogram":
                            item[
                                "established"
                            ][
                                "histogram"
                            ]
                    }
                }
                for item in subset
            ]
        )

        pooled_q90 = (
            dist.conservative_quantile(
                pooled,
                0.90,
            )
        )

        pooled_q95 = (
            dist.conservative_quantile(
                pooled,
                0.95,
            )
        )

        pooled_q975 = (
            dist.conservative_quantile(
                pooled,
                0.975,
            )
        )

        summary[
            f"{mu:.3f}"
        ] = {
            "success_count":
                sum(
                    int(x["success"])
                    for x in subset
                ),

            "q90":
                pooled_q90,

            "q95":
                pooled_q95,

            "q975":
                pooled_q975,

            "per_episode_q95":
                [
                    x["q95"]
                    for x in subset
                ],
        }

    out = {
        "schema":
            "icra27_os_t4p1d3b_marginal_q95_v0",

        "marginal_viable_mu":
            0.152,

        "nearest_failure_mu":
            0.151,

        "results":
            summary,
    }

    path = (
        OUT_DIR
        / "summary.json"
    )

    path.write_text(
        json.dumps(
            out,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 96)
    print("POOLED RESULT")
    print("=" * 96)

    for mu in MUS:
        item = summary[
            f"{mu:.3f}"
        ]

        print(
            f"mu={mu:.3f} "
            f"success={item['success_count']}/3 "
            f"Q90={item['q90']['conservative_upper_mps']} "
            f"Q95={item['q95']['conservative_upper_mps']} "
            f"Q97.5={item['q975']['conservative_upper_mps']}"
        )

    print()
    print("summary:", path)


if __name__ == "__main__":
    main()
