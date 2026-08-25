#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import socket
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


HELPER_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_os_t4p3c_train9_cost_basis_v0.py"
)

TRAIN_ROOT = (
    ROOT
    / "results"
    / "icra27"
    / "phase1a_beta_conditioned_v3_seed27027"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5_checkpoint_refine_v1"
    / "train9_u35"
)

CANDIDATE_UPDATES = (
    35,
)

# Historical Phase-1 semantic gates.
MOTION_MIN = 7
STABILITY_MIN = 6
ENERGY_MIN = 6

# New v3 traction-authority gate.
LOW_TRACTION_MIN = 2


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {path}"
        )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def udp_bindable(port):
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    try:
        sock.bind(
            ("127.0.0.1", int(port))
        )
        return True
    except OSError:
        return False
    finally:
        sock.close()


def choose_base_port():
    offsets = []

    for beta_index in range(4):
        for terrain_index in range(3):
            start = (
                100 * beta_index
                + 10 * terrain_index
            )

            offsets.extend(
                (
                    start,
                    start + 1,
                    start + 2,
                )
            )

    for base in range(
        60000,
        64001,
        500,
    ):
        if all(
            udp_bindable(base + x)
            for x in offsets
        ):
            return base

    raise RuntimeError(
        "No free UDP layout found."
    )


def strict_down(a, b):
    return int(
        float(a) < float(b)
    )


def strict_up(a, b):
    return int(
        float(a) > float(b)
    )


def maybe_down(a, b):
    if a is None or b is None:
        return None

    return strict_down(a, b)


def maybe_up(a, b):
    if a is None or b is None:
        return None

    return strict_up(a, b)


def count_valid(values):
    return sum(
        int(x)
        for x in values
        if x is not None
    )


def denominator(values):
    return sum(
        1
        for x in values
        if x is not None
    )


def metric(row, key):
    value = row.get(key)

    if value is None:
        return None

    return float(value)


def checkpoint_path(update):
    return (
        TRAIN_ROOT
        / f"checkpoint_update_{update:04d}.pt"
    )


def validate_candidate_bank():
    for update in CANDIDATE_UPDATES:
        p = checkpoint_path(update)

        if not p.exists():
            raise RuntimeError(
                f"Missing candidate checkpoint: {p}"
            )


def summarize_candidate(rows):
    by_key = {
        (
            row["terrain_label"],
            int(row["seed"]),
            row["beta_name"],
        ): row
        for row in rows
    }

    motion_cost = []
    motion_vx = []
    motion_time = []

    stability_cost = []

    stability_posture_flat = []
    stability_posture_rough = []

    traction_low = []

    energy_cost = []
    energy_abs = []

    contexts = []

    for terrain_label, _terrain, seeds in (
        helper.TERRAIN_CASES
    ):
        for seed in seeds:
            balanced = by_key[
                (
                    terrain_label,
                    int(seed),
                    "balanced",
                )
            ]

            motion = by_key[
                (
                    terrain_label,
                    int(seed),
                    "motion",
                )
            ]

            stability = by_key[
                (
                    terrain_label,
                    int(seed),
                    "stability",
                )
            ]

            energy = by_key[
                (
                    terrain_label,
                    int(seed),
                    "energy",
                )
            ]

            motion_cost.append(
                strict_down(
                    motion["mean_cost_motion"],
                    balanced["mean_cost_motion"],
                )
            )

            motion_vx.append(
                maybe_up(
                    metric(
                        motion,
                        "mean_applied_vx_mps",
                    ),
                    metric(
                        balanced,
                        "mean_applied_vx_mps",
                    ),
                )
            )

            motion_time.append(
                maybe_down(
                    metric(
                        motion,
                        "decision_time_s",
                    ),
                    metric(
                        balanced,
                        "decision_time_s",
                    ),
                )
            )

            stability_cost.append(
                strict_down(
                    stability[
                        "mean_cost_stability"
                    ],
                    balanced[
                        "mean_cost_stability"
                    ],
                )
            )

            if terrain_label == "flat":
                stability_posture_flat.append(
                    strict_down(
                        stability[
                            "mean_cost_posture"
                        ],
                        balanced[
                            "mean_cost_posture"
                        ],
                    )
                )

            if terrain_label == "rough":
                stability_posture_rough.append(
                    strict_down(
                        stability[
                            "mean_cost_posture"
                        ],
                        balanced[
                            "mean_cost_posture"
                        ],
                    )
                )

            if terrain_label == "low_friction":
                traction_low.append(
                    strict_down(
                        stability[
                            "mean_cost_traction"
                        ],
                        balanced[
                            "mean_cost_traction"
                        ],
                    )
                )

            energy_cost.append(
                strict_down(
                    energy["mean_cost_energy"],
                    balanced["mean_cost_energy"],
                )
            )

            energy_abs.append(
                maybe_down(
                    metric(
                        energy,
                        "energy_abs_j",
                    ),
                    metric(
                        balanced,
                        "energy_abs_j",
                    ),
                )
            )

            contexts.append(
                {
                    "terrain":
                        terrain_label,
                    "seed":
                        int(seed),

                    "motion_cost_down":
                        motion_cost[-1],

                    "motion_vx_up":
                        motion_vx[-1],

                    "motion_time_down":
                        motion_time[-1],

                    "stability_cost_down":
                        stability_cost[-1],

                    "energy_cost_down":
                        energy_cost[-1],
                }
            )

    nm = count_valid(motion_cost)
    ns = count_valid(stability_cost)
    ne = count_valid(energy_cost)
    nt = count_valid(traction_low)

    gate = {
        "motion":
            nm >= MOTION_MIN,

        "stability":
            ns >= STABILITY_MIN,

        "energy":
            ne >= ENERGY_MIN,

        "low_friction_traction":
            nt >= LOW_TRACTION_MIN,
    }

    primary_pass = all(
        gate.values()
    )

    normalized = (
        nm / 9.0,
        ns / 9.0,
        ne / 9.0,
        nt / 3.0,
    )

    success_count = sum(
        int(
            bool(row["success"])
        )
        for row in rows
    )

    m4_count = sum(
        int(
            str(row.get("status", ""))
            .lower()
            .startswith("m4")
        )
        for row in rows
    )

    return {
        "paired_sign_consistency": {
            "motion_cost_down":
                f"{nm}/9",

            "motion_vx_up":
                (
                    f"{count_valid(motion_vx)}/"
                    f"{denominator(motion_vx)}"
                ),

            "motion_time_down":
                (
                    f"{count_valid(motion_time)}/"
                    f"{denominator(motion_time)}"
                ),

            "stability_cost_down":
                f"{ns}/9",

            "stability_posture_flat_down":
                (
                    f"{count_valid(stability_posture_flat)}/3"
                ),

            "stability_posture_rough_down":
                (
                    f"{count_valid(stability_posture_rough)}/3"
                ),

            "low_friction_traction_cost_down":
                f"{nt}/3",

            "energy_cost_down":
                f"{ne}/9",

            "episode_energy_abs_down":
                (
                    f"{count_valid(energy_abs)}/"
                    f"{denominator(energy_abs)}"
                ),
        },

        "primary_counts": {
            "motion_cost_down":
                nm,
            "stability_cost_down":
                ns,
            "energy_cost_down":
                ne,
            "low_friction_traction_cost_down":
                nt,
        },

        "predefined_gate_result":
            gate,

        "gate_pass":
            primary_pass,

        "semantic_worst_ratio":
            min(normalized),

        "semantic_sum_ratio":
            sum(normalized),

        "success_count":
            success_count,

        "episodes":
            len(rows),

        "m4_count":
            m4_count,

        "context_signs":
            contexts,
    }


def rank_key(item):
    """
    Predeclared tie-break.

    1. Gate PASS before FAIL.
    2. More successful deterministic episodes.
    3. Fewer M4 episodes.
    4. Maximize weakest normalized semantic sign ratio.
    5. Maximize total normalized semantic sign ratio.
    6. Earlier update if still tied.
    """

    return (
        int(item["gate_pass"]),
        int(item["success_count"]),
        -int(item["m4_count"]),
        float(item["semantic_worst_ratio"]),
        float(item["semantic_sum_ratio"]),
        -int(item["update"]),
    )


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    global helper

    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists; "
            "refusing overwrite: "
            f"{OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    validate_candidate_bank()

    helper = load_module(
        HELPER_PATH,
        "os_t4p5_t4p3c_helper",
    )

    phase2b = helper.load_module(
        helper.PHASE2B_PATH,
        "os_t4p5_phase2b",
    )

    historical_mode = (
        phase2b.base.EVAL_REWARD_MODE
    )

    if historical_mode != "tracer_cost_v2":
        raise RuntimeError(
            "Unexpected historical common-evaluator mode: "
            f"{historical_mode!r}"
        )

    phase2b.base.EVAL_REWARD_MODE = (
        "tracer_cost_v3"
    )

    beta_bank = tuple(
        (
            str(name),
            tuple(
                float(x)
                for x in beta
            ),
        )
        for name, beta
        in phase2b.BETA_BANK
    )

    expected_names = (
        "balanced",
        "motion",
        "stability",
        "energy",
    )

    if tuple(
        name
        for name, _beta
        in beta_bank
    ) != expected_names:
        raise RuntimeError(
            "Unexpected beta-bank order."
        )

    base_port = choose_base_port()

    print("=" * 112)
    print(
        "ICRA27 OS-T4.5 TRAIN-ONLY "
        "MIDPOINT CHECKPOINT REFINEMENT V1"
    )
    print("=" * 112)

    print(
        "candidates :",
        list(CANDIDATE_UPDATES),
    )
    print(
        "reward     : tracer_cost_v3"
    )
    print(
        "scope      : TRAIN-only diagnostic 9 contexts"
    )
    print(
        "gates      : "
        "Cm 7/9, Cs 6/9, CE 6/9, "
        "low Ctr 2/3"
    )
    print(
        "base port  :",
        base_port,
    )
    print(
        "output     :",
        OUT_DIR,
    )
    print()

    all_episode_rows = []
    candidate_results = []

    for update in CANDIDATE_UPDATES:
        ckpt = checkpoint_path(update)

        print("=" * 112)
        print(
            f"CANDIDATE u{update:02d}: {ckpt.name}"
        )
        print("=" * 112)

        policy = phase2b.base.load_policy(
            name="beta_conditioned",
            path=ckpt,
            expected_reward_mode=(
                "tracer_cost_v3"
            ),
        )

        phase2b.validate_checkpoint_beta_bank(
            policy["payload"]
        )

        candidate_rows = []

        for beta_index, (
            beta_name,
            beta,
        ) in enumerate(beta_bank):

            for terrain_index, (
                terrain_label,
                terrain,
                seeds,
            ) in enumerate(
                helper.TERRAIN_CASES
            ):

                command_port = (
                    base_port
                    + 100 * beta_index
                    + 10 * terrain_index
                )

                log_dir = (
                    OUT_DIR
                    / f"u{update:02d}"
                    / "env_logs"
                    / beta_name
                    / terrain_label
                )

                env, base_env = (
                    helper.make_env(
                        phase2b,
                        terrain=terrain,
                        beta=beta,
                        port=command_port,
                        log_dir=log_dir,
                    )
                )

                try:
                    for seed in seeds:
                        base_row = (
                            phase2b.base.run_episode(
                                env,

                                policy_name=(
                                    "beta_conditioned"
                                ),

                                checkpoint=(
                                    policy["path"]
                                ),

                                trained_reward_mode=(
                                    policy[
                                        "trained_reward_mode"
                                    ]
                                ),

                                model=policy["model"],

                                group=(
                                    f"u{update:02d}_"
                                    f"{terrain_label}"
                                ),

                                terrain=terrain,
                                seed=seed,
                            )
                        )

                        episode_index = int(
                            base_env.episode_index
                        )

                        log_path = (
                            log_dir
                            / (
                                "episode_"
                                f"{episode_index:04d}"
                                ".jsonl"
                            )
                        )

                        all_steps = (
                            helper.read_reward_steps(
                                log_path
                            )
                        )

                        (
                            selected_steps,
                            aggregation_rule,
                        ) = (
                            helper.select_policy_steps(
                                all_steps,
                                base_row,
                            )
                        )

                        cost_summary = (
                            helper.summarize_steps(
                                selected_steps
                            )
                        )

                        row = {
                            "update":
                                int(update),

                            "checkpoint":
                                str(ckpt),

                            "beta_name":
                                beta_name,

                            "terrain_label":
                                terrain_label,

                            "terrain":
                                terrain,

                            "seed":
                                int(seed),

                            "success":
                                bool(
                                    base_row.get(
                                        "success",
                                        False,
                                    )
                                ),

                            "status":
                                base_row.get(
                                    "status"
                                ),

                            "policy_steps":
                                base_row.get(
                                    "policy_steps"
                                ),

                            "decision_time_s":
                                base_row.get(
                                    "decision_time_s"
                                ),

                            "energy_abs_j":
                                base_row.get(
                                    "energy_abs_j"
                                ),

                            "mean_applied_vx_mps":
                                base_row.get(
                                    "mean_applied_vx_mps"
                                ),

                            "aggregation_rule":
                                aggregation_rule,

                            **cost_summary,
                        }

                        candidate_rows.append(
                            row
                        )
                        all_episode_rows.append(
                            row
                        )

                        print(
                            f"  {beta_name:<10} "
                            f"{terrain_label:<13} "
                            f"seed={seed:<6} "
                            f"succ={int(row['success'])} "
                            f"Cm={row['mean_cost_motion']:.4f} "
                            f"Cpost={row['mean_cost_posture']:.4f} "
                            f"Ctr={row['mean_cost_traction']:.4f} "
                            f"Cs={row['mean_cost_stability']:.4f} "
                            f"CE={row['mean_cost_energy']:.4f}"
                        )

                finally:
                    env.close()

        if len(candidate_rows) != 36:
            raise RuntimeError(
                f"u{update:02d}: expected 36 episodes, "
                f"got {len(candidate_rows)}"
            )

        result = summarize_candidate(
            candidate_rows
        )

        result["update"] = int(update)
        result["checkpoint"] = str(ckpt)

        candidate_results.append(
            result
        )

        print()
        print(
            f"u{update:02d} paired signs:"
        )

        for key, value in (
            result[
                "paired_sign_consistency"
            ].items()
        ):
            print(
                f"  {key:<38} {value}"
            )

        print(
            "  gate:",
            "PASS"
            if result["gate_pass"]
            else "FAIL",
        )

        print(
            "  success:",
            f"{result['success_count']}/"
            f"{result['episodes']}",
        )

        print()

    ranked = sorted(
        candidate_results,
        key=rank_key,
        reverse=True,
    )

    passing = [
        x
        for x in ranked
        if x["gate_pass"]
    ]

    selected = (
        passing[0]
        if passing
        else None
    )

    print("=" * 112)
    print("CHECKPOINT SWEEP SUMMARY")
    print("=" * 112)

    print(
        f"{'upd':>5} "
        f"{'Cm':>7} "
        f"{'Cs':>7} "
        f"{'CE':>7} "
        f"{'CtrL':>7} "
        f"{'succ':>8} "
        f"{'worst':>8} "
        f"{'sum':>8} "
        f"{'gate':>7}"
    )

    for x in sorted(
        candidate_results,
        key=lambda z: z["update"],
    ):
        c = x["primary_counts"]

        print(
            f"{x['update']:5d} "
            f"{c['motion_cost_down']:>4}/9 "
            f"{c['stability_cost_down']:>4}/9 "
            f"{c['energy_cost_down']:>4}/9 "
            f"{c['low_friction_traction_cost_down']:>4}/3 "
            f"{x['success_count']:>4}/36 "
            f"{x['semantic_worst_ratio']:8.3f} "
            f"{x['semantic_sum_ratio']:8.3f} "
            f"{'PASS' if x['gate_pass'] else 'FAIL':>7}"
        )

    print()

    if selected is None:
        print(
            "[ICRA27] OS-T4.5 checkpoint semantic gate: FAIL"
        )
        print(
            "No candidate satisfies all predefined gates."
        )
        print(
            "Do NOT relax thresholds post-hoc."
        )
    else:
        print(
            "[ICRA27] OS-T4.5 checkpoint semantic gate: PASS"
        )
        print(
            "selected candidate:",
            f"u{selected['update']:02d}",
        )
        print(
            "checkpoint:",
            selected["checkpoint"],
        )

    manifest = {
        "schema":
            "icra27_os_t4p5_checkpoint_refine_u35_v1",

        "scope":
            "TRAIN-only midpoint refinement after coarse sweep v0 FAIL",

        "refinement_protocol":
            (
                "Predeclared midpoint u35 between coarse candidates "
                "u30 and u40. Gates are unchanged from sweep v0."
            ),

        "reward_mode":
            "tracer_cost_v3",

        "candidate_updates":
            list(CANDIDATE_UPDATES),

        "terrain_cases":
            [
                {
                    "label": label,
                    "terrain": terrain,
                    "seeds": list(seeds),
                }
                for label, terrain, seeds
                in helper.TERRAIN_CASES
            ],

        "predefined_gate": {
            "motion_cost_down_min":
                "7/9",

            "stability_cost_down_min":
                "6/9",

            "energy_cost_down_min":
                "6/9",

            "low_friction_traction_cost_down_min":
                "2/3",
        },

        "tie_break": [
            "gate_pass",
            "success_count",
            "fewer_m4",
            "max_semantic_worst_ratio",
            "max_semantic_sum_ratio",
            "earlier_update",
        ],

        "candidates":
            candidate_results,

        "gate_result":
            (
                "PASS"
                if selected is not None
                else "FAIL"
            ),

        "selected_update":
            (
                int(selected["update"])
                if selected is not None
                else None
            ),

        "selected_checkpoint":
            (
                selected["checkpoint"]
                if selected is not None
                else None
            ),

        "selection_note":
            (
                "Recommendation only; checkpoint is not "
                "copied/frozen by this evaluator."
            ),
    }

    manifest_path = (
        OUT_DIR
        / "checkpoint_sweep_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    write_csv(
        OUT_DIR / "episodes.csv",
        all_episode_rows,
    )

    print()
    print(
        "manifest:",
        manifest_path,
    )

    print(
        "episodes:",
        OUT_DIR / "episodes.csv",
    )


if __name__ == "__main__":
    main()
