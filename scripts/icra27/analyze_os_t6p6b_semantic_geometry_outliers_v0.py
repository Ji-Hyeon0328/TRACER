from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "results/icra27"
    / "os_t6p6a_semantic_geometry_identifiability_v0"
    / "representation_fold_results.csv"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p6b_semantic_geometry_outliers_v0"
)

OUT_CSV = (
    OUT_DIR
    / "paired_context_deltas.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "semantic_geometry_outlier_manifest.json"
)


def read_csv(path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    rows = read_csv(INPUT)

    grouped = {}

    for row in rows:
        grouped.setdefault(
            row["held_context"],
            {},
        )[row["representation"]] = row

    if len(grouped) != 18:
        raise RuntimeError(
            f"Expected 18 held contexts; got {len(grouped)}"
        )

    paired = []

    for context, reps in grouped.items():
        for required in (
            "raw_height",
            "derivative_field",
            "semantic_regions",
        ):
            if required not in reps:
                raise RuntimeError(
                    f"{context}: missing {required}"
                )

        raw = reps["raw_height"]
        der = reps["derivative_field"]
        sem = reps["semantic_regions"]

        row = {
            "held_context":
                context,

            "raw_nn":
                raw["nearest_context"],

            "derivative_nn":
                der["nearest_context"],

            "semantic_nn":
                sem["nearest_context"],

            "raw_Q_excess":
                float(raw["Q_excess"]),

            "derivative_Q_excess":
                float(der["Q_excess"]),

            "semantic_Q_excess":
                float(sem["Q_excess"]),

            "derivative_minus_raw_Q_excess":
                float(der["Q_excess"])
                - float(raw["Q_excess"]),

            "semantic_minus_raw_Q_excess":
                float(sem["Q_excess"])
                - float(raw["Q_excess"]),

            "raw_rho_JS":
                float(raw["rho_J_stability"]),

            "semantic_rho_JS":
                float(sem["rho_J_stability"]),

            "semantic_minus_raw_rho_JS":
                float(sem["rho_J_stability"])
                - float(raw["rho_J_stability"]),

            "raw_rho_JE":
                float(raw["rho_J_energy"]),

            "semantic_rho_JE":
                float(sem["rho_J_energy"]),

            "semantic_minus_raw_rho_JE":
                float(sem["rho_J_energy"])
                - float(raw["rho_J_energy"]),

            "raw_rho_Q":
                float(raw["rho_Q"]),

            "semantic_rho_Q":
                float(sem["rho_Q"]),

            "semantic_minus_raw_rho_Q":
                float(sem["rho_Q"])
                - float(raw["rho_Q"]),

            "raw_rank":
                int(raw["true_rank"]),

            "semantic_rank":
                int(sem["true_rank"]),

            "rank_delta_semantic_minus_raw":
                int(sem["true_rank"])
                - int(raw["true_rank"]),

            "baseline_Q_excess":
                float(raw["baseline_Q_excess"]),
        }

        paired.append(row)


    # Sort worst semantic regressions first.
    paired.sort(
        key=lambda x:
            x["semantic_minus_raw_Q_excess"],
        reverse=True,
    )


    q_delta = np.asarray(
        [
            row["semantic_minus_raw_Q_excess"]
            for row in paired
        ],
        dtype=np.float64,
    )

    rho_s_delta = np.asarray(
        [
            row["semantic_minus_raw_rho_JS"]
            for row in paired
        ],
        dtype=np.float64,
    )

    rho_e_delta = np.asarray(
        [
            row["semantic_minus_raw_rho_JE"]
            for row in paired
        ],
        dtype=np.float64,
    )

    rho_q_delta = np.asarray(
        [
            row["semantic_minus_raw_rho_Q"]
            for row in paired
        ],
        dtype=np.float64,
    )


    aggregate = {
        "contexts":
            len(paired),

        "semantic_lower_Q_excess_contexts":
            int(np.sum(q_delta < 0.0)),

        "semantic_higher_Q_excess_contexts":
            int(np.sum(q_delta > 0.0)),

        "semantic_Q_excess_delta_mean":
            float(np.mean(q_delta)),

        "semantic_Q_excess_delta_median":
            float(np.median(q_delta)),

        "semantic_Q_excess_delta_p10":
            float(np.percentile(q_delta, 10)),

        "semantic_Q_excess_delta_p90":
            float(np.percentile(q_delta, 90)),

        "semantic_rho_JS_delta_mean":
            float(np.mean(rho_s_delta)),

        "semantic_rho_JE_delta_mean":
            float(np.mean(rho_e_delta)),

        "semantic_rho_Q_delta_mean":
            float(np.mean(rho_q_delta)),

        "worst_context":
            paired[0]["held_context"],

        "worst_Q_excess_delta":
            paired[0][
                "semantic_minus_raw_Q_excess"
            ],

        "best_context":
            paired[-1]["held_context"],

        "best_Q_excess_delta":
            paired[-1][
                "semantic_minus_raw_Q_excess"
            ],
    }


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    with OUT_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                paired[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(paired)


    manifest = {
        "schema":
            "icra27_os_t6p6b_semantic_geometry_outliers_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "purpose":
            (
                "Paired per-context audit of the "
                "raw-height versus semantic-terrain "
                "representation to identify whether "
                "the improved median/ranking metrics "
                "are offset by a small number of "
                "catastrophic Q-selection failures."
            ),

        "aggregate":
            aggregate,
    }


    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    print()
    print("=" * 116)
    print(
        "ICRA27 OS-T6.6b SEMANTIC GEOMETRY "
        "PAIRED OUTLIER AUDIT"
    )
    print("=" * 116)

    print()
    print(
        "Worst semantic regressions "
        "(positive ΔQ is worse)"
    )

    for row in paired[:8]:
        print(
            f"{row['held_context']:<16} "
            f"rawNN={row['raw_nn']:<16} "
            f"semNN={row['semantic_nn']:<16} "
            f"rawQ={row['raw_Q_excess']:.4f} "
            f"semQ={row['semantic_Q_excess']:.4f} "
            f"dQ={row['semantic_minus_raw_Q_excess']:+.4f} "
            f"rank={row['raw_rank']}->{row['semantic_rank']} "
            f"dRhoS={row['semantic_minus_raw_rho_JS']:+.3f} "
            f"dRhoE={row['semantic_minus_raw_rho_JE']:+.3f}"
        )

    print()
    print(
        "Best semantic improvements"
    )

    for row in reversed(
        paired[-8:]
    ):
        print(
            f"{row['held_context']:<16} "
            f"rawNN={row['raw_nn']:<16} "
            f"semNN={row['semantic_nn']:<16} "
            f"rawQ={row['raw_Q_excess']:.4f} "
            f"semQ={row['semantic_Q_excess']:.4f} "
            f"dQ={row['semantic_minus_raw_Q_excess']:+.4f} "
            f"rank={row['raw_rank']}->{row['semantic_rank']} "
            f"dRhoS={row['semantic_minus_raw_rho_JS']:+.3f} "
            f"dRhoE={row['semantic_minus_raw_rho_JE']:+.3f}"
        )

    print()
    print("Aggregate")

    for key, value in aggregate.items():
        print(
            f"  {key:<40}: {value}"
        )

    print()
    print(
        "[ICRA27] OS-T6.6b semantic geometry "
        "outlier audit: COMPUTE PASS"
    )
    print("=" * 116)


if __name__ == "__main__":
    main()
