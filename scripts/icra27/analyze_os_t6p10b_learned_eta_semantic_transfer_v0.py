from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PRED_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p9c_augmented_context_regret_surface_loco_v0"
    / "regret_surface_candidate_predictions.csv"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p10b_learned_eta_semantic_transfer_v0"
)

OUT_CONTEXT = OUT_DIR / "learned_eta_context_results.csv"
OUT_SUMMARY = OUT_DIR / "learned_eta_summary.csv"
OUT_MANIFEST = OUT_DIR / "learned_eta_manifest.json"


RHO = 0.01
EPS = 1e-12

ETAS = {
    "balanced":
        np.array([1/3, 1/3, 1/3]),

    "motion_biased":
        np.array([0.70, 0.15, 0.15]),

    "stability_biased":
        np.array([0.15, 0.70, 0.15]),

    "energy_biased":
        np.array([0.15, 0.15, 0.70]),
}

TARGET = {
    "motion_biased": 0,
    "stability_biased": 1,
    "energy_biased": 2,
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )
        writer.writeheader()
        writer.writerows(rows)


def Q(regret, eta):
    weighted = regret * eta[None, :]

    return (
        np.max(weighted, axis=1)
        + RHO * np.sum(weighted, axis=1)
    )


def rank_of(index, values):
    order = np.argsort(
        values,
        kind="stable",
    )

    return int(
        np.where(order == index)[0][0]
    ) + 1


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    rows = read_csv(PRED_CSV)

    reps = sorted(
        set(r["representation"] for r in rows)
    )

    grouped = {}

    for row in rows:
        key = (
            row["representation"],
            row["held_context"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)


    context_results = []


    for (rep, cid), group in sorted(
        grouped.items()
    ):
        group = sorted(
            group,
            key=lambda r: int(r["beta_index"]),
        )

        if len(group) != 21:
            raise RuntimeError(
                f"{rep}/{cid}: expected 21 beta rows"
            )


        true_r = np.asarray(
            [
                [
                    float(r["true_regret_motion"]),
                    float(r["true_regret_stability"]),
                    float(r["true_regret_energy"]),
                ]
                for r in group
            ]
        )

        pred_r = np.asarray(
            [
                [
                    float(r["pred_regret_motion"]),
                    float(r["pred_regret_stability"]),
                    float(r["pred_regret_energy"]),
                ]
                for r in group
            ]
        )

        beta_names = [
            r["beta_name"]
            for r in group
        ]


        selections = {}

        for eta_name, eta in ETAS.items():

            pred_q = Q(
                pred_r,
                eta,
            )

            true_q = Q(
                true_r,
                eta,
            )

            selected = int(
                np.argmin(pred_q)
            )

            oracle = int(
                np.argmin(true_q)
            )

            selections[eta_name] = {
                "selected": selected,
                "oracle": oracle,
                "true_q": true_q,
                "pred_q": pred_q,
            }


        balanced_selected = (
            selections["balanced"]["selected"]
        )


        for eta_name, eta in ETAS.items():

            info = selections[
                eta_name
            ]

            selected = info["selected"]
            oracle = info["oracle"]
            true_q = info["true_q"]

            true_rank = rank_of(
                selected,
                true_q,
            )

            out = {
                "representation": rep,
                "context_id": cid,
                "eta_name": eta_name,

                "selected_beta":
                    beta_names[selected],

                "oracle_beta":
                    beta_names[oracle],

                "true_rank":
                    true_rank,

                "exact":
                    int(true_rank == 1),

                "top3":
                    int(true_rank <= 3),

                "top5":
                    int(true_rank <= 5),

                "Q_excess_eta":
                    float(
                        true_q[selected]
                        - true_q[oracle]
                    ),

                "beta_changed_vs_learned_balanced":
                    int(
                        selected
                        != balanced_selected
                    ),
            }


            if eta_name == "balanced":

                out.update(
                    {
                        "target_improved": 0,
                        "target_tied": 1,
                        "target_worsened": 0,
                        "delta_target_regret": 0.0,
                        "delta_other_regret_mean": 0.0,
                    }
                )

            else:

                k = TARGET[
                    eta_name
                ]

                other = [
                    j for j in range(3)
                    if j != k
                ]

                delta_target = float(
                    true_r[selected, k]
                    - true_r[
                        balanced_selected,
                        k
                    ]
                )

                delta_other = float(
                    np.mean(
                        true_r[
                            selected,
                            other
                        ]
                        - true_r[
                            balanced_selected,
                            other
                        ]
                    )
                )

                out.update(
                    {
                        "target_improved":
                            int(
                                delta_target
                                < -EPS
                            ),

                        "target_tied":
                            int(
                                abs(delta_target)
                                <= EPS
                            ),

                        "target_worsened":
                            int(
                                delta_target
                                > EPS
                            ),

                        "delta_target_regret":
                            delta_target,

                        "delta_other_regret_mean":
                            delta_other,
                    }
                )


            context_results.append(out)


    summary_rows = []


    for rep in reps:
        for eta_name in ETAS:

            selected_rows = [
                r
                for r in context_results
                if (
                    r["representation"] == rep
                    and r["eta_name"] == eta_name
                )
            ]

            q = np.asarray(
                [
                    r["Q_excess_eta"]
                    for r in selected_rows
                ]
            )

            summary = {
                "representation": rep,
                "eta_name": eta_name,
                "contexts": len(selected_rows),

                "beta_changed_fraction":
                    float(
                        np.mean(
                            [
                                r[
                                    "beta_changed_vs_learned_balanced"
                                ]
                                for r in selected_rows
                            ]
                        )
                    ),

                "Q_excess_mean":
                    float(np.mean(q)),

                "Q_excess_median":
                    float(np.median(q)),

                "Q_excess_p95":
                    float(
                        np.percentile(q, 95)
                    ),

                "exact_fraction":
                    float(
                        np.mean(
                            [
                                r["exact"]
                                for r in selected_rows
                            ]
                        )
                    ),

                "top3_fraction":
                    float(
                        np.mean(
                            [
                                r["top3"]
                                for r in selected_rows
                            ]
                        )
                    ),

                "top5_fraction":
                    float(
                        np.mean(
                            [
                                r["top5"]
                                for r in selected_rows
                            ]
                        )
                    ),
            }


            if eta_name != "balanced":

                summary.update(
                    {
                        "target_improved_count":
                            int(
                                sum(
                                    r["target_improved"]
                                    for r in selected_rows
                                )
                            ),

                        "target_tied_count":
                            int(
                                sum(
                                    r["target_tied"]
                                    for r in selected_rows
                                )
                            ),

                        "target_worsened_count":
                            int(
                                sum(
                                    r["target_worsened"]
                                    for r in selected_rows
                                )
                            ),

                        "mean_delta_target_regret":
                            float(
                                np.mean(
                                    [
                                        r["delta_target_regret"]
                                        for r in selected_rows
                                    ]
                                )
                            ),

                        "mean_delta_other_regret":
                            float(
                                np.mean(
                                    [
                                        r[
                                            "delta_other_regret_mean"
                                        ]
                                        for r in selected_rows
                                    ]
                                )
                            ),
                    }
                )


            summary_rows.append(summary)


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_CONTEXT,
        context_results,
    )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
    )


    manifest = {
        "schema":
            "icra27_os_t6p10b_learned_eta_semantic_transfer_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "purpose":
            (
                "Test whether the learned regret "
                "surrogate preserves eta-dependent "
                "physical semantic tradeoffs."
            ),

        "eta_profiles":
            {
                k: v.tolist()
                for k, v in ETAS.items()
            },

        "summary":
            summary_rows,
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
    print("=" * 132)

    print(
        "ICRA27 OS-T6.10b LEARNED ETA "
        "SEMANTIC TRANSFER"
    )

    print("=" * 132)

    print(
        " representation              eta              "
        "| changeB | imp/tie/worse | "
        "dTarget  dOther | Qmean  Qmed  top5"
    )

    print("-" * 132)


    for row in summary_rows:

        if row["eta_name"] == "balanced":
            continue

        print(
            f" {row['representation']:<27} "
            f"{row['eta_name']:<17} | "
            f"{row['beta_changed_fraction']:>7.3f} | "
            f"{row['target_improved_count']:>2}/"
            f"{row['target_tied_count']:>2}/"
            f"{row['target_worsened_count']:>2} | "
            f"{row['mean_delta_target_regret']:+.3f} "
            f"{row['mean_delta_other_regret']:+.3f} | "
            f"{row['Q_excess_mean']:.3f} "
            f"{row['Q_excess_median']:.3f} "
            f"{row['top5_fraction']:.3f}"
        )


    print()
    print(
        "[ICRA27] OS-T6.10b learned eta "
        "semantic transfer: COMPUTE PASS"
    )

    print("=" * 132)


if __name__ == "__main__":
    main()
