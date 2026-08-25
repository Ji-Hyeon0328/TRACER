from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

T69C_PATH = (
    ROOT
    / "scripts/icra27"
    / "train_os_t6p9c_augmented_context_regret_surface_loco_v0.py"
)

PRED_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p9c_augmented_context_regret_surface_loco_v0"
    / "regret_surface_candidate_predictions.csv"
)

CONTEXT_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p9a_augmented_causal_context_v0"
    / "augmented_causal_context.csv"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p10c_eta_specific_baseline_audit_v0"
)

OUT_FOLDS = OUT_DIR / "eta_specific_fold_results.csv"
OUT_SUMMARY = OUT_DIR / "eta_specific_summary.csv"
OUT_MANIFEST = OUT_DIR / "eta_specific_manifest.json"


RHO = 0.01
EPS = 1e-12

ETAS = {
    "balanced":
        np.asarray([1/3, 1/3, 1/3]),

    "motion_biased":
        np.asarray([0.70, 0.15, 0.15]),

    "stability_biased":
        np.asarray([0.15, 0.70, 0.15]),

    "energy_biased":
        np.asarray([0.15, 0.15, 0.70]),
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(path)

    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(rows)


def Q(regret, eta):
    weighted = (
        regret
        * eta[None, :]
    )

    return (
        np.max(
            weighted,
            axis=1,
        )
        + RHO
        * np.sum(
            weighted,
            axis=1,
        )
    )


def rank_of(index, values):
    order = np.argsort(
        values,
        kind="stable",
    )

    return (
        int(
            np.where(
                order == index
            )[0][0]
        )
        + 1
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    for p in (
        T69C_PATH,
        PRED_CSV,
        CONTEXT_CSV,
    ):
        if not p.exists():
            raise FileNotFoundError(p)


    t69c = load_module(
        T69C_PATH,
        "t610c_t69c",
    )

    t66 = t69c.load_module(
        t69c.T66_PATH,
        "t610c_t66",
    )


    context_rows = read_csv(
        CONTEXT_CSV
    )

    original = [
        r["context_id"]
        for r in context_rows
        if r["split_group"]
        == "original_rough_train"
    ]

    extension = [
        r["context_id"]
        for r in context_rows
        if r["split_group"]
        == "extension_rough_train"
    ]

    if len(original) != 18:
        raise RuntimeError(
            "Expected 18 original contexts."
        )

    if len(extension) != 15:
        raise RuntimeError(
            "Expected 15 extension contexts."
        )

    all_contexts = (
        original
        + extension
    )


    # --------------------------------------------------------
    # Reconstruct frozen true regret surfaces for all 33
    # TRAIN-side contexts.
    # --------------------------------------------------------

    physical_rows = t66.read_csv(
        t66.PHYSICAL_CSV
    )

    grouped_phys = {}

    for row in physical_rows:
        cid = row["context_id"]

        if cid in set(all_contexts):
            grouped_phys.setdefault(
                cid,
                [],
            ).append(row)


    true_regret = {}
    beta_names = None


    for cid in all_contexts:

        rows = sorted(
            grouped_phys[cid],
            key=lambda r: r["beta_name"],
        )

        if len(rows) != 21:
            raise RuntimeError(
                f"{cid}: expected 21 rows."
            )

        names = [
            r["beta_name"]
            for r in rows
        ]

        if beta_names is None:
            beta_names = names

        elif names != beta_names:
            raise RuntimeError(
                f"{cid}: beta order mismatch."
            )


        J = np.asarray(
            [
                [
                    float(
                        r["J_motion_s_per_m"]
                    ),
                    float(
                        r["J_stability"]
                    ),
                    float(
                        r["J_energy_j_per_m"]
                    ),
                ]
                for r in rows
            ],
            dtype=np.float64,
        )

        _, regret = (
            t69c.tcheby_quality(J)
        )

        true_regret[cid] = regret


    # --------------------------------------------------------
    # Held-context predictions from T6.9c.
    # --------------------------------------------------------

    pred_rows = read_csv(
        PRED_CSV
    )

    pred_lookup = {}

    for row in pred_rows:

        key = (
            row["representation"],
            row["held_context"],
        )

        pred_lookup.setdefault(
            key,
            [],
        ).append(row)


    representations = sorted(
        set(
            r["representation"]
            for r in pred_rows
        )
    )


    fold_rows = []


    for rep in representations:

        for held in original:

            key = (
                rep,
                held,
            )

            rows = sorted(
                pred_lookup[key],
                key=lambda r:
                    int(r["beta_index"]),
            )

            if len(rows) != 21:
                raise RuntimeError(
                    f"{rep}/{held}: "
                    "expected 21 predictions."
                )


            pred_r = np.asarray(
                [
                    [
                        float(
                            r[
                                "pred_regret_motion"
                            ]
                        ),
                        float(
                            r[
                                "pred_regret_stability"
                            ]
                        ),
                        float(
                            r[
                                "pred_regret_energy"
                            ]
                        ),
                    ]
                    for r in rows
                ],
                dtype=np.float64,
            )


            held_true_r = (
                true_regret[held]
            )

            outer_train = (
                [
                    cid
                    for cid in original
                    if cid != held
                ]
                + extension
            )

            if len(outer_train) != 32:
                raise RuntimeError(
                    "Outer TRAIN size mismatch."
                )


            for eta_name, eta in ETAS.items():

                true_Q = Q(
                    held_true_r,
                    eta,
                )

                pred_Q = Q(
                    pred_r,
                    eta,
                )


                oracle_idx = int(
                    np.argmin(true_Q)
                )

                learned_idx = int(
                    np.argmin(pred_Q)
                )


                mean_train_Q = np.mean(
                    np.stack(
                        [
                            Q(
                                true_regret[cid],
                                eta,
                            )
                            for cid
                            in outer_train
                        ],
                        axis=0,
                    ),
                    axis=0,
                )

                baseline_idx = int(
                    np.argmin(
                        mean_train_Q
                    )
                )


                oracle_Q = float(
                    true_Q[
                        oracle_idx
                    ]
                )

                learned_excess = float(
                    true_Q[
                        learned_idx
                    ]
                    - oracle_Q
                )

                baseline_excess = float(
                    true_Q[
                        baseline_idx
                    ]
                    - oracle_Q
                )


                fold_rows.append(
                    {
                        "representation":
                            rep,

                        "held_context":
                            held,

                        "eta_name":
                            eta_name,

                        "oracle_beta":
                            beta_names[
                                oracle_idx
                            ],

                        "learned_beta":
                            beta_names[
                                learned_idx
                            ],

                        "baseline_beta":
                            beta_names[
                                baseline_idx
                            ],

                        "learned_rank":
                            rank_of(
                                learned_idx,
                                true_Q,
                            ),

                        "baseline_rank":
                            rank_of(
                                baseline_idx,
                                true_Q,
                            ),

                        "learned_Q_excess":
                            learned_excess,

                        "baseline_Q_excess":
                            baseline_excess,

                        "learned_beats_baseline":
                            int(
                                learned_excess
                                <
                                baseline_excess
                                - EPS
                            ),

                        "learned_ties_baseline":
                            int(
                                abs(
                                    learned_excess
                                    -
                                    baseline_excess
                                )
                                <= EPS
                            ),

                        "learned_worse_baseline":
                            int(
                                learned_excess
                                >
                                baseline_excess
                                + EPS
                            ),

                        "learned_exact":
                            int(
                                learned_idx
                                == oracle_idx
                            ),

                        "learned_top3":
                            int(
                                rank_of(
                                    learned_idx,
                                    true_Q,
                                )
                                <= 3
                            ),

                        "learned_top5":
                            int(
                                rank_of(
                                    learned_idx,
                                    true_Q,
                                )
                                <= 5
                            ),
                    }
                )


    summary_rows = []


    for rep in representations:

        for eta_name in ETAS:

            rows = [
                r
                for r in fold_rows
                if (
                    r["representation"]
                    == rep
                    and
                    r["eta_name"]
                    == eta_name
                )
            ]


            learned = np.asarray(
                [
                    r[
                        "learned_Q_excess"
                    ]
                    for r in rows
                ],
                dtype=np.float64,
            )

            baseline = np.asarray(
                [
                    r[
                        "baseline_Q_excess"
                    ]
                    for r in rows
                ],
                dtype=np.float64,
            )


            summary_rows.append(
                {
                    "representation":
                        rep,

                    "eta_name":
                        eta_name,

                    "contexts":
                        len(rows),

                    "learned_Q_excess_mean":
                        float(
                            np.mean(
                                learned
                            )
                        ),

                    "learned_Q_excess_median":
                        float(
                            np.median(
                                learned
                            )
                        ),

                    "learned_Q_excess_p95":
                        float(
                            np.percentile(
                                learned,
                                95,
                            )
                        ),

                    "baseline_Q_excess_mean":
                        float(
                            np.mean(
                                baseline
                            )
                        ),

                    "baseline_Q_excess_median":
                        float(
                            np.median(
                                baseline
                            )
                        ),

                    "learned_minus_baseline_mean":
                        float(
                            np.mean(
                                learned
                                - baseline
                            )
                        ),

                    "beat_tie_worse":
                        (
                            f"{sum(r['learned_beats_baseline'] for r in rows)}"
                            f"/"
                            f"{sum(r['learned_ties_baseline'] for r in rows)}"
                            f"/"
                            f"{sum(r['learned_worse_baseline'] for r in rows)}"
                        ),

                    "exact_fraction":
                        float(
                            np.mean(
                                [
                                    r[
                                        "learned_exact"
                                    ]
                                    for r in rows
                                ]
                            )
                        ),

                    "top3_fraction":
                        float(
                            np.mean(
                                [
                                    r[
                                        "learned_top3"
                                    ]
                                    for r in rows
                                ]
                            )
                        ),

                    "top5_fraction":
                        float(
                            np.mean(
                                [
                                    r[
                                        "learned_top5"
                                    ]
                                    for r in rows
                                ]
                            )
                        ),
                }
            )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_FOLDS,
        fold_rows,
    )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
    )


    manifest = {
        "schema":
            "icra27_os_t6p10c_eta_specific_baseline_audit_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "purpose":
            (
                "Compare learned regret-surrogate "
                "selection against an eta-specific "
                "strict-LOCO no-context baseline and "
                "true oracle under identical physical "
                "scalarization."
            ),

        "eta_profiles":
            {
                k:
                    v.tolist()
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
    print("=" * 138)
    print(
        "ICRA27 OS-T6.10c ETA-SPECIFIC "
        "BASELINE AUDIT"
    )
    print("=" * 138)

    print(
        " representation              eta              "
        "| learned/base Qmean | dMean   | "
        "beat/tie/worse | top5"
    )

    print("-" * 138)


    for r in summary_rows:

        print(
            f" {r['representation']:<27} "
            f"{r['eta_name']:<17} | "
            f"{r['learned_Q_excess_mean']:.3f}/"
            f"{r['baseline_Q_excess_mean']:.3f} | "
            f"{r['learned_minus_baseline_mean']:+.3f} | "
            f"{r['beat_tie_worse']:>11} | "
            f"{r['top5_fraction']:.3f}"
        )


    print()
    print(
        "[ICRA27] OS-T6.10c eta-specific "
        "baseline audit: COMPUTE PASS"
    )
    print("=" * 138)


if __name__ == "__main__":
    main()
