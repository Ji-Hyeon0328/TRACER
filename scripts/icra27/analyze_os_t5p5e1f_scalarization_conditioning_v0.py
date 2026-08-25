from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

ATLAS = (
    ROOT
    / "results/icra27"
    / "os_t5p5c_expanded_physical_atlas_v0"
    / "physical_beta_response_atlas.csv"
)

LABELS = (
    ROOT
    / "results/icra27"
    / "os_t5p5d_expanded_preference_labels_v0"
    / "preference_to_beta_labels.csv"
)

TRANSFER = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1d_context_identifiability_v2_v0"
    / "nearest_context_preference_transfer.csv"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1f_scalarization_conditioning_v0"
)

OUT_CONTEXT = (
    OUT_DIR
    / "context_conditioning.csv"
)

OUT_LABEL = (
    OUT_DIR
    / "label_margin_audit.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "conditioning_manifest.json"
)


OBJ = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)

RHO = 0.01
TOL = 1e-10
EPS = 1e-15


def finite(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(x)

    return y


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("no rows")

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(rows)


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x - x.mean()
    y = y - y.mean()

    d = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if d <= EPS:
        return float("nan")

    return float(
        np.dot(x, y) / d
    )


def pareto(rows):
    v = np.asarray(
        [
            [finite(r[k]) for k in OBJ]
            for r in rows
        ],
        dtype=float,
    )

    keep = []

    for i in range(len(rows)):
        dominated = False

        for j in range(len(rows)):
            if i == j:
                continue

            no_worse = np.all(
                v[j] <= v[i] + TOL
            )

            strictly = np.any(
                v[j] < v[i] - TOL
            )

            if no_worse and strictly:
                dominated = True
                break

        if not dominated:
            keep.append(i)

    return [
        rows[i]
        for i in keep
    ]


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    for p in (
        ATLAS,
        LABELS,
        TRANSFER,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    atlas = read_csv(ATLAS)
    labels = read_csv(LABELS)
    transfer = read_csv(TRANSFER)

    rough = sorted(
        {
            r["context_id"]
            for r in atlas
            if r["context_id"].startswith(
                "rough_seed_"
            )
        },
        key=lambda x: int(
            x.rsplit("_", 1)[1]
        ),
    )

    if len(rough) != 18:
        raise RuntimeError(
            f"expected 18 rough contexts, got {len(rough)}"
        )

    by_context = defaultdict(list)

    for row in atlas:
        by_context[
            row["context_id"]
        ].append(row)

    labels_by_context = defaultdict(list)

    for row in labels:
        labels_by_context[
            row["context_id"]
        ].append(row)

    transfer_by_context = defaultdict(list)

    for row in transfer:
        transfer_by_context[
            row["test_context"]
        ].append(row)


    context_rows = []
    label_rows = []

    inv_motion = []
    inv_stability = []
    inv_energy = []
    transfer_mean = []


    for cid in rough:
        front = pareto(
            by_context[cid]
        )

        values = np.asarray(
            [
                [
                    finite(row[k])
                    for k in OBJ
                ]
                for row in front
            ],
            dtype=float,
        )

        ideal = values.min(axis=0)
        nadir = values.max(axis=0)
        span = nadir - ideal

        if np.any(span < -TOL):
            raise RuntimeError(
                f"negative span: {cid}"
            )

        trows = (
            transfer_by_context[cid]
        )

        mean_transfer_excess = float(
            np.mean(
                [
                    finite(
                        r["score_excess"]
                    )
                    for r in trows
                ]
            )
        )

        context_rows.append(
            {
                "context_id":
                    cid,

                "pareto_count":
                    len(front),

                "motion_ideal":
                    ideal[0],

                "motion_nadir":
                    nadir[0],

                "motion_span":
                    span[0],

                "stability_ideal":
                    ideal[1],

                "stability_nadir":
                    nadir[1],

                "stability_span":
                    span[1],

                "energy_ideal":
                    ideal[2],

                "energy_nadir":
                    nadir[2],

                "energy_span":
                    span[2],

                "inverse_motion_span":
                    (
                        1.0
                        / max(
                            span[0],
                            EPS,
                        )
                    ),

                "inverse_stability_span":
                    (
                        1.0
                        / max(
                            span[1],
                            EPS,
                        )
                    ),

                "inverse_energy_span":
                    (
                        1.0
                        / max(
                            span[2],
                            EPS,
                        )
                    ),

                "v2_transfer_score_excess_mean":
                    mean_transfer_excess,
            }
        )

        inv_motion.append(
            1.0
            / max(
                span[0],
                EPS,
            )
        )

        inv_stability.append(
            1.0
            / max(
                span[1],
                EPS,
            )
        )

        inv_energy.append(
            1.0
            / max(
                span[2],
                EPS,
            )
        )

        transfer_mean.append(
            mean_transfer_excess
        )


        # ----------------------------------------------------
        # Oracle-label score margin:
        # best versus second best on the context Pareto front.
        # ----------------------------------------------------

        for label in (
            labels_by_context[cid]
        ):
            w = np.asarray(
                [
                    finite(
                        label[
                            "w_motion"
                        ]
                    ),
                    finite(
                        label[
                            "w_stability"
                        ]
                    ),
                    finite(
                        label[
                            "w_energy"
                        ]
                    ),
                ],
                dtype=float,
            )

            scores = []

            for row in front:
                j = np.asarray(
                    [
                        finite(row[k])
                        for k in OBJ
                    ],
                    dtype=float,
                )

                regret = np.zeros(
                    3,
                    dtype=float,
                )

                for k in range(3):
                    if span[k] > TOL:
                        regret[k] = (
                            j[k]
                            - ideal[k]
                        ) / span[k]

                weighted = w * regret

                score = float(
                    weighted.max()
                    + RHO
                    * weighted.sum()
                )

                scores.append(
                    (
                        score,
                        row[
                            "beta_name"
                        ],
                    )
                )

            scores.sort(
                key=lambda x: x[0]
            )

            best_score, best_beta = (
                scores[0]
            )

            if len(scores) >= 2:
                second_score, second_beta = (
                    scores[1]
                )

                margin = (
                    second_score
                    - best_score
                )
            else:
                second_score = best_score
                second_beta = best_beta
                margin = float("inf")

            frozen_beta = (
                label[
                    "selected_beta_name"
                ]
            )

            if (
                best_beta
                != frozen_beta
            ):
                raise RuntimeError(
                    "frozen oracle regression mismatch: "
                    f"{cid}/"
                    f"{label['preference_name']} "
                    f"{best_beta} != {frozen_beta}"
                )

            label_rows.append(
                {
                    "context_id":
                        cid,

                    "preference_name":
                        label[
                            "preference_name"
                        ],

                    "preference_kind":
                        label[
                            "preference_kind"
                        ],

                    "best_beta":
                        best_beta,

                    "best_score":
                        best_score,

                    "second_beta":
                        second_beta,

                    "second_score":
                        second_score,

                    "best_second_margin":
                        margin,

                    "motion_span":
                        span[0],

                    "stability_span":
                        span[1],

                    "energy_span":
                        span[2],
                }
            )


    corr_motion = pearson(
        transfer_mean,
        inv_motion,
    )

    corr_stability = pearson(
        transfer_mean,
        inv_stability,
    )

    corr_energy = pearson(
        transfer_mean,
        inv_energy,
    )

    finite_margins = np.asarray(
        [
            finite(
                r[
                    "best_second_margin"
                ]
            )
            for r in label_rows
            if math.isfinite(
                finite(
                    r[
                        "best_second_margin"
                    ]
                )
            )
        ],
        dtype=float,
    )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_CONTEXT,
        context_rows,
    )

    write_csv(
        OUT_LABEL,
        label_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1f_"
                "scalarization_conditioning_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "rough_contexts":
            len(rough),

        "labels_audited":
            len(label_rows),

        "correlation_transfer_mean_vs_inverse_span":
            {
                "motion":
                    corr_motion,

                "stability":
                    corr_stability,

                "energy":
                    corr_energy,
            },

        "span_min":
            {
                "motion":
                    min(
                        r["motion_span"]
                        for r in context_rows
                    ),

                "stability":
                    min(
                        r["stability_span"]
                        for r in context_rows
                    ),

                "energy":
                    min(
                        r["energy_span"]
                        for r in context_rows
                    ),
            },

        "label_margin":
            {
                "median":
                    float(
                        np.median(
                            finite_margins
                        )
                    ),

                "p10":
                    float(
                        np.quantile(
                            finite_margins,
                            0.10,
                        )
                    ),

                "p05":
                    float(
                        np.quantile(
                            finite_margins,
                            0.05,
                        )
                    ),

                "le_0p001_fraction":
                    float(
                        np.mean(
                            finite_margins
                            <= 0.001
                        )
                    ),

                "le_0p01_fraction":
                    float(
                        np.mean(
                            finite_margins
                            <= 0.01
                        )
                    ),
            },

        "interpretation_guard":
            (
                "Read-only diagnostic. "
                "No frozen T5.4b/T5.5d "
                "scalarization or labels are changed."
            ),

        "heldout_used":
            False,
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
    print("=" * 112)
    print(
        "ICRA27 OS-T5.5e1f "
        "SCALARIZATION CONDITIONING AUDIT"
    )
    print("=" * 112)

    print("rough contexts :", len(rough))
    print("labels audited :", len(label_rows))

    print()
    print("MIN PARETO SPANS")

    print(
        "  motion    :",
        manifest[
            "span_min"
        ][
            "motion"
        ],
    )

    print(
        "  stability :",
        manifest[
            "span_min"
        ][
            "stability"
        ],
    )

    print(
        "  energy    :",
        manifest[
            "span_min"
        ][
            "energy"
        ],
    )

    print()
    print(
        "corr(mean transfer excess, 1/span)"
    )

    print(
        "  motion    :",
        corr_motion,
    )

    print(
        "  stability :",
        corr_stability,
    )

    print(
        "  energy    :",
        corr_energy,
    )

    print()
    print("ORACLE BEST-vs-SECOND MARGIN")

    for key, value in (
        manifest[
            "label_margin"
        ].items()
    ):
        print(
            f"  {key:<24}: "
            f"{value}"
        )

    print()
    print("outputs:")
    print(" ", OUT_CONTEXT)
    print(" ", OUT_LABEL)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5e1f "
        "scalarization conditioning: COMPUTE PASS"
    )

    print("=" * 112)


if __name__ == "__main__":
    main()
