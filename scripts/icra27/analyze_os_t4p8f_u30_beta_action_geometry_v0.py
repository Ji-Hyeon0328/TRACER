#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SRC = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p8e_u30_beta_specialization_v0"
    / "u30_episode_action_means.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p8f_u30_beta_action_geometry_v0"
)

BETA_ORDER = (
    "balanced",
    "motion",
    "stability",
    "energy",
)


def vector(row):
    return np.asarray(
        [
            float(
                row["mean_requested_norm_vx"]
            ),
            float(
                row["mean_requested_norm_yaw"]
            ),
            float(
                row["mean_requested_norm_height"]
            ),
        ],
        dtype=np.float64,
    )


def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")

    return float(
        np.dot(a, b)
        / (na * nb)
    )


def main():
    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists; "
            f"refusing overwrite: {OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    with SRC.open(
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != 36:
        raise RuntimeError(
            f"Expected 36 rows, got {len(rows)}"
        )

    contexts = {}

    for row in rows:
        key = (
            row["terrain_label"],
            int(row["seed"]),
        )

        contexts.setdefault(
            key,
            {},
        )[
            row["beta_name"]
        ] = row

    if len(contexts) != 9:
        raise RuntimeError(
            f"Expected 9 contexts, got {len(contexts)}"
        )

    context_results = []
    all_centered = []

    for (
        terrain,
        seed,
    ), by_beta in contexts.items():

        if set(by_beta) != set(
            BETA_ORDER
        ):
            raise RuntimeError(
                f"Incomplete beta bank: "
                f"{terrain}/{seed}"
            )

        v = {
            beta:
                vector(by_beta[beta])
            for beta in BETA_ORDER
        }

        b = v["balanced"]

        d_m = v["motion"] - b
        d_s = v["stability"] - b
        d_e = v["energy"] - b

        matrix = np.stack(
            [
                v[beta]
                for beta
                in BETA_ORDER
            ],
            axis=0,
        )

        centered = (
            matrix
            - matrix.mean(
                axis=0,
                keepdims=True,
            )
        )

        all_centered.append(
            centered
        )

        _u, singular, _vt = (
            np.linalg.svd(
                centered,
                full_matrices=False,
            )
        )

        variance = singular ** 2

        if variance.sum() > 0.0:
            explained = (
                variance
                / variance.sum()
            )
        else:
            explained = np.zeros_like(
                variance
            )

        ranges = (
            matrix.max(axis=0)
            - matrix.min(axis=0)
        )

        result = {
            "terrain":
                terrain,
            "seed":
                seed,

            "range_norm_vx":
                float(ranges[0]),

            "range_norm_yaw":
                float(ranges[1]),

            "range_norm_height":
                float(ranges[2]),

            "pc1_explained":
                float(explained[0]),

            "cos_motion_stability":
                cosine(d_m, d_s),

            "cos_motion_energy":
                cosine(d_m, d_e),

            "cos_stability_energy":
                cosine(d_s, d_e),

            "norm_delta_motion":
                float(
                    np.linalg.norm(d_m)
                ),

            "norm_delta_stability":
                float(
                    np.linalg.norm(d_s)
                ),

            "norm_delta_energy":
                float(
                    np.linalg.norm(d_e)
                ),
        }

        context_results.append(
            result
        )

    global_centered = np.concatenate(
        all_centered,
        axis=0,
    )

    _u, singular, vt = np.linalg.svd(
        global_centered,
        full_matrices=False,
    )

    variance = singular ** 2
    explained = (
        variance
        / variance.sum()
    )

    component = vt[0]

    # Sign PC1 so +PC1 corresponds to increasing vx.
    if component[0] < 0.0:
        component = -component

    summary = {
        "schema":
            "icra27_os_t4p8f_u30_beta_action_geometry_v0",

        "contexts":
            len(context_results),

        "global_pc_explained": [
            float(x)
            for x in explained
        ],

        "global_pc1_direction":
            {
                "normalized_vx":
                    float(component[0]),

                "normalized_yaw":
                    float(component[1]),

                "normalized_height":
                    float(component[2]),
            },

        "mean_dimension_range": {
            "normalized_vx":
                float(
                    np.mean(
                        [
                            x["range_norm_vx"]
                            for x in context_results
                        ]
                    )
                ),

            "normalized_yaw":
                float(
                    np.mean(
                        [
                            x["range_norm_yaw"]
                            for x in context_results
                        ]
                    )
                ),

            "normalized_height":
                float(
                    np.mean(
                        [
                            x["range_norm_height"]
                            for x in context_results
                        ]
                    )
                ),
        },

        "mean_cosines": {
            "motion_stability":
                float(
                    np.mean(
                        [
                            x[
                                "cos_motion_stability"
                            ]
                            for x
                            in context_results
                        ]
                    )
                ),

            "motion_energy":
                float(
                    np.mean(
                        [
                            x[
                                "cos_motion_energy"
                            ]
                            for x
                            in context_results
                        ]
                    )
                ),

            "stability_energy":
                float(
                    np.mean(
                        [
                            x[
                                "cos_stability_energy"
                            ]
                            for x
                            in context_results
                        ]
                    )
                ),
        },
    }

    manifest = (
        OUT_DIR
        / "action_geometry_manifest.json"
    )

    manifest.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    csv_path = (
        OUT_DIR
        / "context_action_geometry.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                context_results[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            context_results
        )

    print("=" * 104)
    print(
        "ICRA27 OS-T4.8f u30 "
        "BETA ACTION GEOMETRY AUDIT"
    )
    print("=" * 104)

    print()
    print("GLOBAL PCA")
    print(
        "  PC1 explained : "
        f"{explained[0]:.6f}"
    )
    print(
        "  PC2 explained : "
        f"{explained[1]:.6f}"
    )
    print(
        "  PC3 explained : "
        f"{explained[2]:.6f}"
    )

    print(
        "  PC1 direction : "
        f"[vx={component[0]:+.6f}, "
        f"yaw={component[1]:+.6f}, "
        f"h={component[2]:+.6f}]"
    )

    print()
    print("MEAN DIMENSION RANGE")

    for key, value in (
        summary[
            "mean_dimension_range"
        ].items()
    ):
        print(
            f"  {key:<20} "
            f"{value:.8f}"
        )

    print()
    print("MEAN DELTA COSINES")

    for key, value in (
        summary[
            "mean_cosines"
        ].items()
    ):
        print(
            f"  {key:<20} "
            f"{value:+.6f}"
        )

    print()
    print("PER CONTEXT")

    for row in context_results:
        print(
            f"  {row['terrain']:<12} "
            f"seed={row['seed']:<6} "
            f"PC1={row['pc1_explained']:.4f} "
            f"range[vx/yaw/h]="
            f"{row['range_norm_vx']:.5f}/"
            f"{row['range_norm_yaw']:.5f}/"
            f"{row['range_norm_height']:.5f} "
            f"cos(M,S)="
            f"{row['cos_motion_stability']:+.4f} "
            f"cos(M,E)="
            f"{row['cos_motion_energy']:+.4f}"
        )

    print()
    print("outputs:")
    print(" ", csv_path)
    print(" ", manifest)

    print()
    print(
        "[ICRA27] OS-T4.8f "
        "u30 beta action geometry: COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
