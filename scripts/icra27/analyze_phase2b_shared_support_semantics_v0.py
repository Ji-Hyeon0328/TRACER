#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


TERRAINS = (
    "flat",
    "low_friction",
    "rough_perlin",
)

MISSIONS = (
    "fast",
    "stable",
    "efficient",
)

GEOM_TOL = 1e-12


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def clip_polygon(
    polygon,
    a,
    b,
    c,
):
    """
    Keep points satisfying:

        a*x + b*y + c >= 0

    where:
        x = beta_motion
        y = beta_stability
        beta_energy = 1 - x - y
    """

    if not polygon:
        return []

    output = []

    def value(p):
        return (
            a * p[0]
            + b * p[1]
            + c
        )

    for i in range(len(polygon)):
        s = polygon[i]
        e = polygon[
            (i + 1) % len(polygon)
        ]

        fs = value(s)
        fe = value(e)

        s_inside = (
            fs >= -GEOM_TOL
        )

        e_inside = (
            fe >= -GEOM_TOL
        )

        if s_inside and e_inside:
            output.append(e)

        elif s_inside and not e_inside:
            denom = fs - fe

            if abs(denom) > GEOM_TOL:
                t = fs / denom

                output.append(
                    (
                        s[0]
                        + t * (
                            e[0] - s[0]
                        ),
                        s[1]
                        + t * (
                            e[1] - s[1]
                        ),
                    )
                )

        elif (
            not s_inside
            and e_inside
        ):
            denom = fs - fe

            if abs(denom) > GEOM_TOL:
                t = fs / denom

                output.append(
                    (
                        s[0]
                        + t * (
                            e[0] - s[0]
                        ),
                        s[1]
                        + t * (
                            e[1] - s[1]
                        ),
                    )
                )

            output.append(e)

    return output


def polygon_area_centroid(
    polygon,
):
    if len(polygon) < 3:
        return (
            0.0,
            None,
        )

    cross_sum = 0.0
    cx_sum = 0.0
    cy_sum = 0.0

    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[
            (i + 1) % len(polygon)
        ]

        cross = (
            x0 * y1
            - x1 * y0
        )

        cross_sum += cross

        cx_sum += (
            (x0 + x1)
            * cross
        )

        cy_sum += (
            (y0 + y1)
            * cross
        )

    signed_area = (
        0.5
        * cross_sum
    )

    area = abs(
        signed_area
    )

    if area <= GEOM_TOL:
        return (
            0.0,
            None,
        )

    cx = (
        cx_sum
        / (
            6.0
            * signed_area
        )
    )

    cy = (
        cy_sum
        / (
            6.0
            * signed_area
        )
    )

    return (
        area,
        (
            cx,
            cy,
        ),
    )


def build_support_polygon(
    deltas,
):
    # Simplex in:
    #
    # x = beta_motion
    # y = beta_stability
    #
    # beta_energy = 1-x-y
    polygon = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    ]

    for dm, ds, de in deltas:
        # beta^T delta >= 0
        #
        # x*dm + y*ds
        # + (1-x-y)*de >= 0
        #
        # (dm-de)x
        # + (ds-de)y
        # + de >= 0

        polygon = clip_polygon(
            polygon,
            dm - de,
            ds - de,
            de,
        )

        if not polygon:
            break

    return polygon


def dominance_polygon(
    support,
    name,
):
    p = list(support)

    if name == "motion":
        # M >= S
        p = clip_polygon(
            p,
            1.0,
            -1.0,
            0.0,
        )

        # M >= E
        # x >= 1-x-y
        # 2x+y-1 >= 0
        p = clip_polygon(
            p,
            2.0,
            1.0,
            -1.0,
        )

    elif name == "stability":
        # S >= M
        p = clip_polygon(
            p,
            -1.0,
            1.0,
            0.0,
        )

        # S >= E
        # y >= 1-x-y
        p = clip_polygon(
            p,
            1.0,
            2.0,
            -1.0,
        )

    elif name == "energy":
        # E >= M
        # 1-x-y >= x
        p = clip_polygon(
            p,
            -2.0,
            -1.0,
            1.0,
        )

        # E >= S
        p = clip_polygon(
            p,
            -1.0,
            -2.0,
            1.0,
        )

    else:
        raise ValueError(name)

    return p


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--preference-pairs",
        required=True,
    )

    ap.add_argument(
        "--shared-groups",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    pair_path = Path(
        args.preference_pairs
    )

    group_path = Path(
        args.shared_groups
    )

    out_dir = Path(
        args.out_dir
    )

    for path in (
        pair_path,
        group_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_rows = read_csv(
        pair_path
    )

    shared_rows = read_csv(
        group_path
    )

    shared_lookup = {
        (
            row["terrain"],
            row["mission"],
        ):
            row
        for row in shared_rows
    }

    # --------------------------------------------------------
    # Deduplicate repeated critical-budget preference events.
    # --------------------------------------------------------

    unique_pairs = {}

    for row in pair_rows:
        key = (
            row["context_id"],
            row["mission"],
            row["selected_beta_name"],
            row["alternative_beta_name"],
            float(
                row[
                    "delta_cost_motion"
                ]
            ),
            float(
                row[
                    "delta_cost_stability"
                ]
            ),
            float(
                row[
                    "delta_cost_energy"
                ]
            ),
        )

        unique_pairs[
            key
        ] = row

    by_observable = defaultdict(
        list
    )

    for row in unique_pairs.values():
        by_observable[
            (
                row["terrain"],
                row["mission"],
            )
        ].append(
            (
                float(
                    row[
                        "delta_cost_motion"
                    ]
                ),
                float(
                    row[
                        "delta_cost_stability"
                    ]
                ),
                float(
                    row[
                        "delta_cost_energy"
                    ]
                ),
            )
        )

    expected = {
        (
            terrain,
            mission,
        )
        for terrain in TERRAINS
        for mission in MISSIONS
    }

    if set(by_observable) != expected:
        raise RuntimeError(
            "Observable-group mismatch."
        )

    records = []

    simplex_area = 0.5

    for terrain, mission in sorted(
        by_observable
    ):
        deltas = by_observable[
            (
                terrain,
                mission,
            )
        ]

        support = build_support_polygon(
            deltas
        )

        area, centroid = (
            polygon_area_centroid(
                support
            )
        )

        shared = shared_lookup[
            (
                terrain,
                mission,
            )
        ]

        exact_class = shared[
            "shared_exact_class"
        ]

        exact_rho = float(
            shared[
                "shared_exact_rho"
            ]
        )

        boltzmann_mean = (
            float(
                shared[
                    "shared_posterior_mean_motion"
                ]
            ),
            float(
                shared[
                    "shared_posterior_mean_stability"
                ]
            ),
            float(
                shared[
                    "shared_posterior_mean_energy"
                ]
            ),
        )

        if area > 0.0:
            x, y = centroid

            support_mean = (
                x,
                y,
                1.0 - x - y,
            )

            dominance = {}

            for name in (
                "motion",
                "stability",
                "energy",
            ):
                dp = dominance_polygon(
                    support,
                    name,
                )

                d_area, _ = (
                    polygon_area_centroid(
                        dp
                    )
                )

                dominance[
                    name
                ] = (
                    d_area
                    / area
                )

            l1 = sum(
                abs(
                    support_mean[i]
                    - boltzmann_mean[i]
                )
                for i in range(3)
            )

        else:
            support_mean = (
                None,
                None,
                None,
            )

            dominance = {
                "motion":
                    None,
                "stability":
                    None,
                "energy":
                    None,
            }

            l1 = None

        records.append(
            {
                "terrain":
                    terrain,

                "mission":
                    mission,

                "unique_pair_constraint_count":
                    len(deltas),

                "shared_exact_class":
                    exact_class,

                "shared_exact_rho":
                    exact_rho,

                "support_nonempty":
                    int(
                        area > 0.0
                    ),

                "support_area":
                    area,

                "support_area_fraction_of_simplex":
                    (
                        area
                        / simplex_area
                    ),

                "support_uniform_mean_motion":
                    (
                        ""
                        if support_mean[0]
                        is None
                        else support_mean[0]
                    ),

                "support_uniform_mean_stability":
                    (
                        ""
                        if support_mean[1]
                        is None
                        else support_mean[1]
                    ),

                "support_uniform_mean_energy":
                    (
                        ""
                        if support_mean[2]
                        is None
                        else support_mean[2]
                    ),

                "support_motion_dominant_fraction":
                    (
                        ""
                        if dominance[
                            "motion"
                        ] is None
                        else dominance[
                            "motion"
                        ]
                    ),

                "support_stability_dominant_fraction":
                    (
                        ""
                        if dominance[
                            "stability"
                        ] is None
                        else dominance[
                            "stability"
                        ]
                    ),

                "support_energy_dominant_fraction":
                    (
                        ""
                        if dominance[
                            "energy"
                        ] is None
                        else dominance[
                            "energy"
                        ]
                    ),

                "boltzmann_mean_motion":
                    boltzmann_mean[0],

                "boltzmann_mean_stability":
                    boltzmann_mean[1],

                "boltzmann_mean_energy":
                    boltzmann_mean[2],

                "support_vs_boltzmann_l1":
                    (
                        ""
                        if l1 is None
                        else l1
                    ),
            }
        )

    csv_out = (
        out_dir
        / "shared_support_semantics.csv"
    )

    with csv_out.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                records[0]
            ),
        )

        writer.writeheader()

        writer.writerows(records)

    summary = {
        "schema":
            (
                "icra27_phase2b_shared_"
                "support_semantics_v0"
            ),

        "purpose":
            (
                "Separate preference-data "
                "identifiability from Boltzmann "
                "margin-likelihood bias."
            ),

        "support_definition":
            (
                "beta dot "
                "(C_alternative-C_selected) >= 0"
            ),

        "support_measure":
            (
                "exact convex polygon area "
                "inside the beta simplex"
            ),
    }

    summary_path = (
        out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print("=" * 126)
    print(
        "ICRA27 PHASE-2B SHARED SUPPORT-REGION "
        "SEMANTIC AUDIT V0"
    )
    print("=" * 126)

    for r in records:
        if not r[
            "support_nonempty"
        ]:
            print(
                f"{r['terrain']:<14} "
                f"{r['mission']:<10} "
                f"support=EMPTY  "
                f"exact="
                f"{r['shared_exact_class']} "
                f"rho="
                f"{r['shared_exact_rho']:+.6f}  "
                f"Boltz="
                f"[{r['boltzmann_mean_motion']:.3f},"
                f"{r['boltzmann_mean_stability']:.3f},"
                f"{r['boltzmann_mean_energy']:.3f}]"
            )

            continue

        print(
            f"{r['terrain']:<14} "
            f"{r['mission']:<10} "
            f"areaFrac="
            f"{r['support_area_fraction_of_simplex']:.4f}  "
            f"supportMean="
            f"[{float(r['support_uniform_mean_motion']):.3f},"
            f"{float(r['support_uniform_mean_stability']):.3f},"
            f"{float(r['support_uniform_mean_energy']):.3f}]  "
            f"dominance(M/S/E)="
            f"{float(r['support_motion_dominant_fraction']):.3f}/"
            f"{float(r['support_stability_dominant_fraction']):.3f}/"
            f"{float(r['support_energy_dominant_fraction']):.3f}  "
            f"Boltz="
            f"[{r['boltzmann_mean_motion']:.3f},"
            f"{r['boltzmann_mean_stability']:.3f},"
            f"{r['boltzmann_mean_energy']:.3f}]  "
            f"L1="
            f"{float(r['support_vs_boltzmann_l1']):.3f}"
        )

    print()
    print("csv    :", csv_out)
    print("summary:", summary_path)

    print()
    print(
        "[ICRA27] Phase-2B shared support-region "
        "semantic audit: COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
