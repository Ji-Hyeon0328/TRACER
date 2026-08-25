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

CONTEXT_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p9a_augmented_causal_context_v0"
    / "augmented_causal_context.csv"
)

CONTEXT_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p9a_augmented_causal_context_v0"
    / "augmented_causal_context_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p10a_oracle_eta_semantic_transfer_v0"
)

OUT_CONTEXT = (
    OUT_DIR
    / "oracle_eta_context_results.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "oracle_eta_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "oracle_eta_semantic_transfer_manifest.json"
)


TCHEBY_RHO = 0.01
EPS = 1.0e-12


ETAS = {
    "balanced":
        np.asarray(
            [1/3, 1/3, 1/3],
            dtype=np.float64,
        ),

    "motion_biased":
        np.asarray(
            [0.70, 0.15, 0.15],
            dtype=np.float64,
        ),

    "stability_biased":
        np.asarray(
            [0.15, 0.70, 0.15],
            dtype=np.float64,
        ),

    "energy_biased":
        np.asarray(
            [0.15, 0.15, 0.70],
            dtype=np.float64,
        ),
}


TARGET_INDEX = {
    "motion_biased": 0,
    "stability_biased": 1,
    "energy_biased": 2,
}


TARGET_NAME = {
    "motion_biased": "motion",
    "stability_biased": "stability",
    "energy_biased": "energy",
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def read_csv(path):
    with path.open(
        "r",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

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


def q_from_regret(
    regret,
    eta,
):
    weighted = (
        regret
        * eta[
            None,
            :
        ]
    )

    return (
        np.max(
            weighted,
            axis=1,
        )
        + TCHEBY_RHO
        * np.sum(
            weighted,
            axis=1,
        )
    )


def classify_delta(x):
    if x < -EPS:
        return "improved"
    if x > EPS:
        return "worsened"
    return "tied"


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        T69C_PATH,
        CONTEXT_CSV,
        CONTEXT_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


    context_manifest = json.loads(
        CONTEXT_MANIFEST.read_text()
    )

    if context_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.9a context is not FREEZE_PASS."
        )

    if bool(
        context_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "T6.9a reports heldout use."
        )


    context_rows = read_csv(
        CONTEXT_CSV
    )

    original_contexts = [
        row["context_id"]
        for row in context_rows
        if row["split_group"]
        == "original_rough_train"
    ]

    if len(original_contexts) != 18:
        raise RuntimeError(
            "Expected 18 original TRAIN contexts."
        )


    t69c = load_module(
        T69C_PATH,
        "os_t6p10a_t69c",
    )

    t66 = t69c.load_module(
        t69c.T66_PATH,
        "os_t6p10a_t66",
    )


    physical_rows = t66.read_csv(
        t66.PHYSICAL_CSV
    )


    grouped = {}

    for row in physical_rows:
        cid = row["context_id"]

        if cid in set(
            original_contexts
        ):
            grouped.setdefault(
                cid,
                [],
            ).append(row)


    context_results = []


    for cid in original_contexts:

        rows = sorted(
            grouped[cid],
            key=lambda row:
                row["beta_name"],
        )

        if len(rows) != 21:
            raise RuntimeError(
                f"{cid}: expected 21 beta rows."
            )

        beta_names = [
            row["beta_name"]
            for row in rows
        ]

        beta_vectors = np.stack(
            [
                t69c.beta_vector(name)
                for name in beta_names
            ],
            axis=0,
        )


        J = np.asarray(
            [
                [
                    float(
                        row[
                            "J_motion_s_per_m"
                        ]
                    ),

                    float(
                        row[
                            "J_stability"
                        ]
                    ),

                    float(
                        row[
                            "J_energy_j_per_m"
                        ]
                    ),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )


        _balanced_Q_check, regret = (
            t69c.tcheby_quality(
                J
            )
        )


        selected = {}

        for eta_name, eta in ETAS.items():

            Q = q_from_regret(
                regret,
                eta,
            )

            index = int(
                np.argmin(Q)
            )

            selected[
                eta_name
            ] = {
                "index":
                    index,

                "Q":
                    Q,

                "beta":
                    beta_vectors[
                        index
                    ],

                "J":
                    J[
                        index
                    ],

                "regret":
                    regret[
                        index
                    ],
            }


        bal = selected[
            "balanced"
        ]


        for eta_name, eta in ETAS.items():

            cur = selected[
                eta_name
            ]

            row = {
                "context_id":
                    cid,

                "eta_name":
                    eta_name,

                "eta_motion":
                    float(
                        eta[0]
                    ),

                "eta_stability":
                    float(
                        eta[1]
                    ),

                "eta_energy":
                    float(
                        eta[2]
                    ),

                "selected_beta":
                    beta_names[
                        cur["index"]
                    ],

                "beta_motion":
                    float(
                        cur["beta"][0]
                    ),

                "beta_stability":
                    float(
                        cur["beta"][1]
                    ),

                "beta_energy":
                    float(
                        cur["beta"][2]
                    ),

                "J_motion":
                    float(
                        cur["J"][0]
                    ),

                "J_stability":
                    float(
                        cur["J"][1]
                    ),

                "J_energy":
                    float(
                        cur["J"][2]
                    ),

                "regret_motion":
                    float(
                        cur["regret"][0]
                    ),

                "regret_stability":
                    float(
                        cur["regret"][1]
                    ),

                "regret_energy":
                    float(
                        cur["regret"][2]
                    ),

                "Q_eta":
                    float(
                        cur["Q"][
                            cur["index"]
                        ]
                    ),

                "balanced_selected_beta":
                    beta_names[
                        bal["index"]
                    ],

                "beta_changed_vs_balanced":
                    int(
                        cur["index"]
                        != bal["index"]
                    ),
            }


            if eta_name == "balanced":

                row.update(
                    {
                        "target_objective":
                            "balanced",

                        "delta_target_J_vs_balanced":
                            0.0,

                        "delta_target_regret_vs_balanced":
                            0.0,

                        "target_semantic_result":
                            "reference",

                        "delta_non_target_regret_mean":
                            0.0,

                        "eta_Q_gain_vs_balanced_selection":
                            0.0,
                    }
                )

            else:

                k = TARGET_INDEX[
                    eta_name
                ]

                other = [
                    j
                    for j in range(3)
                    if j != k
                ]

                delta_J = float(
                    cur["J"][k]
                    - bal["J"][k]
                )

                delta_regret = float(
                    cur["regret"][k]
                    - bal["regret"][k]
                )

                delta_non_target = float(
                    np.mean(
                        cur["regret"][
                            other
                        ]
                        - bal["regret"][
                            other
                        ]
                    )
                )

                q_bal_under_eta = float(
                    cur["Q"][
                        bal["index"]
                    ]
                )

                q_selected = float(
                    cur["Q"][
                        cur["index"]
                    ]
                )


                row.update(
                    {
                        "target_objective":
                            TARGET_NAME[
                                eta_name
                            ],

                        "delta_target_J_vs_balanced":
                            delta_J,

                        "delta_target_regret_vs_balanced":
                            delta_regret,

                        "target_semantic_result":
                            classify_delta(
                                delta_regret
                            ),

                        "delta_non_target_regret_mean":
                            delta_non_target,

                        "eta_Q_gain_vs_balanced_selection":
                            (
                                q_selected
                                - q_bal_under_eta
                            ),
                    }
                )


            context_results.append(
                row
            )


    # ========================================================
    # Summary.
    # ========================================================

    summary_rows = []


    for eta_name, eta in ETAS.items():

        rows = [
            row
            for row in context_results
            if row["eta_name"]
            == eta_name
        ]


        summary = {
            "eta_name":
                eta_name,

            "eta_motion":
                float(eta[0]),

            "eta_stability":
                float(eta[1]),

            "eta_energy":
                float(eta[2]),

            "contexts":
                len(rows),

            "beta_changed_fraction":
                float(
                    np.mean(
                        [
                            int(
                                row[
                                    "beta_changed_vs_balanced"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),

            "mean_beta_motion":
                float(
                    np.mean(
                        [
                            float(
                                row[
                                    "beta_motion"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),

            "mean_beta_stability":
                float(
                    np.mean(
                        [
                            float(
                                row[
                                    "beta_stability"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),

            "mean_beta_energy":
                float(
                    np.mean(
                        [
                            float(
                                row[
                                    "beta_energy"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),

            "mean_J_motion":
                float(
                    np.mean(
                        [
                            float(
                                row[
                                    "J_motion"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),

            "mean_J_stability":
                float(
                    np.mean(
                        [
                            float(
                                row[
                                    "J_stability"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),

            "mean_J_energy":
                float(
                    np.mean(
                        [
                            float(
                                row[
                                    "J_energy"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),
        }


        if eta_name != "balanced":

            delta = np.asarray(
                [
                    float(
                        row[
                            "delta_target_regret_vs_balanced"
                        ]
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )

            delta_J = np.asarray(
                [
                    float(
                        row[
                            "delta_target_J_vs_balanced"
                        ]
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )

            non_target = np.asarray(
                [
                    float(
                        row[
                            "delta_non_target_regret_mean"
                        ]
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )


            summary.update(
                {
                    "target_objective":
                        TARGET_NAME[
                            eta_name
                        ],

                    "target_improved_count":
                        int(
                            np.sum(
                                delta
                                < -EPS
                            )
                        ),

                    "target_tied_count":
                        int(
                            np.sum(
                                np.abs(delta)
                                <= EPS
                            )
                        ),

                    "target_worsened_count":
                        int(
                            np.sum(
                                delta
                                > EPS
                            )
                        ),

                    "mean_delta_target_regret":
                        float(
                            np.mean(delta)
                        ),

                    "median_delta_target_regret":
                        float(
                            np.median(delta)
                        ),

                    "mean_delta_target_J":
                        float(
                            np.mean(delta_J)
                        ),

                    "mean_delta_non_target_regret":
                        float(
                            np.mean(
                                non_target
                            )
                        ),
                }
            )


        summary_rows.append(
            summary
        )


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
            "icra27_os_t6p10a_oracle_eta_semantic_transfer_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "contexts":
            18,

        "eta_profiles":
            {
                key:
                    value.tolist()
                for key, value
                in ETAS.items()
            },

        "purpose":
            (
                "Isolate the semantic validity of the "
                "augmented-Tchebycheff Objective Selector "
                "formulation using true physical regret "
                "surfaces, independently of learned "
                "surrogate error."
            ),

        "interpretation":
            (
                "A biased eta is semantically consistent "
                "when its selected beta reduces the "
                "corresponding true physical regret "
                "relative to the balanced-eta selection, "
                "while potentially trading other "
                "objectives."
            ),

        "important_note":
            (
                "The biased eta need not increase the "
                "matching internal beta component. "
                "Semantic transfer is judged by the "
                "resulting physical objective/regret, "
                "not beta-component identity."
            ),

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
    print("=" * 126)
    print(
        "ICRA27 OS-T6.10a ORACLE ETA "
        "SEMANTIC TRANSFER"
    )
    print("=" * 126)

    print(
        " eta                 | changeB | "
        "mean beta [M,S,E]          | "
        "target imp/tie/worse | "
        "dTargetReg  dOtherReg"
    )

    print("-" * 126)


    for row in summary_rows:

        if row[
            "eta_name"
        ] == "balanced":

            print(
                f" {row['eta_name']:<19} | "
                f"{row['beta_changed_fraction']:>7.3f} | "
                f"[{row['mean_beta_motion']:.3f},"
                f"{row['mean_beta_stability']:.3f},"
                f"{row['mean_beta_energy']:.3f}] | "
                f"{'reference':>20} |"
            )

        else:

            print(
                f" {row['eta_name']:<19} | "
                f"{row['beta_changed_fraction']:>7.3f} | "
                f"[{row['mean_beta_motion']:.3f},"
                f"{row['mean_beta_stability']:.3f},"
                f"{row['mean_beta_energy']:.3f}] | "
                f"{row['target_improved_count']:>2}/"
                f"{row['target_tied_count']:>2}/"
                f"{row['target_worsened_count']:>2} "
                f"| "
                f"{row['mean_delta_target_regret']:+.4f} "
                f"{row['mean_delta_non_target_regret']:+.4f}"
            )


    print()
    print(
        "[ICRA27] OS-T6.10a oracle eta "
        "semantic transfer: COMPUTE PASS"
    )

    print("=" * 126)


if __name__ == "__main__":
    main()
