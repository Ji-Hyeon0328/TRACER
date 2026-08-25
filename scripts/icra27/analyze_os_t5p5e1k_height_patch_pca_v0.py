from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PATCH_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_height_patch_v0"
)

PATCH_CSV = PATCH_ROOT / "oracle_height_patches.csv"
PATCH_CONTRACT = PATCH_ROOT / "oracle_height_patch_contract.json"

E1J_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1j_height_patch_identifiability_v0"
)

RAW_PAIRWISE = E1J_ROOT / "height_patch_pairwise_similarity.csv"
RAW_MANIFEST = E1J_ROOT / "height_patch_identifiability_manifest.json"

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

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1k_height_patch_pca_v0"
)

OUT_SPECTRUM = OUT_DIR / "pca_spectrum.csv"
OUT_SUMMARY = OUT_DIR / "pca_candidate_summary.csv"
OUT_TRANSFER = OUT_DIR / "pca_nearest_transfer.csv"
OUT_MANIFEST = OUT_DIR / "pca_representation_manifest.json"


OBJECTIVES = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)

PATCH_DIM = 48
EXPECTED_ROUGH = 18
EXPECTED_PREFS = 25

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
        raise RuntimeError(f"No rows: {path}")

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denom <= EPS:
        return float("nan")

    return float(
        np.dot(x, y) / denom
    )


def pareto(rows):
    values = np.asarray(
        [
            [finite(row[k]) for k in OBJECTIVES]
            for row in rows
        ],
        dtype=np.float64,
    )

    keep = []

    for i in range(len(rows)):
        dominated = False

        for j in range(len(rows)):
            if i == j:
                continue

            no_worse = np.all(
                values[j] <= values[i] + TOL
            )

            strictly_better = np.any(
                values[j] < values[i] - TOL
            )

            if no_worse and strictly_better:
                dominated = True
                break

        if not dominated:
            keep.append(i)

    return [rows[i] for i in keep]


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    for path in (
        PATCH_CSV,
        PATCH_CONTRACT,
        RAW_PAIRWISE,
        RAW_MANIFEST,
        ATLAS,
        LABELS,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


    # ========================================================
    # Parent contracts
    # ========================================================

    patch_contract = json.loads(
        PATCH_CONTRACT.read_text()
    )

    if patch_contract.get("status") != "FREEZE_PASS":
        raise RuntimeError(
            "Height patch is not FREEZE_PASS."
        )

    if int(patch_contract["patch_dim"]) != PATCH_DIM:
        raise RuntimeError(
            "Unexpected patch dimension."
        )

    raw_manifest = json.loads(
        RAW_MANIFEST.read_text()
    )

    if raw_manifest.get("status") != "COMPUTE_PASS":
        raise RuntimeError(
            "T5.5e1j is not COMPUTE_PASS."
        )

    if bool(raw_manifest["heldout_used"]):
        raise RuntimeError(
            "T5.5e1j used held-out data."
        )


    # ========================================================
    # Rough patch matrix
    # ========================================================

    patch_features = [
        x
        for x in patch_contract["feature_order"]
        if x.startswith("h_l")
    ]

    if len(patch_features) != PATCH_DIM:
        raise RuntimeError(
            "Expected 48 patch features."
        )

    patch_rows = read_csv(PATCH_CSV)

    rough_rows = [
        row
        for row in patch_rows
        if row["context_id"].startswith(
            "rough_seed_"
        )
    ]

    if len(rough_rows) != EXPECTED_ROUGH:
        raise RuntimeError(
            "Expected 18 rough TRAIN contexts."
        )

    rough_contexts = [
        row["context_id"]
        for row in rough_rows
    ]

    X = np.asarray(
        [
            [
                finite(row[f])
                for f in patch_features
            ]
            for row in rough_rows
        ],
        dtype=np.float64,
    )

    if X.shape != (
        EXPECTED_ROUGH,
        PATCH_DIM,
    ):
        raise RuntimeError(
            f"Unexpected X shape: {X.shape}"
        )


    # ========================================================
    # PCA: mean-center only, NO whitening.
    #
    # PCA dimension is selected only from explained terrain
    # variance; beta-response outcomes do not determine k.
    # ========================================================

    x_mean = np.mean(
        X,
        axis=0,
    )

    Xc = X - x_mean

    U, singular_values, Vt = np.linalg.svd(
        Xc,
        full_matrices=False,
    )

    total_energy = float(
        np.sum(
            singular_values ** 2
        )
    )

    if total_energy <= EPS:
        raise RuntimeError(
            "Degenerate patch matrix."
        )

    explained = (
        singular_values ** 2
        / total_energy
    )

    cumulative = np.cumsum(
        explained
    )

    rank = int(
        np.sum(
            singular_values > 1e-12
        )
    )

    if rank < 1:
        raise RuntimeError(
            "Degenerate PCA rank."
        )

    k90 = int(
        np.searchsorted(
            cumulative,
            0.90,
            side="left",
        )
        + 1
    )

    k95 = int(
        np.searchsorted(
            cumulative,
            0.95,
            side="left",
        )
        + 1
    )

    k99 = int(
        np.searchsorted(
            cumulative,
            0.99,
            side="left",
        )
        + 1
    )

    k90 = min(k90, rank)
    k95 = min(k95, rank)
    k99 = min(k99, rank)


    spectrum_rows = []

    for i in range(rank):
        spectrum_rows.append(
            {
                "pc":
                    i + 1,

                "singular_value":
                    float(
                        singular_values[i]
                    ),

                "explained_variance_ratio":
                    float(
                        explained[i]
                    ),

                "cumulative_explained_variance":
                    float(
                        cumulative[i]
                    ),
            }
        )


    # Fixed diagnostic dimensions + unsupervised thresholds.
    candidate_dims = sorted(
        {
            2,
            3,
            4,
            5,
            k90,
            k95,
            k99,
            rank,
        }
    )

    candidate_dims = [
        k
        for k in candidate_dims
        if 1 <= k <= rank
    ]


    # ========================================================
    # Frozen response-distance lookup
    # ========================================================

    raw_pairwise = read_csv(
        RAW_PAIRWISE
    )

    response_lookup = {}
    raw_patch_lookup = {}

    for row in raw_pairwise:
        key = frozenset(
            (
                row["context_a"],
                row["context_b"],
            )
        )

        response_lookup[key] = finite(
            row["response_distance"]
        )

        raw_patch_lookup[key] = finite(
            row[
                "height_patch_rms_distance_m"
            ]
        )


    # ========================================================
    # Frozen atlas + preference labels
    # ========================================================

    atlas = read_csv(ATLAS)

    by_context = defaultdict(list)
    atlas_lookup = {}

    for row in atlas:
        cid = row["context_id"]

        by_context[cid].append(row)

        atlas_lookup[
            (
                cid,
                row["beta_name"],
            )
        ] = row


    labels = read_csv(LABELS)

    label_lookup = {}
    preference_order = []

    first_rough = rough_contexts[0]

    for row in labels:
        key = (
            row["context_id"],
            row["preference_name"],
        )

        label_lookup[key] = row

        if row["context_id"] == first_rough:
            preference_order.append(
                row["preference_name"]
            )

    if len(preference_order) != EXPECTED_PREFS:
        raise RuntimeError(
            "Expected 25 preferences."
        )


    # ========================================================
    # Original frozen T5.5d score
    # ========================================================

    reference = {}

    for cid in rough_contexts:
        front = pareto(
            by_context[cid]
        )

        values = np.asarray(
            [
                [
                    finite(row[k])
                    for k in OBJECTIVES
                ]
                for row in front
            ],
            dtype=np.float64,
        )

        ideal = np.min(
            values,
            axis=0,
        )

        span = (
            np.max(
                values,
                axis=0,
            )
            - ideal
        )

        reference[cid] = (
            ideal,
            span,
        )


    def score(
        *,
        context_id,
        beta_name,
        w,
    ):
        row = atlas_lookup[
            (
                context_id,
                beta_name,
            )
        ]

        values = np.asarray(
            [
                finite(row[k])
                for k in OBJECTIVES
            ],
            dtype=np.float64,
        )

        ideal, span = reference[
            context_id
        ]

        regret = np.zeros(
            3,
            dtype=np.float64,
        )

        for k in range(3):
            if span[k] > TOL:
                regret[k] = (
                    values[k]
                    - ideal[k]
                ) / span[k]

        weighted = w * regret

        return float(
            np.max(weighted)
            + RHO
            * np.sum(weighted)
        )


    # ========================================================
    # Evaluate PCA representations
    # ========================================================

    summary_rows = []
    transfer_rows = []

    full_rank_distance_error = 0.0


    for k in candidate_dims:
        Vk = Vt[:k]

        Z = Xc @ Vk.T

        pair_distances = []
        response_distances = []

        pca_distance_lookup = {}


        for i in range(EXPECTED_ROUGH):
            for j in range(
                i + 1,
                EXPECTED_ROUGH,
            ):
                ci = rough_contexts[i]
                cj = rough_contexts[j]

                key = frozenset(
                    (
                        ci,
                        cj,
                    )
                )

                # Scale by original PATCH_DIM so that full-rank
                # PCA distance equals raw patch RMS distance.
                d = float(
                    np.linalg.norm(
                        Z[i] - Z[j]
                    )
                    / math.sqrt(
                        PATCH_DIM
                    )
                )

                pca_distance_lookup[key] = d

                pair_distances.append(d)
                response_distances.append(
                    response_lookup[key]
                )

                if k == rank:
                    full_rank_distance_error = max(
                        full_rank_distance_error,
                        abs(
                            d
                            - raw_patch_lookup[
                                key
                            ]
                        ),
                    )


        corr = pearson(
            pair_distances,
            response_distances,
        )


        # ----------------------------------------------------
        # 1-NN is diagnostic only.
        # It is NOT the final Objective Selector.
        # ----------------------------------------------------

        nearest = {}

        for i, test_cid in enumerate(
            rough_contexts
        ):
            candidates = []

            for j, train_cid in enumerate(
                rough_contexts
            ):
                if i == j:
                    continue

                key = frozenset(
                    (
                        test_cid,
                        train_cid,
                    )
                )

                candidates.append(
                    (
                        pca_distance_lookup[
                            key
                        ],
                        train_cid,
                    )
                )

            candidates.sort(
                key=lambda x: (
                    x[0],
                    x[1],
                )
            )

            nearest[test_cid] = (
                candidates[0]
            )


        excesses = []
        exacts = []


        for test_cid in rough_contexts:
            distance, source_cid = (
                nearest[test_cid]
            )

            for preference_name in (
                preference_order
            ):
                source = label_lookup[
                    (
                        source_cid,
                        preference_name,
                    )
                ]

                target = label_lookup[
                    (
                        test_cid,
                        preference_name,
                    )
                ]

                transferred_beta = source[
                    "selected_beta_name"
                ]

                oracle_beta = target[
                    "selected_beta_name"
                ]

                w = np.asarray(
                    [
                        finite(
                            target["w_motion"]
                        ),
                        finite(
                            target[
                                "w_stability"
                            ]
                        ),
                        finite(
                            target["w_energy"]
                        ),
                    ],
                    dtype=np.float64,
                )

                oracle_score = finite(
                    target[
                        "scalarization_score"
                    ]
                )

                transferred_score = score(
                    context_id=test_cid,
                    beta_name=(
                        transferred_beta
                    ),
                    w=w,
                )

                excess = (
                    transferred_score
                    - oracle_score
                )

                if excess < -1e-8:
                    raise RuntimeError(
                        "Transferred beta beats "
                        "frozen oracle."
                    )

                excess = max(
                    0.0,
                    excess,
                )

                exact = int(
                    transferred_beta
                    == oracle_beta
                )

                transfer_rows.append(
                    {
                        "pca_dim":
                            k,

                        "test_context":
                            test_cid,

                        "nearest_context":
                            source_cid,

                        "pca_distance":
                            distance,

                        "preference_name":
                            preference_name,

                        "transferred_beta_name":
                            transferred_beta,

                        "oracle_beta_name":
                            oracle_beta,

                        "exact":
                            exact,

                        "score_excess":
                            excess,
                    }
                )

                excesses.append(
                    excess
                )

                exacts.append(
                    exact
                )


        x = np.asarray(
            excesses,
            dtype=np.float64,
        )


        summary_rows.append(
            {
                "pca_dim":
                    k,

                "is_k90":
                    int(
                        k == k90
                    ),

                "is_k95":
                    int(
                        k == k95
                    ),

                "is_k99":
                    int(
                        k == k99
                    ),

                "is_full_rank":
                    int(
                        k == rank
                    ),

                "cumulative_explained_variance":
                    float(
                        cumulative[
                            k - 1
                        ]
                    ),

                "context_response_pearson":
                    corr,

                "nearest_exact_accuracy":
                    float(
                        np.mean(
                            exacts
                        )
                    ),

                "nearest_score_excess_mean":
                    float(
                        np.mean(x)
                    ),

                "nearest_score_excess_median":
                    float(
                        np.median(x)
                    ),

                "nearest_score_excess_p95":
                    float(
                        np.quantile(
                            x,
                            0.95,
                        )
                    ),

                "nearest_score_excess_max":
                    float(
                        np.max(x)
                    ),

                "nearest_score_excess_le_0p05_fraction":
                    float(
                        np.mean(
                            x <= 0.05
                        )
                    ),
            }
        )


    if full_rank_distance_error > 1e-10:
        raise RuntimeError(
            "Full-rank PCA failed raw-patch "
            "distance regression: "
            f"{full_rank_distance_error}"
        )


    # ========================================================
    # Primary representation choice
    #
    # k95 is chosen WITHOUT beta-response outcomes.
    # ========================================================

    k95_summary = next(
        row
        for row in summary_rows
        if row["pca_dim"] == k95
    )


    raw_corr = finite(
        raw_manifest[
            "patch_response_pearson"
        ]
    )

    raw_transfer_mean = finite(
        raw_manifest[
            "nearest_patch_transfer"
        ][
            "score_excess_mean"
        ]
    )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_SPECTRUM,
        spectrum_rows,
    )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
    )

    write_csv(
        OUT_TRANSFER,
        transfer_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1k_"
                "height_patch_pca_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "rough_train_contexts":
            EXPECTED_ROUGH,

        "raw_patch_dim":
            PATCH_DIM,

        "pca_rank":
            rank,

        "k90":
            k90,

        "k95":
            k95,

        "k99":
            k99,

        "primary_representation":
            {
                "name":
                    "patch_pca_k95",

                "dimension":
                    k95,

                "selection_rule":
                    (
                        "smallest PCA dimension "
                        "with >=95% cumulative "
                        "TRAIN rough-patch variance"
                    ),

                "response_outcomes_used_to_choose_dimension":
                    False,

                "whitening":
                    False,

                "mean_centering":
                    True,
            },

        "raw_patch_reference":
            {
                "context_response_pearson":
                    raw_corr,

                "nearest_score_excess_mean":
                    raw_transfer_mean,
            },

        "k95_metrics":
            k95_summary,

        "full_rank_raw_distance_max_error":
            full_rank_distance_error,

        "interpretation_guard":
            (
                "PCA is a simple deterministic "
                "linear terrain encoder baseline. "
                "1-NN remains diagnostic only and "
                "is not the Objective Selector."
            ),

        "loco_guard":
            (
                "For subsequent rough-context LOCO "
                "selector validation, PCA must be "
                "fit independently inside each "
                "outer fold using TRAIN contexts "
                "only. The held-out rough context "
                "must not participate in PCA mean "
                "or basis estimation."
            ),

        "future_interface":
            (
                "PCA latent z_T may later be "
                "replaced by CART/world-model/"
                "learned terrain latent without "
                "changing the Objective Selector "
                "mission-preference interface."
            ),

        "next_stage":
            (
                "Train a small Objective Selector "
                "with fold-local PCA(z_T) + friction "
                "+ w under rough-context LOCO."
            ),
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
    print("=" * 118)
    print(
        "ICRA27 OS-T5.5e1k LOCAL HEIGHT "
        "PATCH PCA AUDIT"
    )
    print("=" * 118)

    print(
        "rough TRAIN contexts:",
        EXPECTED_ROUGH,
    )

    print(
        "raw patch dim       :",
        PATCH_DIM,
    )

    print(
        "PCA rank            :",
        rank,
    )

    print(
        "k90 / k95 / k99     :",
        k90,
        "/",
        k95,
        "/",
        k99,
    )

    print()
    print("PCA CANDIDATES")

    for row in summary_rows:
        flags = []

        if row["is_k90"]:
            flags.append("k90")

        if row["is_k95"]:
            flags.append("k95")

        if row["is_k99"]:
            flags.append("k99")

        if row["is_full_rank"]:
            flags.append("full")

        print(
            f"  k={row['pca_dim']:<2} "
            f"var="
            f"{row['cumulative_explained_variance']:.4f} "
            f"corr="
            f"{row['context_response_pearson']:.4f} "
            f"meanEx="
            f"{row['nearest_score_excess_mean']:.4f} "
            f"p95="
            f"{row['nearest_score_excess_p95']:.4f} "
            f"exact="
            f"{row['nearest_exact_accuracy']:.3f} "
            f"{','.join(flags)}"
        )

    print()
    print("PRIMARY = unsupervised k95")

    print(
        "  dimension:",
        k95,
    )

    print(
        "  explained variance:",
        k95_summary[
            "cumulative_explained_variance"
        ],
    )

    print(
        "  response correlation:",
        k95_summary[
            "context_response_pearson"
        ],
    )

    print(
        "  nearest mean excess:",
        k95_summary[
            "nearest_score_excess_mean"
        ],
    )

    print()
    print(
        "full-rank raw-distance max error:",
        full_rank_distance_error,
    )

    print()
    print("outputs:")
    print(" ", OUT_SPECTRUM)
    print(" ", OUT_SUMMARY)
    print(" ", OUT_TRANSFER)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5e1k "
        "height patch PCA: COMPUTE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
