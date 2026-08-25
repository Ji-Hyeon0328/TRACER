from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

T66_PATH = (
    ROOT
    / "scripts/icra27"
    / "analyze_os_t6p6a_semantic_geometry_identifiability_v0.py"
)

PATCH_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p3a_highres_height_patches_v0"
    / "highres_height_patch_manifest.json"
)

PROBE_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p8b_prepolicy_probe_response_atlas_v0"
    / "prepolicy_probe_response_atlas.csv"
)

PROBE_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p8b_prepolicy_probe_response_atlas_v0"
    / "prepolicy_probe_response_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p9a_augmented_causal_context_v0"
)

OUT_CSV = (
    OUT_DIR
    / "augmented_causal_context.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "augmented_causal_context_manifest.json"
)


PROBE_STEP = 2
PROBE_TIME_S = 0.4

SEMANTIC_DIM = 74
PROP_DIM = 6
PROBE_DIM = 7

EXPECTED_DIMS = {
    "semantic_only": 74,
    "semantic_plus_probe": 81,
    "semantic_plus_prop_probe": 87,
}


PROP_COLUMNS = (
    "vx_mps",
    "vy_mps",
    "yaw_rate_rps",
    "base_z_m",
    "roll_rad",
    "pitch_rad",
)

PROBE_COLUMNS = (
    "probe_progress_m",
    "delta_vx_mps",
    "delta_vy_mps",
    "delta_yaw_rate_rps",
    "delta_base_z_m",
    "delta_roll_rad",
    "delta_pitch_rad",
)


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
        writer.writerows(
            rows
        )


def finite_float(
    value,
    *,
    name,
):
    x = float(value)

    if not math.isfinite(x):
        raise RuntimeError(
            f"{name}: non-finite value {x}"
        )

    return x


def vector_hash(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return hashlib.sha256(
        x.tobytes()
    ).hexdigest()


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        T66_PATH,
        PATCH_MANIFEST,
        PROBE_CSV,
        PROBE_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Frozen TRAIN-only context split.
    # ========================================================

    patch_manifest = json.loads(
        PATCH_MANIFEST.read_text()
    )

    if patch_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.3a patch source is not "
            "FREEZE_PASS."
        )

    if bool(
        patch_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "Patch source reports heldout use."
        )


    original_seeds = [
        int(x)
        for x in patch_manifest[
            "original_seeds"
        ]
    ]

    extension_seeds = [
        int(x)
        for x in patch_manifest[
            "extension_seeds"
        ]
    ]

    if len(original_seeds) != 18:
        raise RuntimeError(
            "Expected 18 original rough contexts."
        )

    if len(extension_seeds) != 15:
        raise RuntimeError(
            "Expected 15 extension rough contexts."
        )


    original_contexts = [
        f"rough_seed_{seed}"
        for seed in original_seeds
    ]

    extension_contexts = [
        f"rough_seed_{seed}"
        for seed in extension_seeds
    ]

    all_contexts = (
        original_contexts
        + extension_contexts
    )

    if len(set(all_contexts)) != 33:
        raise RuntimeError(
            "Expected 33 unique TRAIN contexts."
        )


    # ========================================================
    # Rebuild frozen 74D semantic terrain representation.
    # ========================================================

    t66 = load_module(
        T66_PATH,
        "os_t6p9a_t66",
    )

    terrain_rows = t66.read_csv(
        t66.PATCH_CSV
    )

    patch_cols = t66.patch_columns(
        terrain_rows
    )

    terrain_lookup = {
        row[
            "context_id"
        ]:
            row
        for row in terrain_rows
    }


    semantic_lookup = {}

    for cid in all_contexts:
        if cid not in terrain_lookup:
            raise RuntimeError(
                f"Missing terrain row: {cid}"
            )

        representations = (
            t66.build_representations(
                terrain_lookup[
                    cid
                ],
                patch_cols,
            )
        )

        semantic = np.asarray(
            representations[
                "semantic_regions"
            ],
            dtype=np.float64,
        )

        if semantic.shape != (
            SEMANTIC_DIM,
        ):
            raise RuntimeError(
                f"{cid}: expected semantic "
                f"shape {(SEMANTIC_DIM,)}, "
                f"got {semantic.shape}"
            )

        if not np.all(
            np.isfinite(
                semantic
            )
        ):
            raise RuntimeError(
                f"{cid}: semantic vector "
                "contains non-finite values."
            )

        semantic_lookup[
            cid
        ] = semantic


    # ========================================================
    # Frozen 0.4 s beta-independent probe context.
    # ========================================================

    probe_manifest = json.loads(
        PROBE_MANIFEST.read_text()
    )

    if probe_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.8b probe atlas is not "
            "FREEZE_PASS."
        )

    if bool(
        probe_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "Probe atlas reports heldout use."
        )

    if int(
        probe_manifest[
            "contexts"
        ]
    ) != 33:
        raise RuntimeError(
            "Expected 33 probe contexts."
        )

    if not bool(
        probe_manifest[
            "t6p8a_regression"
        ][
            "pass"
        ]
    ):
        raise RuntimeError(
            "T6.8a deterministic regression "
            "was not PASS."
        )


    probe_rows = read_csv(
        PROBE_CSV
    )

    selected_probe_rows = [
        row
        for row in probe_rows
        if int(
            row[
                "probe_step"
            ]
        )
        == PROBE_STEP
    ]

    if len(
        selected_probe_rows
    ) != 33:
        raise RuntimeError(
            "Expected exactly 33 rows "
            f"at probe step {PROBE_STEP}; "
            f"got {len(selected_probe_rows)}"
        )


    probe_lookup = {
        row[
            "context_id"
        ]:
            row
        for row in selected_probe_rows
    }

    if len(
        probe_lookup
    ) != 33:
        raise RuntimeError(
            "Duplicate 0.4 s probe contexts."
        )


    # ========================================================
    # Build frozen context rows.
    # ========================================================

    output_rows = []

    nonnormal_at_horizon = []

    for cid in all_contexts:

        if cid not in probe_lookup:
            raise RuntimeError(
                f"Missing 0.4 s probe row: {cid}"
            )


        source = probe_lookup[
            cid
        ]

        observed_time = finite_float(
            source[
                "probe_time_s"
            ],
            name=(
                f"{cid}.probe_time_s"
            ),
        )

        if abs(
            observed_time
            - PROBE_TIME_S
        ) > 1.0e-12:
            raise RuntimeError(
                f"{cid}: expected "
                f"probe_time_s={PROBE_TIME_S}, "
                f"got {observed_time}"
            )


        prop = np.asarray(
            [
                finite_float(
                    source[name],
                    name=f"{cid}.{name}",
                )
                for name in PROP_COLUMNS
            ],
            dtype=np.float64,
        )

        probe = np.asarray(
            [
                finite_float(
                    source[name],
                    name=f"{cid}.{name}",
                )
                for name in PROBE_COLUMNS
            ],
            dtype=np.float64,
        )

        if prop.shape != (
            PROP_DIM,
        ):
            raise RuntimeError(
                f"{cid}: prop dim mismatch."
            )

        if probe.shape != (
            PROBE_DIM,
        ):
            raise RuntimeError(
                f"{cid}: probe dim mismatch."
            )


        semantic = semantic_lookup[
            cid
        ]

        r74 = semantic.copy()

        r81 = np.concatenate(
            (
                semantic,
                probe,
            )
        )

        r87 = np.concatenate(
            (
                semantic,
                prop,
                probe,
            )
        )


        if r74.shape != (
            EXPECTED_DIMS[
                "semantic_only"
            ],
        ):
            raise RuntimeError(
                f"{cid}: 74D shape mismatch."
            )

        if r81.shape != (
            EXPECTED_DIMS[
                "semantic_plus_probe"
            ],
        ):
            raise RuntimeError(
                f"{cid}: 81D shape mismatch."
            )

        if r87.shape != (
            EXPECTED_DIMS[
                "semantic_plus_prop_probe"
            ],
        ):
            raise RuntimeError(
                f"{cid}: 87D shape mismatch."
            )


        split_group = (
            "original_rough_train"
            if cid in original_contexts
            else "extension_rough_train"
        )


        row = {
            "context_id":
                cid,

            "split_group":
                split_group,

            "probe_step":
                PROBE_STEP,

            "probe_time_s":
                PROBE_TIME_S,

            "probe_safety_state":
                str(
                    source[
                        "safety_state"
                    ]
                ),

            "semantic_dim":
                SEMANTIC_DIM,

            "prop_dim":
                PROP_DIM,

            "probe_dim":
                PROBE_DIM,

            "semantic_only_dim":
                int(
                    r74.shape[0]
                ),

            "semantic_plus_probe_dim":
                int(
                    r81.shape[0]
                ),

            "semantic_plus_prop_probe_dim":
                int(
                    r87.shape[0]
                ),

            "semantic_sha256":
                vector_hash(
                    r74
                ),

            "semantic_plus_probe_sha256":
                vector_hash(
                    r81
                ),

            "semantic_plus_prop_probe_sha256":
                vector_hash(
                    r87
                ),
        }


        for i, value in enumerate(
            semantic
        ):
            row[
                f"semantic_{i:03d}"
            ] = float(
                value
            )


        for name, value in zip(
            PROP_COLUMNS,
            prop,
        ):
            row[
                f"prop_{name}"
            ] = float(
                value
            )


        for name, value in zip(
            PROBE_COLUMNS,
            probe,
        ):
            row[
                f"probe_{name}"
            ] = float(
                value
            )


        output_rows.append(
            row
        )


        if (
            str(
                source[
                    "safety_state"
                ]
            )
            != "normal"
        ):
            nonnormal_at_horizon.append(
                cid
            )


    if len(
        output_rows
    ) != 33:
        raise RuntimeError(
            f"Expected 33 output rows, "
            f"got {len(output_rows)}"
        )


    # ========================================================
    # Dataset-level checks.
    # ========================================================

    semantic_matrix = np.stack(
        [
            semantic_lookup[
                cid
            ]
            for cid in all_contexts
        ],
        axis=0,
    )

    prop_matrix = np.stack(
        [
            np.asarray(
                [
                    float(
                        probe_lookup[
                            cid
                        ][
                            name
                        ]
                    )
                    for name in PROP_COLUMNS
                ],
                dtype=np.float64,
            )
            for cid in all_contexts
        ],
        axis=0,
    )

    probe_matrix = np.stack(
        [
            np.asarray(
                [
                    float(
                        probe_lookup[
                            cid
                        ][
                            name
                        ]
                    )
                    for name in PROBE_COLUMNS
                ],
                dtype=np.float64,
            )
            for cid in all_contexts
        ],
        axis=0,
    )


    group_ranges = {
        "semantic": {
            "min":
                float(
                    np.min(
                        semantic_matrix
                    )
                ),

            "max":
                float(
                    np.max(
                        semantic_matrix
                    )
                ),
        },

        "proprioceptive": {
            "min":
                float(
                    np.min(
                        prop_matrix
                    )
                ),

            "max":
                float(
                    np.max(
                        prop_matrix
                    )
                ),
        },

        "probe_response": {
            "min":
                float(
                    np.min(
                        probe_matrix
                    )
                ),

            "max":
                float(
                    np.max(
                        probe_matrix
                    )
                ),
        },
    }


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_CSV,
        output_rows,
    )


    manifest = {
        "schema":
            "icra27_os_t6p9a_augmented_causal_context_v0",

        "status":
            "FREEZE_PASS",

        "heldout_used":
            False,

        "contexts":
            33,

        "original_rough_train":
            18,

        "extension_rough_train":
            15,

        "selected_probe_step":
            PROBE_STEP,

        "selected_probe_time_s":
            PROBE_TIME_S,

        "probe_horizon_selection_semantics":
            (
                "0.4 s was selected from TRAIN-only "
                "T6.8c diagnostic results as the "
                "short interaction horizon with the "
                "best observed selector-quality / "
                "safety-exposure tradeoff. It was "
                "not predeclared before T6.8c."
            ),

        "representations":
            {
                "semantic_only": {
                    "dim":
                        74,

                    "components":
                        [
                            "semantic terrain descriptor"
                        ],
                },

                "semantic_plus_probe": {
                    "dim":
                        81,

                    "components":
                        [
                            "74D semantic terrain descriptor",
                            "7D 0.4 s interaction response",
                        ],
                },

                "semantic_plus_prop_probe": {
                    "dim":
                        87,

                    "components":
                        [
                            "74D semantic terrain descriptor",
                            "6D absolute proprioceptive boundary state",
                            "7D 0.4 s interaction response",
                        ],
                },
            },

        "proprioceptive_features":
            list(
                PROP_COLUMNS
            ),

        "interaction_response_features":
            list(
                PROBE_COLUMNS
            ),

        "causal_boundary":
            (
                "All augmented features are available "
                "at or before the end of the fixed "
                "0.4 s beta-independent nominal-motion "
                "probe, prior to the first learned "
                "beta-conditioned policy action."
            ),

        "explicitly_excluded":
            [
                "reward_components",
                "physical J labels",
                "Q labels",
                "selected beta",
                "post-probe learned-policy trajectory",
                "future energy",
                "future stability",
                "future slip",
                "terminal success",
                "M4 future outcome",
                "safety_state as model feature",
            ],

        "probe_safety_state_used_as_feature":
            False,

        "nonnormal_at_0p4s":
            nonnormal_at_horizon,

        "nonnormal_at_0p4s_count":
            len(
                nonnormal_at_horizon
            ),

        "feature_group_ranges":
            group_ranges,

        "source_probe_regression_pass":
            bool(
                probe_manifest[
                    "t6p8a_regression"
                ][
                    "pass"
                ]
            ),

        "scientific_role":
            (
                "Freeze causal Objective Selector "
                "context candidates for controlled "
                "comparison of terrain-only, "
                "terrain+interaction, and "
                "terrain+absolute proprioception+"
                "interaction representations."
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
    print(
        "=" * 118
    )

    print(
        "ICRA27 OS-T6.9a AUGMENTED "
        "CAUSAL CONTEXT FREEZE"
    )

    print(
        "=" * 118
    )

    print(
        "contexts                              :",
        len(
            output_rows
        ),
    )

    print(
        "original rough TRAIN                  :",
        len(
            original_contexts
        ),
    )

    print(
        "extension rough TRAIN                 :",
        len(
            extension_contexts
        ),
    )

    print(
        "probe horizon                         :",
        f"{PROBE_TIME_S:.1f} s",
    )

    print(
        "representation dims                   :",
        EXPECTED_DIMS,
    )

    print(
        "non-normal exactly at 0.4 s           :",
        len(
            nonnormal_at_horizon
        ),
        nonnormal_at_horizon,
    )

    print(
        "source deterministic regression       :",
        probe_manifest[
            "t6p8a_regression"
        ],
    )

    print()
    print(
        "feature group ranges:"
    )

    for name, values in (
        group_ranges.items()
    ):
        print(
            f"  {name:<24} "
            f"[{values['min']:+.6g}, "
            f"{values['max']:+.6g}]"
        )

    print()
    print(
        "[ICRA27] OS-T6.9a augmented "
        "causal context: FREEZE PASS"
    )

    print(
        "=" * 118
    )


if __name__ == "__main__":
    main()
