from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

T63A_PATH = (
    ROOT
    / "scripts/icra27"
    / "freeze_os_t6p3a_highres_height_patches_v0.py"
)

T68B_PATH = (
    ROOT
    / "scripts/icra27"
    / "freeze_os_t6p8b_prepolicy_probe_response_atlas_v0.py"
)

T69A_PATH = (
    ROOT
    / "scripts/icra27"
    / "freeze_os_t6p9a_augmented_causal_context_v0.py"
)

PROTOCOL_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p12b_untouched_eval_protocol_v0"
    / "untouched_eval_protocol_manifest.json"
)

FINAL_SELECTOR_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p12a_final_factorized_regret_selector_v0"
    / "final_selector_manifest.json"
)

TRAIN_CONTEXT_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p9a_augmented_causal_context_v0"
    / "augmented_causal_context_manifest.json"
)


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12c_heldout_augmented_causal_context_v0"
)

OUT_PATCH = (
    OUT_DIR
    / "heldout_highres_height_patches.csv"
)

OUT_PROBE = (
    OUT_DIR
    / "heldout_probe_prefix.csv"
)

OUT_CONTEXT = (
    OUT_DIR
    / "heldout_augmented_causal_context.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "heldout_augmented_causal_context_manifest.json"
)


VAL = (
    1,
    21,
    16,
    14,
)

TEST = (
    27,
    2,
    3,
    19,
    26,
)

HARD = (
    9,
    23,
    24,
)

ALL_EVAL = (
    VAL
    + TEST
    + HARD
)


N_LONG = 16
N_LAT = 12
PATCH_DIM = 192

PROBE_STEP = 2
PROBE_TIME_S = 0.4

EXPECTED_CONTEXT_DIM = 87
EXPECTED_CONTEXTS = 12

# Operational port namespace only.
# Scientific probe semantics remain inherited from T6.8b.
BASE_PORT = 64110


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


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def split_name(seed):
    if seed in VAL:
        return "val"

    if seed in TEST:
        return "test"

    if seed in HARD:
        return "hard"

    raise RuntimeError(
        f"Unknown heldout seed: {seed}"
    )


def main():
    # ========================================================
    # Start-of-run immutability guard.
    # ========================================================

    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        T63A_PATH,
        T68B_PATH,
        T69A_PATH,
        PROTOCOL_MANIFEST,
        FINAL_SELECTOR_MANIFEST,
        TRAIN_CONTEXT_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Frozen protocol / selector guards.
    # ========================================================

    protocol = json.loads(
        PROTOCOL_MANIFEST.read_text()
    )

    selector = json.loads(
        FINAL_SELECTOR_MANIFEST.read_text()
    )

    train_context_manifest = json.loads(
        TRAIN_CONTEXT_MANIFEST.read_text()
    )


    if protocol.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.12b protocol is not FREEZE_PASS."
        )

    if selector.get(
        "status"
    ) != "FINAL_TRAIN_PASS":
        raise RuntimeError(
            "T6.12a final selector is not "
            "FINAL_TRAIN_PASS."
        )

    if train_context_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.9a TRAIN context is not FREEZE_PASS."
        )


    if bool(
        protocol.get(
            "heldout_outcomes_used",
            True,
        )
    ):
        raise RuntimeError(
            "Protocol indicates heldout outcomes "
            "were already used."
        )

    if bool(
        selector.get(
            "heldout_used",
            True,
        )
    ):
        raise RuntimeError(
            "Final selector reports heldout use."
        )


    frozen_splits = protocol[
        "evaluation_splits"
    ]

    if frozen_splits[
        "val"
    ] != list(VAL):
        raise RuntimeError(
            "VAL split drift."
        )

    if frozen_splits[
        "test"
    ] != list(TEST):
        raise RuntimeError(
            "TEST split drift."
        )

    if frozen_splits[
        "hard"
    ] != list(HARD):
        raise RuntimeError(
            "HARD split drift."
        )


    if len(
        set(
            ALL_EVAL
        )
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Heldout seeds are not 12 unique seeds."
        )


    # ========================================================
    # Load exact frozen implementation modules.
    # ========================================================

    t63a = load_module(
        T63A_PATH,
        "os_t6p12c_t63a",
    )

    t68b = load_module(
        T68B_PATH,
        "os_t6p12c_t68b",
    )

    t69a = load_module(
        T69A_PATH,
        "os_t6p12c_t69a",
    )

    t66 = t69a.load_module(
        t69a.T66_PATH,
        "os_t6p12c_t66",
    )


    t55 = t68b.load_module(
        t68b.T55_PATH,
        "os_t6p12c_t55",
    )

    phase = t55.load_module(
        t55.T49_PATH,
        "os_t6p12c_t49",
    )


    # Exact frozen dimensional semantics.
    if int(
        t69a.SEMANTIC_DIM
    ) != 74:
        raise RuntimeError(
            "Semantic dimension drift."
        )

    if int(
        t69a.PROP_DIM
    ) != 6:
        raise RuntimeError(
            "Proprioceptive dimension drift."
        )

    if int(
        t69a.PROBE_DIM
    ) != 7:
        raise RuntimeError(
            "Probe dimension drift."
        )

    if int(
        t69a.PROBE_STEP
    ) != PROBE_STEP:
        raise RuntimeError(
            "Probe step drift."
        )

    if abs(
        float(
            t69a.PROBE_TIME_S
        )
        - PROBE_TIME_S
    ) > 1.0e-12:
        raise RuntimeError(
            "Probe horizon drift."
        )


    if int(
        t63a.N_LONG
    ) != N_LONG:
        raise RuntimeError(
            "High-res longitudinal dimension drift."
        )

    if int(
        t63a.N_LAT
    ) != N_LAT:
        raise RuntimeError(
            "High-res lateral dimension drift."
        )

    if int(
        t63a.PATCH_DIM
    ) != PATCH_DIM:
        raise RuntimeError(
            "High-res patch dimension drift."
        )


    # Explicit patch ordering used by the frozen T6.6a
    # semantic constructor.
    patch_cols = [
        f"h_l{i:02d}_r{j:02d}_m"
        for i in range(N_LONG)
        for j in range(N_LAT)
    ]

    if len(
        patch_cols
    ) != PATCH_DIM:
        raise RuntimeError(
            "Patch-column construction mismatch."
        )


    # ========================================================
    # 1. Heldout high-resolution terrain patches.
    #
    # Uses T6.3a.run_worker(seed), which invokes the exact
    # frozen 16x12 worker and rough_patch_row semantics.
    # No policy rollout here.
    # ========================================================

    patch_rows = []
    patch_lookup = {}

    geometry_hashes = set()


    print()
    print("=" * 118)
    print(
        "T6.12c / PART 1: HELDOUT "
        "HIGH-RES TERRAIN PATCHES"
    )
    print("=" * 118)


    for index, seed in enumerate(
        ALL_EVAL,
        1,
    ):

        row = t63a.run_worker(
            int(seed)
        )

        context_id = (
            f"rough_seed_{seed}"
        )


        if row.get(
            "context_id"
        ) != context_id:
            raise RuntimeError(
                f"{context_id}: worker context mismatch: "
                f"{row.get('context_id')}"
            )


        if int(
            row[
                "seed"
            ]
        ) != seed:
            raise RuntimeError(
                f"{context_id}: worker seed mismatch."
            )


        missing = [
            col
            for col in patch_cols
            if col not in row
        ]

        if missing:
            raise RuntimeError(
                f"{context_id}: missing "
                f"{len(missing)} patch cells."
            )


        if "friction_mu" not in row:
            raise RuntimeError(
                f"{context_id}: missing friction_mu."
            )


        row = dict(
            row
        )

        row[
            "split_group"
        ] = split_name(
            seed
        )


        geometry_hashes.add(
            str(
                row[
                    "geometry_sha256"
                ]
            )
        )

        patch_rows.append(
            row
        )

        patch_lookup[
            context_id
        ] = row


        print(
            f"  {index:02d}/{EXPECTED_CONTEXTS:02d} "
            f"{context_id:<16} "
            f"split={split_name(seed):<4} "
            f"mu={float(row['friction_mu']):.3f} "
            f"range=["
            f"{float(row['patch_min_m']):+.5f},"
            f"{float(row['patch_max_m']):+.5f}]"
        )


    if len(
        patch_rows
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Heldout patch row count mismatch."
        )


    # ========================================================
    # 2. Exact frozen 74D semantic terrain representation.
    # ========================================================

    semantic_lookup = {}


    for seed in ALL_EVAL:

        context_id = (
            f"rough_seed_{seed}"
        )

        source = patch_lookup[
            context_id
        ]


        representations = (
            t66.build_representations(
                source,
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
            74,
        ):
            raise RuntimeError(
                f"{context_id}: expected 74D "
                f"semantic vector, got "
                f"{semantic.shape}"
            )

        if not np.all(
            np.isfinite(
                semantic
            )
        ):
            raise RuntimeError(
                f"{context_id}: non-finite "
                "semantic vector."
            )


        semantic_lookup[
            context_id
        ] = semantic


    # ========================================================
    # 3. Exact frozen active probe, truncated at 0.4 s.
    #
    # Important:
    # - reset
    # - zero normalized action
    # - step 1 = 0.2 s
    # - step 2 = 0.4 s
    # - STOP immediately
    #
    # No 0.6/0.8/1.0 s interaction is performed.
    # ========================================================

    probe_prefix_rows = []
    boundary_lookup = {}

    safety_by_step = {
        0: {},
        1: {},
        2: {},
    }


    print()
    print("=" * 118)
    print(
        "T6.12c / PART 2: HELDOUT "
        "0.4 s CAUSAL PROBE"
    )
    print("=" * 118)


    for context_index, seed in enumerate(
        ALL_EVAL
    ):

        context_id = (
            f"rough_seed_{seed}"
        )

        command_port = (
            BASE_PORT
            + 10
            * context_index
        )

        log_dir = (
            OUT_DIR
            / "env_logs"
            / context_id
        )


        env = phase.make_env(
            terrain="rough_perlin",
            beta=t68b.FIXED_BETA,
            command_port=command_port,
            log_dir=log_dir,
        )


        print()
        print(
            f"  [{context_index + 1:02d}/"
            f"{EXPECTED_CONTEXTS:02d}] "
            f"{context_id} "
            f"split={split_name(seed)}"
        )


        try:
            obs, info = env.reset(
                seed=int(seed)
            )

            obs0 = t68b.base_obs(
                obs
            )

            initial_goal_distance = (
                t68b.finite(
                    "initial_goal_distance",
                    info[
                        "goal_distance"
                    ],
                )
            )


            row0 = t68b.state_row(
                context_id=context_id,
                seed=seed,
                horizon=0,
                obs0=obs0,
                obs=obs0,
                info=info,
                initial_goal_distance=(
                    initial_goal_distance
                ),
            )

            row0[
                "split_group"
            ] = split_name(
                seed
            )

            probe_prefix_rows.append(
                row0
            )


            state = str(
                row0[
                    "safety_state"
                ]
            )

            safety_by_step[
                0
            ][
                state
            ] = (
                safety_by_step[
                    0
                ].get(
                    state,
                    0,
                )
                + 1
            )


            zero = np.zeros(
                env.action_space.shape,
                dtype=np.float32,
            )

            if zero.shape != (
                3,
            ):
                raise RuntimeError(
                    f"{context_id}: expected "
                    "3D learned action space."
                )


            for step in (
                1,
                2,
            ):
                (
                    obs,
                    _reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    zero
                )


                if (
                    terminated
                    or truncated
                ):
                    raise RuntimeError(
                        f"{context_id}: frozen probe "
                        f"terminated/truncated at "
                        f"step={step}, "
                        f"time={step * t68b.DECISION_DT_S:.1f}s"
                    )


                requested = np.asarray(
                    info[
                        "requested_normalized"
                    ],
                    dtype=np.float64,
                )


                if not np.array_equal(
                    requested,
                    np.zeros_like(
                        requested
                    ),
                ):
                    raise RuntimeError(
                        f"{context_id}: probe action "
                        f"drift at step {step}: "
                        f"{requested}"
                    )


                obs_base = t68b.base_obs(
                    obs
                )


                row = t68b.state_row(
                    context_id=context_id,
                    seed=seed,
                    horizon=step,
                    obs0=obs0,
                    obs=obs_base,
                    info=info,
                    initial_goal_distance=(
                        initial_goal_distance
                    ),
                )

                row[
                    "split_group"
                ] = split_name(
                    seed
                )

                probe_prefix_rows.append(
                    row
                )


                state = str(
                    row[
                        "safety_state"
                    ]
                )

                safety_by_step[
                    step
                ][
                    state
                ] = (
                    safety_by_step[
                        step
                    ].get(
                        state,
                        0,
                    )
                    + 1
                )


                print(
                    f"    t="
                    f"{step * t68b.DECISION_DT_S:.1f}s "
                    f"progress="
                    f"{row['probe_progress_m']:+.4f} "
                    f"vx={row['vx_mps']:+.4f} "
                    f"vy={row['vy_mps']:+.4f} "
                    f"yaw={row['yaw_rate_rps']:+.4f} "
                    f"z={row['base_z_m']:+.4f} "
                    f"roll={row['roll_rad']:+.4f} "
                    f"pitch={row['pitch_rad']:+.4f} "
                    f"safety={row['safety_state']}"
                )


                if step == PROBE_STEP:

                    observed_time = float(
                        row[
                            "probe_time_s"
                        ]
                    )

                    if abs(
                        observed_time
                        - PROBE_TIME_S
                    ) > 1.0e-12:
                        raise RuntimeError(
                            f"{context_id}: expected "
                            f"0.4 s boundary, got "
                            f"{observed_time}"
                        )

                    boundary_lookup[
                        context_id
                    ] = row


        finally:
            env.close()


    expected_probe_rows = (
        EXPECTED_CONTEXTS
        * 3
    )

    if len(
        probe_prefix_rows
    ) != expected_probe_rows:
        raise RuntimeError(
            f"Expected {expected_probe_rows} "
            f"probe-prefix rows, got "
            f"{len(probe_prefix_rows)}"
        )

    if len(
        boundary_lookup
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Expected 12 probe boundary rows."
        )


    # ========================================================
    # 4. Exact frozen T6.9a 87D concatenation.
    # ========================================================

    context_rows = []
    nonnormal_at_0p4 = []

    semantic_matrix = []
    prop_matrix = []
    probe_matrix = []
    context_matrix = []


    for seed in ALL_EVAL:

        context_id = (
            f"rough_seed_{seed}"
        )

        semantic = semantic_lookup[
            context_id
        ]

        source = boundary_lookup[
            context_id
        ]


        prop = np.asarray(
            [
                t69a.finite_float(
                    source[
                        name
                    ],
                    name=(
                        f"{context_id}.{name}"
                    ),
                )
                for name
                in t69a.PROP_COLUMNS
            ],
            dtype=np.float64,
        )


        probe = np.asarray(
            [
                t69a.finite_float(
                    source[
                        name
                    ],
                    name=(
                        f"{context_id}.{name}"
                    ),
                )
                for name
                in t69a.PROBE_COLUMNS
            ],
            dtype=np.float64,
        )


        if prop.shape != (
            6,
        ):
            raise RuntimeError(
                f"{context_id}: prop dimension "
                f"mismatch {prop.shape}"
            )

        if probe.shape != (
            7,
        ):
            raise RuntimeError(
                f"{context_id}: probe dimension "
                f"mismatch {probe.shape}"
            )


        r87 = np.concatenate(
            (
                semantic,
                prop,
                probe,
            )
        )


        if r87.shape != (
            EXPECTED_CONTEXT_DIM,
        ):
            raise RuntimeError(
                f"{context_id}: expected 87D, "
                f"got {r87.shape}"
            )

        if not np.all(
            np.isfinite(
                r87
            )
        ):
            raise RuntimeError(
                f"{context_id}: non-finite "
                "87D context."
            )


        row = {
            "context_id":
                context_id,

            "seed":
                int(seed),

            "split_group":
                split_name(
                    seed
                ),

            "probe_step":
                PROBE_STEP,

            "probe_time_s":
                PROBE_TIME_S,

            # Diagnostic only.
            "probe_safety_state":
                str(
                    source[
                        "safety_state"
                    ]
                ),

            "semantic_dim":
                74,

            "prop_dim":
                6,

            "probe_dim":
                7,

            "semantic_plus_prop_probe_dim":
                int(
                    r87.shape[
                        0
                    ]
                ),

            "geometry_sha256":
                str(
                    patch_lookup[
                        context_id
                    ][
                        "geometry_sha256"
                    ]
                ),

            "semantic_sha256":
                t69a.vector_hash(
                    semantic
                ),

            "semantic_plus_prop_probe_sha256":
                t69a.vector_hash(
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
            t69a.PROP_COLUMNS,
            prop,
        ):
            row[
                f"prop_{name}"
            ] = float(
                value
            )


        for name, value in zip(
            t69a.PROBE_COLUMNS,
            probe,
        ):
            row[
                f"probe_{name}"
            ] = float(
                value
            )


        context_rows.append(
            row
        )

        semantic_matrix.append(
            semantic
        )

        prop_matrix.append(
            prop
        )

        probe_matrix.append(
            probe
        )

        context_matrix.append(
            r87
        )


        if str(
            source[
                "safety_state"
            ]
        ) != "normal":
            nonnormal_at_0p4.append(
                context_id
            )


    if len(
        context_rows
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Final heldout context row count "
            "mismatch."
        )


    semantic_matrix = np.stack(
        semantic_matrix,
        axis=0,
    )

    prop_matrix = np.stack(
        prop_matrix,
        axis=0,
    )

    probe_matrix = np.stack(
        probe_matrix,
        axis=0,
    )

    context_matrix = np.stack(
        context_matrix,
        axis=0,
    )


    if context_matrix.shape != (
        EXPECTED_CONTEXTS,
        EXPECTED_CONTEXT_DIM,
    ):
        raise RuntimeError(
            f"Unexpected heldout context "
            f"matrix shape: "
            f"{context_matrix.shape}"
        )


    # ========================================================
    # 5. Persist only causal context artifacts.
    #
    # OUT_DIR already exists because make_env() created
    # env_logs beneath it. Retain the same guarded storage
    # semantics used by repaired T6.8b.
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    for path in (
        OUT_PATCH,
        OUT_PROBE,
        OUT_CONTEXT,
        OUT_MANIFEST,
    ):
        if path.exists():
            raise RuntimeError(
                f"REFUSING TO OVERWRITE: {path}"
            )


    write_csv(
        OUT_PATCH,
        patch_rows,
    )

    write_csv(
        OUT_PROBE,
        probe_prefix_rows,
    )

    write_csv(
        OUT_CONTEXT,
        context_rows,
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


    safety_summary = {}

    for step in (
        0,
        1,
        2,
    ):
        safety_summary[
            str(
                step
            )
        ] = {
            "time_s":
                float(
                    step
                    * t68b.DECISION_DT_S
                ),

            "counts":
                safety_by_step[
                    step
                ],
        }


    manifest = {
        "schema":
            "icra27_os_t6p12c_heldout_augmented_causal_context_v0",

        "status":
            "FREEZE_PASS",

        "heldout_outcomes_used":
            False,

        "contexts":
            EXPECTED_CONTEXTS,

        "splits": {
            "val":
                list(
                    VAL
                ),

            "test":
                list(
                    TEST
                ),

            "hard":
                list(
                    HARD
                ),
        },

        "context_ids": [
            f"rough_seed_{seed}"
            for seed in ALL_EVAL
        ],

        "representation":
            "semantic_plus_prop_probe",

        "context_dim":
            EXPECTED_CONTEXT_DIM,

        "components": {
            "semantic":
                74,

            "absolute_proprioceptive_boundary_state":
                6,

            "probe_response":
                7,
        },

        "probe": {
            "step":
                PROBE_STEP,

            "time_s":
                PROBE_TIME_S,

            "decision_dt_s":
                float(
                    t68b.DECISION_DT_S
                ),

            "normalized_action":
                [
                    0.0,
                    0.0,
                    0.0,
                ],

            "physical_semantics":
                (
                    "Inherited unchanged from the "
                    "frozen T6.8b beta-independent "
                    "nominal-motion probe."
                ),

            "executed_prefix_only":
                (
                    "reset -> 0.2 s -> 0.4 s -> stop"
                ),

            "no_post_0p4_probe_steps":
                True,
        },

        "semantic_source":
            (
                "Frozen T6.6a build_representations() "
                "semantic_regions output."
            ),

        "highres_patch_source":
            (
                "Frozen T6.3a run_worker(seed), "
                "16x12 resolution."
            ),

        "unique_geometry_hashes":
            len(
                geometry_hashes
            ),

        "safety_by_causal_prefix":
            safety_summary,

        "nonnormal_at_0p4s":
            nonnormal_at_0p4,

        "nonnormal_at_0p4s_count":
            len(
                nonnormal_at_0p4
            ),

        "probe_safety_state_used_as_feature":
            False,

        "feature_group_ranges":
            group_ranges,

        "explicitly_excluded": [
            "reward components",
            "physical J labels",
            "physical regret labels",
            "Q labels",
            "oracle beta",
            "selected beta",
            "post-0.4 s learned-policy trajectory",
            "future energy",
            "future stability",
            "future slip",
            "terminal success",
            "future M4 outcome",
            "safety_state as selector feature",
        ],

        "causal_boundary":
            (
                "Every selector feature is available "
                "at or before 0.4 s, prior to the first "
                "learned beta-conditioned policy action."
            ),

        "next_step":
            (
                "Apply the already-frozen T6.12a "
                "nine-checkpoint selector and precommit "
                "all eta-specific beta selections before "
                "generating any heldout 21-beta physical "
                "response atlas."
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
    print("=" * 126)
    print(
        "ICRA27 OS-T6.12c HELDOUT "
        "AUGMENTED CAUSAL CONTEXT"
    )
    print("=" * 126)

    print(
        "contexts                         :",
        len(
            context_rows
        ),
    )

    print(
        "VAL / TEST / HARD                :",
        len(VAL),
        "/",
        len(TEST),
        "/",
        len(HARD),
    )

    print(
        "context matrix                   :",
        context_matrix.shape,
    )

    print(
        "semantic / prop / probe dims     :",
        "74 / 6 / 7",
    )

    print(
        "probe executed                   :",
        "reset -> 0.2 s -> 0.4 s -> STOP",
    )

    print(
        "unique geometry hashes           :",
        len(
            geometry_hashes
        ),
        "/",
        EXPECTED_CONTEXTS,
    )

    print(
        "non-normal exactly at 0.4 s      :",
        len(
            nonnormal_at_0p4
        ),
        nonnormal_at_0p4,
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
        "[ICRA27] OS-T6.12c heldout causal "
        "context: FREEZE PASS"
    )

    print("=" * 126)


if __name__ == "__main__":
    main()
