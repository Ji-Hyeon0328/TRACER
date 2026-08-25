#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import freeze_os_t5p5a_oracle_context_descriptor_v1 as base


ROOT = Path(__file__).resolve().parents[2]

SEED_WORKER = (
    ROOT
    / "scripts/icra27"
    / "probe_os_t5p5a_context_seed_v2.py"
)

V1_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_context_descriptor_v1"
)

V1_CONTEXT_CSV = (
    V1_ROOT
    / "oracle_context_descriptors.csv"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_context_descriptor_v2"
)

OUT_CSV = (
    OUT_DIR
    / "oracle_context_descriptors.csv"
)

OUT_CONTRACT = (
    OUT_DIR
    / "oracle_context_contract.json"
)

OUT_GEOMETRY = (
    OUT_DIR
    / "rough_geometry_manifest.json"
)


MIN_SUBREGION_CELLS = 3

BASE_FEATURES = (
    "friction_mu",
    "height_std_m",
    "height_relief_p95_p05_m",
    "slope_rms",
    "slope_q95",
)

ADDED_FEATURES = (
    "slope_longitudinal_rms",
    "slope_lateral_rms",
    "height_front_minus_rear_mean_m",
    "height_left_minus_right_mean_m",
    "height_front_minus_rear_std_m",
    "height_mid_minus_ends_std_m",
)

CONTEXT_FEATURES = (
    BASE_FEATURES
    + ADDED_FEATURES
)

EXPECTED_CONTEXT_DIM = 11
EXPECTED_ROUGH_TRAIN = 18

TOL = 1.0e-10


def finite(
    value: Any,
) -> float:
    return base.finite(
        value
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields: list[str] = []

    for row in rows:
        for key in row:
            if key not in fields:
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
        writer.writerows(rows)


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def rms(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        raise ValueError(
            "Empty RMS input."
        )

    return float(
        math.sqrt(
            np.mean(
                values
                * values
            )
        )
    )


def checked_values(
    array: np.ndarray,
    mask: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    count = int(
        np.sum(mask)
    )

    if count < MIN_SUBREGION_CELLS:
        raise RuntimeError(
            f"{name}: too few cells: "
            f"{count} < "
            f"{MIN_SUBREGION_CELLS}"
        )

    return np.asarray(
        array[
            mask
        ],
        dtype=np.float64,
    )


def spatial_directional_descriptor(
    *,
    hfield,
    start_x: float,
    start_y: float,
    yaw: float,
    hip_height: float,
) -> dict[str, Any]:
    z = hfield[
        "z"
    ]

    xs = hfield[
        "xs"
    ]

    ys = hfield[
        "ys"
    ]

    dx = finite(
        hfield[
            "dx"
        ]
    )

    dy = finite(
        hfield[
            "dy"
        ]
    )

    X, Y = np.meshgrid(
        xs,
        ys,
    )

    c = math.cos(
        yaw
    )

    s = math.sin(
        yaw
    )

    rx = (
        X
        - start_x
    )

    ry = (
        Y
        - start_y
    )

    longitudinal = (
        c * rx
        + s * ry
    )

    lateral = (
        -s * rx
        + c * ry
    )

    margin = (
        base.CORRIDOR_MARGIN_HIP_MULT
        * hip_height
    )

    longitudinal_lo = (
        -margin
    )

    longitudinal_hi = (
        base.GOAL_DISTANCE_M
        + margin
    )

    mask = (
        (
            longitudinal
            >= longitudinal_lo
        )
        & (
            longitudinal
            <= longitudinal_hi
        )
        & (
            np.abs(
                lateral
            )
            <= margin
        )
    )

    corridor_cells = int(
        np.sum(
            mask
        )
    )

    if corridor_cells < (
        base.MIN_CORRIDOR_CELLS
    ):
        raise RuntimeError(
            "Too few corridor cells: "
            f"{corridor_cells}"
        )


    # ========================================================
    # Directional physical slopes
    # ========================================================

    # np.gradient ordering is [y, x].
    dz_dy, dz_dx = np.gradient(
        z,
        dy,
        dx,
    )

    # Rotate world-frame height gradient into the same
    # body/goal-aligned frame used by the corridor.
    dz_dlongitudinal = (
        c * dz_dx
        + s * dz_dy
    )

    dz_dlateral = (
        -s * dz_dx
        + c * dz_dy
    )

    local_longitudinal_slope = (
        dz_dlongitudinal[
            mask
        ]
    )

    local_lateral_slope = (
        dz_dlateral[
            mask
        ]
    )

    slope_longitudinal_rms = rms(
        local_longitudinal_slope
    )

    slope_lateral_rms = rms(
        local_lateral_slope
    )


    # ========================================================
    # Spatial partitions
    #
    # Use longitudinal thirds rather than a fine patch grid:
    # the current Go2 corridor contains only ~35-44 hfield
    # cells, so this retains usable cell counts.
    # ========================================================

    longitudinal_span = (
        longitudinal_hi
        - longitudinal_lo
    )

    rear_end = (
        longitudinal_lo
        + longitudinal_span
        / 3.0
    )

    front_start = (
        longitudinal_lo
        + 2.0
        * longitudinal_span
        / 3.0
    )

    rear_mask = (
        mask
        & (
            longitudinal
            < rear_end
        )
    )

    mid_mask = (
        mask
        & (
            longitudinal
            >= rear_end
        )
        & (
            longitudinal
            <= front_start
        )
    )

    front_mask = (
        mask
        & (
            longitudinal
            > front_start
        )
    )

    # Positive body-frame lateral coordinate is "left".
    left_mask = (
        mask
        & (
            lateral
            >= 0.0
        )
    )

    right_mask = (
        mask
        & (
            lateral
            < 0.0
        )
    )


    rear_h = checked_values(
        z,
        rear_mask,
        name="rear_third",
    )

    mid_h = checked_values(
        z,
        mid_mask,
        name="middle_third",
    )

    front_h = checked_values(
        z,
        front_mask,
        name="front_third",
    )

    left_h = checked_values(
        z,
        left_mask,
        name="left_half",
    )

    right_h = checked_values(
        z,
        right_mask,
        name="right_half",
    )


    front_mean = float(
        np.mean(
            front_h
        )
    )

    rear_mean = float(
        np.mean(
            rear_h
        )
    )

    left_mean = float(
        np.mean(
            left_h
        )
    )

    right_mean = float(
        np.mean(
            right_h
        )
    )


    front_std = float(
        np.std(
            front_h,
            ddof=0,
        )
    )

    mid_std = float(
        np.std(
            mid_h,
            ddof=0,
        )
    )

    rear_std = float(
        np.std(
            rear_h,
            ddof=0,
        )
    )


    return {
        "slope_longitudinal_rms":
            slope_longitudinal_rms,

        "slope_lateral_rms":
            slope_lateral_rms,

        "height_front_minus_rear_mean_m":
            (
                front_mean
                - rear_mean
            ),

        "height_left_minus_right_mean_m":
            (
                left_mean
                - right_mean
            ),

        "height_front_minus_rear_std_m":
            (
                front_std
                - rear_std
            ),

        "height_mid_minus_ends_std_m":
            (
                mid_std
                - 0.5
                * (
                    front_std
                    + rear_std
                )
            ),

        "rear_third_cells":
            int(
                np.sum(
                    rear_mask
                )
            ),

        "middle_third_cells":
            int(
                np.sum(
                    mid_mask
                )
            ),

        "front_third_cells":
            int(
                np.sum(
                    front_mask
                )
            ),

        "left_half_cells":
            int(
                np.sum(
                    left_mask
                )
            ),

        "right_half_cells":
            int(
                np.sum(
                    right_mask
                )
            ),
    }


def rough_row(
    *,
    seed: int,
    robot_name: str,
    hip_height: float,
):
    # Keep the exact T5.5a-v1 seeded Perlin / environment
    # creation semantics.
    with base.SeededPerlinHooks(
        seed
    ):
        env = base.QuadrupedEnv(
            robot=robot_name,
            scene="perlin",
            sim_dt=0.002,
            ground_friction_coeff=0.8,
        )

        try:
            env.reset(
                seed=seed,
                random=True,
            )

            start_x = finite(
                env.mjData.qpos[
                    0
                ]
            )

            start_y = finite(
                env.mjData.qpos[
                    1
                ]
            )

            yaw = base.yaw_from_wxyz(
                env.mjData.qpos[
                    3:7
                ]
            )

            goal_x = (
                start_x
                + base.GOAL_DISTANCE_M
                * math.cos(
                    yaw
                )
            )

            goal_y = (
                start_y
                + base.GOAL_DISTANCE_M
                * math.sin(
                    yaw
                )
            )

            hfield = base.get_hfield(
                env.mjModel
            )

            # Frozen original v1 descriptor.
            old_descriptor = (
                base.local_descriptor(
                    hfield=hfield,
                    start_x=start_x,
                    start_y=start_y,
                    yaw=yaw,
                    hip_height=(
                        hip_height
                    ),
                )
            )

            # Added directional/spatial descriptor.
            extra = (
                spatial_directional_descriptor(
                    hfield=hfield,
                    start_x=start_x,
                    start_y=start_y,
                    yaw=yaw,
                    hip_height=(
                        hip_height
                    ),
                )
            )


            # Rotation preserves gradient norm:
            #
            # RMS(|grad h|)^2
            # = RMS(dh/dlong)^2 + RMS(dh/dlat)^2
            directional_norm = math.sqrt(
                extra[
                    "slope_longitudinal_rms"
                ]
                ** 2
                + extra[
                    "slope_lateral_rms"
                ]
                ** 2
            )

            if not math.isclose(
                directional_norm,
                finite(
                    old_descriptor[
                        "slope_rms"
                    ]
                ),
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise RuntimeError(
                    "Directional gradient rotation "
                    "consistency failure: "
                    f"{directional_norm} vs "
                    f"{old_descriptor['slope_rms']}"
                )


            geometry_sha256 = (
                hashlib.sha256(
                    np.ascontiguousarray(
                        hfield[
                            "normalized"
                        ]
                    ).tobytes()
                ).hexdigest()
            )

        finally:
            env.close()


    preset = base.get_terrain_preset(
        "rough_perlin"
    )

    return {
        "context_name":
            f"rough_perlin_seed_{seed}",

        "terrain":
            "rough_perlin",

        "seed":
            int(seed),

        "friction_mu":
            float(
                preset.friction_coeff
            ),

        "height_std_m":
            old_descriptor[
                "height_std_m"
            ],

        "height_relief_p95_p05_m":
            old_descriptor[
                "height_relief_p95_p05_m"
            ],

        "slope_rms":
            old_descriptor[
                "slope_rms"
            ],

        "slope_q95":
            old_descriptor[
                "slope_q95"
            ],

        "slope_longitudinal_rms":
            extra[
                "slope_longitudinal_rms"
            ],

        "slope_lateral_rms":
            extra[
                "slope_lateral_rms"
            ],

        "height_front_minus_rear_mean_m":
            extra[
                "height_front_minus_rear_mean_m"
            ],

        "height_left_minus_right_mean_m":
            extra[
                "height_left_minus_right_mean_m"
            ],

        "height_front_minus_rear_std_m":
            extra[
                "height_front_minus_rear_std_m"
            ],

        "height_mid_minus_ends_std_m":
            extra[
                "height_mid_minus_ends_std_m"
            ],

        "start_x_m":
            start_x,

        "start_y_m":
            start_y,

        "start_yaw_rad":
            yaw,

        "goal_x_m":
            goal_x,

        "goal_y_m":
            goal_y,

        "corridor_cells":
            old_descriptor[
                "corridor_cells"
            ],

        "rear_third_cells":
            extra[
                "rear_third_cells"
            ],

        "middle_third_cells":
            extra[
                "middle_third_cells"
            ],

        "front_third_cells":
            extra[
                "front_third_cells"
            ],

        "left_half_cells":
            extra[
                "left_half_cells"
            ],

        "right_half_cells":
            extra[
                "right_half_cells"
            ],

        "hfield_rows":
            hfield[
                "nrow"
            ],

        "hfield_cols":
            hfield[
                "ncol"
            ],

        "hfield_dx_m":
            hfield[
                "dx"
            ],

        "hfield_dy_m":
            hfield[
                "dy"
            ],

        "hfield_elevation_scale_m":
            hfield[
                "elevation_z"
            ],

        "geometry_sha256":
            geometry_sha256,
    }


def planar_row(
    terrain: str,
):
    row = dict(
        base.planar_row(
            terrain
        )
    )

    for feature in (
        ADDED_FEATURES
    ):
        row[
            feature
        ] = 0.0

    for key in (
        "rear_third_cells",
        "middle_third_cells",
        "front_third_cells",
        "left_half_cells",
        "right_half_cells",
    ):
        row[
            key
        ] = ""

    return row


def rough_row_fresh_process(
    *,
    seed: int,
):
    if not SEED_WORKER.exists():
        raise FileNotFoundError(
            SEED_WORKER
        )

    proc = subprocess.run(
        [
            sys.executable,
            str(
                SEED_WORKER
            ),
            "--seed",
            str(
                int(
                    seed
                )
            ),
        ],
        cwd=str(
            ROOT
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        print(
            proc.stdout,
            file=sys.stderr,
        )

        print(
            proc.stderr,
            file=sys.stderr,
        )

        raise RuntimeError(
            "Fresh-process v2 context "
            f"probe failed for seed={seed}; "
            f"rc={proc.returncode}"
        )

    lines = [
        line.strip()
        for line in (
            proc.stdout.splitlines()
        )
        if line.strip()
    ]

    if len(lines) != 1:
        raise RuntimeError(
            "Expected exactly one JSON line "
            "from v2 worker; "
            f"got {len(lines)}"
        )

    row = json.loads(
        lines[
            0
        ]
    )

    if int(
        row[
            "seed"
        ]
    ) != int(
        seed
    ):
        raise RuntimeError(
            "v2 worker seed mismatch."
        )

    return row


def v1_lookup():
    if not V1_CONTEXT_CSV.exists():
        raise FileNotFoundError(
            V1_CONTEXT_CSV
        )

    rows = read_csv(
        V1_CONTEXT_CSV
    )

    lookup = {}

    for row in rows:
        terrain = str(
            row[
                "terrain"
            ]
        )

        raw_seed = str(
            row.get(
                "seed",
                "",
            )
        ).strip()

        seed = (
            int(
                float(
                    raw_seed
                )
            )
            if raw_seed
            else None
        )

        lookup[
            (
                terrain,
                seed,
            )
        ] = row

    return lookup


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    robot_name = (
        base.active_robot_name()
    )

    hip_height = finite(
        base.cfg.hip_height
    )

    cfg_hip = finite(
        base.cfg.robot_cfg.hip_height
    )

    if not math.isclose(
        hip_height,
        cfg_hip,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Quadruped-PyMPC hip-height "
            "contract mismatch."
        )

    train_seeds = list(
        base.load_train_seeds()
    )

    if len(
        train_seeds
    ) != EXPECTED_ROUGH_TRAIN:
        raise RuntimeError(
            "Expected 18 rough TRAIN seeds."
        )

    frozen_v1 = (
        v1_lookup()
    )

    rows = [
        planar_row(
            "flat"
        ),
        planar_row(
            "low_friction"
        ),
    ]

    rough_rows = []


    for index, seed in enumerate(
        train_seeds,
        start=1,
    ):
        print(
            f"rough TRAIN seed "
            f"{index:02d}/"
            f"{len(train_seeds)}: "
            f"{seed}"
        )

        row = (
            rough_row_fresh_process(
                seed=seed
            )
        )

        rough_rows.append(
            row
        )

        rows.append(
            row
        )

        print(
            "  "
            f"std="
            f"{row['height_std_m']:.6f} "
            f"relief="
            f"{row['height_relief_p95_p05_m']:.6f} "
            f"slong="
            f"{row['slope_longitudinal_rms']:.6f} "
            f"slat="
            f"{row['slope_lateral_rms']:.6f} "
            f"dFR="
            f"{row['height_front_minus_rear_mean_m']:+.6f} "
            f"dLR="
            f"{row['height_left_minus_right_mean_m']:+.6f}"
        )


    # ========================================================
    # Regression against frozen v1 descriptors
    # ========================================================

    max_v1_feature_error = 0.0

    for row in rows:
        terrain = str(
            row[
                "terrain"
            ]
        )

        seed = (
            int(
                row[
                    "seed"
                ]
            )
            if terrain
            == "rough_perlin"
            else None
        )

        old = frozen_v1[
            (
                terrain,
                seed,
            )
        ]

        for feature in (
            BASE_FEATURES
        ):
            error = abs(
                finite(
                    row[
                        feature
                    ]
                )
                - finite(
                    old[
                        feature
                    ]
                )
            )

            max_v1_feature_error = max(
                max_v1_feature_error,
                error,
            )

            if error > 1e-12:
                raise RuntimeError(
                    "Frozen v1 feature regression "
                    f"failed for "
                    f"{terrain}/{seed}/{feature}: "
                    f"{error}"
                )

        if terrain == "rough_perlin":
            if (
                row[
                    "geometry_sha256"
                ]
                != old[
                    "geometry_sha256"
                ]
            ):
                raise RuntimeError(
                    "Frozen v1 geometry-hash "
                    f"regression failed for "
                    f"seed={seed}"
                )


    # ========================================================
    # Fresh-process geometry provenance
    # ========================================================

    hash_groups = {}

    for row in rough_rows:
        hash_groups.setdefault(
            row[
                "geometry_sha256"
            ],
            [],
        ).append(
            int(
                row[
                    "seed"
                ]
            )
        )

    print()
    print(
        "unique global hfields:",
        len(
            hash_groups
        ),
        "/",
        len(
            rough_rows
        ),
    )

    if len(
        hash_groups
    ) != EXPECTED_ROUGH_TRAIN:
        raise RuntimeError(
            "v2 fresh-process global hfield "
            "uniqueness failure."
        )


    # ========================================================
    # Feature validity / variation
    # ========================================================

    if len(
        CONTEXT_FEATURES
    ) != EXPECTED_CONTEXT_DIM:
        raise RuntimeError(
            "Context dimension mismatch."
        )

    feature_ranges = {}

    for feature in (
        CONTEXT_FEATURES
    ):
        values = np.asarray(
            [
                finite(
                    row[
                        feature
                    ]
                )
                for row in rough_rows
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise RuntimeError(
                f"Non-finite feature: {feature}"
            )

        feature_ranges[
            feature
        ] = [
            float(
                np.min(
                    values
                )
            ),
            float(
                np.max(
                    values
                )
            ),
        ]

        # friction is constant within the rough family.
        if (
            feature
            != "friction_mu"
            and float(
                np.ptp(
                    values
                )
            )
            <= 1e-12
        ):
            raise RuntimeError(
                f"Degenerate rough feature: "
                f"{feature}"
            )


    subregion_minima = {
        key:
            min(
                int(
                    row[
                        key
                    ]
                )
                for row in rough_rows
            )
        for key in (
            "rear_third_cells",
            "middle_third_cells",
            "front_third_cells",
            "left_half_cells",
            "right_half_cells",
        )
    }


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_CSV,
        rows,
    )


    contract = {
        "schema":
            (
                "icra27_os_t5p5a_"
                "oracle_context_descriptor_v2"
            ),

        "status":
            "FREEZE_PASS",

        "active_robot":
            robot_name,

        "hip_height_m":
            hip_height,

        "context_dim":
            EXPECTED_CONTEXT_DIM,

        "context_feature_order":
            list(
                CONTEXT_FEATURES
            ),

        "base_v1_features":
            list(
                BASE_FEATURES
            ),

        "added_v2_features":
            list(
                ADDED_FEATURES
            ),

        "rough_train_contexts":
            EXPECTED_ROUGH_TRAIN,

        "total_contexts":
            len(
                rows
            ),

        "fresh_process_global_hfields_unique":
            len(
                hash_groups
            ),

        "fresh_process_global_hfields_expected":
            EXPECTED_ROUGH_TRAIN,

        "v1_regression_max_abs_error":
            max_v1_feature_error,

        "subregion_minimum_cell_counts":
            subregion_minima,

        "feature_ranges_rough_train":
            feature_ranges,

        "spatial_frame":
            (
                "reset body/goal-aligned "
                "longitudinal-lateral frame"
            ),

        "directional_gradient":
            (
                "world hfield gradient rotated "
                "into reset body/goal frame"
            ),

        "longitudinal_partition":
            (
                "three equal-length thirds of "
                "the frozen T5.5a reset-to-goal "
                "corridor including margins"
            ),

        "lateral_partition":
            (
                "left/right halves at "
                "body-frame lateral=0"
            ),

        "interpretation_guard":
            (
                "Oracle physical terrain descriptor. "
                "No seed ID, rollout outcome metric, "
                "or held-out context is an Objective "
                "Selector input feature."
            ),

        "parent_contract":
            (
                "results/icra27/"
                "os_t5p5a_oracle_context_descriptor_v1/"
                "oracle_context_contract.json"
            ),

        "next_stage":
            (
                "OS-T5.5e1d: rerun TRAIN-only "
                "context identifiability diagnostics "
                "with c_OS_v2 before retraining "
                "the Objective Selector."
            ),
    }

    OUT_CONTRACT.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    geometry_manifest = {
        "schema":
            (
                "icra27_os_t5p5a_"
                "rough_geometry_manifest_v2"
            ),

        "status":
            "FREEZE_PASS",

        "rough_train_seeds":
            train_seeds,

        "global_hfield_hashes":
            {
                str(
                    row[
                        "seed"
                    ]
                ):
                    row[
                        "geometry_sha256"
                    ]
                for row in rough_rows
            },

        "unique_global_hfields":
            len(
                hash_groups
            ),

        "expected_global_hfields":
            EXPECTED_ROUGH_TRAIN,

        "v1_geometry_hash_regression":
            "PASS",
    }

    OUT_GEOMETRY.write_text(
        json.dumps(
            geometry_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    print()
    print("=" * 112)
    print(
        "ICRA27 OS-T5.5a ORACLE CONTEXT "
        "DESCRIPTOR v2 FREEZE"
    )
    print("=" * 112)

    print(
        "active robot :",
        robot_name,
    )

    print(
        "hip height   :",
        hip_height,
    )

    print(
        "context dim  :",
        EXPECTED_CONTEXT_DIM,
    )

    print(
        "contexts     :",
        len(
            rows
        ),
    )

    print(
        "rough TRAIN  :",
        len(
            rough_rows
        ),
    )

    print(
        "global fields:",
        len(
            hash_groups
        ),
        "/",
        EXPECTED_ROUGH_TRAIN,
        "unique",
    )

    print(
        "v1 max error :",
        max_v1_feature_error,
    )

    print()
    print("SUBREGION MINIMUM CELL COUNTS")

    for key, value in (
        subregion_minima.items()
    ):
        print(
            f"  {key:<24}: "
            f"{value}"
        )

    print()
    print("ADDED FEATURE RANGES")

    for feature in (
        ADDED_FEATURES
    ):
        lo, hi = (
            feature_ranges[
                feature
            ]
        )

        print(
            f"  {feature:<38} "
            f"[{lo:+.9f}, {hi:+.9f}]"
        )

    print()
    print("outputs:")
    print(" ", OUT_CSV)
    print(" ", OUT_CONTRACT)
    print(" ", OUT_GEOMETRY)

    print()
    print(
        "[ICRA27] OS-T5.5a oracle context "
        "descriptor v2: FREEZE PASS"
    )

    print("=" * 112)


if __name__ == "__main__":
    main()
