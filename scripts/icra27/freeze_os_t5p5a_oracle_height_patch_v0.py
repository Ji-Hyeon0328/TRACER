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

import freeze_os_t5p5a_oracle_context_descriptor_v2 as v2


ROOT = Path(__file__).resolve().parents[2]

BASE = v2.base


N_LONGITUDINAL = 8
N_LATERAL = 6
PATCH_DIM = (
    N_LONGITUDINAL
    * N_LATERAL
)

SELECTOR_CONTEXT_DIM = (
    1
    + PATCH_DIM
)

TOL = 1.0e-10


SEED_WORKER = (
    ROOT
    / "scripts/icra27"
    / "probe_os_t5p5a_height_patch_seed_v0.py"
)

V2_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_context_descriptor_v2"
)

V2_CSV = (
    V2_ROOT
    / "oracle_context_descriptors.csv"
)

V2_CONTRACT = (
    V2_ROOT
    / "oracle_context_contract.json"
)


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_height_patch_v0"
)

OUT_CSV = (
    OUT_DIR
    / "oracle_height_patches.csv"
)

OUT_CONTRACT = (
    OUT_DIR
    / "oracle_height_patch_contract.json"
)

OUT_GEOMETRY = (
    OUT_DIR
    / "height_patch_geometry_manifest.json"
)


def finite(
    value: Any,
) -> float:
    x = float(value)

    if not math.isfinite(x):
        raise ValueError(
            f"Non-finite value: {value!r}"
        )

    return x


def read_csv(
    path: Path,
):
    with path.open(
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows,
):
    if not rows:
        raise RuntimeError(
            f"No rows: {path}"
        )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(
                    key
                )

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


def patch_feature_names():
    names = []

    for i in range(
        N_LONGITUDINAL
    ):
        for j in range(
            N_LATERAL
        ):
            names.append(
                f"h_l{i:02d}_r{j:02d}_m"
            )

    return names


PATCH_FEATURES = tuple(
    patch_feature_names()
)


def bilinear_height(
    *,
    hfield,
    x: float,
    y: float,
) -> float:
    z = hfield[
        "z"
    ]

    xs = hfield[
        "xs"
    ]

    ys = hfield[
        "ys"
    ]


    x = finite(
        x
    )

    y = finite(
        y
    )


    if (
        x < xs[0] - TOL
        or x > xs[-1] + TOL
        or y < ys[0] - TOL
        or y > ys[-1] + TOL
    ):
        raise RuntimeError(
            "Patch sample outside compiled "
            "MuJoCo hfield bounds: "
            f"x={x}, y={y}"
        )


    x = float(
        np.clip(
            x,
            xs[0],
            xs[-1],
        )
    )

    y = float(
        np.clip(
            y,
            ys[0],
            ys[-1],
        )
    )


    ix1 = int(
        np.searchsorted(
            xs,
            x,
            side="right",
        )
    )

    iy1 = int(
        np.searchsorted(
            ys,
            y,
            side="right",
        )
    )


    if ix1 <= 0:
        ix0 = 0
        ix1 = 1

    elif ix1 >= len(xs):
        ix1 = (
            len(xs)
            - 1
        )
        ix0 = (
            ix1
            - 1
        )

    else:
        ix0 = (
            ix1
            - 1
        )


    if iy1 <= 0:
        iy0 = 0
        iy1 = 1

    elif iy1 >= len(ys):
        iy1 = (
            len(ys)
            - 1
        )
        iy0 = (
            iy1
            - 1
        )

    else:
        iy0 = (
            iy1
            - 1
        )


    x0 = finite(
        xs[ix0]
    )

    x1 = finite(
        xs[ix1]
    )

    y0 = finite(
        ys[iy0]
    )

    y1 = finite(
        ys[iy1]
    )


    if (
        x1 <= x0
        or y1 <= y0
    ):
        raise RuntimeError(
            "Degenerate hfield interpolation "
            "cell."
        )


    tx = (
        x - x0
    ) / (
        x1 - x0
    )

    ty = (
        y - y0
    ) / (
        y1 - y0
    )


    z00 = finite(
        z[
            iy0,
            ix0
        ]
    )

    z10 = finite(
        z[
            iy0,
            ix1
        ]
    )

    z01 = finite(
        z[
            iy1,
            ix0
        ]
    )

    z11 = finite(
        z[
            iy1,
            ix1
        ]
    )


    return float(
        (
            1.0 - tx
        )
        * (
            1.0 - ty
        )
        * z00

        + tx
        * (
            1.0 - ty
        )
        * z10

        + (
            1.0 - tx
        )
        * ty
        * z01

        + tx
        * ty
        * z11
    )


def local_height_patch(
    *,
    hfield,
    start_x: float,
    start_y: float,
    yaw: float,
    hip_height: float,
):
    margin = (
        BASE.CORRIDOR_MARGIN_HIP_MULT
        * hip_height
    )


    longitudinal_lo = (
        -margin
    )

    longitudinal_hi = (
        BASE.GOAL_DISTANCE_M
        + margin
    )

    lateral_lo = (
        -margin
    )

    lateral_hi = (
        margin
    )


    longitudinal_edges = (
        np.linspace(
            longitudinal_lo,
            longitudinal_hi,
            N_LONGITUDINAL + 1,
            dtype=np.float64,
        )
    )

    lateral_edges = (
        np.linspace(
            lateral_lo,
            lateral_hi,
            N_LATERAL + 1,
            dtype=np.float64,
        )
    )


    longitudinal_centers = (
        0.5
        * (
            longitudinal_edges[:-1]
            + longitudinal_edges[1:]
        )
    )

    lateral_centers = (
        0.5
        * (
            lateral_edges[:-1]
            + lateral_edges[1:]
        )
    )


    c = math.cos(
        yaw
    )

    s = math.sin(
        yaw
    )


    start_height = (
        bilinear_height(
            hfield=hfield,
            x=start_x,
            y=start_y,
        )
    )


    patch = np.zeros(
        (
            N_LONGITUDINAL,
            N_LATERAL,
        ),
        dtype=np.float64,
    )


    world_x = np.zeros_like(
        patch
    )

    world_y = np.zeros_like(
        patch
    )


    for i, longitudinal in enumerate(
        longitudinal_centers
    ):
        for j, lateral in enumerate(
            lateral_centers
        ):
            # Inverse transform from the frozen
            # body/goal-aligned frame:
            #
            # longitudinal = c*dx + s*dy
            # lateral      = -s*dx + c*dy
            #
            # therefore:
            # dx = c*l - s*r
            # dy = s*l + c*r

            dx_world = (
                c * longitudinal
                - s * lateral
            )

            dy_world = (
                s * longitudinal
                + c * lateral
            )


            x = (
                start_x
                + dx_world
            )

            y = (
                start_y
                + dy_world
            )


            world_x[
                i,
                j
            ] = x

            world_y[
                i,
                j
            ] = y


            height = (
                bilinear_height(
                    hfield=hfield,
                    x=x,
                    y=y,
                )
            )


            patch[
                i,
                j
            ] = (
                height
                - start_height
            )


    if not np.all(
        np.isfinite(
            patch
        )
    ):
        raise RuntimeError(
            "Non-finite local height patch."
        )


    return {
        "patch":
            patch,

        "start_height_m":
            start_height,

        "longitudinal_centers_m":
            longitudinal_centers,

        "lateral_centers_m":
            lateral_centers,

        "longitudinal_lo_m":
            longitudinal_lo,

        "longitudinal_hi_m":
            longitudinal_hi,

        "lateral_lo_m":
            lateral_lo,

        "lateral_hi_m":
            lateral_hi,

        "sample_world_x":
            world_x,

        "sample_world_y":
            world_y,
    }


def rough_patch_row(
    *,
    seed: int,
    robot_name: str,
    hip_height: float,
):
    # Preserve the same seeded Perlin and reset
    # semantics as T5.5a-v1/v2.
    with BASE.SeededPerlinHooks(
        seed
    ):
        env = BASE.QuadrupedEnv(
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

            yaw = (
                BASE.yaw_from_wxyz(
                    env.mjData.qpos[
                        3:7
                    ]
                )
            )


            hfield = (
                BASE.get_hfield(
                    env.mjModel
                )
            )


            patch_info = (
                local_height_patch(
                    hfield=hfield,
                    start_x=start_x,
                    start_y=start_y,
                    yaw=yaw,
                    hip_height=(
                        hip_height
                    ),
                )
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


    preset = (
        BASE.get_terrain_preset(
            "rough_perlin"
        )
    )


    patch = (
        patch_info[
            "patch"
        ]
    )

    flat_patch = (
        patch.reshape(
            -1
        )
    )


    row = {
        "context_id":
            f"rough_seed_{seed}",

        "context_name":
            f"rough_perlin_seed_{seed}",

        "terrain":
            "rough_perlin",

        "seed":
            int(
                seed
            ),

        "friction_mu":
            float(
                preset.friction_coeff
            ),

        "start_x_m":
            start_x,

        "start_y_m":
            start_y,

        "start_yaw_rad":
            yaw,

        "start_height_m":
            patch_info[
                "start_height_m"
            ],

        "patch_min_m":
            float(
                np.min(
                    patch
                )
            ),

        "patch_max_m":
            float(
                np.max(
                    patch
                )
            ),

        "patch_std_m":
            float(
                np.std(
                    patch,
                    ddof=0,
                )
            ),

        "geometry_sha256":
            geometry_sha256,
    }


    for name, value in zip(
        PATCH_FEATURES,
        flat_patch,
    ):
        row[
            name
        ] = finite(
            value
        )


    return row


def planar_row(
    terrain: str,
):
    preset = (
        BASE.get_terrain_preset(
            terrain
        )
    )

    row = {
        "context_id":
            terrain,

        "context_name":
            terrain,

        "terrain":
            terrain,

        "seed":
            "",

        "friction_mu":
            float(
                preset.friction_coeff
            ),

        "start_x_m":
            "",

        "start_y_m":
            "",

        "start_yaw_rad":
            "",

        "start_height_m":
            "",

        "patch_min_m":
            0.0,

        "patch_max_m":
            0.0,

        "patch_std_m":
            0.0,

        "geometry_sha256":
            "",
    }

    for name in (
        PATCH_FEATURES
    ):
        row[
            name
        ] = 0.0

    return row


def rough_patch_row_fresh_process(
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
                int(seed)
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
            "Fresh-process height patch "
            f"worker failed for seed={seed}; "
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
            "Expected exactly one JSON "
            "line from patch worker; "
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
            "Height-patch worker seed "
            "mismatch."
        )


    return row


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )


    # ========================================================
    # Parent v2 contract
    # ========================================================

    if not V2_CONTRACT.exists():
        raise FileNotFoundError(
            V2_CONTRACT
        )

    if not V2_CSV.exists():
        raise FileNotFoundError(
            V2_CSV
        )


    v2_contract = json.loads(
        V2_CONTRACT.read_text()
    )

    if v2_contract.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Parent T5.5a-v2 is not "
            "FREEZE_PASS."
        )


    v2_rows = read_csv(
        V2_CSV
    )

    v2_lookup = {
        row[
            "context_name"
        ]:
            row
        for row in v2_rows
    }


    robot_name = (
        BASE.active_robot_name()
    )

    hip_height = finite(
        BASE.cfg.hip_height
    )

    cfg_hip = finite(
        BASE.cfg.robot_cfg.hip_height
    )


    if not math.isclose(
        hip_height,
        cfg_hip,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Hip-height contract mismatch."
        )


    train_seeds = list(
        BASE.load_train_seeds()
    )


    if len(
        train_seeds
    ) != 18:
        raise RuntimeError(
            "Expected 18 rough TRAIN seeds."
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
            rough_patch_row_fresh_process(
                seed=seed
            )
        )


        parent = (
            v2_lookup[
                f"rough_perlin_seed_{seed}"
            ]
        )


        if (
            row[
                "geometry_sha256"
            ]
            != parent[
                "geometry_sha256"
            ]
        ):
            raise RuntimeError(
                "Parent v2 geometry-hash "
                f"regression failed: seed={seed}"
            )


        for key in (
            "start_x_m",
            "start_y_m",
            "start_yaw_rad",
        ):
            error = abs(
                finite(
                    row[
                        key
                    ]
                )
                - finite(
                    parent[
                        key
                    ]
                )
            )

            if error > 1e-12:
                raise RuntimeError(
                    "Parent v2 reset-state "
                    "regression failure: "
                    f"seed={seed}, "
                    f"key={key}, "
                    f"error={error}"
                )


        rough_rows.append(
            row
        )

        rows.append(
            row
        )


        print(
            "  "
            f"range="
            f"[{row['patch_min_m']:+.6f}, "
            f"{row['patch_max_m']:+.6f}] "
            f"std="
            f"{row['patch_std_m']:.6f}"
        )


    # ========================================================
    # Geometry provenance
    # ========================================================

    hashes = {
        row[
            "geometry_sha256"
        ]
        for row in rough_rows
    }


    if len(
        hashes
    ) != 18:
        raise RuntimeError(
            "Expected 18/18 unique "
            "fresh-process hfields."
        )


    # ========================================================
    # Patch validity
    # ========================================================

    matrix = np.asarray(
        [
            [
                finite(
                    row[
                        feature
                    ]
                )
                for feature in (
                    PATCH_FEATURES
                )
            ]
            for row in rough_rows
        ],
        dtype=np.float64,
    )


    if matrix.shape != (
        18,
        PATCH_DIM,
    ):
        raise RuntimeError(
            f"Unexpected patch matrix shape: "
            f"{matrix.shape}"
        )


    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        raise RuntimeError(
            "Non-finite patch matrix."
        )


    # Ensure representation is not degenerate.
    unique_patch_rows = np.unique(
        np.round(
            matrix,
            decimals=12,
        ),
        axis=0,
    )


    if len(
        unique_patch_rows
    ) != 18:
        raise RuntimeError(
            "Expected 18/18 unique rough "
            "local height patches."
        )


    # Every spatial location should not be required
    # to vary; however at least a substantial number
    # of cells must carry variation.
    varying_cells = int(
        np.sum(
            np.ptp(
                matrix,
                axis=0,
            )
            > 1e-12
        )
    )


    if varying_cells < (
        PATCH_DIM // 2
    ):
        raise RuntimeError(
            "Too few varying height-patch cells: "
            f"{varying_cells}/{PATCH_DIM}"
        )


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
                "oracle_height_patch_v0"
            ),

        "status":
            "FREEZE_PASS",

        "heldout_used":
            False,

        "active_robot":
            robot_name,

        "hip_height_m":
            hip_height,

        "rough_train_contexts":
            18,

        "total_contexts":
            20,

        "patch_shape":
            [
                N_LONGITUDINAL,
                N_LATERAL,
            ],

        "patch_dim":
            PATCH_DIM,

        "selector_context_dim_with_friction":
            SELECTOR_CONTEXT_DIM,

        "feature_order":
            [
                "friction_mu",
                *PATCH_FEATURES,
            ],

        "height_reference":
            (
                "Each sampled terrain height "
                "minus the bilinearly interpolated "
                "terrain height at the reset "
                "base x-y position."
            ),

        "sampling_frame":
            (
                "Frozen reset body/goal-aligned "
                "longitudinal-lateral frame."
            ),

        "sampling_domain":
            {
                "longitudinal_m":
                    [
                        (
                            -BASE.CORRIDOR_MARGIN_HIP_MULT
                            * hip_height
                        ),
                        (
                            BASE.GOAL_DISTANCE_M
                            + BASE.CORRIDOR_MARGIN_HIP_MULT
                            * hip_height
                        ),
                    ],

                "lateral_m":
                    [
                        (
                            -BASE.CORRIDOR_MARGIN_HIP_MULT
                            * hip_height
                        ),
                        (
                            BASE.CORRIDOR_MARGIN_HIP_MULT
                            * hip_height
                        ),
                    ],
            },

        "sampling_method":
            (
                "Uniform cell-center grid with "
                "bilinear interpolation on the "
                "compiled MuJoCo hfield."
            ),

        "absolute_elevation_removed":
            True,

        "fresh_process_global_hfields_unique":
            len(
                hashes
            ),

        "unique_local_patches":
            len(
                unique_patch_rows
            ),

        "varying_patch_cells":
            varying_cells,

        "parent":
            (
                "os_t5p5a_oracle_context_"
                "descriptor_v2"
            ),

        "future_interface_guard":
            (
                "This artifact defines a simple "
                "oracle terrain representation. "
                "A future CART/world-model/learned "
                "representation may replace the "
                "representation provider without "
                "changing the Objective Selector "
                "mission-preference semantics."
            ),

        "interpretation_guard":
            (
                "No seed ID, rollout outcome "
                "metric, held-out terrain outcome, "
                "or beta-response metric is an "
                "Objective Selector input."
            ),

        "next_stage":
            (
                "TRAIN-only patch-distance versus "
                "beta-response-distance and "
                "nearest-context preference-transfer "
                "audit before selector retraining."
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
                "oracle_height_patch_geometry_v0"
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
                hashes
            ),

        "unique_local_patches":
            len(
                unique_patch_rows
            ),

        "v2_geometry_and_reset_regression":
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
    print("=" * 116)
    print(
        "ICRA27 OS-T5.5a ORACLE LOCAL "
        "HEIGHT PATCH v0 FREEZE"
    )
    print("=" * 116)

    print(
        "active robot     :",
        robot_name,
    )

    print(
        "patch shape      :",
        (
            N_LONGITUDINAL,
            N_LATERAL,
        ),
    )

    print(
        "patch dim        :",
        PATCH_DIM,
    )

    print(
        "selector c dim   :",
        SELECTOR_CONTEXT_DIM,
    )

    print(
        "rough TRAIN      :",
        len(
            rough_rows
        ),
    )

    print(
        "global hfields   :",
        len(
            hashes
        ),
        "/18 unique",
    )

    print(
        "local patches    :",
        len(
            unique_patch_rows
        ),
        "/18 unique",
    )

    print(
        "varying cells    :",
        varying_cells,
        "/",
        PATCH_DIM,
    )

    print()
    print("outputs:")
    print(" ", OUT_CSV)
    print(" ", OUT_CONTRACT)
    print(" ", OUT_GEOMETRY)

    print()
    print(
        "[ICRA27] OS-T5.5a oracle local "
        "height patch v0: FREEZE PASS"
    )

    print("=" * 116)


if __name__ == "__main__":
    main()
