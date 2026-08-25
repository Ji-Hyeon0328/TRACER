from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

QPYMPC_ROOT = (
    ROOT
    / "external_baselines"
    / "Quadruped-PyMPC"
)

sys.path.insert(
    0,
    str(ROOT),
)

sys.path.insert(
    0,
    str(QPYMPC_ROOT),
)


import quadruped_pympc.config as cfg

import gym_quadruped.quadruped_env as gym_env_module

from gym_quadruped.quadruped_env import (
    QuadrupedEnv,
)

from gym_quadruped.utils.mujoco import (
    terrain as gym_terrain,
)

from tracer_core.highlevel_rl.terrain import (
    get_terrain_preset,
)


SPLIT_PATH = (
    ROOT
    / "configs/icra27"
    / "m7_perlin_seed_split_v0.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_context_descriptor_v1"
)

OUT_CSV = (
    OUT_DIR
    / "oracle_context_descriptors.csv"
)

OUT_JSON = (
    OUT_DIR
    / "oracle_context_contract.json"
)

ROUGH_MANIFEST = (
    OUT_DIR
    / "rough_geometry_manifest.json"
)

SEED_WORKER = (
    ROOT
    / "scripts"
    / "icra27"
    / "probe_os_t5p5a_context_seed_v1.py"
)


EXPECTED_ROUGH_TRAIN_SEEDS = 18

GOAL_DISTANCE_M = 0.50

# Robot-scale dilation around the actual reset->goal segment.
#
# Longitudinal:
#   start - 2 hip heights
#   goal  + 2 hip heights
#
# Lateral:
#   +/- 2 hip heights
#
# This captures terrain under/around the body and likely
# foothold region rather than only the mathematical centerline.
CORRIDOR_MARGIN_HIP_MULT = 2.0

MIN_CORRIDOR_CELLS = 20

TOL = 1.0e-10


def finite(
    value: Any,
) -> float:
    x = float(value)

    if not math.isfinite(x):
        raise ValueError(
            f"Non-finite value: {value!r}"
        )

    return x


def yaw_from_wxyz(
    quat,
) -> float:
    q = np.asarray(
        quat,
        dtype=np.float64,
    ).reshape(4)

    w, x, y, z = q

    yaw = math.atan2(
        2.0 * (
            w * z
            + x * y
        ),
        1.0
        - 2.0 * (
            y * y
            + z * z
        ),
    )

    return float(yaw)


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


def active_robot_name() -> str:
    filename = Path(
        cfg.robot_cfg.mjcf_filename
    )

    if len(filename.parts) >= 2:
        return filename.parts[0]

    return filename.stem


def load_train_seeds():
    data = json.loads(
        SPLIT_PATH.read_text()
    )

    if data.get("schema") != (
        "icra27_m7_perlin_seed_split_v0"
    ):
        raise RuntimeError(
            "Unexpected Perlin split schema."
        )

    train = [
        int(x)
        for x in (
            data[
                "splits"
            ][
                "train"
            ]
        )
    ]

    if len(train) != (
        EXPECTED_ROUGH_TRAIN_SEEDS
    ):
        raise RuntimeError(
            "Expected exactly 18 rough "
            f"TRAIN seeds; got {len(train)}"
        )

    other = set(
        int(x)
        for key in (
            "validation",
            "test",
        )
        for x in (
            data[
                "splits"
            ][key]
        )
    )

    other.update(
        int(x)
        for x in (
            data[
                "stress_banks"
            ][
                "hard"
            ]
        )
    )

    if set(train) & other:
        raise RuntimeError(
            "TRAIN seeds overlap held-out "
            "or hard banks."
        )

    return train


class SeededPerlinHooks:
    def __init__(
        self,
        seed: int,
    ):
        self.seed = int(seed)

        self.orig_env_generate = (
            gym_env_module.generate_terrain
        )

        self.orig_module_generate = (
            gym_terrain.generate_terrain
        )

        self.orig_pnoise2 = (
            gym_terrain.noise.pnoise2
        )

    def __enter__(self):
        seed = self.seed

        orig_generate = (
            self.orig_module_generate
        )

        orig_pnoise2 = (
            self.orig_pnoise2
        )

        def seeded_pnoise2(
            *args,
            **kwargs,
        ):
            kwargs.setdefault(
                "base",
                seed,
            )

            return orig_pnoise2(
                *args,
                **kwargs,
            )

        def seeded_generate_terrain(
            *args,
            **kwargs,
        ):
            # Match validated M7 terrain-seed runner:
            # ignore package-internal seed=10.
            kwargs["seed"] = seed

            return orig_generate(
                *args,
                **kwargs,
            )

        gym_env_module.generate_terrain = (
            seeded_generate_terrain
        )

        gym_terrain.generate_terrain = (
            seeded_generate_terrain
        )

        gym_terrain.noise.pnoise2 = (
            seeded_pnoise2
        )

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        gym_env_module.generate_terrain = (
            self.orig_env_generate
        )

        gym_terrain.generate_terrain = (
            self.orig_module_generate
        )

        gym_terrain.noise.pnoise2 = (
            self.orig_pnoise2
        )


def get_hfield(
    model,
):
    if int(model.nhfield) != 1:
        raise RuntimeError(
            "Expected exactly one Perlin "
            f"hfield; got {model.nhfield}"
        )

    index = 0

    nrow = int(
        model.hfield_nrow[index]
    )

    ncol = int(
        model.hfield_ncol[index]
    )

    adr = int(
        model.hfield_adr[index]
    )

    count = (
        nrow
        * ncol
    )

    normalized = np.asarray(
        model.hfield_data[
            adr:
            adr + count
        ],
        dtype=np.float64,
    ).reshape(
        nrow,
        ncol,
    )

    if (
        np.min(normalized) < -TOL
        or np.max(normalized) > 1.0 + TOL
    ):
        raise RuntimeError(
            "MuJoCo hfield_data outside "
            "expected [0,1]."
        )

    size = np.asarray(
        model.hfield_size[index],
        dtype=np.float64,
    )

    radius_x = finite(
        size[0]
    )

    radius_y = finite(
        size[1]
    )

    elevation_z = finite(
        size[2]
    )

    base_depth = finite(
        size[3]
    )

    z = (
        normalized
        * elevation_z
    )

    dx = (
        2.0
        * radius_x
        / (
            ncol - 1
        )
    )

    dy = (
        2.0
        * radius_y
        / (
            nrow - 1
        )
    )

    xs = np.linspace(
        -radius_x,
        radius_x,
        ncol,
        dtype=np.float64,
    )

    ys = np.linspace(
        -radius_y,
        radius_y,
        nrow,
        dtype=np.float64,
    )

    return {
        "normalized":
            normalized,

        "z":
            z,

        "nrow":
            nrow,

        "ncol":
            ncol,

        "radius_x":
            radius_x,

        "radius_y":
            radius_y,

        "elevation_z":
            elevation_z,

        "base_depth":
            base_depth,

        "dx":
            dx,

        "dy":
            dy,

        "xs":
            xs,

        "ys":
            ys,
    }


def local_descriptor(
    *,
    hfield,
    start_x: float,
    start_y: float,
    yaw: float,
    hip_height: float,
):
    z = hfield[
        "z"
    ]

    xs = hfield[
        "xs"
    ]

    ys = hfield[
        "ys"
    ]

    dx = hfield[
        "dx"
    ]

    dy = hfield[
        "dy"
    ]

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

    # Body/goal-aligned coordinates.
    longitudinal = (
        c * rx
        + s * ry
    )

    lateral = (
        -s * rx
        + c * ry
    )

    margin = (
        CORRIDOR_MARGIN_HIP_MULT
        * hip_height
    )

    mask = (
        (
            longitudinal
            >= -margin
        )
        & (
            longitudinal
            <= (
                GOAL_DISTANCE_M
                + margin
            )
        )
        & (
            np.abs(
                lateral
            )
            <= margin
        )
    )

    count = int(
        np.sum(mask)
    )

    if count < MIN_CORRIDOR_CELLS:
        raise RuntimeError(
            "Too few local hfield cells: "
            f"{count}"
        )

    heights = z[
        mask
    ]

    # np.gradient ordering is [y, x].
    dz_dy, dz_dx = np.gradient(
        z,
        dy,
        dx,
    )

    slope = np.sqrt(
        dz_dx * dz_dx
        + dz_dy * dz_dy
    )

    local_slope = slope[
        mask
    ]

    q05 = float(
        np.quantile(
            heights,
            0.05,
            method="linear",
        )
    )

    q95 = float(
        np.quantile(
            heights,
            0.95,
            method="linear",
        )
    )

    return {
        "corridor_cells":
            count,

        "height_std_m":
            float(
                np.std(
                    heights,
                    ddof=0,
                )
            ),

        "height_relief_p95_p05_m":
            (
                q95
                - q05
            ),

        "slope_rms":
            float(
                math.sqrt(
                    np.mean(
                        local_slope
                        * local_slope
                    )
                )
            ),

        "slope_q95":
            float(
                np.quantile(
                    local_slope,
                    0.95,
                    method="linear",
                )
            ),

        "height_mean_m":
            float(
                np.mean(
                    heights
                )
            ),

        "height_min_m":
            float(
                np.min(
                    heights
                )
            ),

        "height_max_m":
            float(
                np.max(
                    heights
                )
            ),
    }


def rough_row(
    *,
    seed: int,
    robot_name: str,
    hip_height: float,
):
    with SeededPerlinHooks(
        seed
    ):
        env = QuadrupedEnv(
            robot=robot_name,
            scene="perlin",
            sim_dt=0.002,
            ground_friction_coeff=0.8,
        )

        try:
            # Match native runner reset semantics.
            env.reset(
                seed=seed,
                random=True,
            )

            start_x = finite(
                env.mjData.qpos[0]
            )

            start_y = finite(
                env.mjData.qpos[1]
            )

            yaw = yaw_from_wxyz(
                env.mjData.qpos[
                    3:7
                ]
            )

            goal_x = (
                start_x
                + GOAL_DISTANCE_M
                * math.cos(
                    yaw
                )
            )

            goal_y = (
                start_y
                + GOAL_DISTANCE_M
                * math.sin(
                    yaw
                )
            )

            hfield = get_hfield(
                env.mjModel
            )

            descriptor = (
                local_descriptor(
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

    preset = get_terrain_preset(
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
            descriptor[
                "height_std_m"
            ],

        "height_relief_p95_p05_m":
            descriptor[
                "height_relief_p95_p05_m"
            ],

        "slope_rms":
            descriptor[
                "slope_rms"
            ],

        "slope_q95":
            descriptor[
                "slope_q95"
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
            descriptor[
                "corridor_cells"
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
            str(SEED_WORKER),
            "--seed",
            str(int(seed)),
        ],
        cwd=str(ROOT),
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
            "Fresh-process context probe "
            f"failed for seed={seed}; "
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
        print(
            "worker stdout:",
            repr(proc.stdout),
            file=sys.stderr,
        )

        print(
            "worker stderr:",
            repr(proc.stderr),
            file=sys.stderr,
        )

        raise RuntimeError(
            "Expected exactly one JSON "
            "line from seed worker; "
            f"got {len(lines)}"
        )

    row = json.loads(
        lines[0]
    )

    if int(
        row["seed"]
    ) != int(seed):
        raise RuntimeError(
            "Worker seed mismatch: "
            f"requested={seed} "
            f"returned={row['seed']}"
        )

    return row


def planar_row(
    terrain: str,
):
    preset = get_terrain_preset(
        terrain
    )

    return {
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

        "height_std_m":
            0.0,

        "height_relief_p95_p05_m":
            0.0,

        "slope_rms":
            0.0,

        "slope_q95":
            0.0,

        "start_x_m":
            "",

        "start_y_m":
            "",

        "start_yaw_rad":
            "",

        "goal_x_m":
            "",

        "goal_y_m":
            "",

        "corridor_cells":
            "",

        "hfield_rows":
            "",

        "hfield_cols":
            "",

        "hfield_dx_m":
            "",

        "hfield_dy_m":
            "",

        "hfield_elevation_scale_m":
            "",

        "geometry_sha256":
            "",
    }


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    robot_name = (
        active_robot_name()
    )

    hip_height = finite(
        cfg.hip_height
    )

    cfg_hip = finite(
        cfg.robot_cfg.hip_height
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

    train_seeds = (
        load_train_seeds()
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
                seed=seed,
            )
        )

        if (
            row["terrain"]
            != "rough_perlin"
        ):
            raise RuntimeError(
                "Unexpected worker terrain: "
                f"{row['terrain']!r}"
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
            f"slope_rms="
            f"{row['slope_rms']:.6f} "
            f"slope95="
            f"{row['slope_q95']:.6f} "
            f"cells="
            f"{row['corridor_cells']}"
        )

    hashes = [
        row[
            "geometry_sha256"
        ]
        for row in rough_rows
    ]

    hash_groups = {}

    for row in rough_rows:
        hash_groups.setdefault(
            row["geometry_sha256"],
            [],
        ).append(
            int(row["seed"])
        )

    print()
    print("GLOBAL HFIELD HASH GROUPS")

    for hash_value, seeds in (
        hash_groups.items()
    ):
        print(
            f"  {hash_value[:16]}... "
            f"seeds={seeds}"
        )

    print(
        "unique global hfields:",
        len(hash_groups),
        "/",
        len(rough_rows),
    )

    if len(hash_groups) != len(
        rough_rows
    ):
        raise RuntimeError(
            "Fresh-process rough TRAIN "
            "geometry uniqueness failure: "
            f"{len(hash_groups)}/"
            f"{len(rough_rows)} unique"
        )

    # Fresh-process geometry matches the
    # actual M7 per-episode subprocess
    # execution semantics.
    # diagnostic, not an Objective-Selector
    # context requirement. Different seeded reset
    # poses may traverse different local corridors
    # on the same global hfield.

    geometry_features = (
        "height_std_m",
        "height_relief_p95_p05_m",
        "slope_rms",
        "slope_q95",
    )

    # At least one rough seed must differ in
    # every geometry descriptor.
    for feature in (
        geometry_features
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

        if float(
            np.ptp(
                values
            )
        ) <= 1e-12:
            raise RuntimeError(
                f"Degenerate rough feature: "
                f"{feature}"
            )

    context_vectors = [
        (
            float(
                row["friction_mu"]
            ),
            float(
                row["height_std_m"]
            ),
            float(
                row[
                    "height_relief_p95_p05_m"
                ]
            ),
            float(
                row["slope_rms"]
            ),
            float(
                row["slope_q95"]
            ),
        )
        for row in rough_rows
    ]

    rounded_context_vectors = [
        tuple(
            round(
                value,
                12,
            )
            for value in vector
        )
        for vector
        in context_vectors
    ]

    unique_context_count = len(
        set(
            rounded_context_vectors
        )
    )

    if unique_context_count != len(
        rough_rows
    ):
        raise RuntimeError(
            "Rough TRAIN local context "
            "descriptor collision: "
            f"{unique_context_count}/"
            f"{len(rough_rows)} unique"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_CSV,
        rows,
    )

    rough_manifest = {
        "schema":
            "icra27_os_t5p5a_rough_geometry_manifest_v1",

        "robot":
            robot_name,

        "hip_height_m":
            hip_height,

        "train_seed_count":
            len(
                train_seeds
            ),

        "train_seeds":
            train_seeds,

        "global_hfield_unique_count":
            len(
                hash_groups
            ),

        "global_hfield_hash_groups":
            {
                key:
                    value
                for key, value
                in hash_groups.items()
            },

        "global_hfield_uniqueness_required":
            True,

        "geometry_reconstruction":
            (
                "fresh Python subprocess "
                "per rough TRAIN seed, matching "
                "M7 per-episode runner semantics"
            ),

        "context_semantics":
            (
                "Objective Selector context is "
                "defined from the actual seeded "
                "reset-to-goal local corridor, "
                "not from global terrain identity."
            ),

        "rows":
            rough_rows,
    }

    ROUGH_MANIFEST.write_text(
        json.dumps(
            rough_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    contract = {
        "schema":
            "icra27_os_t5p5a_oracle_context_descriptor_v1",

        "status":
            "FREEZE_PASS",

        "role":
            (
                "Objective-Selector-only oracle "
                "physical terrain context"
            ),

        "does_not_modify_frozen_ppo_observation":
            True,

        "context_vector":
            [
                "friction_mu",
                "height_std_m",
                "height_relief_p95_p05_m",
                "slope_rms",
                "slope_q95",
            ],

        "semantics":
            {
                "friction_mu":
                    (
                        "terrain tangential "
                        "friction coefficient"
                    ),

                "height_std_m":
                    (
                        "local reset-to-goal "
                        "corridor height standard "
                        "deviation"
                    ),

                "height_relief_p95_p05_m":
                    (
                        "local robust vertical "
                        "relief Q95(h)-Q05(h)"
                    ),

                "slope_rms":
                    (
                        "local RMS magnitude of "
                        "physical height gradient"
                    ),

                "slope_q95":
                    (
                        "local 95th percentile "
                        "height-gradient magnitude"
                    ),
            },

        "corridor":
            {
                "goal_distance_m":
                    GOAL_DISTANCE_M,

                "longitudinal_margin":
                    (
                        "2 * active robot hip height "
                        "before reset and beyond goal"
                    ),

                "lateral_half_width":
                    (
                        "2 * active robot hip height"
                    ),

                "aligned_with":
                    (
                        "actual seeded reset yaw"
                    ),
            },

        "terrain_geometry_source":
            (
                "compiled MuJoCo hfield_data "
                "from a fresh Python subprocess "
                "per TRAIN seed under the "
                "validated seeded Perlin hooks"
            ),

        "runtime_semantics_match":
            (
                "Matches M7 environment behavior "
                "where reset(seed) launches a "
                "fresh terrain runner subprocess."
            ),

        "rough_train_seeds_only":
            True,

        "heldout_used":
            False,

        "robot":
            robot_name,

        "hip_height_m":
            hip_height,

        "context_count":
            len(
                rows
            ),

        "unique_physical_contexts":
            (
                "1 flat + 1 low_friction + "
                "18 rough TRAIN realizations"
            ),

        "next_stage":
            (
                "OS-T5.5b remaining 15 rough "
                "TRAIN beta-response atlas "
                "expansion"
            ),
    }

    OUT_JSON.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 104)
    print(
        "ICRA27 OS-T5.5a "
        "ORACLE CONTEXT DESCRIPTOR FREEZE"
    )
    print("=" * 104)

    print(
        "active robot    :",
        robot_name,
    )

    print(
        "hip height      :",
        hip_height,
    )

    print(
        "context dim     : 5"
    )

    print(
        "contexts        :",
        len(
            rows
        ),
    )

    print(
        "rough TRAIN     :",
        len(
            rough_rows
        ),
    )

    print(
        "global hfields  :",
        len(hash_groups),
        "/",
        len(rough_rows),
        "unique",
    )

    print()
    print("ROUGH FEATURE RANGES")

    for feature in (
        geometry_features
    ):
        values = [
            finite(
                row[
                    feature
                ]
            )
            for row in rough_rows
        ]

        print(
            f"  {feature:<30} "
            f"[{min(values):.9f}, "
            f"{max(values):.9f}]"
        )

    print()
    print("outputs:")
    print(" ", OUT_CSV)
    print(" ", OUT_JSON)
    print(" ", ROUGH_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5a "
        "oracle context descriptor: FREEZE PASS"
    )


if __name__ == "__main__":
    main()
