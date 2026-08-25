from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

WORKER = (
    ROOT
    / "scripts/icra27"
    / "probe_os_t6p3a_highres_height_patch_seed_v0.py"
)

BASE_CONTRACT = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_height_patch_v0"
    / "oracle_height_patch_contract.json"
)

BASE_GEOMETRY = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_height_patch_v0"
    / "height_patch_geometry_manifest.json"
)

EXT_PATCH = (
    ROOT
    / "results/icra27"
    / "os_t5p5f1_selector_train_extension_height_patches_v0"
    / "extension_height_patches.csv"
)

F3_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3_expanded_physical_atlas_v0"
    / "expanded_physical_atlas_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p3a_highres_height_patches_v0"
)

OUT_CSV = (
    OUT_DIR
    / "highres_height_patches.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "highres_height_patch_manifest.json"
)


N_LONG = 16
N_LAT = 12
PATCH_DIM = 192

EXPECTED_OLD = 18
EXPECTED_EXT = 15
EXPECTED_TOTAL = 33


def read_csv(path):
    with path.open(
        "r",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def run_worker(seed):
    proc = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            "--seed",
            str(seed),
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
            f"High-res worker failed: "
            f"seed={seed}, "
            f"rc={proc.returncode}"
        )

    lines = [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise RuntimeError(
            f"Expected one JSON line "
            f"for seed={seed}; "
            f"got {len(lines)}."
        )

    row = json.loads(
        lines[0]
    )

    if int(
        row[
            "seed"
        ]
    ) != seed:
        raise RuntimeError(
            f"Seed mismatch for {seed}."
        )

    return row


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        WORKER,
        BASE_CONTRACT,
        BASE_GEOMETRY,
        EXT_PATCH,
        F3_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    base = json.loads(
        BASE_CONTRACT.read_text()
    )

    geometry = json.loads(
        BASE_GEOMETRY.read_text()
    )

    f3 = json.loads(
        F3_MANIFEST.read_text()
    )

    if base.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Base patch is not FREEZE_PASS."
        )

    if geometry.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Base geometry is not FREEZE_PASS."
        )

    if f3.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T5.5f3 atlas is not FREEZE_PASS."
        )

    if list(
        base[
            "patch_shape"
        ]
    ) != [8, 6]:
        raise RuntimeError(
            "Expected frozen base shape [8,6]."
        )

    old_seeds = [
        int(x)
        for x in geometry[
            "rough_train_seeds"
        ]
    ]

    ext_seeds = [
        int(x)
        for x in f3[
            "full_grid_extension_seeds"
        ]
    ]

    if len(
        old_seeds
    ) != EXPECTED_OLD:
        raise RuntimeError(
            "Expected 18 original seeds."
        )

    if len(
        ext_seeds
    ) != EXPECTED_EXT:
        raise RuntimeError(
            "Expected 15 extension seeds."
        )

    if set(
        old_seeds
    ) & set(
        ext_seeds
    ):
        raise RuntimeError(
            "Old/extension seed overlap."
        )

    seeds = (
        old_seeds
        + ext_seeds
    )

    if len(
        seeds
    ) != EXPECTED_TOTAL:
        raise RuntimeError(
            "Expected 33 total seeds."
        )


    # ========================================================
    # Frozen geometry-hash references
    # ========================================================

    old_hash = {
        int(seed):
            value
        for seed, value in geometry[
            "global_hfield_hashes"
        ].items()
    }

    ext_rows = read_csv(
        EXT_PATCH
    )

    ext_hash = {
        int(
            row[
                "seed"
            ]
        ):
            row[
                "geometry_sha256"
            ]
        for row in ext_rows
    }


    rows = []

    hashes = set()

    print(
        "sampling high-resolution "
        "16x12 patches..."
    )

    for idx, seed in enumerate(
        seeds,
        1,
    ):
        row = run_worker(
            seed
        )

        expected_hash = (
            old_hash[
                seed
            ]
            if seed in old_hash
            else ext_hash[
                seed
            ]
        )

        if row[
            "geometry_sha256"
        ] != expected_hash:
            raise RuntimeError(
                f"Compiled hfield changed "
                f"for seed={seed}."
            )

        patch_fields = [
            key
            for key in row
            if key.startswith(
                "h_l"
            )
            and key.endswith(
                "_m"
            )
        ]

        if len(
            patch_fields
        ) != PATCH_DIM:
            raise RuntimeError(
                f"seed={seed}: expected "
                f"{PATCH_DIM} cells; "
                f"got {len(patch_fields)}."
            )

        hashes.add(
            row[
                "geometry_sha256"
            ]
        )

        rows.append(
            row
        )

        print(
            f"  {idx:02d}/{len(seeds)} "
            f"seed={seed:<2} "
            f"range=["
            f"{float(row['patch_min_m']):+.5f},"
            f"{float(row['patch_max_m']):+.5f}] "
            f"std="
            f"{float(row['patch_std_m']):.5f}"
        )


    if len(
        hashes
    ) != EXPECTED_TOTAL:
        raise RuntimeError(
            "High-res sample hfields are "
            "not unique across 33 contexts."
        )


    expected_features = [
        f"h_l{i:02d}_r{j:02d}_m"
        for i in range(N_LONG)
        for j in range(N_LAT)
    ]

    for row in rows:
        missing = [
            name
            for name in expected_features
            if name not in row
        ]

        if missing:
            raise RuntimeError(
                f"{row['context_id']}: "
                f"missing high-res cells."
            )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(
                    key
                )

    with OUT_CSV.open(
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


    manifest = {
        "schema":
            "icra27_os_t6p3a_highres_height_patches_v0",

        "status":
            "FREEZE_PASS",

        "heldout_used":
            False,

        "contexts":
            EXPECTED_TOTAL,

        "original_rough_contexts":
            EXPECTED_OLD,

        "extension_rough_contexts":
            EXPECTED_EXT,

        "original_seeds":
            old_seeds,

        "extension_seeds":
            ext_seeds,

        "patch_shape":
            [
                N_LONG,
                N_LAT,
            ],

        "patch_dim":
            PATCH_DIM,

        "parent_patch_shape":
            [
                8,
                6,
            ],

        "controlled_change":
            (
                "Sampling resolution only: "
                "8x6 -> 16x12."
            ),

        "unchanged_semantics": [
            "same Perlin seed",
            "same compiled MuJoCo hfield",
            "same reset semantics",
            "same reset body/goal-aligned frame",
            "same longitudinal domain",
            "same lateral domain",
            "same bilinear interpolation",
            "same start-height subtraction"
        ],

        "sampling_domain":
            base[
                "sampling_domain"
            ],

        "sampling_frame":
            base[
                "sampling_frame"
            ],

        "height_reference":
            base[
                "height_reference"
            ],

        "all_geometry_hashes_regressed":
            True,

        "unique_global_hfields":
            len(
                hashes
            ),

        "next_step":
            (
                "Strict LOCO 1-NN Q-surface "
                "identifiability audit using "
                "the 16x12 context."
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
        "ICRA27 OS-T6.3a HIGH-RESOLUTION "
        "HEIGHT PATCH FREEZE"
    )
    print("=" * 118)

    print(
        "contexts          :",
        len(
            rows
        ),
    )

    print(
        "patch shape       :",
        [
            N_LONG,
            N_LAT,
        ],
    )

    print(
        "patch dimension   :",
        PATCH_DIM,
    )

    print(
        "geometry hashes   :",
        f"{len(hashes)}/{EXPECTED_TOTAL}",
    )

    print(
        "controlled change : "
        "resolution only"
    )

    print()
    print(
        " ",
        OUT_CSV,
    )
    print(
        " ",
        OUT_MANIFEST,
    )

    print()
    print(
        "[ICRA27] OS-T6.3a high-resolution "
        "height patches: FREEZE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
