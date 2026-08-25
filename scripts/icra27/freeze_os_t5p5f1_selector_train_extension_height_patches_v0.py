from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EXT_CFG = (
    ROOT
    / "configs/icra27"
    / "os_selector_train_extension_v0.json"
)

WORKER = (
    ROOT
    / "scripts/icra27"
    / "probe_os_t5p5a_height_patch_seed_v0.py"
)

BASE_PATCH_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_height_patch_v0"
)

BASE_CONTRACT = (
    BASE_PATCH_ROOT
    / "oracle_height_patch_contract.json"
)

BASE_GEOMETRY = (
    BASE_PATCH_ROOT
    / "height_patch_geometry_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5f1_selector_train_extension_height_patches_v0"
)

OUT_CSV = (
    OUT_DIR
    / "extension_height_patches.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "extension_height_patch_manifest.json"
)


EXPECTED_SEEDS = list(
    range(30, 48)
)

EXPECTED_COUNT = 18
EXPECTED_PATCH_DIM = 48


def write_csv(
    path: Path,
    rows: list[dict],
):
    if not rows:
        raise RuntimeError(
            f"No rows: {path}"
        )

    fields = []

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


def run_worker(
    seed: int,
) -> dict:
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
            f"Patch worker failed for "
            f"seed={seed}, rc={proc.returncode}"
        )

    lines = [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise RuntimeError(
            f"Expected exactly one JSON line "
            f"for seed={seed}; got {len(lines)}"
        )

    row = json.loads(
        lines[0]
    )

    if int(
        row["seed"]
    ) != seed:
        raise RuntimeError(
            f"Seed mismatch: "
            f"requested={seed}, "
            f"returned={row['seed']}"
        )

    if row["context_id"] != (
        f"rough_seed_{seed}"
    ):
        raise RuntimeError(
            f"Context-id mismatch: "
            f"{row['context_id']}"
        )

    return row


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        EXT_CFG,
        WORKER,
        BASE_CONTRACT,
        BASE_GEOMETRY,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Frozen extension contract
    # ========================================================

    extension = json.loads(
        EXT_CFG.read_text()
    )

    if extension.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Extension config is not "
            "FREEZE_PASS."
        )

    if bool(
        extension[
            "original_split_modified"
        ]
    ):
        raise RuntimeError(
            "Extension config reports "
            "original split modification."
        )

    if bool(
        extension[
            "external_heldout_used"
        ]
    ):
        raise RuntimeError(
            "Extension config reports "
            "heldout usage."
        )

    seeds = [
        int(x)
        for x in extension[
            "rough_train_extension_seeds"
        ]
    ]

    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            f"Unexpected extension seeds: "
            f"{seeds}"
        )


    # ========================================================
    # Parent patch representation contract
    # ========================================================

    base_contract = json.loads(
        BASE_CONTRACT.read_text()
    )

    if base_contract.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Base height patch is not "
            "FREEZE_PASS."
        )

    if bool(
        base_contract[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "Base patch reports heldout use."
        )

    if int(
        base_contract[
            "patch_dim"
        ]
    ) != EXPECTED_PATCH_DIM:
        raise RuntimeError(
            "Base patch dimension mismatch."
        )

    if list(
        base_contract[
            "patch_shape"
        ]
    ) != [8, 6]:
        raise RuntimeError(
            "Expected frozen patch shape [8,6]."
        )

    patch_features = [
        feature
        for feature in (
            base_contract[
                "feature_order"
            ]
        )
        if feature.startswith(
            "h_l"
        )
    ]

    if len(
        patch_features
    ) != EXPECTED_PATCH_DIM:
        raise RuntimeError(
            "Expected 48 frozen patch features."
        )


    # ========================================================
    # Original rough geometry hashes
    # ========================================================

    base_geometry = json.loads(
        BASE_GEOMETRY.read_text()
    )

    if base_geometry.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Base geometry manifest is not "
            "FREEZE_PASS."
        )

    old_hashes = set(
        base_geometry[
            "global_hfield_hashes"
        ].values()
    )

    if len(
        old_hashes
    ) != 18:
        raise RuntimeError(
            "Expected 18 unique original "
            "rough hfields."
        )


    # ========================================================
    # Generate extension patches
    # ========================================================

    rows = []

    extension_hashes = set()

    patch_vectors = []


    for index, seed in enumerate(
        seeds,
        start=1,
    ):
        print(
            f"extension rough seed "
            f"{index:02d}/{len(seeds)}: "
            f"{seed}"
        )

        row = run_worker(
            seed
        )

        h = row[
            "geometry_sha256"
        ]

        if not h:
            raise RuntimeError(
                f"Missing geometry hash: "
                f"seed={seed}"
            )

        if h in extension_hashes:
            raise RuntimeError(
                f"Duplicate extension hfield: "
                f"seed={seed}"
            )

        if h in old_hashes:
            raise RuntimeError(
                "Extension hfield duplicates "
                "an original TRAIN hfield: "
                f"seed={seed}"
            )

        extension_hashes.add(
            h
        )


        vector = tuple(
            round(
                float(
                    row[
                        feature
                    ]
                ),
                12,
            )
            for feature in (
                patch_features
            )
        )

        patch_vectors.append(
            vector
        )

        rows.append(
            row
        )

        print(
            "  "
            f"range="
            f"[{float(row['patch_min_m']):+.6f}, "
            f"{float(row['patch_max_m']):+.6f}] "
            f"std="
            f"{float(row['patch_std_m']):.6f}"
        )


    if len(
        rows
    ) != EXPECTED_COUNT:
        raise RuntimeError(
            "Expected 18 extension rows."
        )

    if len(
        extension_hashes
    ) != EXPECTED_COUNT:
        raise RuntimeError(
            "Expected 18/18 unique extension "
            "global hfields."
        )

    unique_patches = len(
        set(
            patch_vectors
        )
    )

    if unique_patches != EXPECTED_COUNT:
        raise RuntimeError(
            "Expected 18/18 unique extension "
            "local patches."
        )


    # ========================================================
    # Combined geometry sanity
    # ========================================================

    combined_hashes = (
        old_hashes
        | extension_hashes
    )

    if len(
        combined_hashes
    ) != 36:
        raise RuntimeError(
            "Expected 36/36 unique original + "
            "extension global hfields."
        )


    # ========================================================
    # Write frozen extension artifact
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_CSV,
        rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5f1_"
                "selector_train_extension_"
                "height_patches_v0"
            ),

        "status":
            "FREEZE_PASS",

        "external_heldout_used":
            False,

        "original_split_modified":
            False,

        "extension_seed_contract":
            (
                "configs/icra27/"
                "os_selector_train_extension_v0.json"
            ),

        "representation_parent":
            (
                "os_t5p5a_oracle_"
                "height_patch_v0"
            ),

        "patch_shape":
            [8, 6],

        "patch_dim":
            EXPECTED_PATCH_DIM,

        "extension_seeds":
            seeds,

        "extension_contexts":
            EXPECTED_COUNT,

        "extension_unique_global_hfields":
            len(
                extension_hashes
            ),

        "extension_unique_local_patches":
            unique_patches,

        "overlap_with_original_global_hfields":
            0,

        "combined_original_plus_extension_rough":
            36,

        "combined_unique_global_hfields":
            len(
                combined_hashes
            ),

        "generation":
            (
                "Existing frozen "
                "probe_os_t5p5a_height_patch_"
                "seed_v0.py worker reused "
                "unchanged, one fresh process "
                "per extension seed."
            ),

        "interpretation_guard":
            (
                "TRAIN-only terrain-context "
                "coverage extension. "
                "No beta response, mission "
                "preference, validation/test/hard "
                "terrain outcome, or selector "
                "performance was used to choose "
                "or filter these contexts."
            ),

        "next_stage":
            (
                "Generate the frozen 21-beta "
                "physical response atlas for "
                "these 18 extension terrains: "
                "18 x 21 = 378 episodes."
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
        "ICRA27 OS-T5.5f1 SELECTOR TRAIN "
        "EXTENSION HEIGHT PATCH FREEZE"
    )

    print("=" * 118)

    print(
        "extension seeds             :",
        len(
            seeds
        ),
    )

    print(
        "extension hfields unique    :",
        len(
            extension_hashes
        ),
        "/18",
    )

    print(
        "extension patches unique    :",
        unique_patches,
        "/18",
    )

    print(
        "original overlap            :",
        0,
    )

    print(
        "combined rough hfields      :",
        len(
            combined_hashes
        ),
        "/36 unique",
    )

    print()
    print("outputs:")
    print(" ", OUT_CSV)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5f1 selector "
        "TRAIN extension height patches: "
        "FREEZE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
