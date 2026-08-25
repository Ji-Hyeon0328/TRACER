from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


PREFIX_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5f2_selector_train_extension_atlas_v0"
)

PREFIX_CSV = (
    PREFIX_ROOT
    / "episodes_partial.csv"
)


TAIL_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5f2_tail_selector_train_extension_atlas_v0"
)

TAIL_CSV = (
    TAIL_ROOT
    / "episodes.csv"
)

TAIL_MANIFEST = (
    TAIL_ROOT
    / "atlas_manifest.json"
)


EXTENSION_CONFIG = (
    ROOT
    / "configs/icra27"
    / "os_selector_train_extension_v0.json"
)


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5f2r_recovered_selector_extension_atlas_v0"
)

OUT_CSV = (
    OUT_DIR
    / "episodes.csv"
)

OUT_CONTEXT = (
    OUT_DIR
    / "context_feasibility_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "recovered_atlas_manifest.json"
)


EXPECTED_SEEDS = list(
    range(30, 48)
)

EXPECTED_BETAS = 21
EXPECTED_PREFIX_BETAS = 16
EXPECTED_TAIL_BETAS = 5

EXPECTED_PREFIX_ROWS = (
    EXPECTED_PREFIX_BETAS
    * len(EXPECTED_SEEDS)
)

EXPECTED_TAIL_ROWS = (
    EXPECTED_TAIL_BETAS
    * len(EXPECTED_SEEDS)
)

EXPECTED_TOTAL_ROWS = (
    EXPECTED_BETAS
    * len(EXPECTED_SEEDS)
)


TOL = 1.0e-10


def read_csv(path):
    with path.open(
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path,
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


def as_int(x):
    return int(
        float(x)
    )


def as_float(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            x
        )

    return y


def as_bool(x):
    value = str(
        x
    ).strip().lower()

    if value in (
        "1",
        "true",
        "yes",
    ):
        return True

    if value in (
        "0",
        "false",
        "no",
    ):
        return False

    raise ValueError(
        f"Unexpected boolean: {x!r}"
    )


def row_feasible(
    row,
):
    # Frozen OS-T5.2b feasibility contract.
    #
    # F(tau) =
    # success AND no M4 AND positive progress.

    return (
        row[
            "status"
        ] == "success"
        and as_bool(
            row[
                "success"
            ]
        )
        and as_int(
            row[
                "m4_interventions"
            ]
        ) == 0
        and as_float(
            row[
                "progress_m"
            ]
        ) > 0.0
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )


    for path in (
        PREFIX_CSV,
        TAIL_CSV,
        TAIL_MANIFEST,
        EXTENSION_CONFIG,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    extension = json.loads(
        EXTENSION_CONFIG.read_text()
    )

    if extension.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Extension seed contract "
            "is not FREEZE_PASS."
        )

    if extension[
        "rough_train_extension_seeds"
    ] != EXPECTED_SEEDS:
        raise RuntimeError(
            "Extension seeds changed."
        )

    if bool(
        extension[
            "external_heldout_used"
        ]
    ):
        raise RuntimeError(
            "Extension contract used heldout."
        )


    tail_manifest = json.loads(
        TAIL_MANIFEST.read_text()
    )

    if tail_manifest.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "Tail atlas is not COMPUTE_PASS."
        )

    if bool(
        tail_manifest[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "Tail atlas used heldout."
        )

    if int(
        tail_manifest[
            "episodes"
        ]
    ) != EXPECTED_TAIL_ROWS:
        raise RuntimeError(
            "Tail manifest episode mismatch."
        )


    prefix = read_csv(
        PREFIX_CSV
    )

    tail = read_csv(
        TAIL_CSV
    )


    if len(
        prefix
    ) != EXPECTED_PREFIX_ROWS:
        raise RuntimeError(
            f"Prefix rows={len(prefix)}, "
            f"expected={EXPECTED_PREFIX_ROWS}"
        )

    if len(
        tail
    ) != EXPECTED_TAIL_ROWS:
        raise RuntimeError(
            f"Tail rows={len(tail)}, "
            f"expected={EXPECTED_TAIL_ROWS}"
        )


    # --------------------------------------------------------
    # Source annotation.
    #
    # Do not mutate original source artifacts.
    # --------------------------------------------------------

    combined = []

    for source_part, source_root, rows in (
        (
            "prefix_partial",
            PREFIX_ROOT,
            prefix,
        ),
        (
            "recovery_tail",
            TAIL_ROOT,
            tail,
        ),
    ):
        for row in rows:
            out = dict(
                row
            )

            out[
                "f2r_source_part"
            ] = source_part

            out[
                "f2r_raw_root"
            ] = str(
                (
                    source_root
                    / "env_logs"
                ).relative_to(
                    ROOT
                )
            )

            out[
                "f2r_feasible"
            ] = int(
                row_feasible(
                    row
                )
            )

            combined.append(
                out
            )


    if len(
        combined
    ) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            "Combined row mismatch."
        )


    # --------------------------------------------------------
    # Unique beta x seed full grid.
    # --------------------------------------------------------

    pairs = [
        (
            row[
                "beta_name"
            ],
            as_int(
                row[
                    "seed"
                ]
            ),
        )
        for row in combined
    ]


    if len(
        set(
            pairs
        )
    ) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            "Duplicate beta x seed rows."
        )


    beta_names = []

    for row in combined:
        name = row[
            "beta_name"
        ]

        if name not in beta_names:
            beta_names.append(
                name
            )


    if len(
        beta_names
    ) != EXPECTED_BETAS:
        raise RuntimeError(
            f"Expected 21 betas, "
            f"got {len(beta_names)}"
        )


    for beta_name in beta_names:
        beta_rows = [
            row
            for row in combined
            if row[
                "beta_name"
            ] == beta_name
        ]

        seeds = sorted(
            as_int(
                row[
                    "seed"
                ]
            )
            for row in beta_rows
        )

        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"{beta_name}: seed grid "
                f"is incomplete."
            )


    # --------------------------------------------------------
    # Provenance invariants.
    # --------------------------------------------------------

    checkpoints = {
        row[
            "checkpoint"
        ]
        for row in combined
    }

    trained_modes = {
        row[
            "trained_reward_mode"
        ]
        for row in combined
    }

    eval_modes = {
        row[
            "eval_reward_mode"
        ]
        for row in combined
    }

    groups = {
        row[
            "group"
        ]
        for row in combined
    }

    terrains = {
        row[
            "terrain"
        ]
        for row in combined
    }


    if len(
        checkpoints
    ) != 1:
        raise RuntimeError(
            "Checkpoint provenance mismatch."
        )

    if trained_modes != {
        "tracer_cost_v4"
    }:
        raise RuntimeError(
            f"Unexpected trained modes: "
            f"{trained_modes}"
        )

    if eval_modes != {
        "tracer_cost_v4"
    }:
        raise RuntimeError(
            f"Unexpected eval modes: "
            f"{eval_modes}"
        )

    if groups != {
        "rough"
    }:
        raise RuntimeError(
            f"Unexpected groups: "
            f"{groups}"
        )

    if terrains != {
        "rough_perlin"
    }:
        raise RuntimeError(
            f"Unexpected terrains: "
            f"{terrains}"
        )


    # --------------------------------------------------------
    # Raw log provenance.
    #
    # Validate counts in original source roots.
    # No copying or renaming.
    # --------------------------------------------------------

    for beta_index, beta_name in enumerate(
        beta_names
    ):
        if beta_index < EXPECTED_PREFIX_BETAS:
            raw_root = (
                PREFIX_ROOT
                / "env_logs"
            )

        else:
            raw_root = (
                TAIL_ROOT
                / "env_logs"
            )


        raw_dir = (
            raw_root
            / beta_name
            / "rough"
        )


        files = sorted(
            raw_dir.glob(
                "episode_*.jsonl"
            )
        )


        if len(
            files
        ) != len(
            EXPECTED_SEEDS
        ):
            raise RuntimeError(
                f"{beta_name}: "
                f"expected 18 raw logs, "
                f"got {len(files)} "
                f"in {raw_dir}"
            )


    # --------------------------------------------------------
    # Episode-level status.
    # --------------------------------------------------------

    status_counts = Counter(
        row[
            "status"
        ]
        for row in combined
    )


    feasible_count = sum(
        int(
            row[
                "f2r_feasible"
            ]
        )
        for row in combined
    )


    # --------------------------------------------------------
    # Context-level feasibility summary.
    # --------------------------------------------------------

    by_seed = defaultdict(
        list
    )

    for row in combined:
        by_seed[
            as_int(
                row[
                    "seed"
                ]
            )
        ].append(
            row
        )


    context_rows = []

    full_grid_eligible = []

    partial_response = []

    zero_feasible = []


    for seed in EXPECTED_SEEDS:
        rows = by_seed[
            seed
        ]

        if len(
            rows
        ) != EXPECTED_BETAS:
            raise RuntimeError(
                f"seed {seed}: "
                f"expected 21 rows."
            )


        counts = Counter(
            row[
                "status"
            ]
            for row in rows
        )


        feasible = sum(
            int(
                row[
                    "f2r_feasible"
                ]
            )
            for row in rows
        )


        if feasible == EXPECTED_BETAS:
            eligibility = (
                "full_21_beta_eligible"
            )

            full_grid_eligible.append(
                seed
            )

        elif feasible == 0:
            eligibility = (
                "zero_feasible_beta"
            )

            zero_feasible.append(
                seed
            )

        else:
            eligibility = (
                "partial_beta_response"
            )

            partial_response.append(
                seed
            )


        context_rows.append(
            {
                "context_id":
                    f"rough_seed_{seed}",

                "seed":
                    seed,

                "beta_rows":
                    EXPECTED_BETAS,

                "feasible_beta_rows":
                    feasible,

                "success_rows":
                    counts.get(
                        "success",
                        0,
                    ),

                "m4_rows":
                    counts.get(
                        "m4",
                        0,
                    ),

                "time_limit_rows":
                    counts.get(
                        "time_limit",
                        0,
                    ),

                "full_grid_eligible":
                    int(
                        feasible
                        == EXPECTED_BETAS
                    ),

                "eligibility":
                    eligibility,
            }
        )


    # --------------------------------------------------------
    # Frozen expectations from observed recovery artifact.
    #
    # These are outcome summaries, NOT seed-selection rules.
    # --------------------------------------------------------

    expected_full = [
        seed
        for seed in EXPECTED_SEEDS
        if seed not in (
            34,
            43,
            46,
        )
    ]


    if full_grid_eligible != expected_full:
        raise RuntimeError(
            "Unexpected full-grid eligible "
            f"contexts: {full_grid_eligible}"
        )


    if partial_response != [
        46
    ]:
        raise RuntimeError(
            "Unexpected partial-response "
            f"contexts: {partial_response}"
        )


    if zero_feasible != [
        34,
        43,
    ]:
        raise RuntimeError(
            "Unexpected zero-feasible "
            f"contexts: {zero_feasible}"
        )


    # --------------------------------------------------------
    # Write immutable recovered artifact.
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    write_csv(
        OUT_CSV,
        combined,
    )

    write_csv(
        OUT_CONTEXT,
        context_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5f2r_"
                "recovered_selector_extension_"
                "atlas_v0"
            ),

        "status":
            "FREEZE_PASS",

        "heldout_used":
            False,

        "original_split_modified":
            False,

        "checkpoint":
            next(
                iter(
                    checkpoints
                )
            ),

        "reward_mode":
            "tracer_cost_v4",

        "extension_seeds":
            EXPECTED_SEEDS,

        "beta_points":
            EXPECTED_BETAS,

        "episodes":
            EXPECTED_TOTAL_ROWS,

        "episode_status_counts":
            dict(
                sorted(
                    status_counts.items()
                )
            ),

        "feasible_episode_rows":
            feasible_count,

        "infeasible_episode_rows":
            (
                EXPECTED_TOTAL_ROWS
                - feasible_count
            ),

        "full_21_beta_eligible_contexts":
            full_grid_eligible,

        "full_21_beta_eligible_count":
            len(
                full_grid_eligible
            ),

        "partial_response_contexts":
            partial_response,

        "zero_feasible_contexts":
            zero_feasible,

        "prefix_source":
            str(
                PREFIX_CSV.relative_to(
                    ROOT
                )
            ),

        "tail_source":
            str(
                TAIL_CSV.relative_to(
                    ROOT
                )
            ),

        "raw_log_storage":
            (
                "Raw logs remain in their "
                "original prefix and recovery-tail "
                "source roots. The merged CSV "
                "records f2r_source_part and "
                "f2r_raw_root for provenance."
            ),

        "feasibility_contract":
            (
                "Frozen OS-T5.2b: "
                "success AND no M4 AND "
                "positive progress."
            ),

        "selector_training_guard":
            (
                "For an apples-to-apples "
                "T5.5d-compatible 21-point "
                "Pareto/scalarization atlas, "
                "use only contexts with all "
                "21 beta rows feasible. "
                "Do not replace, resample, or "
                "silently delete the other "
                "extension contexts."
            ),

        "next_stage":
            (
                "T5.5f3: reconstruct physical "
                "J_M/J_S/J_E for the 15 "
                "full-grid eligible extension "
                "contexts using the frozen "
                "T5.2b metric contract, then "
                "append them to the immutable "
                "420-row T5.5c atlas."
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
        "ICRA27 OS-T5.5f2r RECOVERED "
        "SELECTOR EXTENSION ATLAS FREEZE"
    )

    print("=" * 118)

    print(
        "episodes               :",
        len(
            combined
        ),
    )

    print(
        "status counts          :",
        dict(
            status_counts
        ),
    )

    print(
        "feasible rows          :",
        feasible_count,
        "/",
        EXPECTED_TOTAL_ROWS,
    )

    print(
        "full-grid eligible     :",
        len(
            full_grid_eligible
        ),
        "/18",
    )

    print(
        "full-grid seeds        :",
        full_grid_eligible,
    )

    print(
        "partial-response seeds :",
        partial_response,
    )

    print(
        "zero-feasible seeds    :",
        zero_feasible,
    )

    print()
    print("outputs:")
    print(" ", OUT_CSV)
    print(" ", OUT_CONTEXT)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5f2r recovered "
        "selector extension atlas: "
        "FREEZE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
