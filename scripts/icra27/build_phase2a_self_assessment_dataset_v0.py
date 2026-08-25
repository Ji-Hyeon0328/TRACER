#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


EXPECTED_BETAS = (
    "balanced",
    "motion",
    "stability",
    "energy",
)

COST_KEYS = (
    "mean_cost_motion",
    "mean_cost_stability",
    "mean_cost_energy",
)

PARETO_TOL = 1e-12


def as_bool(value):
    value = str(value).strip().lower()

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
        "",
    ):
        return False

    raise ValueError(
        f"Cannot parse bool value: {value!r}"
    )


def as_float(row, key):
    return float(
        row[key]
    )


def is_feasible(row):
    """
    Preference-independent locomotion feasibility.

    WATCH/proximity alone is not treated as failure.
    The trajectory must:
      - complete settling,
      - reach task success,
      - avoid terminal Low-Level Safety Supervisor abort.
    """
    settled = as_bool(
        row["settled"]
    )

    success = as_bool(
        row["success"]
    )

    safety_abort = as_bool(
        row["m4_terminal"]
    )

    return (
        settled
        and success
        and not safety_abort
    )


def is_strict_feasible(row):
    """
    Supporting diagnostic.

    In addition to feasibility, require zero recorded
    controller-side safety interventions.
    """
    return (
        is_feasible(row)
        and int(
            row["m4_interventions"]
        ) == 0
    )


def dominates(a, b):
    """
    True iff feasible candidate a Pareto-dominates b.

    All objectives are zero-best costs.
    """
    a_cost = [
        as_float(a, key)
        for key in COST_KEYS
    ]

    b_cost = [
        as_float(b, key)
        for key in COST_KEYS
    ]

    no_worse = all(
        x <= y + PARETO_TOL
        for x, y
        in zip(
            a_cost,
            b_cost,
        )
    )

    strictly_better = any(
        x < y - PARETO_TOL
        for x, y
        in zip(
            a_cost,
            b_cost,
        )
    )

    return (
        no_worse
        and strictly_better
    )


def rank_for_cost(
    candidates,
    key,
):
    feasible = [
        row
        for row in candidates
        if row["_feasible"]
    ]

    ordered = sorted(
        feasible,
        key=lambda row:
            as_float(
                row,
                key,
            ),
    )

    return {
        id(row):
            rank + 1
        for rank, row
        in enumerate(
            ordered
        )
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--episodes",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    episodes_path = Path(
        args.episodes
    )

    out_dir = Path(
        args.out_dir
    )

    if not episodes_path.is_file():
        raise FileNotFoundError(
            episodes_path
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with episodes_path.open(
        newline=""
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if not rows:
        raise RuntimeError(
            "Episodes CSV is empty."
        )

    required_columns = {
        "checkpoint",
        "trained_reward_mode",
        "group",
        "terrain",
        "seed",
        "status",
        "settled",
        "success",
        "m4_terminal",
        "m4_interventions",
        "beta_name",
        "beta_motion",
        "beta_stability",
        "beta_energy",
        "mean_cost_motion",
        "mean_cost_stability",
        "mean_cost_energy",
        "energy_abs_j",
        "decision_time_s",
        "roll_rms_rad",
        "pitch_rms_rad",
        "mean_applied_vx_mps",
    }

    missing = (
        required_columns
        - set(rows[0])
    )

    if missing:
        raise RuntimeError(
            "Missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    checkpoints = {
        row["checkpoint"]
        for row in rows
    }

    if len(checkpoints) != 1:
        raise RuntimeError(
            "Expected exactly one checkpoint; "
            f"found {sorted(checkpoints)}"
        )

    trained_modes = {
        row["trained_reward_mode"]
        for row in rows
    }

    if len(trained_modes) != 1:
        raise RuntimeError(
            "Expected one trained_reward_mode; "
            f"found {sorted(trained_modes)}"
        )

    # --------------------------------------------------------
    # Exact counterfactual context:
    #
    # group + terrain + seed uniquely identifies the physical
    # evaluation condition within the current Phase-1 protocol.
    # --------------------------------------------------------

    contexts = {}

    for row in rows:
        key = (
            row["group"],
            row["terrain"],
            int(row["seed"]),
        )

        contexts.setdefault(
            key,
            []
        ).append(
            row
        )

    expected_context_count = 29

    if len(contexts) != expected_context_count:
        raise RuntimeError(
            "Expected "
            f"{expected_context_count} contexts, "
            f"found {len(contexts)}"
        )

    candidate_records = []
    context_records = []

    total_pareto = 0
    total_feasible = 0
    total_strict_feasible = 0

    pareto_count_histogram = {}

    for context_index, key in enumerate(
        sorted(
            contexts,
            key=lambda x: (
                x[0],
                x[1],
                x[2],
            ),
        )
    ):
        group, terrain, seed = key

        candidates = contexts[
            key
        ]

        beta_names = {
            row["beta_name"]
            for row in candidates
        }

        if beta_names != set(
            EXPECTED_BETAS
        ):
            raise RuntimeError(
                f"{key}: beta bank mismatch: "
                f"{sorted(beta_names)}"
            )

        if len(candidates) != len(
            EXPECTED_BETAS
        ):
            raise RuntimeError(
                f"{key}: expected 4 candidates, "
                f"got {len(candidates)}"
            )

        for row in candidates:
            row["_feasible"] = (
                is_feasible(row)
            )

            row["_strict_feasible"] = (
                is_strict_feasible(row)
            )

        feasible_candidates = [
            row
            for row in candidates
            if row["_feasible"]
        ]

        total_feasible += len(
            feasible_candidates
        )

        total_strict_feasible += sum(
            int(
                row["_strict_feasible"]
            )
            for row in candidates
        )

        # ----------------------------------------------------
        # Pareto membership is computed only within the
        # feasible anchor candidates of this exact context.
        #
        # This is an anchor-Pareto set, not a claim about the
        # global continuous-policy Pareto frontier.
        # ----------------------------------------------------

        pareto_members = []

        for candidate in candidates:
            if not candidate[
                "_feasible"
            ]:
                continue

            dominated = any(
                dominates(
                    other,
                    candidate,
                )
                for other in feasible_candidates
                if other is not candidate
            )

            if not dominated:
                pareto_members.append(
                    candidate
                )

        pareto_ids = {
            id(row)
            for row in pareto_members
        }

        total_pareto += len(
            pareto_members
        )

        pareto_count_histogram[
            str(len(pareto_members))
        ] = (
            pareto_count_histogram.get(
                str(len(pareto_members)),
                0,
            )
            + 1
        )

        motion_rank = rank_for_cost(
            candidates,
            "mean_cost_motion",
        )

        stability_rank = rank_for_cost(
            candidates,
            "mean_cost_stability",
        )

        energy_rank = rank_for_cost(
            candidates,
            "mean_cost_energy",
        )

        context_id = (
            f"{group}:"
            f"{terrain}:"
            f"{seed}"
        )

        context_record = {
            "context_index":
                context_index,

            "context_id":
                context_id,

            "group":
                group,

            "terrain":
                terrain,

            "seed":
                seed,

            "candidate_count":
                len(candidates),

            "feasible_count":
                len(feasible_candidates),

            "pareto_count":
                len(pareto_members),

            "pareto_beta_names":
                [
                    row["beta_name"]
                    for row
                    in pareto_members
                ],
        }

        context_records.append(
            context_record
        )

        for row in candidates:
            feasible = bool(
                row["_feasible"]
            )

            pareto = (
                id(row)
                in pareto_ids
            )

            # "good" is deliberately preference-independent.
            good = (
                feasible
                and pareto
            )

            record = {
                "context_index":
                    context_index,

                "context_id":
                    context_id,

                "group":
                    group,

                "terrain":
                    terrain,

                "seed":
                    seed,

                "beta_name":
                    row["beta_name"],

                "beta_motion":
                    as_float(
                        row,
                        "beta_motion",
                    ),

                "beta_stability":
                    as_float(
                        row,
                        "beta_stability",
                    ),

                "beta_energy":
                    as_float(
                        row,
                        "beta_energy",
                    ),

                "feasible":
                    feasible,

                "strict_feasible":
                    bool(
                        row[
                            "_strict_feasible"
                        ]
                    ),

                "pareto_nondominated":
                    pareto,

                "good_candidate":
                    good,

                "cost_motion":
                    as_float(
                        row,
                        "mean_cost_motion",
                    ),

                "cost_stability":
                    as_float(
                        row,
                        "mean_cost_stability",
                    ),

                "cost_energy":
                    as_float(
                        row,
                        "mean_cost_energy",
                    ),

                "motion_rank":
                    (
                        motion_rank.get(
                            id(row)
                        )
                        if feasible
                        else None
                    ),

                "stability_rank":
                    (
                        stability_rank.get(
                            id(row)
                        )
                        if feasible
                        else None
                    ),

                "energy_rank":
                    (
                        energy_rank.get(
                            id(row)
                        )
                        if feasible
                        else None
                    ),

                "episode_energy_abs_j":
                    as_float(
                        row,
                        "energy_abs_j",
                    ),

                "decision_time_s":
                    as_float(
                        row,
                        "decision_time_s",
                    ),

                "roll_rms_rad":
                    as_float(
                        row,
                        "roll_rms_rad",
                    ),

                "pitch_rms_rad":
                    as_float(
                        row,
                        "pitch_rms_rad",
                    ),

                "mean_applied_vx_mps":
                    as_float(
                        row,
                        "mean_applied_vx_mps",
                    ),
            }

            candidate_records.append(
                record
            )

    if len(candidate_records) != 116:
        raise RuntimeError(
            "Expected 116 candidate records, "
            f"got {len(candidate_records)}"
        )

    # --------------------------------------------------------
    # Outputs.
    # --------------------------------------------------------

    candidate_csv = (
        out_dir
        / "self_assessment_candidates.csv"
    )

    with candidate_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                candidate_records[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            candidate_records
        )

    contexts_json = (
        out_dir
        / "self_assessment_contexts.json"
    )

    contexts_json.write_text(
        json.dumps(
            context_records,
            indent=2,
        )
        + "\n"
    )

    good_count = sum(
        int(
            row["good_candidate"]
        )
        for row in candidate_records
    )

    summary = {
        "schema":
            "icra27_phase2a_self_assessment_v0",

        "source_episodes":
            str(episodes_path),

        "checkpoint":
            next(
                iter(checkpoints)
            ),

        "trained_reward_mode":
            next(
                iter(trained_modes)
            ),

        "context_definition":
            [
                "group",
                "terrain",
                "seed",
            ],

        "objective_definition":
            [
                "mean_cost_motion",
                "mean_cost_stability",
                "mean_cost_energy",
            ],

        "feasibility_definition":
            (
                "settled AND success AND "
                "NOT m4_terminal"
            ),

        "good_candidate_definition":
            (
                "feasible AND "
                "anchor-Pareto-nondominated"
            ),

        "pareto_scope":
            (
                "four beta-anchor candidates "
                "within the exact paired context"
            ),

        "context_count":
            len(context_records),

        "candidate_count":
            len(candidate_records),

        "feasible_candidate_count":
            total_feasible,

        "strict_feasible_candidate_count":
            total_strict_feasible,

        "pareto_candidate_count":
            total_pareto,

        "good_candidate_count":
            good_count,

        "pareto_count_per_context_histogram":
            pareto_count_histogram,

        "important_note":
            (
                "This dataset performs self-assessment only. "
                "It does not assign a unique preferred beta. "
                "Preference inference requires expert, mission, "
                "or other selection evidence among good alternatives."
            ),
    }

    summary_path = (
        out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print(
        "=" * 80
    )

    print(
        "ICRA27 PHASE-2A SELF-ASSESSMENT DATASET V0"
    )

    print(
        "=" * 80
    )

    print(
        "contexts          :",
        len(context_records),
    )

    print(
        "candidates        :",
        len(candidate_records),
    )

    print(
        "feasible          :",
        total_feasible,
    )

    print(
        "strict feasible   :",
        total_strict_feasible,
    )

    print(
        "Pareto candidates :",
        total_pareto,
    )

    print(
        "good candidates   :",
        good_count,
    )

    print(
        "Pareto/context hist:",
        pareto_count_histogram,
    )

    print()
    print(
        "candidate csv     :",
        candidate_csv,
    )

    print(
        "contexts json     :",
        contexts_json,
    )

    print(
        "summary           :",
        summary_path,
    )

    print()
    print(
        "[ICRA27] Phase-2A self-assessment "
        "dataset: PASS"
    )


if __name__ == "__main__":
    main()
