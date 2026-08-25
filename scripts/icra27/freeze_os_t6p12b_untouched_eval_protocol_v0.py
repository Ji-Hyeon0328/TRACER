from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_FREEZE = (
    ROOT
    / "results/icra27"
    / "os_t6p11b_selector_candidate_freeze_v0"
    / "selector_candidate_freeze_manifest.json"
)

FINAL_SELECTOR = (
    ROOT
    / "results/icra27"
    / "os_t6p12a_final_factorized_regret_selector_v0"
    / "final_selector_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12b_untouched_eval_protocol_v0"
)

OUT_MANIFEST = (
    OUT_DIR
    / "untouched_eval_protocol_manifest.json"
)


VAL = [1, 21, 16, 14]
TEST = [27, 2, 3, 19, 26]
HARD = [9, 23, 24]

ETA_PROFILES = {
    "balanced": [
        1/3,
        1/3,
        1/3,
    ],

    "motion_biased": [
        0.70,
        0.15,
        0.15,
    ],

    "stability_biased": [
        0.15,
        0.70,
        0.15,
    ],

    "energy_biased": [
        0.15,
        0.15,
        0.70,
    ],
}


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    for path in (
        CANDIDATE_FREEZE,
        FINAL_SELECTOR,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


    candidate = json.loads(
        CANDIDATE_FREEZE.read_text()
    )

    selector = json.loads(
        FINAL_SELECTOR.read_text()
    )


    if candidate.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.11b candidate is not FREEZE_PASS."
        )

    if selector.get(
        "status"
    ) != "FINAL_TRAIN_PASS":
        raise RuntimeError(
            "T6.12a selector is not FINAL_TRAIN_PASS."
        )

    if bool(
        candidate.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "Candidate freeze reports heldout use."
        )

    if bool(
        selector.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "Final selector reports heldout use."
        )


    expected = candidate[
        "untouched_evaluation_contexts"
    ]

    if expected["val"] != VAL:
        raise RuntimeError(
            "VAL split drift."
        )

    if expected["test"] != TEST:
        raise RuntimeError(
            "TEST split drift."
        )

    if expected["hard"] != HARD:
        raise RuntimeError(
            "HARD split drift."
        )


    all_eval = (
        VAL
        + TEST
        + HARD
    )

    if len(set(all_eval)) != 12:
        raise RuntimeError(
            "Evaluation seeds are not 12 unique seeds."
        )


    manifest = {
        "schema":
            "icra27_os_t6p12b_untouched_eval_protocol_v0",

        "status":
            "FREEZE_PASS",

        "heldout_outcomes_used":
            False,

        "selector_source":
            str(
                FINAL_SELECTOR.relative_to(
                    ROOT
                )
            ),

        "evaluation_splits": {
            "val":
                VAL,

            "test":
                TEST,

            "hard":
                HARD,
        },

        "evaluation_roles": {
            "val":
                (
                    "Untouched secondary heldout "
                    "evaluation. No tuning permitted."
                ),

            "test":
                (
                    "Untouched primary standard "
                    "heldout evaluation."
                ),

            "hard":
                (
                    "Untouched stress-test set. "
                    "Reported separately."
                ),
        },

        "standard_heldout_aggregate":
            (
                VAL
                + TEST
            ),

        "standard_heldout_count":
            9,

        "hard_count":
            3,

        "representation":
            "semantic_plus_prop_probe",

        "context_dim":
            87,

        "probe_horizon_s":
            0.4,

        "probe_semantics":
            (
                "Fixed beta-independent nominal-motion "
                "interaction. Only information available "
                "at or before 0.4 s may enter the "
                "Objective Selector."
            ),

        "eta_profiles":
            ETA_PROFILES,

        "primary_eta":
            "balanced",

        "primary_evaluation": {
            "population":
                "TEST",

            "metric":
                (
                    "true balanced augmented-"
                    "Tchebycheff Q excess"
                ),

            "comparison":
                (
                    "frozen context-aware selector "
                    "versus TRAIN-only no-context "
                    "eta-specific baseline"
                ),

            "lower_is_better":
                True,

            "additional_metrics": [
                "beat/tie/worse versus baseline",
                "true beta rank",
                "top3 fraction",
                "top5 fraction",
            ],
        },

        "secondary_evaluation": {
            "standard_heldout":
                "VAL+TEST, n=9",

            "hard":
                "HARD stress test, n=3",

            "eta_semantics": [
                "motion_biased",
                "stability_biased",
                "energy_biased",
            ],
        },

        "frozen_physical_objectives": {
            "motion":
                "J_M = decision_time / progress",

            "stability":
                (
                    "J_S = max(attitude normalized "
                    "RMS, established slip fraction)"
                ),

            "energy":
                (
                    "J_E = mechanical absolute "
                    "energy / progress"
                ),
        },

        "evaluation_order": [
            (
                "1. Generate heldout terrain geometry "
                "and 0.4 s causal probe observations "
                "without policy-outcome labels."
            ),

            (
                "2. Construct frozen 87D context."
            ),

            (
                "3. Apply the frozen nine-checkpoint "
                "T6.12a selector and save all predicted "
                "regret surfaces and selected beta."
            ),

            (
                "4. Freeze those predictions before "
                "opening heldout 21-beta physical "
                "outcomes."
            ),

            (
                "5. Generate the physical 21-beta atlas "
                "using the already-frozen policy/"
                "evaluator."
            ),

            (
                "6. Evaluate saved selections against "
                "true physical Q and TRAIN-only "
                "eta-specific baselines."
            ),
        ],

        "forbidden_after_this_freeze": [
            "change context representation",
            "change probe horizon",
            "change probe command",
            "change MLP architecture",
            "change hidden width",
            "change ensemble seeds",
            "change training epochs",
            "change regret target",
            "change beta lattice",
            "change physical objective definitions",
            "change Tchebycheff rho",
            "change primary eta",
            "retrain using VAL",
            "retrain using TEST",
            "retrain using HARD",
            "select a different architecture based on heldout outcomes",
        ],

        "scientific_interpretation": (
            "All 12 evaluation terrains remain "
            "untouched with respect to selector "
            "architecture and training. Any failure "
            "after this point is retained as an "
            "untouched generalization result rather "
            "than used for model selection."
        ),
    }


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    print("=" * 118)
    print(
        "ICRA27 OS-T6.12b UNTOUCHED "
        "EVALUATION PROTOCOL FREEZE"
    )
    print("=" * 118)

    print(
        "VAL                  :",
        VAL,
    )

    print(
        "TEST                 :",
        TEST,
    )

    print(
        "HARD                 :",
        HARD,
    )

    print(
        "standard heldout     :",
        VAL + TEST,
    )

    print(
        "primary eta          :",
        "balanced",
    )

    print(
        "context              :",
        "87D @ 0.4 s probe boundary",
    )

    print(
        "evaluation ordering  :",
        (
            "context -> prediction freeze "
            "-> physical atlas -> score"
        ),
    )

    print()
    print(
        "[ICRA27] OS-T6.12b untouched "
        "evaluation protocol: FREEZE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
