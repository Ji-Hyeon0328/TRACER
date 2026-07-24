#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = "phase_l1_v0"
ROW_GRANULARITY = "rollout_candidate_outcome"

LABEL_MAP = {
    "reference": "empirical",
    "reject": "reject",
    "shadow_only": "shadow_only",
}

SUPERVISION_ROLE_MAP = {
    "empirical": "reference_anchor",
    "reject": "negative",
    "shadow_only": "near_miss",
    "candidate_positive_later": "future_positive",
}


OUTPUT_FIELDS = [
    "schema_version",
    "row_granularity",

    # provenance
    "source_phase",
    "source_row_index",
    "source_rollout_id",
    "manifest",
    "d5_log_dir",
    "j7_csv",
    "run_dir",

    # situation
    "world_name",
    "context_label",
    "next_context",
    "segment_id",
    "segment_progress",
    "x_start",
    "y_start",
    "abs_y_start",

    # objective / adaptation
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "ram_rho_summary",
    "sigma_uncertainty",

    # baseline action
    "baseline_vx",
    "baseline_yaw",
    "baseline_body_h",
    "baseline_clearance",
    "baseline_enable",

    # candidate
    "candidate_mode",
    "candidate_action_id",
    "candidate_mode_family",
    "candidate_theta_desc",

    "candidate_vx",
    "candidate_yaw",
    "candidate_body_h",
    "candidate_clearance",
    "candidate_enable",

    "candidate_vx_scale",
    "candidate_vx_delta",
    "candidate_yaw_gain",
    "candidate_body_h_delta",
    "candidate_clearance_delta",

    # routing / guard
    "active_used",
    "j7_accepted",
    "j7_projected",
    "j7_failsafe",
    "flat_rows",
    "flat_accepted",
    "flat_accept_rate",

    # outcome
    "status",
    "goal_reached",
    "final_x",
    "final_y",
    "max_abs_y",
    "mean_abs_y",
    "risk_score_lower_better",
    "delta_progress",
    "time_to_goal",
    "termination_reason",

    # supervision
    "decision_raw",
    "decision_label",
    "supervision_role",
    "decision_reason",
]


def normalize_label(raw_decision: str) -> str:
    raw_decision = (raw_decision or "").strip()

    if raw_decision not in LABEL_MAP:
        raise ValueError(
            f"Unsupported K6 decision label: {raw_decision!r}. "
            f"Expected one of {sorted(LABEL_MAP)}"
        )

    return LABEL_MAP[raw_decision]


def build_row(src: dict, row_index: int) -> dict:
    decision_raw = (src.get("decision") or "").strip()
    decision_label = normalize_label(decision_raw)

    source_phase = (src.get("source_phase") or "").strip()
    candidate_mode = (src.get("mode") or "").strip()
    trial_idx = (src.get("idx") or "").strip()

    source_rollout_id = f"{source_phase}:{candidate_mode}:{trial_idx}"

    row = {key: "" for key in OUTPUT_FIELDS}

    row.update({
        "schema_version": SCHEMA_VERSION,
        "row_granularity": ROW_GRANULARITY,

        "source_phase": source_phase,
        "source_row_index": row_index,
        "source_rollout_id": source_rollout_id,

        "manifest": src.get("manifest", ""),
        "d5_log_dir": src.get("d5_log_dir", ""),
        "j7_csv": src.get("j7_csv", ""),
        "run_dir": src.get("run_dir", ""),

        # Do not infer unavailable terrain/context fields.
        "world_name": "",
        "context_label": "",
        "next_context": "",
        "segment_id": "",
        "segment_progress": "",
        "x_start": "",
        "y_start": "",
        "abs_y_start": "",

        # Do not infer beta/RAM values.
        "beta_motion": "",
        "beta_stability": "",
        "beta_energy": "",
        "ram_rho_summary": "",
        "sigma_uncertainty": "",

        # Numerical baseline reference is not present in K6.
        "baseline_vx": "",
        "baseline_yaw": "",
        "baseline_body_h": "",
        "baseline_clearance": "",
        "baseline_enable": "",

        # Preserve categorical candidate identity only.
        "candidate_mode": candidate_mode,
        "candidate_action_id": candidate_mode,
        "candidate_mode_family": src.get("mode_family", ""),
        "candidate_theta_desc": src.get("candidate_theta_desc", ""),

        # Numerical candidate action is intentionally not inferred.
        "candidate_vx": "",
        "candidate_yaw": "",
        "candidate_body_h": "",
        "candidate_clearance": "",
        "candidate_enable": "",

        "candidate_vx_scale": "",
        "candidate_vx_delta": "",
        "candidate_yaw_gain": "",
        "candidate_body_h_delta": "",
        "candidate_clearance_delta": "",

        "active_used": src.get("active_used", ""),
        "j7_accepted": src.get("j7_accepted", ""),
        "j7_projected": src.get("j7_projected", ""),
        "j7_failsafe": src.get("j7_failsafe", ""),
        "flat_rows": src.get("flat_rows", ""),
        "flat_accepted": src.get("flat_accepted", ""),
        "flat_accept_rate": src.get("flat_accept_rate", ""),

        "status": src.get("status", ""),
        "goal_reached": src.get("goal_reached", ""),
        "final_x": src.get("final_x", ""),
        "final_y": src.get("final_y", ""),
        "max_abs_y": src.get("max_abs_y", ""),
        "mean_abs_y": src.get("mean_abs_y", ""),
        "risk_score_lower_better": src.get(
            "risk_score_lower_better", ""
        ),

        # Not available in K6 aggregate rows.
        "delta_progress": "",
        "time_to_goal": "",
        "termination_reason": "",

        "decision_raw": decision_raw,
        "decision_label": decision_label,
        "supervision_role": SUPERVISION_ROLE_MAP[decision_label],
        "decision_reason": src.get("decision_reason", ""),
    })

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="datasets/phase_k/k6_selector_candidate_outcome_dataset_v0.csv",
    )
    parser.add_argument(
        "--output",
        default="datasets/phase_l/l1_selector_dataset_seed_from_k6_v0.csv",
    )
    parser.add_argument(
        "--report",
        default="reports/phase_l/l1_selector_dataset_seed_from_k6_summary_v0.md",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row_index, src in enumerate(reader, start=1):
            rows.append(build_row(src, row_index))

    if not rows:
        raise RuntimeError("No rows found in input dataset")

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(row["decision_label"] for row in rows)
    mode_counts = Counter(row["candidate_mode"] for row in rows)
    source_counts = Counter(row["source_phase"] for row in rows)

    report_lines = [
        "# TRACER Phase-L1 K6 Seed Dataset Summary v0",
        "",
        f"- source: `{input_path}`",
        f"- output: `{output_path}`",
        f"- schema version: `{SCHEMA_VERSION}`",
        f"- row granularity: `{ROW_GRANULARITY}`",
        f"- rows: {len(rows)}",
        "",
        "## Decision labels",
        "",
    ]

    for label, count in sorted(label_counts.items()):
        report_lines.append(f"- `{label}`: {count}")

    report_lines += [
        "",
        "## Source phases",
        "",
    ]

    for source, count in sorted(source_counts.items()):
        report_lines.append(f"- `{source}`: {count}")

    report_lines += [
        "",
        "## Candidate modes",
        "",
    ]

    for mode, count in sorted(mode_counts.items()):
        report_lines.append(f"- `{mode}`: {count}")

    report_lines += [
        "",
        "## Interpretation",
        "",
        "- This is a rollout-level candidate outcome seed dataset.",
        "- It is not yet a state/context-conditioned selector training dataset.",
        "- K6 does not contain explicit context, beta, RAM rho/sigma, or numerical candidate action fields.",
        "- Missing input features are intentionally left empty rather than inferred from candidate names.",
        "- `reference` is normalized to the canonical `empirical` label.",
        "- `shadow_only` is treated as near-miss supervision, not a positive deployment label.",
        "- No `candidate_positive_later` samples are created in this step.",
        "- Active runtime remains empirical/default.",
        "",
        "## Next step",
        "",
        "Build a broader Phase-L dataset that logs state/context/objective/action features",
        "at segment or candidate-decision time, then join those inputs with rollout outcomes.",
        "",
    ]

    report_path.write_text("\n".join(report_lines))

    print(f"[L1] source rows: {len(rows)}")
    print(f"[L1] labels: {dict(sorted(label_counts.items()))}")
    print(f"[L1] output: {output_path}")
    print(f"[L1] report: {report_path}")


if __name__ == "__main__":
    main()
