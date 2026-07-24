#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean


LABEL_MAP = {
    "reference": "empirical",
    "reject": "reject",
    "shadow_only": "shadow_only",
}


FIELDS = [
    "schema_version",
    "row_granularity",

    "source_phase",
    "candidate_mode",
    "trial_idx",
    "source_rollout_id",

    "window_id",
    "window_type",
    "context",
    "action_id",

    "t_start",
    "t_end",
    "duration_s",

    "window_rows",
    "state_rows",
    "state_coverage_ratio",

    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "mean_abs_y_window",
    "max_abs_y_window",

    "beta_motion_mean",
    "beta_stability_mean",
    "beta_energy_mean",

    "ram_slip_mean",
    "ram_roughness_mean",
    "ram_sigma_mean",

    "emp_vx_mean",
    "emp_yaw_rate_mean",
    "emp_body_h_mean",
    "emp_clearance_mean",
    "emp_enable_mean",

    "proj_vx_mean",
    "proj_yaw_rate_mean",
    "proj_body_h_mean",
    "proj_clearance_mean",
    "proj_enable_mean",

    "out_vx_mean",
    "out_yaw_rate_mean",
    "out_body_h_mean",
    "out_clearance_mean",
    "out_enable_mean",

    "delta_vx_mean",
    "delta_yaw_rate_mean",
    "delta_body_h_mean",
    "delta_clearance_mean",

    "vx_scale_mean",
    "vx_delta_mean",
    "yaw_gain_scale_mean",
    "body_h_delta_mean",
    "clearance_delta_mean",

    "goal_reached",
    "final_x",
    "final_y",
    "rollout_max_abs_y",
    "rollout_mean_abs_y",
    "rollout_risk_score",

    "rollout_decision_raw",
    "rollout_decision_label",

    "evidence_role",
    "direct_action_label_allowed",

    "manifest",
    "d5_log_dir",
    "j7_csv",
    "run_dir",
]


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        s = str(value).strip()
        if not s or s.lower() == "nan":
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def mean_field(rows, field):
    values = [
        v for v in (to_float(r.get(field)) for r in rows)
        if v is not None
    ]
    if not values:
        return ""
    return mean(values)


def first_float(rows, field):
    for row in rows:
        value = to_float(row.get(field))
        if value is not None:
            return value
    return ""


def last_float(rows, field):
    for row in reversed(rows):
        value = to_float(row.get(field))
        if value is not None:
            return value
    return ""


def abs_y_stats(rows):
    values = [
        abs(v)
        for v in (to_float(r.get("y")) for r in rows)
        if v is not None
    ]

    if not values:
        return "", ""

    return mean(values), max(values)


def is_true(value):
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def known_context(row):
    context = str(row.get("context", "")).strip()
    return context not in {"", "unknown"}


def timestamp(row, field="t_wall"):
    return to_float(row.get(field))


def group_contiguous(
    rows,
    time_field,
    key_fields,
    max_gap_s=0.25,
):
    rows = [
        row for row in rows
        if timestamp(row, time_field) is not None
    ]

    rows.sort(key=lambda r: timestamp(r, time_field))

    groups = []
    current = []

    for row in rows:
        if not current:
            current = [row]
            continue

        prev = current[-1]

        same_key = all(
            str(row.get(k, "")) == str(prev.get(k, ""))
            for k in key_fields
        )

        gap = (
            timestamp(row, time_field)
            - timestamp(prev, time_field)
        )

        if same_key and gap <= max_gap_s:
            current.append(row)
        else:
            groups.append(current)
            current = [row]

    if current:
        groups.append(current)

    return groups


def rows_in_window(
    rows,
    t_start,
    t_end,
    context=None,
    padding_s=0.0,
):
    selected = []

    for row in rows:
        t = timestamp(row)

        if t is None:
            continue

        if t < t_start - padding_s:
            continue

        if t > t_end + padding_s:
            continue

        if context is not None:
            if str(row.get("context", "")) != context:
                continue

        selected.append(row)

    return selected


def empty_row():
    return {field: "" for field in FIELDS}


def rollout_label(k6_row):
    raw = str(k6_row.get("decision", "")).strip()
    return LABEL_MAP.get(raw, raw)


def add_rollout_fields(out, k6):
    out.update({
        "goal_reached": k6.get("goal_reached", ""),
        "final_x": k6.get("final_x", ""),
        "final_y": k6.get("final_y", ""),
        "rollout_max_abs_y": k6.get("max_abs_y", ""),
        "rollout_mean_abs_y": k6.get("mean_abs_y", ""),
        "rollout_risk_score": k6.get(
            "risk_score_lower_better",
            "",
        ),
        "rollout_decision_raw": k6.get("decision", ""),
        "rollout_decision_label": rollout_label(k6),
        "manifest": k6.get("manifest", ""),
        "d5_log_dir": k6.get("d5_log_dir", ""),
        "j7_csv": k6.get("j7_csv", ""),
        "run_dir": k6.get("run_dir", ""),
    })


def build_baseline_windows(k6, policy_rows):
    valid = [
        row for row in policy_rows
        if known_context(row)
        and timestamp(row) is not None
    ]

    groups = group_contiguous(
        valid,
        time_field="t_wall",
        key_fields=["context"],
    )

    outputs = []

    for group_idx, group in enumerate(groups, start=1):
        context = group[0]["context"]

        t_start = timestamp(group[0])
        t_end = timestamp(group[-1])

        mean_abs_y, max_abs_y = abs_y_stats(group)

        out = empty_row()

        out.update({
            "schema_version": "phase_l2_v0",
            "row_granularity": "context_action_window",

            "source_phase": k6.get("source_phase", ""),
            "candidate_mode": "baseline",
            "trial_idx": k6.get("idx", ""),
            "source_rollout_id": (
                f"{k6.get('source_phase', '')}:"
                f"baseline:"
                f"{k6.get('idx', '')}"
            ),

            "window_id": f"baseline:{context}:{group_idx}",
            "window_type": "empirical_context_window",
            "context": context,
            "action_id": "empirical",

            "t_start": t_start,
            "t_end": t_end,
            "duration_s": t_end - t_start,

            "window_rows": len(group),
            "state_rows": len(group),
            "state_coverage_ratio": 1.0,

            "x_start": first_float(group, "x"),
            "y_start": first_float(group, "y"),
            "x_end": last_float(group, "x"),
            "y_end": last_float(group, "y"),
            "mean_abs_y_window": mean_abs_y,
            "max_abs_y_window": max_abs_y,

            "beta_motion_mean": mean_field(
                group, "beta_motion"
            ),
            "beta_stability_mean": mean_field(
                group, "beta_stability"
            ),
            "beta_energy_mean": mean_field(
                group, "beta_energy"
            ),

            "ram_slip_mean": mean_field(
                group, "ram_slip_proxy"
            ),
            "ram_roughness_mean": mean_field(
                group, "ram_roughness_proxy"
            ),
            "ram_sigma_mean": mean_field(
                group, "ram_sigma"
            ),

            "emp_vx_mean": mean_field(group, "ref_vx"),
            "emp_yaw_rate_mean": mean_field(
                group, "ref_yaw_rate"
            ),
            "emp_body_h_mean": mean_field(
                group, "ref_body_h"
            ),
            "emp_clearance_mean": mean_field(
                group, "ref_clearance"
            ),
            "emp_enable_mean": mean_field(
                group, "ref_enable"
            ),

            "evidence_role": "reference_anchor",
            "direct_action_label_allowed": 0,
        })

        add_rollout_fields(out, k6)
        outputs.append(out)

    return outputs


def build_candidate_windows(
    k6,
    policy_rows,
    j4_rows,
    j7_rows,
):
    accepted = [
        row for row in j7_rows
        if known_context(row)
        and timestamp(row) is not None
        and is_true(row.get("j7_accept"))
        and str(row.get("j7_source", "")).strip()
        == "projected"
    ]

    groups = group_contiguous(
        accepted,
        time_field="t_wall",
        key_fields=["context", "action_id"],
    )

    outputs = []

    for group_idx, group in enumerate(groups, start=1):
        context = group[0]["context"]
        action_id = group[0].get("action_id", "")

        t_start = timestamp(group[0])
        t_end = timestamp(group[-1])

        state_rows = rows_in_window(
            policy_rows,
            t_start,
            t_end,
            context=context,
            padding_s=0.05,
        )

        j4_window = rows_in_window(
            j4_rows,
            t_start,
            t_end,
            context=context,
            padding_s=0.15,
        )

        j4_window = [
            row for row in j4_window
            if str(row.get("action_id", "")) == str(action_id)
        ]

        mean_abs_y, max_abs_y = abs_y_stats(group)

        delta_means = {
            "vx": mean_field(group, "delta_vx"),
            "yaw": mean_field(group, "delta_yaw_rate"),
            "body_h": mean_field(group, "delta_body_h"),
            "clearance": mean_field(
                group, "delta_clearance"
            ),
        }

        # Determine whether the intended meta-action itself is a no-op.
        #
        # Do NOT use J7 observed reference deltas for this decision.
        # Empirical and projected references may have different sequence/timestamp
        # ages, producing small apparent yaw differences even when the meta-action
        # applies no yaw modification.
        #
        # J4 action parameters are the authoritative representation of the
        # intended candidate intervention.
        vx_scale_param = mean_field(
            j4_window, "vx_scale"
        )
        vx_delta_param = mean_field(
            j4_window, "vx_delta"
        )
        yaw_gain_scale_param = mean_field(
            j4_window, "yaw_gain_scale"
        )
        body_h_delta_param = mean_field(
            j4_window, "body_h_delta"
        )
        clearance_delta_param = mean_field(
            j4_window, "clearance_delta"
        )

        noop_params = [
            vx_scale_param,
            vx_delta_param,
            yaw_gain_scale_param,
            body_h_delta_param,
            clearance_delta_param,
        ]

        params_available = all(
            isinstance(v, (int, float))
            for v in noop_params
        )

        is_noop = (
            params_available
            and abs(vx_scale_param - 1.0) < 1e-6
            and abs(vx_delta_param) < 1e-6
            and abs(yaw_gain_scale_param - 1.0) < 1e-6
            and abs(body_h_delta_param) < 1e-6
            and abs(clearance_delta_param) < 1e-6
        )

        label = rollout_label(k6)

        if is_noop:
            evidence_role = "control_equivalent_noop"
        elif label == "shadow_only":
            evidence_role = "near_miss_exposure"
        elif label == "reject":
            evidence_role = (
                "rollout_associated_negative_unpaired"
            )
        else:
            evidence_role = "candidate_exposure"

        coverage = (
            min(1.0, len(state_rows) / len(group))
            if group
            else 0.0
        )

        out = empty_row()

        out.update({
            "schema_version": "phase_l2_v0",
            "row_granularity": "context_action_window",

            "source_phase": k6.get("source_phase", ""),
            "candidate_mode": k6.get("mode", ""),
            "trial_idx": k6.get("idx", ""),
            "source_rollout_id": (
                f"{k6.get('source_phase', '')}:"
                f"{k6.get('mode', '')}:"
                f"{k6.get('idx', '')}"
            ),

            "window_id": (
                f"{k6.get('mode', '')}:"
                f"{context}:"
                f"{action_id}:"
                f"{group_idx}"
            ),
            "window_type": "accepted_candidate_exposure",
            "context": context,
            "action_id": action_id,

            "t_start": t_start,
            "t_end": t_end,
            "duration_s": t_end - t_start,

            "window_rows": len(group),
            "state_rows": len(state_rows),
            "state_coverage_ratio": coverage,

            "x_start": first_float(group, "x"),
            "y_start": first_float(group, "y"),
            "x_end": last_float(group, "x"),
            "y_end": last_float(group, "y"),
            "mean_abs_y_window": mean_abs_y,
            "max_abs_y_window": max_abs_y,

            "beta_motion_mean": mean_field(
                state_rows, "beta_motion"
            ),
            "beta_stability_mean": mean_field(
                state_rows, "beta_stability"
            ),
            "beta_energy_mean": mean_field(
                state_rows, "beta_energy"
            ),

            "ram_slip_mean": mean_field(
                state_rows, "ram_slip_proxy"
            ),
            "ram_roughness_mean": mean_field(
                state_rows, "ram_roughness_proxy"
            ),
            "ram_sigma_mean": mean_field(
                state_rows, "ram_sigma"
            ),

            "emp_vx_mean": mean_field(group, "emp_vx"),
            "emp_yaw_rate_mean": mean_field(
                group, "emp_yaw_rate"
            ),
            "emp_body_h_mean": mean_field(
                group, "emp_body_h"
            ),
            "emp_clearance_mean": mean_field(
                group, "emp_clearance"
            ),
            "emp_enable_mean": mean_field(
                group, "emp_enable"
            ),

            "proj_vx_mean": mean_field(group, "proj_vx"),
            "proj_yaw_rate_mean": mean_field(
                group, "proj_yaw_rate"
            ),
            "proj_body_h_mean": mean_field(
                group, "proj_body_h"
            ),
            "proj_clearance_mean": mean_field(
                group, "proj_clearance"
            ),
            "proj_enable_mean": mean_field(
                group, "proj_enable"
            ),

            "out_vx_mean": mean_field(group, "out_vx"),
            "out_yaw_rate_mean": mean_field(
                group, "out_yaw_rate"
            ),
            "out_body_h_mean": mean_field(
                group, "out_body_h"
            ),
            "out_clearance_mean": mean_field(
                group, "out_clearance"
            ),
            "out_enable_mean": mean_field(
                group, "out_enable"
            ),

            "delta_vx_mean": delta_means["vx"],
            "delta_yaw_rate_mean": delta_means["yaw"],
            "delta_body_h_mean": delta_means["body_h"],
            "delta_clearance_mean": delta_means[
                "clearance"
            ],

            "vx_scale_mean": vx_scale_param,
            "vx_delta_mean": vx_delta_param,
            "yaw_gain_scale_mean": yaw_gain_scale_param,
            "body_h_delta_mean": body_h_delta_param,
            "clearance_delta_mean": clearance_delta_param,

            "evidence_role": evidence_role,

            # K6 labels are rollout-level decisions.
            # They are not direct causal labels for this window.
            "direct_action_label_allowed": 0,
        })

        add_rollout_fields(out, k6)
        outputs.append(out)

    return outputs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k6",
        default=(
            "datasets/phase_k/"
            "k6_selector_candidate_outcome_dataset_v0.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "datasets/phase_l/"
            "l2_window_action_evidence_from_k6_v0.csv"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "reports/phase_l/"
            "l2_window_action_evidence_from_k6_summary_v0.md"
        ),
    )

    args = parser.parse_args()

    k6_path = Path(args.k6)
    output_path = Path(args.output)
    report_path = Path(args.report)

    with k6_path.open(newline="") as f:
        k6_rows = list(csv.DictReader(f))

    outputs = []
    skipped = []

    for k6 in k6_rows:
        mode = k6.get("mode", "")

        d5_raw = str(
            k6.get("d5_log_dir", "")
        ).strip()

        if not d5_raw or d5_raw == "NA":
            skipped.append(
                (mode, "missing_d5_log_dir_metadata")
            )
            continue

        d5_dir = Path(d5_raw).expanduser()
        policy_path = d5_dir / "policy_inputs_v0.csv"

        if not policy_path.exists():
            skipped.append(
                (mode, "missing_policy_inputs_file")
            )
            continue

        policy_rows = load_csv(policy_path)

        if not policy_rows:
            skipped.append(
                (mode, "empty_policy_inputs_file")
            )
            continue

        if mode == "baseline":
            outputs.extend(
                build_baseline_windows(
                    k6,
                    policy_rows,
                )
            )
            continue

        run_dir = Path(k6.get("run_dir", ""))

        j4_paths = list(
            run_dir.rglob(
                "meta_action_ref_projection_shadow_j4_v0.csv"
            )
        )

        j7_paths = list(
            run_dir.rglob(
                "guarded_meta_action_ref_gate_j7_v0.csv"
            )
        )

        if not j7_paths:
            skipped.append((mode, "missing_j7"))
            continue

        j7_rows = load_csv(j7_paths[0])

        j4_rows = (
            load_csv(j4_paths[0])
            if j4_paths
            else []
        )

        candidate_windows = build_candidate_windows(
            k6,
            policy_rows,
            j4_rows,
            j7_rows,
        )

        if not candidate_windows:
            skipped.append(
                (mode, "no_accepted_candidate_exposure")
            )
            continue

        outputs.extend(candidate_windows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )
        writer.writeheader()
        writer.writerows(outputs)

    window_types = Counter(
        row["window_type"]
        for row in outputs
    )

    roles = Counter(
        row["evidence_role"]
        for row in outputs
    )

    modes = Counter(
        row["candidate_mode"]
        for row in outputs
    )

    contexts = Counter(
        row["context"]
        for row in outputs
    )

    lines = [
        "# TRACER Phase-L2 Window Action Evidence Summary v0",
        "",
        f"- source K6: `{k6_path}`",
        f"- output: `{output_path}`",
        f"- output rows: {len(outputs)}",
        f"- skipped source rows: {len(skipped)}",
        "",
        "## Window types",
        "",
    ]

    for key, value in sorted(window_types.items()):
        lines.append(f"- `{key}`: {value}")

    lines += [
        "",
        "## Evidence roles",
        "",
    ]

    for key, value in sorted(roles.items()):
        lines.append(f"- `{key}`: {value}")

    lines += [
        "",
        "## Candidate modes",
        "",
    ]

    for key, value in sorted(modes.items()):
        lines.append(f"- `{key}`: {value}")

    lines += [
        "",
        "## Contexts",
        "",
    ]

    for key, value in sorted(contexts.items()):
        lines.append(f"- `{key}`: {value}")

    if skipped:
        lines += [
            "",
            "## Skipped source rows",
            "",
        ]

        for mode, reason in skipped:
            lines.append(
                f"- `{mode}`: `{reason}`"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Rows represent context/action windows, not independent 10 Hz samples.",
        "- Candidate rows require realized J7 projected exposure.",
        "- K6 decisions are preserved only as rollout-level labels.",
        "- `direct_action_label_allowed` remains false for all L2 rows.",
        "- Control-equivalent no-op windows are explicitly separated from causal negative evidence.",
        "- This dataset is an evidence table for later effect estimation and selector training design.",
        "",
    ]

    report_path.write_text(
        "\n".join(lines)
    )

    print("[L2] rows =", len(outputs))
    print("[L2] window types =", dict(window_types))
    print("[L2] evidence roles =", dict(roles))
    print("[L2] modes =", dict(modes))
    print("[L2] contexts =", dict(contexts))
    print("[L2] skipped =", skipped)
    print("[L2] output =", output_path)
    print("[L2] report =", report_path)


if __name__ == "__main__":
    main()
