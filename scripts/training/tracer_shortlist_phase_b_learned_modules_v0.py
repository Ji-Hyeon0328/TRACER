#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


PRIMARY_PATTERNS = {
    "objective_selector": [
        "artifacts/objective_selector_v1/objective_selector_v1.pt",
        "artifacts/objective_selector_v2/objective_selector_v2.pt",
        "artifacts/objective_preference_v2/objective_preference_v2.pt",
        "artifacts/phase_a_objective_selector_pref_v0/phase_a_objective_selector_pref_v0.pt",
        "artifacts/phase_a_objective_selector_pref_with_risk_v0/phase_a_objective_selector_pref_v0.pt",
        "artifacts/tracer_objective_irl_v0/model.pt",
        "artifacts/tracer_objective_irl_v1/model.pt",
        "artifacts/tracer_highlevel_selector_v0/model.pt",
        "artifacts/preference_reward_slide_v0/model.pt",
        "artifacts/preference_reward_v0/model.pt",
        "configs/learned_models/tracer_preference_objective_irl_v0_model.json",
        "configs/learned_models/tracer_preference_objective_irl_v1_model.json",
        "configs/highlevel_policy/tracer_objective_conditioned_selector_v0.json"
    ],
    "ram": [
        "artifacts/tracer_ram_scalar_v3/model.pt",
        "configs/learned_models/tracer_ram_scalar_v3_model.pt",
        "artifacts/tracer_ram_v0/model.pt",
        "artifacts/tracer_ram_v2/model.pt",
        "artifacts/tracer_ram_v2c/model.pt",
        "artifacts/phase_a_ram_empirical_v1/phase_a_ram_empirical_v1.pt",
        "artifacts/phase_a_ram_teacher_smoke_v0/phase_a_ram_teacher_smoke_v0.pt",
        "artifacts/ram_shadow_v2/ram_shadow_v2.pt",
        "artifacts/ram_intervention_v1/ram_intervention_v1.pt",
        "configs/phase_b_objective_ram_bootstrap_v0/current_model.json",
        "configs/phase_b_objective_ram_uncertainty_v0/current_registry.json"
    ],
    "runtime_scripts": [
        "scripts/runtime/check_objective_selector_runtime_v0.py",
        "scripts/runtime/tracer_ensure_objective_selector_beta_node_v0.sh",
        "scripts/runtime/tracer_publish_objective_beta_once.sh",
        "scripts/runtime/tracer_run_learned_stack_v3_beta_blend_robust30.sh",
        "scripts/runtime/tracer_run_objective_selector_clean_routing_smoke.sh",
        "scripts/runtime/tracer_run_objective_selector_overlay_smoke.sh",
        "scripts/training/tracer_check_objective_selector_v1.py",
        "scripts/training/tracer_check_objective_conditioned_selector_v0.py",
        "scripts/training/tracer_check_preference_objective_irl_v0.py",
        "scripts/training/tracer_report_learned_stack_v3_beta_blend.py",
        "scripts/training/tracer_report_learned_stack_v3_beta_blend_robust_eval.py",
        "scripts/training/tracer_report_learned_stack_v3_beta_shadow.py"
    ],
    "datasets_reports": [
        "data/preference_datasets/tracer_objective_selector_runtime_v0.jsonl",
        "data/preference_datasets/tracer_objective_selector_data_v1_core3_hard3_objective_conditioned_pairs_clean.jsonl",
        "data/training/tracer_highlevel_selector_dataset_v0.csv",
        "data/training/tracer_irl_trajectory_features_v0.csv",
        "reports/learned_stack_v3_beta_blend_report.md",
        "reports/learned_stack_v3_beta_blend_robust_eval_report.md",
        "reports/learned_stack_v3_beta_shadow_report.md",
        "reports/tracer_preference_objective_irl_v1_summary.json"
    ]
}


def safe_json_head(path):
    try:
        if path.suffix.lower() != ".json":
            return None
        obj = json.load(open(path))
        if isinstance(obj, dict):
            return {
                "keys": sorted(list(obj.keys()))[:30],
                "schema": obj.get("schema"),
                "description": obj.get("description"),
                "metrics": obj.get("metrics"),
            }
        return {"type": type(obj).__name__}
    except Exception as e:
        return {"error": str(e)}


def safe_text_head(path, n=1200):
    try:
        if path.suffix.lower() not in [".md", ".txt", ".csv", ".jsonl", ".py", ".sh"]:
            return ""
        return path.read_text(errors="ignore")[:n]
    except Exception:
        return ""


def classify_connectability(path, group):
    s = str(path)
    if group == "ram":
        if "tracer_ram_scalar_v3" in s:
            return "high_priority_phase_b_candidate"
        if "phase_b_objective_ram" in s:
            return "phase_b_proxy_candidate"
        if "tracer_ram_v2" in s or "phase_a_ram" in s:
            return "needs_input_schema_check"
        return "needs_architecture_check"

    if group == "objective_selector":
        if "tracer_objective_irl_v1" in s or "phase_a_objective_selector_pref_with_risk" in s:
            return "high_priority_objective_candidate"
        if "tracer_preference_objective_irl" in s or s.endswith(".json"):
            return "runtime_export_candidate"
        if "highlevel_selector" in s:
            return "theta_selection_candidate"
        return "needs_architecture_check"

    if group == "runtime_scripts":
        return "integration_or_diagnostic_script"

    return "supporting_data_or_report"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory-json", default="reports/phase_b_learned_module_inventory_v0.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = []

    for group, paths in PRIMARY_PATTERNS.items():
        for raw in paths:
            p = Path(raw)
            exists = p.exists()
            row = {
                "group": group,
                "path": raw,
                "exists": exists,
                "size_bytes": p.stat().st_size if exists else None,
                "suffix": p.suffix if exists else Path(raw).suffix,
                "connectability": classify_connectability(p, group),
                "json_head": safe_json_head(p) if exists else None,
                "text_head": safe_text_head(p) if exists else "",
            }
            rows.append(row)

    report = {
        "schema": "phase_b_learned_module_shortlist_v0",
        "purpose": "Shortlist previously learned Objective Selector and RAM artifacts that may be connected to the current Phase-B PPO/Gazebo runner.",
        "inventory_json": args.inventory_json,
        "interpretation": {
            "main_point": "Learned Objective Selector and RAM artifacts exist, but they are not yet wired into the current Phase-B PPO action-selection loop.",
            "next_step": "Inspect checkpoint metadata and input/output schemas for high-priority candidates, then add a learned-module inference adapter."
        },
        "rows": rows,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Learned Module Shortlist")
    lines.append("")
    lines.append("This shortlist filters the broad inventory down to likely Objective Selector / RAM candidates for Phase-B integration.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| group | exists | connectability | path | size |")
    lines.append("|---|---:|---|---|---:|")
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['exists']} | {r['connectability']} | `{r['path']}` | {r['size_bytes'] or ''} |"
        )

    lines.append("")
    lines.append("## Recommended Integration Priority")
    lines.append("")
    lines.append("1. Inspect `tracer_ram_scalar_v3` input/output schema.")
    lines.append("2. Inspect `tracer_objective_irl_v1` and `phase_a_objective_selector_pref_with_risk_v0` schema.")
    lines.append("3. Check whether existing runtime scripts already export β or selector decisions.")
    lines.append("4. Build a Phase-B adapter that maps current episode/window features into those learned modules.")
    lines.append("5. Compare raw PPO vs learned Objective Selector/RAM adapter, using reach, stability, speed, and energy proxy.")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "num_rows": len(rows),
        "exists": sum(1 for r in rows if r["exists"]),
        "missing": sum(1 for r in rows if not r["exists"]),
    }, indent=2, sort_keys=True))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
