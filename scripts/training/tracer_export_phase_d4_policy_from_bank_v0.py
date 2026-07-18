#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_action_table(text: str):
    table = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        key, vals = item.split(":", 1)
        vx, body_h, clearance = [float(x.strip()) for x in vals.split(",")]
        table[key.strip()] = {
            "vx": vx,
            "body_h": body_h,
            "clearance": clearance,
        }
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    best_path = Path(args.best_json)
    data = json.loads(best_path.read_text())
    best = data["best"]

    action_table_text = best["action_table"]
    theta_by_context = parse_action_table(action_table_text)

    policy = {
        "policy_name": f"phase_d4_context_meta_{best['name']}_v0",
        "policy_type": "rollout_selected_context_meta_table",
        "source_best_json": str(best_path),
        "source_method": "D4.2 context table action-bank search",
        "context_input": "/tracer/terrain_context_label",
        "output_topic": "/tracer/mpc_reference",
        "output_protocol": "[seq, vx, yaw_rate, body_h, clearance, enable]",
        "selected_name": best["name"],
        "hold_vx": float(best["hold_vx"]),
        "action_table": action_table_text,
        "theta_by_context": theta_by_context,
        "selection_metrics": {
            "score": best.get("score"),
            "success_rate": best.get("success_rate"),
            "goal_rate": best.get("goal_rate"),
            "startup_failed_rate": best.get("startup_failed_rate"),
            "out_lane_rate": best.get("out_lane_rate"),
            "first_goal_time_mean": best.get("first_goal_time_mean"),
            "final_goal_error_mean": best.get("final_goal_error_mean"),
            "hold_drift_mean": best.get("hold_drift_mean"),
            "max_abs_y_mean": best.get("max_abs_y_mean"),
            "mean_abs_y_mean": best.get("mean_abs_y_mean"),
        },
        "known_limitations": [
            "This is a rollout-selected table policy, not a neural RL policy.",
            "Validated on stress v5 with oracle terrain context labels.",
            "Lateral drift is acceptable but still a tuning target.",
        ],
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(policy, indent=2))

    lines = []
    lines.append("# TRACER Phase-D4 Context Meta Policy Export v0")
    lines.append("")
    lines.append(f"- policy_name: `{policy['policy_name']}`")
    lines.append(f"- selected_name: `{policy['selected_name']}`")
    lines.append(f"- hold_vx: `{policy['hold_vx']}`")
    lines.append(f"- source_best_json: `{best_path}`")
    lines.append("")
    lines.append("## Selection metrics")
    lines.append("")
    for k, v in policy["selection_metrics"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Action table")
    lines.append("")
    lines.append("```")
    lines.append(policy["action_table"])
    lines.append("```")
    lines.append("")
    lines.append("## Theta by context")
    lines.append("")
    lines.append("| context | vx | body_h | clearance |")
    lines.append("|---|---:|---:|---:|")
    for ctx, th in policy["theta_by_context"].items():
        lines.append(f"| {ctx} | {th['vx']:.4f} | {th['body_h']:.4f} | {th['clearance']:.4f} |")

    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {out_json}")
    print(f"[TRACER] wrote {out_md}")
    print(f"[TRACER] exported policy={policy['policy_name']}")


if __name__ == "__main__":
    main()
