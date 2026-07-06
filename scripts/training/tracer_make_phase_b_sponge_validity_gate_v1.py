#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validity-v1c-json", required=True)
    ap.add_argument("--validity-score-v1b-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    v1c = load_json(args.validity_v1c_json)
    v1b_score = load_json(args.validity_score_v1b_json)

    by_action_v1c = v1c.get("by_action", {})
    score_rows = {
        r["world_action"]: r
        for r in v1b_score.get("rows", [])
    }

    rows = []
    for key, r in by_action_v1c.items():
        s = score_rows.get(key, {})

        reach = float(r.get("reach_rate", 0.0))
        hard = float(r.get("hard_invalid_rate", 1.0))
        warn = float(r.get("warning_rate", 1.0))
        final = float(r.get("mean_final_dist", 999.0))
        drift = float(r.get("mean_drift_proxy", 999.0))
        deploy_success = float(r.get("deploy_valid_success_rate_v1c", 0.0))
        deploy_score = float(r.get("mean_deploy_valid_score_v1c", -999.0))
        score_v1b = float(s.get("validity_aware_score_v1b", -999.0))

        is_deploy_valid = bool(
            deploy_success >= 0.2
            and deploy_score > 1.0
            and hard <= 0.1
            and final < 1.0
            and drift < 0.75
        )

        is_reach_safe_probe = bool(
            reach >= 0.6
            and hard <= 0.1
        )

        is_low_drift_risky = bool(
            final < 1.25
            and drift < 1.0
            and hard > 0.1
        )

        reject = bool(
            deploy_score < 0.0
            or warn >= 0.6
        )

        rows.append({
            "world_action": key,
            "reach_rate": reach,
            "hard_invalid_rate": hard,
            "warning_rate": warn,
            "mean_final_dist": final,
            "mean_drift_proxy": drift,
            "deploy_valid_success_rate_v1c": deploy_success,
            "deploy_valid_score_v1c": deploy_score,
            "validity_aware_score_v1b": score_v1b,
            "is_deploy_valid": is_deploy_valid,
            "is_reach_safe_probe": is_reach_safe_probe,
            "is_low_drift_risky": is_low_drift_risky,
            "reject_or_recovery_needed": reject,
        })

    rows_sorted = sorted(rows, key=lambda x: x["deploy_valid_score_v1c"], reverse=True)

    deploy_valid = [r for r in rows_sorted if r["is_deploy_valid"]]
    reach_safe = [r for r in rows_sorted if r["is_reach_safe_probe"]]
    low_drift_risky = [r for r in rows_sorted if r["is_low_drift_risky"]]

    decision = {
        "schema": "phase_b_sponge_validity_gate_v1",
        "deploy_valid_available": bool(deploy_valid),
        "recovery_needed": not bool(deploy_valid),
        "recommended_deploy_action": deploy_valid[0]["world_action"] if deploy_valid else None,
        "recommended_reach_safe_probe": reach_safe[0]["world_action"] if reach_safe else None,
        "recommended_low_drift_risky_probe": low_drift_risky[0]["world_action"] if low_drift_risky else None,
        "decision_note": (
            "No deploy-valid sponge primitive found in the current frozen low-level theta-lite action bank. "
            "Use reach-safe probe only for data collection, not as a deployable traversal primitive."
            if not deploy_valid else
            "A deploy-valid sponge primitive exists under current thresholds."
        ),
        "rows": rows_sorted,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(decision, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Sponge Validity Gate v1")
    lines.append("")
    lines.append(f"- Deploy-valid available: `{decision['deploy_valid_available']}`")
    lines.append(f"- Recovery needed: `{decision['recovery_needed']}`")
    lines.append(f"- Recommended deploy action: `{decision['recommended_deploy_action']}`")
    lines.append(f"- Reach-safe probe: `{decision['recommended_reach_safe_probe']}`")
    lines.append(f"- Low-drift risky probe: `{decision['recommended_low_drift_risky_probe']}`")
    lines.append("")
    lines.append(decision["decision_note"])
    lines.append("")
    lines.append("| action | reach | hard | warning | final | drift | deploy_success | score_v1c | deploy_valid | reach_safe_probe | low_drift_risky |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in rows_sorted:
        lines.append(
            f"| {r['world_action']} | {r['reach_rate']:.3f} | "
            f"{r['hard_invalid_rate']:.3f} | {r['warning_rate']:.3f} | "
            f"{r['mean_final_dist']:.3f} | {r['mean_drift_proxy']:.3f} | "
            f"{r['deploy_valid_success_rate_v1c']:.3f} | "
            f"{r['deploy_valid_score_v1c']:.3f} | "
            f"{r['is_deploy_valid']} | {r['is_reach_safe_probe']} | {r['is_low_drift_risky']} |"
        )

    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
