#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def extract_reach_by_action(reach):
    for key in ["by_world_action", "by_world_and_action", "by_action"]:
        if isinstance(reach, dict) and isinstance(reach.get(key), dict):
            return reach[key]

    out = {}
    if isinstance(reach, dict):
        for world, w in reach.items():
            if not isinstance(w, dict):
                continue
            if isinstance(w.get("by_action"), dict):
                for action, a in w["by_action"].items():
                    out[f"{world}::{action}"] = a
            if isinstance(w.get("actions"), dict):
                for action, a in w["actions"].items():
                    out[f"{world}::{action}"] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach-json", required=True)
    ap.add_argument("--invalid-v1b-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    reach = load_json(args.reach_json)
    invalid = load_json(args.invalid_v1b_json)

    by_action_reach = extract_reach_by_action(reach)
    by_action_invalid = invalid.get("by_action", {})

    rows = []

    for key, inv in by_action_invalid.items():
        r = by_action_reach.get(key, {})

        reach_rate = float(r.get("reach_rate", 0.0))
        final_dist = float(r.get("mean_final_rel_dist", r.get("mean_final_dist", 999.0)))
        drift = float(r.get("mean_post_reach_drift", 999.0))
        R_s = float(r.get("mean_R_s", -999.0))
        R_v = float(r.get("mean_R_v", 0.0))

        hard_invalid_rate = float(inv.get("hard_invalid_rate", 1.0))
        warning_rate = float(inv.get("warning_rate", 1.0))

        # v1b score:
        # - reach and forward reward are useful
        # - R_s and post-reach drift are now strongly weighted
        # - hard invalid is treated as a deployability blocker
        # - warning is softer: low posture/sink can be useful signal, but should not dominate
        score = (
            2.0 * reach_rate
            + 0.20 * R_v
            + 1.00 * R_s
            - 3.00 * hard_invalid_rate
            - 0.75 * warning_rate
            - 0.70 * max(0.0, drift)
            - 0.30 * max(0.0, final_dist)
        )

        rows.append({
            "world_action": key,
            "n": inv.get("n"),
            "reach_rate": reach_rate,
            "mean_final_dist": final_dist,
            "mean_post_reach_drift": drift,
            "mean_R_v": R_v,
            "mean_R_s": R_s,
            "hard_invalid_rate": hard_invalid_rate,
            "warning_rate": warning_rate,
            "validity_aware_score_v1b": score,
            "top_hard_reasons": inv.get("top_hard_reasons", []),
            "top_warning_reasons": inv.get("top_warning_reasons", []),
        })

    rows.sort(key=lambda x: x["validity_aware_score_v1b"], reverse=True)

    out = {
        "schema": "phase_b_validity_aware_action_score_v1b",
        "reach_json": args.reach_json,
        "invalid_v1b_json": args.invalid_v1b_json,
        "score_formula": "2*reach + 0.20*R_v + 1.00*R_s - 3*hard_invalid - 0.75*warning - 0.70*drift - 0.30*final_dist",
        "rows": rows,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Validity-Aware Action Score v1b")
    lines.append("")
    lines.append("Formula:")
    lines.append("")
    lines.append("`2*reach + 0.20*R_v + 1.00*R_s - 3*hard_invalid - 0.75*warning - 0.70*drift - 0.30*final_dist`")
    lines.append("")
    lines.append("| rank | world::action | n | reach | final | drift | R_v | R_s | hard_invalid | warning | score |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['world_action']} | {r['n']} | "
            f"{r['reach_rate']:.3f} | {r['mean_final_dist']:.3f} | "
            f"{r['mean_post_reach_drift']:.3f} | {r['mean_R_v']:.3f} | "
            f"{r['mean_R_s']:.3f} | {r['hard_invalid_rate']:.3f} | "
            f"{r['warning_rate']:.3f} | {r['validity_aware_score_v1b']:.3f} |"
        )

    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
