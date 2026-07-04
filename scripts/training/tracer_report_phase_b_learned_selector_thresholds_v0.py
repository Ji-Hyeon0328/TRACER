#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def fmt(x):
    if x is None:
        return ""
    return f"{float(x):.3f}"


def required_alpha(raw, challenger):
    """
    final = log_prob + alpha * learned_z - missing_penalty
    Returns alpha needed for challenger to beat raw.
    """
    raw_log = float(raw.get("log_prob", math.log(max(float(raw.get("prob", 1e-12)), 1e-12))))
    ch_log = float(challenger.get("log_prob", math.log(max(float(challenger.get("prob", 1e-12)), 1e-12))))

    raw_z = float(raw.get("learned_score_z", 0.0))
    ch_z = float(challenger.get("learned_score_z", 0.0))

    raw_pen = float(raw.get("missing_score_penalty", 0.0))
    ch_pen = float(challenger.get("missing_score_penalty", 0.0))

    # Need:
    # ch_log + a*ch_z - ch_pen > raw_log + a*raw_z - raw_pen
    # a*(ch_z - raw_z) > raw_log - ch_log + ch_pen - raw_pen
    denom = ch_z - raw_z
    numer = raw_log - ch_log + ch_pen - raw_pen

    if denom <= 1e-12:
        return None

    return numer / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = []

    for p in sorted(Path(args.result_root).rglob("ppo_policy_episode_result_v0.json")):
        r = load_json(p)
        sel = r.get("learned_score_selector", {})
        ranked = sel.get("ranked_actions", [])

        if not ranked:
            continue

        raw_name = sel.get("raw_argmax_action")
        raw = None
        for a in ranked:
            if a.get("action_name") == raw_name:
                raw = a
                break
        if raw is None:
            raw = ranked[0]

        challengers = []
        for a in ranked:
            if a.get("action_name") == raw.get("action_name"):
                continue
            alpha_req = required_alpha(raw, a)
            challengers.append({
                "action_name": a.get("action_name"),
                "prob": a.get("prob"),
                "learned_score_z": a.get("learned_score_z"),
                "has_learned_score": a.get("has_learned_score"),
                "required_alpha_to_beat_raw": alpha_req,
                "final_score_at_current_alpha": a.get("final_score"),
            })

        challengers = sorted(
            challengers,
            key=lambda x: float("inf") if x["required_alpha_to_beat_raw"] is None else x["required_alpha_to_beat_raw"]
        )

        rows.append({
            "path": str(p),
            "world_name": r.get("world_name"),
            "current_alpha": r.get("alpha"),
            "raw_action": raw.get("action_name"),
            "raw_prob": raw.get("prob"),
            "raw_learned_z": raw.get("learned_score_z"),
            "selected_action": r.get("selected_action_name"),
            "changed": sel.get("changed_action"),
            "best_challenger": challengers[0] if challengers else None,
            "challengers": challengers[:8],
        })

    report = {
        "schema": "phase_b_learned_selector_threshold_report_v0",
        "result_root": args.result_root,
        "interpretation": {
            "purpose": "Estimate alpha needed for learned Objective/RAM score to override the raw PPO action.",
            "note": "If required alpha is high, PPO prior dominates and a risk gate/top-k gate may be more appropriate than increasing alpha."
        },
        "rows": rows,
    }

    save_json(Path(args.out_json), report)

    lines = []
    lines.append("# Phase-B Learned Selector Alpha Threshold Report")
    lines.append("")
    lines.append("This report estimates the α needed for the best learned-score challenger to beat the raw PPO action.")
    lines.append("")
    lines.append("| world | raw action | raw p | raw z | best challenger | challenger p | challenger z | required α | current α | changed |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|")

    for r in rows:
        b = r.get("best_challenger") or {}
        lines.append(
            f"| {r['world_name']} | {r['raw_action']} | {fmt(r['raw_prob'])} | {fmt(r['raw_learned_z'])} | "
            f"{b.get('action_name', '')} | {fmt(b.get('prob'))} | {fmt(b.get('learned_score_z'))} | "
            f"{fmt(b.get('required_alpha_to_beat_raw'))} | {fmt(r['current_alpha'])} | {r['changed']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If required α is around 0.3–0.6, a stronger but still moderate intervention can be tested.")
    lines.append("- If required α is above 1.0, a pure additive score is probably not the right mechanism.")
    lines.append("- For high-risk sponge behavior, a RAM risk gate/top-k gate may be safer than large α.")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
