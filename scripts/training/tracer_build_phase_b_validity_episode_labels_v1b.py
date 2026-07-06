#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrated-invalid-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = load_jsonl(args.calibrated_invalid_jsonl)
    out_rows = []

    for r in rows:
        f = r.get("features", {})

        reached = bool(f.get("reached_stop_distance", False))
        min_dist = ff(f.get("min_rel_dist"), 999.0)
        final_dist = ff(f.get("final_rel_dist"), 999.0)
        progress = ff(f.get("progress_initial_minus_min"), 0.0)

        hard_invalid = bool(r.get("hard_invalid_gait", False))
        warning = bool(r.get("warning_gait", False))

        # Approx drift when report-level drift is unavailable per episode.
        # If reached, final-min is a post-reach/after-best proxy.
        # If not reached, final-min still captures late degradation after best approach.
        drift_proxy = max(0.0, final_dist - min_dist)

        reach_score = 1.0 if reached else 0.0
        progress_score = max(0.0, min(1.0, progress / 0.50))
        final_score = max(0.0, min(1.0, 1.0 - final_dist / 2.5))
        drift_score = max(0.0, min(1.0, 1.0 - drift_proxy / 2.0))

        # Validity target for training.
        # This is intentionally not just action average: each rollout receives its own penalty.
        valid_score = (
            2.0 * reach_score
            + 0.8 * progress_score
            + 0.8 * final_score
            + 0.8 * drift_score
            - 3.0 * float(hard_invalid)
            - 0.75 * float(warning)
        )

        valid_success = bool(
            reached
            and not hard_invalid
            and final_dist < 1.5
            and drift_proxy < 1.25
        )

        out_rows.append({
            "schema": "phase_b_validity_episode_label_v1b",
            "world": r.get("world"),
            "action": r.get("action"),
            "episode": r.get("episode"),
            "csv_path": r.get("csv_path"),
            "summary_path": r.get("summary_path"),
            "reached_stop_distance": reached,
            "hard_invalid_gait": hard_invalid,
            "warning_gait": warning,
            "hard_reasons": r.get("hard_reasons", []),
            "warning_reasons": r.get("warning_reasons", []),
            "min_rel_dist": min_dist,
            "final_rel_dist": final_dist,
            "progress_initial_minus_min": progress,
            "drift_proxy": drift_proxy,
            "reach_score": reach_score,
            "progress_score": progress_score,
            "final_score": final_score,
            "drift_score": drift_score,
            "valid_score_v1b": valid_score,
            "valid_success_v1b": valid_success,
        })

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    by_action = {}
    for r in out_rows:
        key = f"{r['world']}::{r['action']}"
        by_action.setdefault(key, []).append(r)

    report = {
        "schema": "phase_b_validity_episode_label_report_v1b",
        "input": args.calibrated_invalid_jsonl,
        "num_episodes": len(out_rows),
        "mean_valid_score_v1b": sum(r["valid_score_v1b"] for r in out_rows) / max(1, len(out_rows)),
        "valid_success_rate_v1b": sum(r["valid_success_v1b"] for r in out_rows) / max(1, len(out_rows)),
        "by_action": {},
    }

    for key, rs in sorted(by_action.items()):
        n = len(rs)
        report["by_action"][key] = {
            "n": n,
            "mean_valid_score_v1b": sum(r["valid_score_v1b"] for r in rs) / max(1, n),
            "valid_success_rate_v1b": sum(r["valid_success_v1b"] for r in rs) / max(1, n),
            "reach_rate": sum(r["reached_stop_distance"] for r in rs) / max(1, n),
            "hard_invalid_rate": sum(r["hard_invalid_gait"] for r in rs) / max(1, n),
            "warning_rate": sum(r["warning_gait"] for r in rs) / max(1, n),
            "mean_final_dist": sum(r["final_rel_dist"] for r in rs) / max(1, n),
            "mean_drift_proxy": sum(r["drift_proxy"] for r in rs) / max(1, n),
        }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Validity Episode Labels v1b")
    lines.append("")
    lines.append(f"- Episodes: `{report['num_episodes']}`")
    lines.append(f"- Mean valid score: `{report['mean_valid_score_v1b']:.3f}`")
    lines.append(f"- Valid success rate: `{report['valid_success_rate_v1b']:.3f}`")
    lines.append("")
    lines.append("| world::action | n | valid_success | valid_score | reach | hard_invalid | warning | final | drift_proxy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, v in sorted(
        report["by_action"].items(),
        key=lambda kv: kv[1]["mean_valid_score_v1b"],
        reverse=True,
    ):
        lines.append(
            f"| {key} | {v['n']} | {v['valid_success_rate_v1b']:.3f} | "
            f"{v['mean_valid_score_v1b']:.3f} | {v['reach_rate']:.3f} | "
            f"{v['hard_invalid_rate']:.3f} | {v['warning_rate']:.3f} | "
            f"{v['mean_final_dist']:.3f} | {v['mean_drift_proxy']:.3f} |"
        )

    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
