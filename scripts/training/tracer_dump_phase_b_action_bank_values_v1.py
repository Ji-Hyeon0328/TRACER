#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from pprint import pformat

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime.tracer_run_phase_b_training_campaign_v1 import build_action_bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rows = []

    for cfg_path in args.config:
        cfg = json.loads(Path(cfg_path).read_text())
        teacher_json = cfg["teacher_json"]
        worlds = cfg.get("worlds", [])
        actions = cfg.get("actions", [])

        bank = build_action_bank(teacher_json)

        for action in actions:
            theta = bank.get(action)
            rows.append({
                "config": cfg_path,
                "teacher_json": teacher_json,
                "worlds": worlds,
                "action": action,
                "found": theta is not None,
                "theta": theta,
            })

    out = {
        "schema": "phase_b_action_bank_values_v1",
        "rows": rows,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Action Bank Values v1")
    lines.append("")
    lines.append("This report uses the same `build_action_bank()` parser as the Phase-B campaign runner.")
    lines.append("")
    lines.append("| world(s) | action | found | vx_far | vx_near | slow_dist | stop_dist | body_h | clearance |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        theta = r["theta"] or {}
        worlds_s = ", ".join(r["worlds"])
        lines.append(
            f"| {worlds_s} | {r['action']} | {r['found']} | "
            f"{theta.get('vx_far', '')} | "
            f"{theta.get('vx_near', '')} | "
            f"{theta.get('goal_slow_distance', '')} | "
            f"{theta.get('goal_stop_distance', '')} | "
            f"{theta.get('body_height', '')} | "
            f"{theta.get('swing_clearance', '')} |"
        )

    lines.append("")
    lines.append("## Raw theta entries")
    lines.append("")
    for r in rows:
        lines.append(f"### {r['action']}")
        lines.append("")
        lines.append(f"- config: `{r['config']}`")
        lines.append(f"- teacher: `{r['teacher_json']}`")
        lines.append(f"- worlds: `{', '.join(r['worlds'])}`")
        lines.append(f"- found: `{r['found']}`")
        lines.append("")
        lines.append("```python")
        lines.append(pformat(r["theta"], width=120))
        lines.append("```")
        lines.append("")

    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
