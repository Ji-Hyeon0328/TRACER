#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def summarize_csv(path: Path):
    with path.open() as fobj:
        rows = list(csv.DictReader(fobj))

    ok = [r for r in rows if str(r.get("learned_ok", "0")) == "1"]

    def mean(key):
        vals = [f(r.get(key, 0.0)) for r in ok]
        return sum(vals) / max(1, len(vals))

    def counts(key):
        d = defaultdict(int)
        for r in ok:
            d[str(r.get(key, ""))] += 1
        return dict(sorted(d.items()))

    terrain = rows[0].get("terrain", path.stem) if rows else path.stem

    return {
        "csv": str(path),
        "terrain": terrain,
        "n": len(rows),
        "ok_n": len(ok),
        "ok_rate": len(ok) / max(1, len(rows)),
        "gms_disagreement_rate": mean("gms_disagree"),
        "ram_intervention_mean": mean("learned_ram_intervention_score"),
        "ram_override_mean": mean("learned_ram_future_override_mean"),
        "rule_counts": counts("rule_gms_label"),
        "learned_counts": counts("learned_gms_label"),
        "errors": counts("error"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/shadow_logs_v3/learned_stack_shadow_v3_*.csv")
    ap.add_argument("--out-json", default="reports/learned_stack_shadow_v3_report.json")
    ap.add_argument("--out-md", default="reports/learned_stack_shadow_v3_report.md")
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.glob))
    summaries = [summarize_csv(p) for p in paths]

    by_terrain = {}
    for s in summaries:
        by_terrain[s["terrain"]] = s

    report = {
        "version": "learned_stack_shadow_v3_report",
        "stack": {
            "objective": "objective_selector_v2",
            "ram": "ram_shadow_v2",
            "gms": "gms_classifier_v1",
            "server_port": 50430,
        },
        "summary": by_terrain,
        "interpretation": {
            "flat_normal": "Expected low RAM intervention and fast agreement.",
            "rough_mid": "Expected elevated RAM intervention and conservative-biased transition.",
            "slope_5deg": "Expected elevated/moderate RAM intervention and conservative-biased transition.",
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("# TRACER Learned Stack Shadow v3 Report")
    lines.append("")
    lines.append("Stack: objective_selector_v2 + ram_shadow_v2 + gms_classifier_v1")
    lines.append("")
    lines.append("| terrain | n | ok_rate | gms_disagreement | ram_intervention | ram_override | rule_counts | learned_counts |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")

    for terrain, s in by_terrain.items():
        lines.append(
            f"| {terrain} | {s['n']} | {s['ok_rate']:.3f} | "
            f"{s['gms_disagreement_rate']:.4f} | "
            f"{s['ram_intervention_mean']:.4f} | "
            f"{s['ram_override_mean']:.4f} | "
            f"`{json.dumps(s['rule_counts'])}` | "
            f"`{json.dumps(s['learned_counts'])}` |"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- This is a shadow-runtime validation report, not an active deployment result.")
    lines.append("- RAM shadow v2 is trained from shadow teacher/proxy labels, not true privileged physics labels.")
    lines.append("- Learned stack v3 is acceptable as a deploy candidate only behind an explicit runtime flag.")

    out_md.write_text("\n".join(lines) + "\n")

    print("[TRACER] wrote learned stack shadow v3 report")
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "terrains": list(by_terrain.keys()),
    }, indent=2))


if __name__ == "__main__":
    main()
