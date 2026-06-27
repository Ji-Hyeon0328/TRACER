#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/shadow_logs_v2/learned_stack_shadow_v2_*.csv")
    ap.add_argument("--out-json", default="data/shadow_logs_v2/learned_stack_shadow_v2_summary.json")
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.glob))

    by_terrain = defaultdict(lambda: {
        "n": 0,
        "ok": 0,
        "disagree": 0,
        "ram": [],
        "override": [],
        "rule_counts": defaultdict(int),
        "learned_counts": defaultdict(int),
        "errors": defaultdict(int),
    })

    for p in paths:
        with p.open() as fobj:
            for row in csv.DictReader(fobj):
                terrain = row.get("terrain", "unknown")
                d = by_terrain[terrain]
                d["n"] += 1
                d["ok"] += int(f(row.get("learned_ok", 0.0)) >= 0.5)
                d["disagree"] += int(f(row.get("gms_disagree", 0.0)) >= 0.5)
                d["ram"].append(f(row.get("learned_ram_intervention_score", 0.0)))
                d["override"].append(f(row.get("learned_ram_future_override_mean", 0.0)))
                d["rule_counts"][row.get("rule_gms_label", "")] += 1
                d["learned_counts"][row.get("learned_gms_label", "")] += 1
                if row.get("error", ""):
                    d["errors"][row.get("error", "")[:120]] += 1

    report = {}
    for terrain, d in sorted(by_terrain.items()):
        n = max(1, d["n"])
        report[terrain] = {
            "n": d["n"],
            "ok_rate": d["ok"] / n,
            "gms_disagreement_rate": d["disagree"] / n,
            "ram_intervention_mean": sum(d["ram"]) / max(1, len(d["ram"])),
            "ram_override_mean": sum(d["override"]) / max(1, len(d["override"])),
            "rule_counts": dict(d["rule_counts"]),
            "learned_counts": dict(d["learned_counts"]),
            "errors": dict(d["errors"]),
        }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"[TRACER] wrote shadow v2 summary: {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
