#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default


def b(x: Any) -> float:
    return float(bool(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/rollouts/phase_a_runtime_v0")
    ap.add_argument("--terrain", default="")
    ap.add_argument("--policy-contains", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    root = Path(args.root)
    summaries = sorted(root.glob("runtime_rollout_*/summaries/*.json"))

    rows: list[dict[str, Any]] = []
    for p in summaries:
        try:
            s = json.loads(p.read_text())
        except Exception:
            continue

        if args.terrain and s.get("terrain") != args.terrain:
            continue

        if args.policy_contains and args.policy_contains not in str(s.get("policy_id", "")):
            continue

        s["_summary_path"] = str(p)
        rows.append(s)

    print(f"[TRACER] root={root}")
    print(f"[TRACER] n={len(rows)}")
    if not rows:
        raise SystemExit(1)

    scalar_keys = [
        "success_proxy",
        "valid_data",
        "proprio_base_height_stable",
        "distance_xy_proxy",
        "proprio_base_z_min",
        "proprio_base_z_p50",
        "proprio_base_z_below_0p18_frac",
        "proprio_base_z_below_0p22_frac",
        "proprio_roll_abs_max",
        "proprio_pitch_abs_max",
        "debug_fresh_rate_0p5s",
        "age_proprio_p90",
        "age_odom_p90",
    ]

    print()
    for i, s in enumerate(rows, 1):
        print(f"---- row {i} {s.get('episode_id')}")
        for k in scalar_keys:
            print(f"{k}: {s.get(k)}")
        print()

    agg = {
        "n": len(rows),
        "terrain": args.terrain,
        "policy_contains": args.policy_contains,
        "success_rate": mean(b(s.get("success_proxy")) for s in rows),
        "valid_rate": mean(b(s.get("valid_data")) for s in rows),
        "height_stable_rate": mean(b(s.get("proprio_base_height_stable")) for s in rows),
        "distance_mean": mean(f(s.get("distance_xy_proxy")) for s in rows),
        "zmin_mean": mean(f(s.get("proprio_base_z_min")) for s in rows),
        "z50_mean": mean(f(s.get("proprio_base_z_p50")) for s in rows),
        "below18_mean": mean(f(s.get("proprio_base_z_below_0p18_frac")) for s in rows),
        "below22_mean": mean(f(s.get("proprio_base_z_below_0p22_frac")) for s in rows),
        "rollmax_mean": mean(f(s.get("proprio_roll_abs_max")) for s in rows),
        "pitchmax_mean": mean(f(s.get("proprio_pitch_abs_max")) for s in rows),
        "debug_fresh_rate_0p5s_mean": mean(f(s.get("debug_fresh_rate_0p5s")) for s in rows),
        "age_proprio_p90_mean": mean(f(s.get("age_proprio_p90")) for s in rows),
        "age_odom_p90_mean": mean(f(s.get("age_odom_p90")) for s in rows),
    }

    print("==== aggregate ====")
    print(json.dumps(agg, indent=2, ensure_ascii=False))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2, ensure_ascii=False))
        print(f"[TRACER] wrote json: {out}")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r.keys()})
        with out.open("w", newline="", encoding="utf-8") as fcsv:
            wr = csv.DictWriter(fcsv, fieldnames=keys)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
        print(f"[TRACER] wrote csv: {out}")


if __name__ == "__main__":
    main()
