#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", default="data/rollout_dataset_v0/summaries")
    ap.add_argument("--out-csv", default="data/rollout_dataset_v0/rollout_summary_table_v0.csv")
    ap.add_argument("--out-jsonl", default="data/rollout_dataset_v0/rollout_summary_table_v0.jsonl")
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    paths = sorted(summary_dir.glob("*.json"))

    rows = []
    for p in paths:
        try:
            rows.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[TRACER][WARN] failed to read {p}: {e}")

    if not rows:
        print(f"[TRACER] no summaries found in {summary_dir}")
        return

    # Stable field order with fallback for future-added keys.
    preferred = [
        "episode_id",
        "terrain",
        "policy_id",
        "duration_sec",
        "n_rows",
        "success_proxy",
        "distance_xy_proxy",
        "mpc_vx_mean",
        "mpc_vx_p50",
        "mpc_vx_p90",
        "mpc_enable_mean",
        "mpc_body_height_mean",
        "mpc_clearance_mean",
        "ram_run_fallen_mean",
        "ram_run_fallen_p90",
        "ram_recovery_needed_mean",
        "gate_level_mean",
        "gate_action_mean",
        "gate_override_mean",
        "proprio_abs_mean",
        "proprio_abs_p90",
        "step_csv",
        "summary_json",
    ]
    extra = sorted({k for r in rows for k in r.keys()} - set(preferred))
    fields = preferred + extra

    out_csv = Path(args.out_csv)
    out_jsonl = Path(args.out_jsonl)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    with out_jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[TRACER] wrote {len(rows)} summaries")
    print(f"[TRACER] csv:   {out_csv}")
    print(f"[TRACER] jsonl: {out_jsonl}")

    print()
    by_key = {}
    for r in rows:
        key = (r.get("terrain", "unknown"), r.get("policy_id", "unknown"))
        by_key.setdefault(key, []).append(r)

    for (terrain, policy), rs in sorted(by_key.items()):
        def avg(k):
            vals = []
            for r in rs:
                try:
                    vals.append(float(r.get(k, 0.0)))
                except Exception:
                    pass
            return sum(vals) / max(1, len(vals))

        print(
            f"{terrain:16s} {policy:22s} n={len(rs):3d} "
            f"success_proxy={avg('success_proxy'):.2f} "
            f"vx={avg('mpc_vx_mean'):.4f} "
            f"enable={avg('mpc_enable_mean'):.2f} "
            f"fallen={avg('ram_run_fallen_mean'):.3f} "
            f"gate={avg('gate_level_mean'):.2f}/{avg('gate_action_mean'):.2f}"
        )


if __name__ == "__main__":
    main()
