#!/usr/bin/env python3
import csv
import glob
import json
import math
import os
import sys

def fnum(x, nd=3):
    if x is None:
        return ""
    try:
        x = float(x)
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)

def main():
    if len(sys.argv) < 2:
        print("usage: tracer_summarize_objective_ablation.py <ablation_dir>")
        sys.exit(2)

    out_dir = sys.argv[1]
    paths = sorted(glob.glob(os.path.join(out_dir, "*_summary.json")))

    rows = []
    for p in paths:
        with open(p) as f:
            s = json.load(f)

        label = s.get("ablation_label", "")
        rep = s.get("ablation_repeat", "")
        rows.append({
            "label": label,
            "rep": rep,
            "success": s.get("success_inferred_by_radius", None),
            "duration_sec": s.get("duration_sec", None),
            "path_length": s.get("odom_path_length", None),
            "net_displacement": s.get("odom_net_displacement", None),
            "vx_mean": s.get("mpc_vx_mean", None),
            "vx_max": s.get("mpc_vx_max", None),
            "body_height_mean": s.get("mpc_body_height_mean", None),
            "clearance_mean": s.get("mpc_clearance_mean", None),
            "enable_fraction": s.get("mpc_enable_fraction", None),
            "relative_goal_min_dist": s.get("relative_goal_min_dist", None),
            "objective_mean": s.get("objective_mean", None),
            "summary_path": p,
        })

    order = {"balanced": 0, "motion": 1, "stability": 2, "energy": 3}
    rows.sort(key=lambda r: (order.get(r["label"], 99), int(r["rep"]) if str(r["rep"]).isdigit() else 999))

    csv_path = os.path.join(out_dir, "objective_ablation_summary.csv")
    md_path = os.path.join(out_dir, "objective_ablation_summary.md")

    fields = [
        "label", "rep", "success", "duration_sec", "path_length",
        "net_displacement", "vx_mean", "vx_max", "body_height_mean",
        "clearance_mean", "enable_fraction", "relative_goal_min_dist",
        "objective_mean", "summary_path",
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["objective_mean"] = json.dumps(rr["objective_mean"])
            w.writerow(rr)

    with open(md_path, "w") as f:
        f.write("# TRACER Objective Ablation Summary\n\n")
        f.write("| label | rep | success | duration | path | net disp | vx mean | vx max | height | clearance | enable |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['label']} | {r['rep']} | {r['success']} | "
                f"{fnum(r['duration_sec'],2)} | {fnum(r['path_length'],2)} | "
                f"{fnum(r['net_displacement'],2)} | {fnum(r['vx_mean'],3)} | "
                f"{fnum(r['vx_max'],3)} | {fnum(r['body_height_mean'],3)} | "
                f"{fnum(r['clearance_mean'],3)} | {fnum(r['enable_fraction'],2)} |\n"
            )

    print(f"[TRACER] wrote {csv_path}")
    print(f"[TRACER] wrote {md_path}")
    print()
    with open(md_path) as f:
        print(f.read())

if __name__ == "__main__":
    main()
