#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import csv


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def short_name(fname):
    name = fname
    if name.startswith("tracer_goal_tracer_"):
        name = name[len("tracer_goal_tracer_"):]
    if name.endswith(".csv"):
        name = name[:-4]

    # Remove timestamp suffix if present.
    parts = name.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        name = "_".join(parts[:-2])

    return name


def goal_label(name):
    if "fwd100_lat050" in name:
        return "Forward 1.0 m + lateral 0.5 m"
    if "fwd050_lat100" in name:
        return "Forward 0.5 m + lateral 1.0 m"
    if "fwd100" in name or "fwd_100" in name:
        return "Forward 1.0 m"
    return "Unknown"


def main():
    if len(sys.argv) < 2:
        summary_path = "/root/TRACER/logs/tracer_goal_tracer_summary.csv"
    else:
        summary_path = sys.argv[1]

    if not os.path.exists(summary_path):
        print("Summary file not found:", summary_path)
        sys.exit(1)

    out_csv = os.path.join(os.path.dirname(summary_path), "tracer_goal_tracer_report_table.csv")
    out_md = os.path.join(os.path.dirname(summary_path), "tracer_goal_tracer_report_table.md")

    rows = []
    with open(summary_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            fname = row.get("file", "")
            name = short_name(fname)

            # M16: report table focuses on config-driven representative runs.
            if not name.startswith("cfg_"):
                continue

            if safe_int(row.get("valid_for_report", 0)) != 1:
                continue

            rows.append(row)

    if not rows:
        print("No valid cfg_ rows found in:", summary_path)
        sys.exit(0)

    rows = sorted(rows, key=lambda x: x.get("file", ""))

    fields = [
        "experiment",
        "goal",
        "duration_sec",
        "start_dist",
        "end_dist",
        "distance_reduction",
        "travel_distance",
        "reached",
        "emergency_stop_rows",
        "vx_ratio",
        "yaw_ratio",
        "rho",
        "sigma",
        "beta_v",
        "beta_s",
        "beta_e",
        "nominal_count",
        "cautious_count",
        "conservative_count",
    ]

    report = []
    for row in rows:
        name = short_name(row.get("file", ""))

        report.append({
            "experiment": name,
            "goal": goal_label(name),
            "duration_sec": "%.2f" % safe_float(row.get("duration_sec")),
            "start_dist": "%.3f" % safe_float(row.get("start_dist")),
            "end_dist": "%.3f" % safe_float(row.get("end_dist")),
            "distance_reduction": "%.3f" % safe_float(row.get("distance_reduction")),
            "travel_distance": "%.3f" % safe_float(row.get("travel_distance")),
            "reached": str(safe_int(row.get("reached"))),
            "emergency_stop_rows": str(safe_int(row.get("emergency_stop_rows"))),
            "vx_ratio": "%.3f" % safe_float(row.get("vx_modulation_ratio")),
            "yaw_ratio": "%.3f" % safe_float(row.get("yaw_modulation_ratio")),
            "rho": "%.3f" % safe_float(row.get("avg_rho_v_mean")),
            "sigma": "%.3f" % safe_float(row.get("avg_sigma_v")),
            "beta_v": "%.3f" % safe_float(row.get("avg_beta_v")),
            "beta_s": "%.3f" % safe_float(row.get("avg_beta_s")),
            "beta_e": "%.3f" % safe_float(row.get("avg_beta_e")),
            "nominal_count": str(safe_int(row.get("mode_nominal"))),
            "cautious_count": str(safe_int(row.get("mode_cautious_mismatch"))),
            "conservative_count": str(safe_int(row.get("mode_conservative_mismatch"))),
        })

    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in report:
            w.writerow(row)

    with open(out_md, "w") as f:
        f.write("# Goal-conditioned TRACER report table\n\n")
        f.write("| Experiment | Goal | Reached | End Dist. | Dist. Reduction | Duration | vx Ratio | yaw Ratio | rho | sigma | Modes N/C/CM |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in report:
            modes = "%s/%s/%s" % (
                row["nominal_count"],
                row["cautious_count"],
                row["conservative_count"],
            )
            f.write("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
                row["experiment"],
                row["goal"],
                row["reached"],
                row["end_dist"],
                row["distance_reduction"],
                row["duration_sec"],
                row["vx_ratio"],
                row["yaw_ratio"],
                row["rho"],
                row["sigma"],
                modes,
            ))

    print("Wrote:", out_csv)
    print("Wrote:", out_md)
    print("")
    print("Report rows:", len(report))
    print("")
    with open(out_md, "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
