#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import csv
import glob


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

    parts = name.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        name = "_".join(parts[:-2])

    return name


def classify_goal(name):
    if "fwd100_lat050" in name:
        return "Fwd 1.0 + Lat 0.5"
    if "fwd050_lat100" in name:
        return "Fwd 0.5 + Lat 1.0"
    if "fwd100" in name or "fwd_100" in name:
        return "Fwd 1.0"
    return "Unknown"


def classify_ram_type(name):
    if name.startswith("cfg_learnedram_"):
        return "Learned RAM"
    if name.startswith("cfg_rel_"):
        return "Proxy RAM"
    return "Other"


def goal_key(name):
    if "fwd100_lat050" in name:
        return "1_fwd100_lat050"
    if "fwd050_lat100" in name:
        return "2_fwd050_lat100"
    if "fwd100" in name or "fwd_100" in name:
        return "0_fwd100"
    return "9_unknown"


def ram_key(ram_type):
    if ram_type == "Proxy RAM":
        return 0
    if ram_type == "Learned RAM":
        return 1
    return 9


def count_ram_sources(log_dir, fname):
    path = os.path.join(log_dir, fname)
    counts = {}

    if not os.path.exists(path):
        return counts

    with open(path, "r") as f:
        r = csv.DictReader(f)
        if "ram_source" not in r.fieldnames:
            return counts

        for row in r:
            src = row.get("ram_source", "")
            if src:
                counts[src] = counts.get(src, 0) + 1

    return counts


def main():
    if len(sys.argv) < 2:
        summary_path = "/root/TRACER/logs/tracer_goal_tracer_summary.csv"
    else:
        summary_path = sys.argv[1]

    log_dir = os.path.dirname(summary_path)

    if not os.path.exists(summary_path):
        print("Summary file not found:", summary_path)
        sys.exit(1)

    rows = []
    with open(summary_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            fname = row.get("file", "")
            name = short_name(fname)
            ram_type = classify_ram_type(name)

            if ram_type == "Other":
                continue

            if safe_int(row.get("valid_for_report", 0)) != 1:
                continue

            # Keep only representative config runs.
            if not (
                name.startswith("cfg_rel_")
                or name.startswith("cfg_learnedram_")
            ):
                continue

            row["_short_name"] = name
            row["_ram_type"] = ram_type
            row["_goal"] = classify_goal(name)
            row["_goal_key"] = goal_key(name)
            row["_ram_key"] = ram_key(ram_type)
            row["_ram_sources"] = count_ram_sources(log_dir, fname)
            rows.append(row)

    if not rows:
        print("No representative cfg rows found.")
        return

    rows = sorted(rows, key=lambda x: (x["_goal_key"], x["_ram_key"], x["_short_name"]))

    out_csv = os.path.join(log_dir, "tracer_ram_comparison_table.csv")
    out_md = os.path.join(log_dir, "tracer_ram_comparison_table.md")

    fields = [
        "goal",
        "ram_type",
        "experiment",
        "reached",
        "duration_sec",
        "end_dist",
        "distance_reduction",
        "travel_distance",
        "vx_ratio",
        "yaw_ratio",
        "rho",
        "sigma",
        "beta_v",
        "beta_s",
        "beta_e",
        "nominal",
        "cautious",
        "conservative",
        "ram_sources",
    ]

    table = []
    for row in rows:
        ram_sources = row.get("_ram_sources", {})
        ram_source_str = ";".join(
            ["%s:%d" % (k, ram_sources[k]) for k in sorted(ram_sources.keys())]
        )

        table.append({
            "goal": row["_goal"],
            "ram_type": row["_ram_type"],
            "experiment": row["_short_name"],
            "reached": str(safe_int(row.get("reached"))),
            "duration_sec": "%.2f" % safe_float(row.get("duration_sec")),
            "end_dist": "%.3f" % safe_float(row.get("end_dist")),
            "distance_reduction": "%.3f" % safe_float(row.get("distance_reduction")),
            "travel_distance": "%.3f" % safe_float(row.get("travel_distance")),
            "vx_ratio": "%.3f" % safe_float(row.get("vx_modulation_ratio")),
            "yaw_ratio": "%.3f" % safe_float(row.get("yaw_modulation_ratio")),
            "rho": "%.3f" % safe_float(row.get("avg_rho_v_mean")),
            "sigma": "%.3f" % safe_float(row.get("avg_sigma_v")),
            "beta_v": "%.3f" % safe_float(row.get("avg_beta_v")),
            "beta_s": "%.3f" % safe_float(row.get("avg_beta_s")),
            "beta_e": "%.3f" % safe_float(row.get("avg_beta_e")),
            "nominal": str(safe_int(row.get("mode_nominal"))),
            "cautious": str(safe_int(row.get("mode_cautious_mismatch"))),
            "conservative": str(safe_int(row.get("mode_conservative_mismatch"))),
            "ram_sources": ram_source_str,
        })

    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in table:
            w.writerow(row)

    with open(out_md, "w") as f:
        f.write("# Proxy RAM vs Learned RAM Comparison\n\n")
        f.write("| Goal | RAM | Reached | End Dist. | Dist. Reduction | Duration | vx Ratio | yaw Ratio | rho | sigma | Modes N/C/CM | RAM sources |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")

        for row in table:
            modes = "%s/%s/%s" % (
                row["nominal"],
                row["cautious"],
                row["conservative"],
            )
            f.write("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
                row["goal"],
                row["ram_type"],
                row["reached"],
                row["end_dist"],
                row["distance_reduction"],
                row["duration_sec"],
                row["vx_ratio"],
                row["yaw_ratio"],
                row["rho"],
                row["sigma"],
                modes,
                row["ram_sources"],
            ))

    print("Wrote:", out_csv)
    print("Wrote:", out_md)
    print("")
    print("Rows:", len(table))
    print("")
    with open(out_md, "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
