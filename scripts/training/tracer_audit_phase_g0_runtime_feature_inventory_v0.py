#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import defaultdict

INPUTS = {
    "f4_rollout_true_metrics": Path("datasets/phase_f/f4_true_metric_training_table_v0.csv"),
    "f18_robust_rollout_targets": Path("datasets/phase_f/f18_rollout_with_robust_group_beta_targets_v0.csv"),
    "f19_d7_robust_timestep": Path("datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv"),
    "f21_mean_vs_robust": Path("datasets/phase_f/f21_mean_vs_robust_calibrator_comparison_v0.csv"),
    "f22b_feature_ablation": Path("datasets/phase_f/f22b_runtime_feature_basis_ablation_imputed_v0.csv"),
}

OUT_MD = Path("reports/phase_g/g0_runtime_feature_inventory_summary_v0.md")
OUT_CSV = Path("datasets/phase_g/g0_runtime_feature_inventory_v0.csv")

CATEGORIES = {
    "target_beta": ["target_beta", "beta_true", "robust_true_metric"],
    "d7_beta": ["pred_beta", "raw_beta", "prior_beta", "actual_beta"],
    "position_lateral": ["x", "y", "lateral", "abs_y", "drift"],
    "reference": ["ref_", "cmd", "vx", "yaw", "clearance", "body_h"],
    "ram_proxy": ["ram", "slip_proxy", "roughness", "sigma"],
    "motion_score": ["motion_score", "stability_score", "energy_score", "effort_proxy"],
    "true_metric": ["true_metric", "base_vx", "base_vy", "joint_abs_power", "imu", "contact_force"],
    "contact": ["contact", "foot", "force"],
    "energy": ["energy", "power", "effort"],
    "manifest_source": ["manifest", "csv", "source", "tag"],
}

DESIRED_G_FEATURES = {
    "command_tracking": [
        "commanded-vs-real velocity tracking error",
        "yaw tracking error",
        "reference-vs-odom mismatch",
    ],
    "slip_mismatch": [
        "base_vx/ref_vx ratio",
        "lateral velocity while commanded yaw is small",
        "distance progress per commanded velocity",
    ],
    "contact_health": [
        "contact asymmetry",
        "contact loss ratio",
        "contact force variance",
    ],
    "energy_efficiency": [
        "energy per distance",
        "power per achieved velocity",
    ],
    "temporal_recovery": [
        "early/mid/late lateral drift trend",
        "hold drift after goal",
        "recovery slope",
    ],
    "terrain_history": [
        "terrain segment transition history",
        "time spent in rough/slope/goal segments",
    ],
}

def read_header_and_count(path):
    if not path.exists():
        return [], 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0
        n = sum(1 for _ in reader)
    return header, n

def categorize(col):
    c = col.lower()
    hits = []
    for cat, keys in CATEGORIES.items():
        if any(k in c for k in keys):
            hits.append(cat)
    return hits or ["uncategorized"]

rows = []
file_infos = {}

for name, path in INPUTS.items():
    header, nrows = read_header_and_count(path)
    file_infos[name] = {"path": path, "nrows": nrows, "ncols": len(header), "header": header}

    for col in header:
        cats = categorize(col)
        rows.append({
            "source": name,
            "path": str(path),
            "column": col,
            "categories": ";".join(cats),
        })

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    fields = ["source", "path", "column", "categories"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# Summaries
cat_counts = defaultdict(lambda: defaultdict(int))
source_cat_cols = defaultdict(lambda: defaultdict(list))

for r in rows:
    for cat in r["categories"].split(";"):
        cat_counts[r["source"]][cat] += 1
        source_cat_cols[r["source"]][cat].append(r["column"])

def has_any(header, keys):
    lower = [h.lower() for h in header]
    return any(any(k.lower() in h for h in lower) for k in keys)

availability = []
for feat_group, descs in DESIRED_G_FEATURES.items():
    for desc in descs:
        keys = desc.replace("-", " ").replace("/", " ").split()
        support = []
        for src, info in file_infos.items():
            if has_any(info["header"], keys):
                support.append(src)
        availability.append({
            "feature_group": feat_group,
            "candidate": desc,
            "supported_by": ", ".join(support) if support else "",
            "status": "available_or_partially_available" if support else "missing_or_needs_derivation",
        })

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-G0 Runtime Feature Inventory v0\n\n")
    f.write("This audits which runtime/log columns are currently available for Phase-G RAM/context feature strengthening.\n\n")

    f.write("## Input files\n\n")
    f.write("| source | path | rows | columns |\n")
    f.write("|---|---|---:|---:|\n")
    for src, info in file_infos.items():
        f.write(f"| {src} | `{info['path']}` | {info['nrows']} | {info['ncols']} |\n")

    f.write("\n## Column category counts\n\n")
    f.write("| source | category | column count |\n")
    f.write("|---|---|---:|\n")
    for src in sorted(cat_counts):
        for cat in sorted(cat_counts[src]):
            f.write(f"| {src} | {cat} | {cat_counts[src][cat]} |\n")

    f.write("\n## Representative columns by category\n\n")
    for src in sorted(source_cat_cols):
        f.write(f"\n### {src}\n\n")
        for cat in sorted(source_cat_cols[src]):
            cols = source_cat_cols[src][cat][:12]
            f.write(f"- **{cat}**: " + ", ".join(f"`{c}`" for c in cols) + "\n")

    f.write("\n## Desired Phase-G feature availability\n\n")
    f.write("| feature group | candidate | status | supported by |\n")
    f.write("|---|---|---|---|\n")
    for a in availability:
        f.write(
            f"| {a['feature_group']} | {a['candidate']} | "
            f"{a['status']} | {a['supported_by']} |\n"
        )

    f.write("\n## Phase-G interpretation\n\n")
    f.write(
        "Phase-G should prioritize features that are runtime-observable and explain true-metric beta targets without leaking target metrics. "
        "If a desired feature is absent, it should be derived from synchronized odom/reference/contact logs or added to the runtime logger in a later subphase.\n"
    )

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
for src, info in file_infos.items():
    print(f"[TRACER] {src}: rows={info['nrows']} cols={info['ncols']}")
