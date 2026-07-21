#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import Counter, defaultdict

IN_F12 = Path("datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv")
OUT_CSV = Path("datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv")
OUT_MD = Path("reports/phase_f/f13_d7_grouped_beta_calibration_dataset_summary_v0.md")

ROOT_PREFIX = "/home/kraken/Tracer/TRACER/"

def rel_path(p):
    p = str(p)
    if p.startswith(ROOT_PREFIX):
        p = p.replace(ROOT_PREFIX, "", 1)
    return Path(p)

def read_manifest(manifest):
    mp = rel_path(manifest)
    if not mp.exists():
        raise FileNotFoundError(f"missing manifest: {mp}")
    rows = list(csv.DictReader(open(mp), delimiter="\t"))
    if not rows:
        raise RuntimeError(f"empty manifest: {mp}")
    return rows[0]

def find_d7_csv(manifest_row):
    d5_dir = manifest_row.get("d5_log_dir", "")
    if not d5_dir:
        raise RuntimeError("manifest row has no d5_log_dir")
    p = rel_path(d5_dir) / "d7_objective_selector_shadow_v0.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing D7 csv: {p}")
    return p

def get_first(row, keys, default=""):
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    return default

rollouts = list(csv.DictReader(open(IN_F12)))
if not rollouts:
    raise SystemExit(f"empty input: {IN_F12}")

out_rows = []
source_counts = Counter()
d7_counts = {}
missing = []

for rr in rollouts:
    tag = rr.get("tag", "unknown")
    base_tag = rr.get("base_tag", tag)

    try:
        mrow = read_manifest(rr.get("manifest", ""))
        d7_csv = find_d7_csv(mrow)
    except Exception as e:
        missing.append((tag, rr.get("manifest", ""), str(e)))
        continue

    d7_rows = list(csv.DictReader(open(d7_csv)))
    d7_counts[tag] = len(d7_rows)

    beta_m = get_first(rr, ["group_group_beta_motion_true_seed", "group_beta_motion_true_seed"])
    beta_s = get_first(rr, ["group_group_beta_stability_true_seed", "group_beta_stability_true_seed"])
    beta_e = get_first(rr, ["group_group_beta_energy_true_seed", "group_beta_energy_true_seed"])
    dominant = get_first(rr, ["group_group_dominant_beta_true_seed", "group_dominant_beta_true_seed"])

    motion_score = get_first(rr, ["group_group_motion_score", "group_motion_score"])
    stability_score = get_first(rr, ["group_group_stability_score", "group_stability_score"])
    energy_score = get_first(rr, ["group_group_energy_score", "group_energy_score"])
    balanced_score = get_first(rr, ["group_group_balanced_score", "group_balanced_score"])

    for r in d7_rows:
        out = dict(r)

        out["f13_source_tag"] = tag
        out["f13_base_tag"] = base_tag
        out["f13_manifest"] = rr.get("manifest", "")
        out["f13_d7_csv"] = str(d7_csv)
        out["f13_true_csv"] = rr.get("true_csv", "")
        out["f13_summary_csv"] = rr.get("summary_csv", "")

        out["target_beta_motion_grouped_true_metric"] = beta_m
        out["target_beta_stability_grouped_true_metric"] = beta_s
        out["target_beta_energy_grouped_true_metric"] = beta_e
        out["target_beta_dominant_grouped_true_metric"] = dominant

        out["target_motion_score_grouped_true_metric"] = motion_score
        out["target_stability_score_grouped_true_metric"] = stability_score
        out["target_energy_score_grouped_true_metric"] = energy_score
        out["target_balanced_score_grouped_true_metric"] = balanced_score

        for k in [
            "base_vx_mean",
            "base_vy_abs_mean",
            "joint_abs_power_mean",
            "imu_ang_vel_norm_mean",
            "contact_force_z_sum_mean",
        ]:
            out[f"rollout_true_metric_{k}"] = rr.get(k, "")

        for k in [
            "group_base_vx_mean_mean",
            "group_base_vy_abs_mean_mean",
            "group_joint_abs_power_mean_mean",
            "group_imu_ang_vel_norm_mean_mean",
            "group_contact_force_z_sum_mean_mean",
        ]:
            out[f"group_true_metric_{k}"] = rr.get(k, "")

        out_rows.append(out)
        source_counts[base_tag] += 1

if missing:
    print("[TRACER][WARN] missing sources:")
    for x in missing:
        print("  ", x)

if not out_rows:
    raise SystemExit("no rows produced")

fields = list(out_rows[0].keys())
for r in out_rows:
    for k in r:
        if k not in fields:
            fields.append(k)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

beta_sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
rollout_by_base = defaultdict(set)

for r in out_rows:
    bt = r["f13_base_tag"]
    rollout_by_base[bt].add(r["f13_source_tag"])
    try:
        bm = float(r["target_beta_motion_grouped_true_metric"])
        bs = float(r["target_beta_stability_grouped_true_metric"])
        be = float(r["target_beta_energy_grouped_true_metric"])
    except Exception:
        continue
    beta_sums[bt][0] += bm
    beta_sums[bt][1] += bs
    beta_sums[bt][2] += be
    beta_sums[bt][3] += 1

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F13 D7 Grouped-Beta Calibration Dataset v0\n\n")
    f.write("This attaches Phase-F12 grouped true-metric beta targets to D7 runtime shadow logs from Phase-F3/F11 rollouts.\n\n")
    f.write(f"- input F12 rollout table: `{IN_F12}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- total rows: `{len(out_rows)}`\n")
    f.write(f"- missing sources: `{len(missing)}`\n\n")

    f.write("## Rows by base condition\n\n")
    f.write("| base_tag | rollouts | output_rows | beta_motion | beta_stability | beta_energy |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for bt in sorted(beta_sums):
        bm, bs, be, n = beta_sums[bt]
        f.write(
            f"| {bt} | {len(rollout_by_base[bt])} | {source_counts[bt]} | "
            f"{bm/n:.4f} | {bs/n:.4f} | {be/n:.4f} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("This is the grouped-target version of F7. It is better conditioned than per-rollout F7 because repeated rollouts under the same base condition share the same grouped true-metric beta target.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print(f"[TRACER] rows={len(out_rows)}")
for bt in sorted(beta_sums):
    bm, bs, be, n = beta_sums[bt]
    print(f"[TRACER] {bt}: rows={source_counts[bt]}, beta=({bm/n:.4f}, {bs/n:.4f}, {be/n:.4f})")
