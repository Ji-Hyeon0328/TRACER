#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import Counter, defaultdict

IN_F6 = Path("datasets/phase_f/f6_true_metric_beta_targets_v0.csv")
OUT_CSV = Path("datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv")
OUT_MD = Path("reports/phase_f/f7_d7_true_metric_calibration_dataset_summary_v0.md")

ROOT = Path.cwd()

def rel_or_abs(p):
    p = str(p)
    if p.startswith("/home/kraken/Tracer/TRACER/"):
        return Path(p.replace("/home/kraken/Tracer/TRACER/", "", 1))
    return Path(p)

def read_manifest(manifest_path):
    mp = rel_or_abs(manifest_path)
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
    p = rel_or_abs(d5_dir) / "d7_objective_selector_shadow_v0.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing D7 csv: {p}")
    return p

f6_rows = list(csv.DictReader(open(IN_F6)))
if not f6_rows:
    raise SystemExit(f"empty input: {IN_F6}")

out_rows = []
source_counts = Counter()
d7_counts = {}
missing = []

for seed in f6_rows:
    tag = seed.get("tag", "unknown")
    manifest = seed.get("manifest", "")
    try:
        mrow = read_manifest(manifest)
        d7_csv = find_d7_csv(mrow)
    except Exception as e:
        missing.append((tag, manifest, str(e)))
        continue

    d7_rows = list(csv.DictReader(open(d7_csv)))
    d7_counts[tag] = len(d7_rows)

    for r in d7_rows:
        out = dict(r)

        # Source identifiers
        out["f7_source_tag"] = tag
        out["f7_reset_y"] = seed.get("reset_y", "")
        out["f7_manifest"] = manifest
        out["f7_d7_csv"] = str(d7_csv)
        out["f7_true_csv"] = seed.get("true_csv", "")

        # True-metric beta targets from F6
        out["target_beta_motion_true_metric"] = seed.get("beta_motion_true_seed", "")
        out["target_beta_stability_true_metric"] = seed.get("beta_stability_true_seed", "")
        out["target_beta_energy_true_metric"] = seed.get("beta_energy_true_seed", "")
        out["target_beta_dominant_true_metric"] = seed.get("dominant_beta_true_seed", "")

        # Objective scores from F5
        out["target_motion_score_true_metric"] = seed.get("objective_motion_score", "")
        out["target_stability_score_true_metric"] = seed.get("objective_stability_score", "")
        out["target_energy_score_true_metric"] = seed.get("objective_energy_score", "")
        out["target_balanced_score_true_metric"] = seed.get("objective_balanced_score", "")

        # Physical proxies from F4/F5
        for k in [
            "base_vx_mean",
            "base_vy_abs_mean",
            "joint_abs_power_mean",
            "imu_ang_vel_norm_mean",
            "contact_force_z_sum_mean",
            "motion_proxy",
            "lateral_drift_proxy",
            "energy_proxy",
            "stability_proxy",
            "contact_load_proxy",
        ]:
            if k in seed:
                out[f"true_metric_{k}"] = seed[k]

        out_rows.append(out)
        source_counts[tag] += 1

if missing:
    print("[TRACER][WARN] missing sources:")
    for x in missing:
        print("  ", x)

if not out_rows:
    raise SystemExit("no rows produced")

fields = list(out_rows[0].keys())
for r in out_rows:
    for k in r.keys():
        if k not in fields:
            fields.append(k)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

# Summary
beta_sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
for r in out_rows:
    tag = r["f7_source_tag"]
    try:
        bm = float(r["target_beta_motion_true_metric"])
        bs = float(r["target_beta_stability_true_metric"])
        be = float(r["target_beta_energy_true_metric"])
    except Exception:
        continue
    beta_sums[tag][0] += bm
    beta_sums[tag][1] += bs
    beta_sums[tag][2] += be
    beta_sums[tag][3] += 1

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F7 D7 True-Metric Calibration Dataset v0\n\n")
    f.write("This dataset attaches Phase-F6 true-metric beta target seeds to D7 runtime shadow logs from Phase-F3 rollouts.\n\n")
    f.write(f"- input F6: `{IN_F6}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- total rows: `{len(out_rows)}`\n")
    f.write(f"- missing sources: `{len(missing)}`\n\n")

    f.write("## Rows by source condition\n\n")
    f.write("| tag | d7_rows | output_rows | beta_motion | beta_stability | beta_energy |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for tag in sorted(source_counts):
        bm, bs, be, n = beta_sums[tag]
        f.write(
            f"| {tag} | {d7_counts.get(tag, 0)} | {source_counts[tag]} | "
            f"{bm/n:.4f} | {bs/n:.4f} | {be/n:.4f} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("This is a calibration dataset, not a final IRL Objective Selector dataset. Rows are timestep-expanded from only three rollout conditions, so they are useful for wiring/calibration and sanity checks, not for strong generalization claims.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print(f"[TRACER] rows={len(out_rows)}")
for tag in sorted(source_counts):
    print(f"[TRACER] {tag}: rows={source_counts[tag]}, d7_rows={d7_counts.get(tag, 0)}")
