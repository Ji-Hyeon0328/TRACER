#!/usr/bin/env python3
import csv
from pathlib import Path

IN_CSV = Path("datasets/phase_f/f4_true_metric_training_table_v0.csv")
OUT_CSV = Path("datasets/phase_f/f5_true_metric_objective_scores_v0.csv")
OUT_MD = Path("reports/phase_f/f5_true_metric_objective_scores_summary_v0.md")

rows = list(csv.DictReader(open(IN_CSV)))
if not rows:
    raise SystemExit(f"empty input: {IN_CSV}")

def ff(r, k):
    try:
        return float(r.get(k, ""))
    except Exception:
        return 0.0

def minmax_good_high(values):
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

def minmax_good_low(values):
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(hi - v) / (hi - lo) for v in values]

vx = [ff(r, "base_vx_mean") for r in rows]
lat = [ff(r, "base_vy_abs_mean") for r in rows]
energy = [ff(r, "joint_abs_power_mean") for r in rows]
imu = [ff(r, "imu_ang_vel_norm_mean") for r in rows]
contact = [ff(r, "contact_force_z_sum_mean") for r in rows]

motion_s = minmax_good_high(vx)
lat_s = minmax_good_low(lat)
energy_s = minmax_good_low(energy)
imu_s = minmax_good_low(imu)
contact_s = minmax_good_low(contact)

out_rows = []
for i, r in enumerate(rows):
    out = dict(r)
    out["motion_score_true"] = motion_s[i]
    out["lateral_score_true"] = lat_s[i]
    out["energy_score_true"] = energy_s[i]
    out["imu_stability_score_true"] = imu_s[i]
    out["contact_score_true"] = contact_s[i]

    # 초기 objective scores.
    # motion: forward speed 중심
    # stability: lateral + IMU + contact
    # energy: effort/power 중심
    out["objective_motion_score"] = motion_s[i]
    out["objective_stability_score"] = 0.45 * lat_s[i] + 0.40 * imu_s[i] + 0.15 * contact_s[i]
    out["objective_energy_score"] = energy_s[i]
    out["objective_balanced_score"] = (
        0.34 * float(out["objective_motion_score"]) +
        0.33 * float(out["objective_stability_score"]) +
        0.33 * float(out["objective_energy_score"])
    )

    out_rows.append(out)

fields = list(out_rows[0].keys())
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

ranked = sorted(out_rows, key=lambda r: float(r["objective_balanced_score"]), reverse=True)

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F5 True Metric Objective Scores v0\n\n")
    f.write("This converts Phase-F4 true rollout metrics into normalized objective scores for Objective Selector / RAM follow-up training.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- output: `{OUT_CSV}`\n")
    f.write(f"- rows: `{len(out_rows)}`\n\n")

    f.write("## Ranking by balanced score\n\n")
    f.write("| rank | tag | reset_y | motion | stability | energy | balanced | vx | lat_abs | power | imu |\n")
    f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for j, r in enumerate(ranked, 1):
        f.write(
            f"| {j} | {r['tag']} | {r['reset_y']} | "
            f"{float(r['objective_motion_score']):.4f} | "
            f"{float(r['objective_stability_score']):.4f} | "
            f"{float(r['objective_energy_score']):.4f} | "
            f"{float(r['objective_balanced_score']):.4f} | "
            f"{float(r['base_vx_mean']):.4f} | "
            f"{float(r['base_vy_abs_mean']):.4f} | "
            f"{float(r['joint_abs_power_mean']):.4f} | "
            f"{float(r['imu_ang_vel_norm_mean']):.4f} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("This is a bootstrap true-metric objective score table, not yet a full IRL Objective Selector. It is suitable for calibrating beta/objective targets and checking whether physical proxies agree with previous deployment choices.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print("[TRACER] balanced ranking:")
for r in ranked:
    print(
        f"  {r['tag']}: balanced={float(r['objective_balanced_score']):.4f}, "
        f"motion={float(r['objective_motion_score']):.4f}, "
        f"stability={float(r['objective_stability_score']):.4f}, "
        f"energy={float(r['objective_energy_score']):.4f}"
    )
