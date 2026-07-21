#!/usr/bin/env python3
import csv
from pathlib import Path

IN_CSV = Path("datasets/phase_f/f5_true_metric_objective_scores_v0.csv")
OUT_CSV = Path("datasets/phase_f/f6_true_metric_beta_targets_v0.csv")
OUT_MD = Path("reports/phase_f/f6_true_metric_beta_targets_summary_v0.md")

BETA_FLOOR = 0.05

rows = list(csv.DictReader(open(IN_CSV)))
if not rows:
    raise SystemExit(f"empty input: {IN_CSV}")

def ff(r, k, default=0.0):
    try:
        return float(r.get(k, default))
    except Exception:
        return default

out_rows = []
for r in rows:
    m = ff(r, "objective_motion_score")
    s = ff(r, "objective_stability_score")
    e = ff(r, "objective_energy_score")

    # Convert true objective quality scores into beta target seed.
    # Higher score means this rollout performed well for that objective.
    # Add a floor so no objective collapses to zero in early bootstrap data.
    wm = m + BETA_FLOOR
    ws = s + BETA_FLOOR
    we = e + BETA_FLOOR
    z = wm + ws + we

    out = dict(r)
    out["beta_motion_true_seed"] = wm / z
    out["beta_stability_true_seed"] = ws / z
    out["beta_energy_true_seed"] = we / z
    out["beta_floor"] = BETA_FLOOR

    # A simple suggested deployment preference label.
    # This is descriptive, not a final policy.
    vals = {
        "motion": out["beta_motion_true_seed"],
        "stability": out["beta_stability_true_seed"],
        "energy": out["beta_energy_true_seed"],
    }
    out["dominant_beta_true_seed"] = max(vals, key=vals.get)

    out_rows.append(out)

fields = list(out_rows[0].keys())

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F6 True Metric Beta Targets v0\n\n")
    f.write("This converts Phase-F5 true metric objective scores into bootstrap beta target seeds for future Objective Selector calibration.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- output: `{OUT_CSV}`\n")
    f.write(f"- beta_floor: `{BETA_FLOOR}`\n")
    f.write(f"- rows: `{len(out_rows)}`\n\n")

    f.write("## Beta target seeds\n\n")
    f.write("| tag | reset_y | beta_motion | beta_stability | beta_energy | dominant | motion_score | stability_score | energy_score | balanced |\n")
    f.write("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|\n")
    for r in out_rows:
        f.write(
            f"| {r['tag']} | {r['reset_y']} | "
            f"{float(r['beta_motion_true_seed']):.4f} | "
            f"{float(r['beta_stability_true_seed']):.4f} | "
            f"{float(r['beta_energy_true_seed']):.4f} | "
            f"{r['dominant_beta_true_seed']} | "
            f"{float(r['objective_motion_score']):.4f} | "
            f"{float(r['objective_stability_score']):.4f} | "
            f"{float(r['objective_energy_score']):.4f} | "
            f"{float(r['objective_balanced_score']):.4f} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("This is a true-metric beta target seed table, not a completed IRL Objective Selector. With only three rollout conditions, it should be used for calibration and sanity checking, not for a strong learned model claim.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
for r in out_rows:
    print(
        f"[TRACER] {r['tag']}: "
        f"beta=({float(r['beta_motion_true_seed']):.4f}, "
        f"{float(r['beta_stability_true_seed']):.4f}, "
        f"{float(r['beta_energy_true_seed']):.4f}), "
        f"dominant={r['dominant_beta_true_seed']}"
    )
