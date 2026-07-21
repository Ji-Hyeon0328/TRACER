#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

F14_JSON = Path("models/phase_f/f14_grouped_runtime_safe_beta_calibrator_ridge_v0.json")
F20_JSON = Path("models/phase_f/f20_robust_runtime_safe_beta_calibrator_ridge_v0.json")
F18_CSV = Path("datasets/phase_f/f18_robust_grouped_true_metric_beta_targets_v0.csv")
F17_CSV = Path("datasets/phase_f/f17_rollout_outlier_variance_audit_v0.csv")

OUT_CSV = Path("datasets/phase_f/f21_mean_vs_robust_calibrator_comparison_v0.csv")
OUT_MD = Path("reports/phase_f/f21_mean_vs_robust_calibrator_comparison_summary_v0.md")

def load_json(p):
    with open(p) as f:
        return json.load(f)

def as_tag_summary(model):
    return {r["tag"]: r for r in model.get("tag_summary", [])}

def as_loto(model):
    return {r["heldout"]: r for r in model.get("leave_one_tag_out", [])}

def fmt_vec(v):
    return "(" + ", ".join(f"{float(x):.4f}" for x in v) + ")"

def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

f14 = load_json(F14_JSON)
f20 = load_json(F20_JSON)

f14_train = as_tag_summary(f14)
f20_train = as_tag_summary(f20)
f14_loto = as_loto(f14)
f20_loto = as_loto(f20)

f18_rows = list(csv.DictReader(open(F18_CSV)))
robust_shift = {
    r["base_tag"]: f(r.get("mean_vs_robust_beta_l1"))
    for r in f18_rows
}

mean_beta = {
    r["base_tag"]: [
        f(r.get("mean_beta_motion_true_seed")),
        f(r.get("mean_beta_stability_true_seed")),
        f(r.get("mean_beta_energy_true_seed")),
    ]
    for r in f18_rows
}
robust_beta = {
    r["base_tag"]: [
        f(r.get("robust_median_beta_motion_true_seed")),
        f(r.get("robust_median_beta_stability_true_seed")),
        f(r.get("robust_median_beta_energy_true_seed")),
    ]
    for r in f18_rows
}

f17_rows = list(csv.DictReader(open(F17_CSV)))
group_max_outlier_delta = defaultdict(float)
group_mean_outlier_delta = defaultdict(list)
for r in f17_rows:
    bt = r.get("base_tag", "")
    d = f(r.get("beta_leave_one_out_delta_l1"))
    group_max_outlier_delta[bt] = max(group_max_outlier_delta[bt], d)
    group_mean_outlier_delta[bt].append(d)

tags = sorted(set(f14_loto) | set(f20_loto) | set(f14_train) | set(f20_train))

rows = []
for tag in tags:
    has_f14_train = tag in f14_train
    has_f14_loto = tag in f14_loto

    train14 = f(f14_train.get(tag, {}).get("l1")) if has_f14_train else ""
    train20 = f(f20_train.get(tag, {}).get("l1"))
    loto14 = f(f14_loto.get(tag, {}).get("l1_mean")) if has_f14_loto else ""
    loto20 = f(f20_loto.get(tag, {}).get("l1_mean"))

    rows.append({
        "base_tag": tag,
        "mean_beta": fmt_vec(mean_beta.get(tag, [0, 0, 0])),
        "robust_beta": fmt_vec(robust_beta.get(tag, [0, 0, 0])),
        "mean_vs_robust_beta_l1": robust_shift.get(tag, 0.0),
        "f17_max_rollout_loo_beta_delta_l1": group_max_outlier_delta.get(tag, 0.0),
        "f17_mean_rollout_loo_beta_delta_l1": mean(group_mean_outlier_delta[tag]) if tag in group_mean_outlier_delta else 0.0,
        "f14_train_l1": train14,
        "f20_train_l1": train20,
        "train_l1_improvement_f14_minus_f20": (train14 - train20) if train14 != "" else "",
        "f14_loto_l1": loto14,
        "f20_loto_l1": loto20,
        "loto_l1_improvement_f14_minus_f20": (loto14 - loto20) if loto14 != "" else "",
    })

rows.sort(key=lambda r: r["loto_l1_improvement_f14_minus_f20"], reverse=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    fields = list(rows[0].keys())
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

improved = [r for r in rows if r["loto_l1_improvement_f14_minus_f20"] > 0]
worse = [r for r in rows if r["loto_l1_improvement_f14_minus_f20"] < 0]

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F21 Mean vs Robust Calibrator Comparison v0\n\n")
    f.write("This compares the Phase-F14 mean-target runtime-safe beta calibrator against the Phase-F20 robust-median-target calibrator.\n\n")
    f.write(f"- F14 model: `{F14_JSON}`\n")
    f.write(f"- F20 model: `{F20_JSON}`\n")
    f.write(f"- F18 robust target table: `{F18_CSV}`\n")
    f.write(f"- F17 rollout outlier audit: `{F17_CSV}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n\n")

    f.write("## Leave-one-tag-out comparison\\n\\n")
    f.write("| base_tag | mean beta | robust beta | target shift L1 | F14 LOTO L1 | F20 LOTO L1 | improvement | F17 max rollout delta |\\n")
    f.write("|---|---|---|---:|---:|---:|---:|---:|\\n")
    for r in rows:
        f14_loto_s = f"{r['f14_loto_l1']:.6f}" if r["f14_loto_l1"] != "" else "NA"
        f20_loto_s = f"{r['f20_loto_l1']:.6f}"
        improvement_s = (
            f"{r['loto_l1_improvement_f14_minus_f20']:.6f}"
            if r["loto_l1_improvement_f14_minus_f20"] != ""
            else "NA"
        )
        f.write(
            f"| {r['base_tag']} | {r['mean_beta']} | {r['robust_beta']} | "
            f"{r['mean_vs_robust_beta_l1']:.6f} | "
            f"{f14_loto_s} | "
            f"{f20_loto_s} | "
            f"{improvement_s} | "
            f"{r['f17_max_rollout_loo_beta_delta_l1']:.6f} |\\n"
        )

    f.write("\n## Summary\n\n")
    f.write(f"- comparable conditions: `{len(comparable)}`\n")
    f.write(f"- new conditions without F14 baseline: `{len(new_conditions)}`\n")
    f.write(f"- conditions improved by robust target: `{len(improved)}` / `{len(comparable)}`\n")
    f.write(f"- conditions worsened by robust target: `{len(worse)}` / `{len(comparable)}`\n\n")

    if improved:
        best = improved[0]
        f.write(f"- strongest improvement: `{best['base_tag']}` with LOTO L1 improvement `{best['loto_l1_improvement_f14_minus_f20']:.6f}`\n")
    if worse:
        worst = sorted(worse, key=lambda r: r["loto_l1_improvement_f14_minus_f20"])[0]
        f.write(f"- strongest degradation: `{worst['base_tag']}` with LOTO L1 change `{worst['loto_l1_improvement_f14_minus_f20']:.6f}`\n")

    f.write("\n## Safe interpretation\n\n")
    f.write("F20 is still diagnostic, not deployable. If robust targets improve outlier-sensitive conditions but high-error conditions remain, the bottleneck is likely runtime feature expressiveness rather than only target noise. The next step should be feature/context design, not simply a larger linear ridge model.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print("[TRACER] LOTO improvement F14 - F20:")
for r in rows:
    print(
        f"  {r['base_tag']}: "
        f"F14={r['f14_loto_l1']:.6f} "
        f"F20={r['f20_loto_l1']:.6f} "
        f"improvement={(f'{r[\'loto_l1_improvement_f14_minus_f20\']:.6f}' if r['loto_l1_improvement_f14_minus_f20'] != '' else 'NA')}"
    )
