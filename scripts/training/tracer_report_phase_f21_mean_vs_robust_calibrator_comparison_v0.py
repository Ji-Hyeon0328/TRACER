#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path

F18 = Path("datasets/phase_f/f18_robust_grouped_true_metric_beta_targets_v0.csv")
F17 = Path("datasets/phase_f/f17_rollout_outlier_variance_audit_v0.csv")
F14_REPORT = Path("reports/phase_f/f14_grouped_runtime_safe_beta_calibrator_summary_v0.md")
F20_REPORT = Path("reports/phase_f/f20_robust_runtime_safe_beta_calibrator_summary_v0.md")

OUT_CSV = Path("datasets/phase_f/f21_mean_vs_robust_calibrator_comparison_v0.csv")
OUT_MD = Path("reports/phase_f/f21_mean_vs_robust_calibrator_comparison_summary_v0.md")

def ff(x, default=""):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default

def fmt_float(v):
    return "NA" if v == "" else f"{float(v):.6f}"

def fmt_beta(b):
    if not b:
        return ""
    return "(" + ", ".join(f"{float(x):.4f}" for x in b) + ")"

def pick(row, names, default=""):
    for n in names:
        if n in row and row[n] != "":
            return row[n]
    return default

def parse_report_l1s(path):
    """
    Parse lines like:
      p045: target=(...) pred=(...) L1=0.586065
    and leave-one-tag-out blocks.
    """
    train = {}
    loto = {}
    if not path.exists():
        return train, loto

    mode = "train"
    for line in path.read_text(errors="ignore").splitlines():
        if "leave-one-tag-out" in line.lower() or "leave-one" in line.lower():
            mode = "loto"

        m = re.search(r"^\s*([A-Za-z0-9_]+):.*?L1=([0-9.]+)", line)
        if not m:
            continue

        tag = m.group(1)
        l1 = float(m.group(2))
        if mode == "loto":
            loto[tag] = l1
        else:
            train[tag] = l1

    return train, loto

def load_f17_max_delta():
    out = {}
    if not F17.exists():
        return out

    rows = list(csv.DictReader(open(F17)))
    for r in rows:
        tag = pick(r, ["base_tag", "group_base_tag", "tag"], "")
        if not tag:
            continue
        val = pick(r, [
            "max_rollout_loo_beta_delta_l1",
            "f17_max_rollout_loo_beta_delta_l1",
            "rollout_loo_beta_delta_l1",
            "loo_beta_delta_l1",
        ], "")
        if val == "":
            continue
        cur = out.get(tag, 0.0)
        out[tag] = max(cur, float(val))
    return out

def load_f18_rows():
    if not F18.exists():
        raise SystemExit(f"[TRACER] missing {F18}")

    raw = list(csv.DictReader(open(F18)))
    rows = []
    seen = set()

    for r in raw:
        tag = pick(r, ["group_base_tag", "base_tag", "tag"], "")
        if not tag or tag in seen:
            continue
        seen.add(tag)

        mean_beta = [
            ff(pick(r, ["group_mean_beta_motion_true_seed", "mean_beta_motion_true_seed", "beta_motion_mean_true_metric"])),
            ff(pick(r, ["group_mean_beta_stability_true_seed", "mean_beta_stability_true_seed", "beta_stability_mean_true_metric"])),
            ff(pick(r, ["group_mean_beta_energy_true_seed", "mean_beta_energy_true_seed", "beta_energy_mean_true_metric"])),
        ]

        robust_beta = [
            ff(pick(r, ["group_robust_median_beta_motion_true_seed", "robust_beta_motion_true_metric", "target_beta_motion_robust_true_metric"])),
            ff(pick(r, ["group_robust_median_beta_stability_true_seed", "robust_beta_stability_true_metric", "target_beta_stability_robust_true_metric"])),
            ff(pick(r, ["group_robust_median_beta_energy_true_seed", "robust_beta_energy_true_metric", "target_beta_energy_robust_true_metric"])),
        ]

        if any(v == "" for v in mean_beta) or any(v == "" for v in robust_beta):
            continue

        shift = sum(abs(float(a) - float(b)) for a, b in zip(mean_beta, robust_beta))

        rows.append({
            "base_tag": tag,
            "mean_beta": mean_beta,
            "robust_beta": robust_beta,
            "mean_vs_robust_beta_l1": shift,
        })

    return rows

f14_train, f14_loto = parse_report_l1s(F14_REPORT)
f20_train, f20_loto = parse_report_l1s(F20_REPORT)
f17_delta = load_f17_max_delta()

rows = []
for r in load_f18_rows():
    tag = r["base_tag"]

    train14 = f14_train.get(tag, "")
    train20 = f20_train.get(tag, "")
    loto14 = f14_loto.get(tag, "")
    loto20 = f20_loto.get(tag, "")

    row = {
        "base_tag": tag,
        "mean_beta": fmt_beta(r["mean_beta"]),
        "robust_beta": fmt_beta(r["robust_beta"]),
        "mean_vs_robust_beta_l1": r["mean_vs_robust_beta_l1"],
        "f17_max_rollout_loo_beta_delta_l1": f17_delta.get(tag, ""),
        "f14_train_l1": train14,
        "f20_train_l1": train20,
        "train_l1_improvement_f14_minus_f20": (train14 - train20) if train14 != "" and train20 != "" else "",
        "f14_loto_l1": loto14,
        "f20_loto_l1": loto20,
        "loto_l1_improvement_f14_minus_f20": (loto14 - loto20) if loto14 != "" and loto20 != "" else "",
    }
    rows.append(row)

def sort_key(r):
    v = r["loto_l1_improvement_f14_minus_f20"]
    return float("-inf") if v == "" else v

rows.sort(key=sort_key, reverse=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "base_tag",
    "mean_beta",
    "robust_beta",
    "mean_vs_robust_beta_l1",
    "f17_max_rollout_loo_beta_delta_l1",
    "f14_train_l1",
    "f20_train_l1",
    "train_l1_improvement_f14_minus_f20",
    "f14_loto_l1",
    "f20_loto_l1",
    "loto_l1_improvement_f14_minus_f20",
]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

comparable = [r for r in rows if r["loto_l1_improvement_f14_minus_f20"] != ""]
new_conditions = [r for r in rows if r["loto_l1_improvement_f14_minus_f20"] == ""]
improved = [r for r in comparable if r["loto_l1_improvement_f14_minus_f20"] > 0]
worse = [r for r in comparable if r["loto_l1_improvement_f14_minus_f20"] < 0]

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F21 Mean vs Robust Calibrator Comparison v0\n\n")
    f.write("This compares the earlier mean-target grouped calibrator against the robust-target calibrator. Conditions absent from the older F14 baseline are marked as `NA` rather than treated as zero.\n\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- conditions: `{len(rows)}`\n")
    f.write(f"- comparable conditions: `{len(comparable)}`\n")
    f.write(f"- new conditions without F14 baseline: `{len(new_conditions)}`\n")
    f.write(f"- conditions improved by robust target: `{len(improved)}` / `{len(comparable)}`\n")
    f.write(f"- conditions worsened by robust target: `{len(worse)}` / `{len(comparable)}`\n\n")

    f.write("## Leave-one-tag-out comparison\n\n")
    f.write("| base_tag | mean beta | robust beta | target shift L1 | F14 LOTO L1 | F20 LOTO L1 | improvement | F17 max rollout delta |\n")
    f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['base_tag']} | {r['mean_beta']} | {r['robust_beta']} | "
            f"{float(r['mean_vs_robust_beta_l1']):.6f} | "
            f"{fmt_float(r['f14_loto_l1'])} | "
            f"{fmt_float(r['f20_loto_l1'])} | "
            f"{fmt_float(r['loto_l1_improvement_f14_minus_f20'])} | "
            f"{fmt_float(r['f17_max_rollout_loo_beta_delta_l1'])} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("Rows with `NA` in the F14 columns are new anchor conditions and should not be interpreted as degradation or improvement against the older baseline.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print("[TRACER] LOTO improvement F14 - F20:")
for r in rows:
    print(
        f"  {r['base_tag']}: "
        f"F14={fmt_float(r['f14_loto_l1'])} "
        f"F20={fmt_float(r['f20_loto_l1'])} "
        f"improvement={fmt_float(r['loto_l1_improvement_f14_minus_f20'])}"
    )
