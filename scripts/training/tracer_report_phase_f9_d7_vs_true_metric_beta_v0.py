#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

IN_CSV = Path("datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv")
MODEL_JSON = Path("models/phase_f/f8_d7_true_metric_beta_calibrator_ridge_v0.json")
OUT_CSV = Path("datasets/phase_f/f9_d7_vs_true_metric_beta_comparison_v0.csv")
OUT_MD = Path("reports/phase_f/f9_d7_vs_true_metric_beta_comparison_summary_v0.md")

TARGET = [
    "target_beta_motion_true_metric",
    "target_beta_stability_true_metric",
    "target_beta_energy_true_metric",
]

METHODS = {
    "d7_pred": ["pred_beta_motion", "pred_beta_stability", "pred_beta_energy"],
    "d7_raw": ["raw_beta_motion", "raw_beta_stability", "raw_beta_energy"],
    "d7_prior": ["prior_beta_motion", "prior_beta_stability", "prior_beta_energy"],
    "d7_actual": ["actual_beta_motion", "actual_beta_stability", "actual_beta_energy"],
}

def ff(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default

def beta_from_cols(row, cols):
    vals = [ff(row.get(c), None) for c in cols]
    if any(v is None for v in vals):
        return None
    s = sum(max(v, 1e-9) for v in vals)
    if s <= 0:
        return None
    return [max(v, 1e-9) / s for v in vals]

def abs_err(a, b):
    return [abs(x - y) for x, y in zip(a, b)]

def rmse(a_list, b_list):
    n = 0
    acc = [0.0, 0.0, 0.0]
    for a, b in zip(a_list, b_list):
        for i in range(3):
            acc[i] += (a[i] - b[i]) ** 2
        n += 1
    if n == 0:
        return ["", "", ""]
    return [math.sqrt(x / n) for x in acc]

def mean_vec(vs):
    if not vs:
        return ["", "", ""]
    return [sum(v[i] for v in vs) / len(vs) for i in range(3)]

def mean_scalar(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return ""
    return sum(xs) / len(xs)

def build_f8_pred(rows):
    model = json.loads(MODEL_JSON.read_text())
    spec = model["feature_spec"]
    weights = model["weights"]

    contexts = spec["contexts"]
    ctx_to_i = {c: i for i, c in enumerate(contexts)}
    nums = spec["numeric_features"]
    mu = spec["numeric_mean"]
    sd = spec["numeric_std"]

    preds = []
    for r in rows:
        x = [1.0]

        ctx = r.get("context", "unknown") or "unknown"
        for c in contexts:
            x.append(1.0 if ctx == c else 0.0)

        for k, m, s in zip(nums, mu, sd):
            v = ff(r.get(k), 0.0)
            if s == 0:
                x.append(0.0)
            else:
                x.append((v - m) / s)

        y = []
        for j in range(3):
            yj = 0.0
            for i in range(len(x)):
                yj += x[i] * weights[i][j]
            y.append(max(yj, 1e-9))

        z = sum(y)
        preds.append([v / z for v in y])

    return preds

def fmt_vec(v):
    if not v or v[0] == "":
        return ""
    return "(" + ", ".join(f"{x:.4f}" for x in v) + ")"

def fmt(x):
    if x == "":
        return ""
    return f"{float(x):.6f}"

def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    if not rows:
        raise SystemExit(f"empty input: {IN_CSV}")

    targets = [beta_from_cols(r, TARGET) for r in rows]
    f8_preds = build_f8_pred(rows)

    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[r.get("f7_source_tag", "unknown")].append(i)

    out_rows = []

    method_preds = {}
    for name, cols in METHODS.items():
        method_preds[name] = [beta_from_cols(r, cols) for r in rows]
    method_preds["f8_calibrated_trainfit"] = f8_preds

    for tag, idxs in sorted(grouped.items()):
        target_tag = [targets[i] for i in idxs if targets[i] is not None]
        target_mean = mean_vec(target_tag)

        for method, preds in method_preds.items():
            pred_tag = []
            targ_tag = []
            abs_e_all = []
            l1_all = []

            for i in idxs:
                p = preds[i]
                t = targets[i]
                if p is None or t is None:
                    continue
                pred_tag.append(p)
                targ_tag.append(t)
                e = abs_err(p, t)
                abs_e_all.append(e)
                l1_all.append(sum(e))

            pred_mean = mean_vec(pred_tag)
            rmse_v = rmse(pred_tag, targ_tag)
            mae_v = mean_vec(abs_e_all)
            l1 = mean_scalar(l1_all)

            out_rows.append({
                "tag": tag,
                "method": method,
                "rows": len(pred_tag),
                "target_beta_motion": target_mean[0],
                "target_beta_stability": target_mean[1],
                "target_beta_energy": target_mean[2],
                "pred_beta_motion": pred_mean[0],
                "pred_beta_stability": pred_mean[1],
                "pred_beta_energy": pred_mean[2],
                "rmse_motion": rmse_v[0],
                "rmse_stability": rmse_v[1],
                "rmse_energy": rmse_v[2],
                "mae_motion": mae_v[0],
                "mae_stability": mae_v[1],
                "mae_energy": mae_v[2],
                "mean_l1_beta_error": l1,
            })

    fields = [
        "tag", "method", "rows",
        "target_beta_motion", "target_beta_stability", "target_beta_energy",
        "pred_beta_motion", "pred_beta_stability", "pred_beta_energy",
        "rmse_motion", "rmse_stability", "rmse_energy",
        "mae_motion", "mae_stability", "mae_energy",
        "mean_l1_beta_error",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("# TRACER Phase-F9 D7 vs True-Metric Beta Comparison v0\n\n")
        f.write("This report compares existing D7 runtime beta outputs against Phase-F6 true-metric beta target seeds, and also includes the Phase-F8 train-fit calibrated beta.\n\n")
        f.write(f"- input dataset: `{IN_CSV}`\n")
        f.write(f"- F8 model: `{MODEL_JSON}`\n")
        f.write(f"- output csv: `{OUT_CSV}`\n")
        f.write(f"- rows: `{len(rows)}`\n\n")

        f.write("## Per-condition beta comparison\n\n")
        f.write("| tag | method | rows | target beta | predicted beta | mean L1 error | RMSE |\n")
        f.write("|---|---|---:|---|---|---:|---|\n")
        for r in out_rows:
            tb = [float(r["target_beta_motion"]), float(r["target_beta_stability"]), float(r["target_beta_energy"])]
            pb = [float(r["pred_beta_motion"]), float(r["pred_beta_stability"]), float(r["pred_beta_energy"])]
            rb = [float(r["rmse_motion"]), float(r["rmse_stability"]), float(r["rmse_energy"])]
            f.write(
                f"| {r['tag']} | {r['method']} | {r['rows']} | "
                f"{fmt_vec(tb)} | {fmt_vec(pb)} | {fmt(r['mean_l1_beta_error'])} | {fmt_vec(rb)} |\n"
            )

        f.write("\n## Safe interpretation\n\n")
        f.write("- `d7_pred`, `d7_raw`, `d7_prior`, and `d7_actual` show how the existing D7 runtime beta behavior differs from true-metric beta seeds.\n")
        f.write("- `f8_calibrated_trainfit` is included only as an upper-bound diagnostic. It should not be deployed because F8 is trained on only three source rollout conditions and its leave-one-tag-out diagnostic was weak.\n")
        f.write("- This report is suitable for calibration evidence, not for claiming a completed IRL Objective Selector.\n")

    print(f"[TRACER] wrote {OUT_CSV}")
    print(f"[TRACER] wrote {OUT_MD}")

    print("[TRACER] lowest mean L1 error by tag:")
    for tag in sorted(grouped):
        candidates = [r for r in out_rows if r["tag"] == tag]
        candidates.sort(key=lambda r: float(r["mean_l1_beta_error"]))
        best = candidates[0]
        print(
            f"  {tag}: best={best['method']} "
            f"L1={float(best['mean_l1_beta_error']):.6f} "
            f"pred=({float(best['pred_beta_motion']):.4f}, "
            f"{float(best['pred_beta_stability']):.4f}, "
            f"{float(best['pred_beta_energy']):.4f})"
        )

if __name__ == "__main__":
    main()
