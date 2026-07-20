#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


FEATURES = [
    "reset_y",
    "y_mean",
    "mean_abs_y_context",
    "max_abs_y_context",
    "ram_slip_proxy_mean",
    "ram_roughness_proxy_mean",
    "ram_sigma_mean",
    "motion_score",
    "stability_score",
    "energy_score",
    "effort_proxy",
    "hold_drift",
    "rollout_max_abs_y",
    "rollout_mean_abs_y",
]


def f(row, key, default=float("nan")):
    try:
        if key in row and row[key] not in ("", None):
            return float(row[key])
        if key == "y_mean" and "y" in row and row["y"] not in ("", None):
            return float(row["y"])
        return default
    except Exception:
        return default


def finite(xs):
    return [x for x in xs if math.isfinite(x)]


def mean(xs):
    xs = finite(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = finite(xs)
    if len(xs) < 2:
        return 0.0 if len(xs) == 1 else float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def q(xs, pct):
    xs = sorted(finite(xs))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def load_csv(path):
    with Path(path).open(newline="") as fp:
        return list(csv.DictReader(fp))


def stats(rows, feature):
    vals = [f(r, feature) for r in rows]
    vals = finite(vals)
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": std(vals),
        "min": min(vals) if vals else float("nan"),
        "p05": q(vals, 0.05),
        "p50": q(vals, 0.50),
        "p95": q(vals, 0.95),
        "max": max(vals) if vals else float("nan"),
    }


def by_context(rows):
    d = defaultdict(list)
    for r in rows:
        d[r.get("context", "unknown")].append(r)
    return d


def fmt(x):
    return "nan" if not math.isfinite(x) else f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-csv", required=True)
    ap.add_argument("--runtime-csv", nargs="+", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    offline = load_csv(args.offline_csv)

    runtime_paths = []
    for item in args.runtime_csv:
        for part in str(item).replace("\\n", "\n").splitlines():
            for token in part.split():
                token = token.strip()
                if token:
                    runtime_paths.append(token)

    runtime = []
    for p in runtime_paths:
        rows = load_csv(p)
        for r in rows:
            r["_runtime_source"] = str(p)
        runtime.extend(rows)

    if not offline:
        raise SystemExit("[ERROR] empty offline csv")
    if not runtime:
        raise SystemExit("[ERROR] empty runtime csv")

    lines = []
    lines.append("# TRACER Phase-D7.1 Runtime Feature Alignment Report v0")
    lines.append("")
    lines.append(f"- offline_csv: `{args.offline_csv}`")
    lines.append(f"- offline_rows: `{len(offline)}`")
    lines.append(f"- runtime_csv_count: `{len(runtime_paths)}`")
    lines.append(f"- runtime_rows: `{len(runtime)}`")
    lines.append("")
    lines.append("## Global feature distribution shift")
    lines.append("")
    lines.append("| feature | off_mean | run_mean | mean_delta | off_std | run_std | std_ratio | off_p05 | run_p05 | off_p95 | run_p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    shift_rows = []
    for feat in FEATURES:
        os = stats(offline, feat)
        rs = stats(runtime, feat)
        std_ratio = rs["std"] / os["std"] if math.isfinite(os["std"]) and os["std"] > 1e-9 else float("nan")
        mean_delta = rs["mean"] - os["mean"] if math.isfinite(rs["mean"]) and math.isfinite(os["mean"]) else float("nan")
        shift_rows.append((feat, abs(mean_delta) if math.isfinite(mean_delta) else -1.0, os, rs, mean_delta, std_ratio))
        lines.append(
            f"| {feat} | {fmt(os['mean'])} | {fmt(rs['mean'])} | {fmt(mean_delta)} | "
            f"{fmt(os['std'])} | {fmt(rs['std'])} | {fmt(std_ratio)} | "
            f"{fmt(os['p05'])} | {fmt(rs['p05'])} | {fmt(os['p95'])} | {fmt(rs['p95'])} |"
        )

    lines.append("")
    lines.append("## Largest mean shifts")
    lines.append("")
    lines.append("| rank | feature | abs_mean_delta | off_mean | run_mean | comment |")
    lines.append("|---:|---|---:|---:|---:|---|")

    for i, (feat, _, os, rs, mean_delta, std_ratio) in enumerate(sorted(shift_rows, key=lambda x: x[1], reverse=True)[:10], 1):
        comment = ""
        if feat in ("motion_score", "stability_score", "energy_score", "effort_proxy"):
            comment = "objective-score feature shift"
        elif feat in ("hold_drift", "rollout_max_abs_y", "rollout_mean_abs_y"):
            comment = "rollout/window statistic shift"
        elif feat.startswith("ram_"):
            comment = "RAM proxy distribution shift"
        else:
            comment = "state/context statistic shift"
        lines.append(
            f"| {i} | {feat} | {fmt(abs(mean_delta))} | {fmt(os['mean'])} | {fmt(rs['mean'])} | {comment} |"
        )

    off_ctx = by_context(offline)
    run_ctx = by_context(runtime)

    lines.append("")
    lines.append("## Context-level key feature means")
    lines.append("")
    lines.append("| context | off_n | run_n | off_stability_score | run_stability_score | off_energy_score | run_energy_score | off_hold_drift | run_hold_drift | off_max_abs_y | run_max_abs_y |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    all_ctx = sorted(set(off_ctx) | set(run_ctx))
    for ctx in all_ctx:
        o = off_ctx.get(ctx, [])
        r = run_ctx.get(ctx, [])
        lines.append(
            f"| {ctx} | {len(o)} | {len(r)} | "
            f"{fmt(stats(o,'stability_score')['mean'])} | {fmt(stats(r,'stability_score')['mean'])} | "
            f"{fmt(stats(o,'energy_score')['mean'])} | {fmt(stats(r,'energy_score')['mean'])} | "
            f"{fmt(stats(o,'hold_drift')['mean'])} | {fmt(stats(r,'hold_drift')['mean'])} | "
            f"{fmt(stats(o,'rollout_max_abs_y')['mean'])} | {fmt(stats(r,'rollout_max_abs_y')['mean'])} |"
        )

    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    lines.append("- This report compares the D7 offline bootstrap training distribution with runtime shadow features.")
    lines.append("- Large shifts indicate why the raw ridge Objective Selector can collapse under online/window statistics.")
    lines.append("- The calibrated runtime beta should remain shadow-only until the feature definition is aligned.")
    lines.append("- Recommended next step: build a runtime-aligned D7.1 dataset using the same online/window feature computation.")
    lines.append("")

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {out}")
    print(f"[TRACER] offline_rows={len(offline)} runtime_rows={len(runtime)}")
    print("[TRACER] largest mean shifts:")
    for feat, _, os, rs, mean_delta, _ in sorted(shift_rows, key=lambda x: x[1], reverse=True)[:8]:
        print(f"  {feat}: offline={fmt(os['mean'])}, runtime={fmt(rs['mean'])}, delta={fmt(mean_delta)}")


if __name__ == "__main__":
    main()
