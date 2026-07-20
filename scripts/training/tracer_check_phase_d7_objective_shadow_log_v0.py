#!/usr/bin/env python3
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def f(row, key, default=float("nan")):
    try:
        v = row.get(key, "")
        if v in ("", None):
            return default
        return float(v)
    except Exception:
        return default


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def rmse(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return math.sqrt(sum(x*x for x in xs) / len(xs)) if xs else float("nan")


def fmt(x):
    return "nan" if not math.isfinite(x) else f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    p = Path(args.csv)
    if not p.exists():
        raise SystemExit(f"[ERROR] csv not found: {p}")

    rows = list(csv.DictReader(p.open()))
    print(f"[TRACER] csv={p}")
    print(f"[TRACER] rows={len(rows)}")
    if not rows:
        return

    ctx_counts = Counter(r.get("context", "unknown") for r in rows)
    print("[TRACER] context_counts:")
    for k, v in sorted(ctx_counts.items()):
        print(f"  {k}: {v}")

    print("[TRACER] pred beta ranges:")
    for k in ["pred_beta_motion", "pred_beta_stability", "pred_beta_energy"]:
        vals = [f(r, k) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        print(f"  {k}: min={min(vals):.6f}, max={max(vals):.6f}, mean={mean(vals):.6f}")

    if "raw_beta_motion" in rows[0]:
        print("[TRACER] raw beta ranges before floor/prior blend:")
        for k in ["raw_beta_motion", "raw_beta_stability", "raw_beta_energy"]:
            vals = [f(r, k) for r in rows]
            vals = [v for v in vals if math.isfinite(v)]
            print(f"  {k}: min={min(vals):.6f}, max={max(vals):.6f}, mean={mean(vals):.6f}")

    sums = [
        f(r, "pred_beta_motion") + f(r, "pred_beta_stability") + f(r, "pred_beta_energy")
        for r in rows
    ]
    sums = [s for s in sums if math.isfinite(s)]
    print("[TRACER] beta_sum:")
    print(f"  min={min(sums):.9f}, max={max(sums):.9f}, mean={mean(sums):.9f}")

    print("[TRACER] delta to current/static beta:")
    for k in ["err_beta_motion", "err_beta_stability", "err_beta_energy"]:
        vals = [f(r, k) for r in rows]
        print(f"  {k}: rmse={fmt(rmse(vals))}, mean={fmt(mean(vals))}")

    by = defaultdict(list)
    for r in rows:
        by[r.get("context", "unknown")].append(r)

    print("[TRACER] context-level predicted beta means:")
    for ctx in sorted(by):
        rs = by[ctx]
        bm = mean(f(r, "pred_beta_motion") for r in rs)
        bs = mean(f(r, "pred_beta_stability") for r in rs)
        be = mean(f(r, "pred_beta_energy") for r in rs)
        ms = mean(f(r, "motion_score") for r in rs)
        ss = mean(f(r, "stability_score") for r in rs)
        es = mean(f(r, "energy_score") for r in rs)
        print(
            f"  {ctx}: n={len(rs)} "
            f"beta=({fmt(bm)}, {fmt(bs)}, {fmt(be)}) "
            f"scores=({fmt(ms)}, {fmt(ss)}, {fmt(es)})"
        )

    print("[TRACER] tail rows:")
    for r in rows[-5:]:
        print(
            "  "
            f"ctx={r.get('context','')} x={f(r,'x'):.3f} y={f(r,'y'):.3f} "
            f"beta=({f(r,'pred_beta_motion'):.3f},{f(r,'pred_beta_stability'):.3f},{f(r,'pred_beta_energy'):.3f}) "
            f"score=({f(r,'motion_score'):.3f},{f(r,'stability_score'):.3f},{f(r,'energy_score'):.3f}) "
            f"hold={f(r,'hold_drift'):.3f}"
        )


if __name__ == "__main__":
    main()
