#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from collections import Counter


ERR_COLS = ["err_vx", "err_yaw_rate", "err_body_h", "err_clearance", "err_enable"]


def fnum(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def rmse(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def mae(xs):
    xs = [abs(x) for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    return sum(xs) / len(xs)


def fmt(x):
    return "NA" if x is None else f"{x:.8f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    p = Path(args.csv)
    if not p.is_file():
        raise SystemExit(f"[ERROR] D6 MLP shadow log not found: {p}")

    rows = list(csv.DictReader(p.open()))
    print(f"[TRACER] csv={p}")
    print(f"[TRACER] rows={len(rows)}")

    ctx = Counter(r.get("context", "unknown") for r in rows)
    print("[TRACER] context_counts:")
    for k, v in ctx.items():
        print(f"  {k}: {v}")

    print("[TRACER] prediction error:")
    for col in ERR_COLS:
        vals = [fnum(r.get(col)) for r in rows]
        print(f"  {col}: rmse={fmt(rmse(vals))}, mae={fmt(mae(vals))}")

    print("[TRACER] prediction ranges:")
    for name in ["vx", "yaw_rate", "body_h", "clearance", "enable"]:
        vals = [fnum(r.get(f"pred_{name}")) for r in rows]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        if vals:
            print(f"  pred_{name}: min={min(vals):.8f}, max={max(vals):.8f}")


if __name__ == "__main__":
    main()
