#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from collections import Counter


PAIRS = [
    ("vx", "actual_vx", "pred_vx", "err_vx"),
    ("yaw_rate", "actual_yaw_rate", "pred_yaw_rate", "err_yaw_rate"),
    ("body_h", "actual_body_h", "pred_body_h", "err_body_h"),
    ("clearance", "actual_clearance", "pred_clearance", "err_clearance"),
    ("enable", "actual_enable", "pred_enable", "err_enable"),
]


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
        raise SystemExit(f"[ERROR] learned shadow log not found: {p}")

    rows = list(csv.DictReader(p.open()))
    print(f"[TRACER] csv={p}")
    print(f"[TRACER] rows={len(rows)}")

    ctx = Counter(r.get("context", "unknown") for r in rows)
    print("[TRACER] context_counts:")
    for k, v in ctx.items():
        print(f"  {k}: {v}")

    print("[TRACER] prediction error:")
    for name, actual_col, pred_col, err_col in PAIRS:
        errs = []
        preds = []
        actuals = []
        for r in rows:
            e = fnum(r.get(err_col))
            pred = fnum(r.get(pred_col))
            actual = fnum(r.get(actual_col))
            if e is not None:
                errs.append(e)
            if pred is not None:
                preds.append(pred)
            if actual is not None:
                actuals.append(actual)
        print(
            f"  {name}: "
            f"rmse={fmt(rmse(errs))}, mae={fmt(mae(errs))}, "
            f"pred_min={fmt(min(preds) if preds else None)}, pred_max={fmt(max(preds) if preds else None)}, "
            f"actual_min={fmt(min(actuals) if actuals else None)}, actual_max={fmt(max(actuals) if actuals else None)}"
        )


if __name__ == "__main__":
    main()
