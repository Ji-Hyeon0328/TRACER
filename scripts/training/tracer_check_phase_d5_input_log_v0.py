#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from collections import Counter, defaultdict

def fnum(x):
    try:
        if x == "" or x is None:
            return None
        return float(x)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    if not args.csv.strip():
        raise SystemExit("[ERROR] empty --csv path; no D5 policy input log was found")

    p = Path(args.csv)
    if not p.is_file():
        raise SystemExit(f"[ERROR] D5 policy input log not found or not a file: {p}")

    rows = list(csv.DictReader(p.open()))
    print(f"[TRACER] csv={p}")
    print(f"[TRACER] rows={len(rows)}")

    ctx = Counter(r.get("context", "unknown") for r in rows)
    print("[TRACER] context_counts:")
    for k, v in ctx.items():
        print(f"  {k}: {v}")

    yaw_vals = [fnum(r.get("ref_yaw_rate")) for r in rows]
    yaw_vals = [v for v in yaw_vals if v is not None]
    nonzero = [v for v in yaw_vals if abs(v) > 1e-6]

    print(f"[TRACER] yaw_rate_min={min(yaw_vals) if yaw_vals else None}")
    print(f"[TRACER] yaw_rate_max={max(yaw_vals) if yaw_vals else None}")
    print(f"[TRACER] yaw_rate_nonzero_count={len(nonzero)}")

    by_ctx = defaultdict(list)
    for r in rows:
        y = fnum(r.get("y"))
        yr = fnum(r.get("ref_yaw_rate"))
        if y is not None and yr is not None:
            by_ctx[r.get("context", "unknown")].append((y, yr))

    print("[TRACER] yaw samples by context:")
    for k, vals in by_ctx.items():
        vals = vals[:3] + vals[-3:] if len(vals) > 6 else vals
        samples = ", ".join([f"(y={y:.3f}, yaw={yr:.5f})" for y, yr in vals])
        print(f"  {k}: {samples}")

if __name__ == "__main__":
    main()
