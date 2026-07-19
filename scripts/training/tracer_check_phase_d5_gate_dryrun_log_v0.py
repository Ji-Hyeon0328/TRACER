#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from collections import Counter, defaultdict


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
        raise SystemExit(f"[ERROR] gate dry-run log not found: {p}")

    rows = list(csv.DictReader(p.open()))
    print(f"[TRACER] csv={p}")
    print(f"[TRACER] rows={len(rows)}")

    ctx = Counter(r.get("context", "unknown") for r in rows)
    print("[TRACER] context_counts:")
    for k, v in ctx.items():
        print(f"  {k}: {v}")

    accepted = [r for r in rows if str(r.get("gate_accept", "")).lower() == "true"]
    rejected = [r for r in rows if str(r.get("gate_accept", "")).lower() != "true"]
    accept_rate = len(accepted) / len(rows) if rows else 0.0

    print(f"[TRACER] accept_count={len(accepted)}")
    print(f"[TRACER] reject_count={len(rejected)}")
    print(f"[TRACER] accept_rate={accept_rate:.6f}")

    reasons = Counter(r.get("reject_reason", "") for r in rejected)
    print("[TRACER] reject_reasons:")
    for k, v in reasons.most_common():
        print(f"  {k}: {v}")

    print("[TRACER] error metrics:")
    for col in ERR_COLS:
        vals = [fnum(r.get(col)) for r in rows]
        print(f"  {col}: rmse={fmt(rmse(vals))}, mae={fmt(mae(vals))}")

    per_context = defaultdict(lambda: [0, 0])
    for r in rows:
        c = r.get("context", "unknown")
        per_context[c][0] += 1
        if str(r.get("gate_accept", "")).lower() == "true":
            per_context[c][1] += 1

    print("[TRACER] accept_rate_by_context:")
    for c, (n, a) in per_context.items():
        print(f"  {c}: {a}/{n} = {a/n if n else 0.0:.6f}")

    moving_rows = [r for r in rows if r.get("context", "unknown") != "goal_flat"]
    hold_rows = [r for r in rows if r.get("context", "unknown") == "goal_flat"]

    moving_accept = [r for r in moving_rows if str(r.get("gate_accept", "")).lower() == "true"]
    hold_accept = [r for r in hold_rows if str(r.get("gate_accept", "")).lower() == "true"]

    print("[TRACER] phase_accept_rates:")
    print(f"  moving_accept_rate={len(moving_accept)}/{len(moving_rows)} = {len(moving_accept)/len(moving_rows) if moving_rows else 0.0:.6f}")
    print(f"  hold_accept_rate={len(hold_accept)}/{len(hold_rows)} = {len(hold_accept)/len(hold_rows) if hold_rows else 0.0:.6f}")


if __name__ == "__main__":
    main()
