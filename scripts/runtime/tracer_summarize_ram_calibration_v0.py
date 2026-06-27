#!/usr/bin/python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


NUMERIC_KEYS = [
    "mpc_vx",
    "mpc_body_height",
    "mpc_clearance",
    "mpc_enable",
    "ram_future_risk",
    "ram_future_slip",
    "ram_future_invalid",
    "ram_run_valid",
    "ram_run_fallen",
    "ram_recovery_needed",
    "ram_sigma_mean",
    "ram_rho_norm",
    "ram_ctrl_ema",
    "gate_level_code",
    "gate_action_code",
    "gate_would_override",
    "gate_control_risk",
    "gate_ctrl_ema",
    "gate_future_risk",
    "gate_fallen_prob",
    "gate_recovery_prob",
    "gate_sigma_mean",
    "gate_rho_norm",
    "gate_vx_scale",
    "gate_h_delta",
    "gate_clr_delta",
    "proprio_abs_mean",
]


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def pct(xs, q):
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = int(round((len(ys) - 1) * q))
    i = max(0, min(len(ys) - 1, i))
    return ys[i]


def summarize_one(path: Path):
    rows = []
    with path.open() as fobj:
        reader = csv.DictReader(fobj)
        rows = list(reader)

    if not rows:
        return None

    terrain = rows[0].get("terrain", "unknown")
    out = {
        "file": str(path),
        "terrain": terrain,
        "n": len(rows),
    }

    for k in NUMERIC_KEYS:
        xs = [f(r.get(k, 0.0)) for r in rows]
        out[f"{k}_mean"] = sum(xs) / max(1, len(xs))
        out[f"{k}_p50"] = pct(xs, 0.50)
        out[f"{k}_p90"] = pct(xs, 0.90)
        out[f"{k}_max"] = max(xs) if xs else 0.0

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="data/ram_calibration_live/ram_calib_summary_v0.csv")
    args = ap.parse_args()

    summaries = []
    for p in args.paths:
        s = summarize_one(Path(p))
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("[TRACER] no rows")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summaries[0].keys())

    with out.open("w", newline="") as fobj:
        w = csv.DictWriter(fobj, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    print(f"[TRACER] wrote {out}")
    print()
    for s in summaries:
        print(f"========== {s['terrain']} ==========")
        print(f"n={s['n']}")
        print(
            "mpc: "
            f"vx_mean={s['mpc_vx_mean']:.4f}, "
            f"enable_mean={s['mpc_enable_mean']:.3f}, "
            f"h_mean={s['mpc_body_height_mean']:.3f}, "
            f"clr_mean={s['mpc_clearance_mean']:.3f}"
        )
        print(
            "ram: "
            f"fallen_mean={s['ram_run_fallen_mean']:.3f}, "
            f"recovery_mean={s['ram_recovery_needed_mean']:.3f}, "
            f"sigma_mean={s['ram_sigma_mean_mean']:.3f}, "
            f"rho_norm_mean={s['ram_rho_norm_mean']:.3f}, "
            f"ctrl_ema_mean={s['ram_ctrl_ema_mean']:.3f}"
        )
        print(
            "gate: "
            f"level_mean={s['gate_level_code_mean']:.3f}, "
            f"action_mean={s['gate_action_code_mean']:.3f}, "
            f"override_mean={s['gate_would_override_mean']:.3f}, "
            f"vx_scale_mean={s['gate_vx_scale_mean']:.3f}, "
            f"h_delta_mean={s['gate_h_delta_mean']:.3f}, "
            f"clr_delta_mean={s['gate_clr_delta_mean']:.3f}"
        )
        print()


if __name__ == "__main__":
    main()
