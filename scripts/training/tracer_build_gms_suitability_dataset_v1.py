#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def infer_preset(policy_id: str, episode_id: str) -> str:
    s = policy_id or episode_id
    for prefix in [
        "reference_sweep_v0_",
        "flat_normal_",
    ]:
        s = s.replace(prefix, "")
    return s


def classify(row: dict[str, Any]) -> str:
    if not row["valid_data"]:
        return "invalid_data"

    if row["low_height"]:
        return "negative_low_height"

    if row["unstable_attitude"]:
        return "negative_unstable_attitude"

    if row["lateral_drift"]:
        if row["dx"] >= 0.09:
            return "variable_forward_with_lateral_drift"
        return "negative_lateral_drift"

    if row["low_progress"]:
        return "negative_low_progress"

    if row["dx"] >= 0.13 and row["z_p10"] >= 0.32:
        return "preferred_fast_flat"

    if row["dx"] >= 0.09 and row["z_p10"] >= 0.32:
        return "preferred_balanced_flat"

    if row["dx"] >= 0.06 and row["z_p10"] >= 0.30:
        return "acceptable_slow_flat"

    return "variable_or_unknown"


def compute_cost(row: dict[str, Any]) -> float:
    return (
        8.0 * (not row["valid_data"])
        + 4.0 * row["low_height"]
        + 3.0 * row["low_progress"]
        + 2.0 * row["lateral_drift"]
        + 2.0 * row["unstable_attitude"]
        + (1.5 * row["roll_abs_max"] if row["has_attitude"] else 0.25)
        + (1.5 * row["pitch_abs_max"] if row["has_attitude"] else 0.25)
        + (0.8 * abs(row["yaw_delta"]) if row["has_attitude"] else 0.10)
        - 2.0 * row["dx"]
    )


def load_episode_csv(path: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(path)
    if len(df) == 0:
        raise RuntimeError(f"empty episode csv: {path}")

    duration = max(1e-6, f(df["elapsed"].iloc[-1]) - f(df["elapsed"].iloc[0]))

    dx = f(df["odom_x"].iloc[-1]) - f(df["odom_x"].iloc[0])
    dy = f(df["odom_y"].iloc[-1]) - f(df["odom_y"].iloc[0])

    vx = float(df["mpc_vx"].mean())
    h = float(df["mpc_body_height"].mean())
    clr = float(df["mpc_clearance"].mean())

    z_mean = float(df["proprio_base_z"].mean())
    z_min = float(df["proprio_base_z"].min())
    z_p10 = float(df["proprio_base_z"].quantile(0.10))

    has_attitude = (
        "proprio_roll" in df.columns
        and "proprio_pitch" in df.columns
        and "proprio_yaw" in df.columns
    )

    roll_abs_max = float(df["proprio_roll"].abs().max()) if has_attitude else float("nan")
    pitch_abs_max = float(df["proprio_pitch"].abs().max()) if has_attitude else float("nan")
    yaw_delta = (
        f(df["proprio_yaw"].iloc[-1]) - f(df["proprio_yaw"].iloc[0])
        if has_attitude
        else float("nan")
    )

    age_mpc_p90 = float(df["age_mpc"].quantile(0.90)) if "age_mpc" in df.columns else 9999.0
    age_proprio_p90 = float(df["age_proprio"].quantile(0.90)) if "age_proprio" in df.columns else 9999.0
    age_odom_p90 = float(df["age_odom"].quantile(0.90)) if "age_odom" in df.columns else 9999.0

    valid_data = (
        vx > 0.001
        and h > 0.1
        and z_mean > 0.1
        and age_mpc_p90 < 1.0
        and age_proprio_p90 < 1.0
        and age_odom_p90 < 1.0
    )

    row = {
        "episode_id": path.stem,
        "terrain": "unknown",
        "preset": "",
        "policy_id": "",
        "step_csv": str(path),

        "vx": vx,
        "body_height": h,
        "clearance": clr,

        "valid_data": bool(valid_data),
        "z_mean": z_mean,
        "z_min": z_min,
        "z_p10": z_p10,
        "dx": dx,
        "dy": dy,
        "dx_per_sec": dx / duration,
        "abs_dy_per_dx": abs(dy) / max(1e-6, abs(dx)),

        "has_attitude": bool(has_attitude),
        "roll_abs_max": roll_abs_max,
        "pitch_abs_max": pitch_abs_max,
        "yaw_delta": yaw_delta,

        "age_mpc_p90": age_mpc_p90,
        "age_proprio_p90": age_proprio_p90,
        "age_odom_p90": age_odom_p90,
    }

    row["low_height"] = bool(row["z_p10"] < 0.28)
    row["low_progress"] = bool(row["dx"] < 0.05)
    row["lateral_drift"] = bool(row["abs_dy_per_dx"] > 0.30)
    row["unstable_attitude"] = bool(
        row["has_attitude"]
        and (
            row["roll_abs_max"] > 0.20
            or row["pitch_abs_max"] > 0.25
            or abs(row["yaw_delta"]) > 0.20
        )
    )

    row["cost"] = compute_cost(row)
    row["suitability_label"] = classify(row)
    return row


def maybe_merge_summary(row: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return row

    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return row

    row["terrain"] = str(data.get("terrain", row["terrain"]))
    row["policy_id"] = str(data.get("policy_id", row["policy_id"]))
    row["preset"] = infer_preset(row["policy_id"], row["episode_id"])
    row["summary_json"] = str(summary_path)

    if "valid_data" in data:
        row["summary_valid_data"] = bool(data.get("valid_data"))

    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="data/rollouts/reference_sweep_v0/ref_sweep_v0_*/episodes/*.csv",
    )
    ap.add_argument(
        "--out-csv",
        default="data/rollout_metrics/gms_suitability_dataset_v1.csv",
    )
    ap.add_argument(
        "--out-json",
        default="data/rollout_metrics/gms_suitability_dataset_v1_summary.json",
    )
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.glob))
    if not paths:
        raise SystemExit(f"No episode CSVs matched: {args.glob}")

    rows = []
    for p in paths:
        try:
            row = load_episode_csv(p)
            summary_path = p.parent.parent / "summaries" / f"{p.stem}.json"
            row = maybe_merge_summary(row, summary_path)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        wr = csv.DictWriter(fcsv, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    valid_rows = [r for r in rows if r.get("valid_data")]
    by_label: dict[str, int] = {}
    for r in rows:
        lab = str(r.get("suitability_label", "unknown"))
        by_label[lab] = by_label.get(lab, 0) + 1

    summary = {
        "n_rows": len(rows),
        "n_valid": len(valid_rows),
        "n_invalid": len(rows) - len(valid_rows),
        "labels": by_label,
        "best_by_cost": sorted(rows, key=lambda r: f(r.get("cost", 9999.0)))[:10],
    }

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[TRACER] wrote:", out_csv)
    print("[TRACER] wrote:", out_json)
    print(json.dumps(summary, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
