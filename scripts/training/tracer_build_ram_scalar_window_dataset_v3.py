#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


BASE_FEATURE_CANDIDATES = [
    "mpc_vx",
    "mpc_yaw",
    "mpc_body_height",
    "mpc_clearance",
    "mpc_enable",
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "proprio_abs_mean",
    "proprio_abs_max",
    "proprio_base_x",
    "proprio_base_y",
    "proprio_base_z",
    "odom_x",
    "odom_y",
    "odom_z",
    "odom_vx",
    "age_mpc",
    "age_beta",
    "age_proprio",
    "age_odom",
    "age_debug",
]


def f(x, default=0.0):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def b(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def read_step_csv(path: Path):
    with path.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise RuntimeError(f"empty step csv: {path}")
    fields = list(rows[0].keys())
    feature_cols = [c for c in BASE_FEATURE_CANDIDATES if c in fields]
    if not feature_cols:
        raise RuntimeError(f"no usable scalar feature cols in {path}")

    X = np.asarray([[f(r.get(c, 0.0)) for c in feature_cols] for r in rows], dtype=np.float32)

    def col(name: str, fallback=0.0):
        if name in fields:
            return np.asarray([f(r.get(name, fallback)) for r in rows], dtype=np.float32)
        return np.full(len(rows), fallback, dtype=np.float32)

    S = {
        "mpc_vx": col("mpc_vx"),
        "mpc_enable": col("mpc_enable", 1.0),
        "proprio_base_x": col("proprio_base_x"),
        "proprio_base_y": col("proprio_base_y"),
        "proprio_base_z": col("proprio_base_z"),
        "odom_x": col("odom_x"),
        "odom_y": col("odom_y"),
        "odom_z": col("odom_z"),
        "odom_vx": col("odom_vx"),
    }
    return X, S, feature_cols


def choose_pose_source(S: Dict[str, np.ndarray]) -> str:
    # Prefer proprio base pose when it has meaningful variation/height.
    pz = S["proprio_base_z"]
    oz = S["odom_z"]
    p_valid = np.isfinite(pz).all() and (np.nanmax(np.abs(pz)) > 1e-6)
    o_valid = np.isfinite(oz).all() and (np.nanmax(np.abs(oz)) > 1e-6)

    if p_valid:
        return "proprio"
    if o_valid:
        return "odom"
    return "proprio"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-rows-csv", default="reports/tracer_ram_gms_data_v2_core_hard_valid_rows.csv")
    ap.add_argument("--out-npz", default="data/training/tracer_ram_scalar_window_dataset_v3.npz")
    ap.add_argument("--summary-json", default="reports/tracer_ram_scalar_window_dataset_v3_summary.json")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--future", type=int, default=10)
    ap.add_argument("--sample-hz", type=float, default=5.0)
    args = ap.parse_args()

    with Path(args.valid_rows_csv).open(newline="") as fp:
        valid_rows = list(csv.DictReader(fp))

    windows = []
    labels = []
    meta = []
    skipped = []
    feature_cols_ref = None

    for rr in valid_rows:
        summary_path = Path(rr["summary_json"])
        if not summary_path.exists():
            skipped.append({"summary": str(summary_path), "reason": "summary_missing"})
            continue

        summary = json.loads(summary_path.read_text())
        step_csv = Path(summary.get("step_csv", ""))
        if not step_csv.exists():
            skipped.append({"summary": str(summary_path), "reason": "step_csv_missing", "step_csv": str(step_csv)})
            continue

        try:
            X, S, feature_cols = read_step_csv(step_csv)
        except Exception as exc:
            skipped.append({"summary": str(summary_path), "reason": f"read_failed:{exc}"})
            continue

        if feature_cols_ref is None:
            feature_cols_ref = feature_cols
        elif feature_cols != feature_cols_ref:
            skipped.append({
                "summary": str(summary_path),
                "reason": "feature_cols_mismatch",
                "feature_cols": feature_cols,
                "expected": feature_cols_ref,
            })
            continue

        T = len(X)
        if T < args.window:
            skipped.append({"summary": str(summary_path), "reason": f"too_short:T={T}"})
            continue

        pose_src = choose_pose_source(S)
        if pose_src == "proprio":
            px, py, pz = S["proprio_base_x"], S["proprio_base_y"], S["proprio_base_z"]
        else:
            px, py, pz = S["odom_x"], S["odom_y"], S["odom_z"]

        episode_success = b(rr.get("success_proxy"))
        episode_fallen_p90 = f(rr.get("ram_run_fallen_p90"))
        episode_fresh = f(rr.get("debug_fresh_rate_1p0s"))
        episode_bad = bool((not episode_success) or episode_fallen_p90 >= 0.70 or episode_fresh < 0.90)

        for start in range(0, T - args.window + 1, args.stride):
            end = start + args.window
            fut_end = min(T, end + args.future)

            if fut_end <= end:
                continue

            z_now = float(np.median(pz[start:end]))
            z_future_min = float(np.min(pz[end:fut_end]))

            xy_now = np.asarray([px[end - 1], py[end - 1]], dtype=np.float32)
            xy_future = np.asarray([px[fut_end - 1], py[fut_end - 1]], dtype=np.float32)
            future_dist = float(np.linalg.norm(xy_future - xy_now))

            vx_cmd = float(np.mean(S["mpc_vx"][start:end]))
            enable = float(np.mean(S["mpc_enable"][start:end]))
            horizon_sec = max(1e-6, (fut_end - end) / args.sample_hz)
            expected_dist = abs(vx_cmd) * horizon_sec * max(0.0, min(1.0, enable))

            commanded_forward = bool(abs(vx_cmd) > 0.03 and enable > 0.5)
            low_progress = bool(commanded_forward and expected_dist > 0.03 and future_dist < 0.20 * expected_dist)

            # Relative height labels are safer than absolute z labels.
            low_height = bool(z_future_min < z_now - 0.08)
            fallen_height = bool(z_future_min < z_now - 0.14)

            bad = bool(episode_bad or fallen_height or low_height or low_progress)
            good = bool(not bad)

            windows.append(X[start:end])
            labels.append([
                float(fallen_height),
                float(low_height),
                float(low_progress),
                float(bad),
                float(good),
            ])

            meta.append({
                "terrain": rr.get("terrain", ""),
                "style": rr.get("forced_style", ""),
                "summary_json": str(summary_path),
                "step_csv": str(step_csv),
                "start": start,
                "end": end,
                "future_end": fut_end,
                "pose_source": pose_src,
                "z_now": z_now,
                "z_future_min": z_future_min,
                "future_dist": future_dist,
                "expected_dist": expected_dist,
                "vx_cmd": vx_cmd,
                "enable": enable,
                "episode_success": episode_success,
                "episode_bad": episode_bad,
                "fallen_height": fallen_height,
                "low_height": low_height,
                "low_progress": low_progress,
                "bad": bad,
                "good": good,
            })

    if not windows:
        raise SystemExit("No scalar RAM windows built")

    Xw = np.asarray(windows, dtype=np.float32)
    Y = np.asarray(labels, dtype=np.float32)

    label_names = np.asarray([
        "future_fallen_height",
        "future_low_height",
        "future_low_progress",
        "future_bad_locomotion",
        "future_good_locomotion",
    ])

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        windows=Xw,
        labels=Y,
        label_names=label_names,
        feature_names=np.asarray(feature_cols_ref),
        meta_json=np.asarray([json.dumps(m) for m in meta]),
    )

    summary = {
        "valid_rows_csv": args.valid_rows_csv,
        "out_npz": str(out_npz),
        "window": args.window,
        "stride": args.stride,
        "future": args.future,
        "sample_hz": args.sample_hz,
        "n_episodes": len(valid_rows),
        "n_windows": int(Xw.shape[0]),
        "window_shape": list(Xw.shape),
        "label_shape": list(Y.shape),
        "feature_names": feature_cols_ref,
        "label_names": [str(x) for x in label_names],
        "label_means": {str(label_names[i]): float(Y[:, i].mean()) for i in range(Y.shape[1])},
        "skipped": skipped,
    }

    Path(args.summary_json).write_text(json.dumps(summary, indent=2))

    print("[TRACER] scalar RAM window dataset v3")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
