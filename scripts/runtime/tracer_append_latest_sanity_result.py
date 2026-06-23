#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CSV = ROOT / "data/sanity_results/tracer_fusion_sanity_results.csv"


def infer_geom_from_world_name(world: str) -> str:
    w = (world or "").lower()

    if "downslope_10deg" in w:
        return "downslope_10deg"
    if "downslope_5deg" in w:
        return "downslope_5deg"
    if "slope_10deg" in w and "downslope" not in w:
        return "upslope_10deg"
    if "slope_5deg" in w and "downslope" not in w:
        return "upslope_5deg"

    return "flat"


def ground_z_sloped(
    x: np.ndarray,
    deg: float,
    direction: str,
    center_z: float = -0.04981,
    thickness: float = 0.10,
) -> np.ndarray:
    theta = math.radians(deg)
    top_at_origin = center_z + thickness / (2.0 * math.cos(theta))

    if direction == "downslope":
        return top_at_origin - np.tan(theta) * x
    if direction == "upslope":
        return top_at_origin + np.tan(theta) * x

    raise ValueError(f"unknown direction: {direction}")


def compute_ground_z(x: np.ndarray, geom: str) -> np.ndarray:
    if geom == "flat":
        return np.zeros_like(x)
    if geom == "downslope_5deg":
        return ground_z_sloped(x, 5.0, "downslope")
    if geom == "downslope_10deg":
        return ground_z_sloped(x, 10.0, "downslope")
    if geom == "upslope_5deg":
        return ground_z_sloped(x, 5.0, "upslope")
    if geom == "upslope_10deg":
        return ground_z_sloped(x, 10.0, "upslope")

    raise ValueError(f"unknown geometry: {geom}")


def latest_file(pattern: str) -> Path:
    files = sorted((ROOT / "data/mission_logs").glob(pattern))
    if not files:
        raise FileNotFoundError(f"no file matching {pattern}")
    return files[-1]


def load_policy_entry(policy_json: Path, terrain_key: str) -> dict:
    if not policy_json.exists():
        return {}

    try:
        data = json.loads(policy_json.read_text())
    except Exception:
        return {}

    return data.get("terrains", {}).get(terrain_key, {})


def scalar(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def main() -> None:
    terrain_key = os.environ.get("TRACER_TERRAIN_KEY", "")
    world_name = os.environ.get("TRACER_WORLD_NAME", "")
    sanity_tag = os.environ.get("TRACER_SANITY_TAG", "")
    policy_json = Path(
        os.environ.get(
            "TRACER_FUSION_POLICY_JSON",
            str(ROOT / "configs/highlevel_policy/tracer_fusion_policy_v0.json"),
        )
    ).expanduser()

    out_csv = Path(os.environ.get("TRACER_SANITY_RESULT_CSV", str(DEFAULT_CSV))).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    npz_env = os.environ.get("TRACER_MISSION_LOG_NPZ", "")
    summary_env = os.environ.get("TRACER_MISSION_SUMMARY_JSON", "")

    if npz_env:
        npz_path = Path(npz_env).expanduser()
        if not npz_path.is_absolute():
            npz_path = ROOT / npz_path
    else:
        npz_path = latest_file("tracer_mission_log_*.npz")

    if summary_env:
        summary_path = Path(summary_env).expanduser()
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
    else:
        summary_path = latest_file("tracer_mission_summary_*.json")

    d = np.load(npz_path, allow_pickle=True)
    summary = json.loads(summary_path.read_text())

    proprio = d["proprio"]
    mpc = d["mpc_reference"]

    valid_p = np.isfinite(proprio).all(axis=1)
    valid_m = np.isfinite(mpc).all(axis=1)

    pv = proprio[valid_p]
    mv = mpc[valid_m]

    if len(pv) < 2:
        raise SystemExit("[TRACER] not enough valid proprio samples")

    x = pv[:, 1]
    y = pv[:, 2]
    z = pv[:, 3]
    roll = pv[:, 4]
    pitch = pv[:, 5]

    geom = os.environ.get("TRACER_TERRAIN_GEOM", "") or infer_geom_from_world_name(world_name)
    ground_z = compute_ground_z(x, geom)
    rel_z = z - ground_z

    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    min_abs_z = float(z.min())
    min_rel_z = float(rel_z.min())
    mean_rel_z = float(rel_z.mean())
    max_roll = float(np.abs(roll).max())
    max_pitch = float(np.abs(pitch).max())

    fallen_abs = bool((min_abs_z < 0.12) or (max_roll > 1.2) or (max_pitch > 1.0))
    fallen_rel = bool((min_rel_z < 0.12) or (max_roll > 1.2) or (max_pitch > 1.0))

    if len(mv) >= 1:
        vx_mean = float(np.mean(mv[:, 1]))
        yaw_mean = float(np.mean(mv[:, 2]))
        height_mean = float(np.mean(mv[:, 3]))
        clearance_mean = float(np.mean(mv[:, 4]))
        enable_fraction = float(np.mean(mv[:, 5] > 0.5))
        mpc_first = mv[0].tolist()
        mpc_last = mv[-1].tolist()
    else:
        vx_mean = yaw_mean = height_mean = clearance_mean = enable_fraction = float("nan")
        mpc_first = []
        mpc_last = []

    policy_entry = load_policy_entry(policy_json, terrain_key)
    command = policy_entry.get("command", {}) if policy_entry else {}

    fused_mode = policy_entry.get("fused_mode", "")
    semantic_mode = policy_entry.get("semantic_mode", "")
    style = policy_entry.get("suggested_style", "")
    runtime_fallback = policy_entry.get("runtime_fallback", "")
    policy_version = ""

    if policy_json.exists():
        try:
            policy_version = json.loads(policy_json.read_text()).get("policy_version", policy_json.stem)
        except Exception:
            policy_version = policy_json.stem

    valid_locomotion_candidate = bool(
        enable_fraction > 0.5
        and not fallen_rel
        and abs(dx) > 0.05
        and max_roll < 0.8
        and max_pitch < 0.8
    )

    stable_hold_candidate = bool(
        enable_fraction <= 0.5
        and not fallen_rel
        and abs(dx) < 0.15
        and abs(dy) < 0.15
        and max_roll < 0.8
        and max_pitch < 0.8
    )

    if semantic_mode:
        decision = semantic_mode
    elif valid_locomotion_candidate:
        decision = "valid_locomotion_candidate"
    elif stable_hold_candidate:
        decision = "stable_hold_candidate"
    elif fallen_rel:
        decision = "fallen_or_collapsed"
    else:
        decision = "needs_review"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sanity_tag": sanity_tag,
        "terrain_key": terrain_key,
        "world_name": world_name,
        "terrain_geom": geom,
        "policy_json": str(policy_json),
        "policy_version": policy_version,
        "fused_mode": fused_mode,
        "semantic_mode": semantic_mode,
        "runtime_fallback": runtime_fallback,
        "style": style,
        "cmd_vx": scalar(command.get("vx"), vx_mean),
        "cmd_yaw_rate": scalar(command.get("yaw_rate"), yaw_mean),
        "cmd_body_height": scalar(command.get("body_height"), height_mean),
        "cmd_clearance": scalar(command.get("swing_clearance"), clearance_mean),
        "cmd_enable": scalar(command.get("enable"), enable_fraction),
        "summary_duration_sec": summary.get("duration_sec"),
        "summary_num_samples": summary.get("num_samples"),
        "mpc_enable_fraction": enable_fraction,
        "mpc_vx_mean": vx_mean,
        "mpc_body_height_mean": height_mean,
        "mpc_clearance_mean": clearance_mean,
        "start_x": float(x[0]),
        "start_y": float(y[0]),
        "start_z": float(z[0]),
        "end_x": float(x[-1]),
        "end_y": float(y[-1]),
        "end_z": float(z[-1]),
        "delta_x": dx,
        "delta_y": dy,
        "min_abs_z": min_abs_z,
        "min_rel_z": min_rel_z,
        "mean_rel_z": mean_rel_z,
        "max_abs_roll": max_roll,
        "max_abs_pitch": max_pitch,
        "fallen_abs_z_based": fallen_abs,
        "fallen_relative_z_based": fallen_rel,
        "valid_locomotion_candidate": valid_locomotion_candidate,
        "stable_hold_candidate": stable_hold_candidate,
        "decision": decision,
        "npz_path": str(npz_path.relative_to(ROOT)),
        "summary_path": str(summary_path.relative_to(ROOT)),
        "mpc_first": json.dumps(mpc_first),
        "mpc_last": json.dumps(mpc_last),
    }

    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("[TRACER] appended sanity result:", out_csv)
    print(
        "[TRACER] row:",
        f"terrain={terrain_key}",
        f"mode={fused_mode}",
        f"semantic={semantic_mode}",
        f"style={style}",
        f"dx={dx:.3f}",
        f"dy={dy:.3f}",
        f"min_rel_z={min_rel_z:.3f}",
        f"fallen_rel={fallen_rel}",
        f"decision={decision}",
    )


if __name__ == "__main__":
    main()
