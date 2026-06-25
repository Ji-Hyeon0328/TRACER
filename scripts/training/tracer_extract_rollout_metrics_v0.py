#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def finite_row_mask(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.isfinite(x)
    return np.all(np.isfinite(x), axis=1)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def first_last_valid_rows(x: np.ndarray, mask: np.ndarray):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None, None, None, None
    return int(idx[0]), int(idx[-1]), x[idx[0]], x[idx[-1]]


def mean_valid(x: np.ndarray, mask: np.ndarray, default: float = float("nan")) -> float:
    if x.size == 0 or mask.size == 0 or not np.any(mask):
        return default
    return float(np.mean(x[mask]))


def max_abs_valid(x: np.ndarray, mask: np.ndarray, default: float = float("nan")) -> float:
    if x.size == 0 or mask.size == 0 or not np.any(mask):
        return default
    return float(np.max(np.abs(x[mask])))


def min_valid(x: np.ndarray, mask: np.ndarray, default: float = float("nan")) -> float:
    if x.size == 0 or mask.size == 0 or not np.any(mask):
        return default
    return float(np.min(x[mask]))


def infer_tag(npz_path: Path) -> str:
    # Mission logger currently stores timestamp in filename.
    # A richer tag can later be injected from summary metadata if available.
    return npz_path.stem.replace("tracer_mission_log_", "")


def load_summary(npz_path: Path) -> dict[str, Any]:
    summary_path = npz_path.with_name(npz_path.name.replace("tracer_mission_log_", "tracer_mission_summary_")).with_suffix(".json")
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text())
        except Exception:
            return {}
    return {}


def extract_one(
    npz_path: Path,
    *,
    min_z_fall_threshold: float,
    max_roll_fall_threshold: float,
    max_pitch_fall_threshold: float,
) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    summary = load_summary(npz_path)

    wall_time = np.asarray(data["wall_time"], dtype=float) if "wall_time" in data else np.array([], dtype=float)
    wall_mask = np.isfinite(wall_time)
    duration = safe_float(summary.get("duration_sec"), float("nan"))
    if not math.isfinite(duration) and np.any(wall_mask):
        wt = wall_time[wall_mask]
        duration = float(wt[-1] - wt[0]) if wt.size >= 2 else 0.0

    proprio = np.asarray(data["proprio"], dtype=float) if "proprio" in data else np.empty((0, 0), dtype=float)
    mpc = np.asarray(data["mpc_reference"], dtype=float) if "mpc_reference" in data else np.empty((0, 0), dtype=float)
    objective = np.asarray(data["objective"], dtype=float) if "objective" in data else np.empty((0, 0), dtype=float)

    proprio_mask = finite_row_mask(proprio) if proprio.size else np.zeros((0,), dtype=bool)
    mpc_mask = finite_row_mask(mpc) if mpc.size else np.zeros((0,), dtype=bool)
    objective_mask = finite_row_mask(objective) if objective.size else np.zeros((0,), dtype=bool)

    result: dict[str, Any] = {
        "npz_path": str(npz_path),
        "summary_path": str(npz_path.with_name(npz_path.name.replace("tracer_mission_log_", "tracer_mission_summary_")).with_suffix(".json")),
        "tag": infer_tag(npz_path),
        "num_samples": int(summary.get("num_samples", len(wall_time))),
        "duration_sec": duration,
        "proprio_rows": int(proprio.shape[0]) if proprio.ndim >= 1 else 0,
        "proprio_finite_rows": int(np.sum(proprio_mask)),
        "mpc_rows": int(mpc.shape[0]) if mpc.ndim >= 1 else 0,
        "mpc_finite_rows": int(np.sum(mpc_mask)),
    }

    # MPC reference convention:
    # [counter, vx, yaw_rate, body_height, swing_clearance, enable]
    if mpc.ndim == 2 and mpc.shape[1] >= 6 and np.any(mpc_mask):
        result.update({
            "mpc_vx_mean": mean_valid(mpc[:, 1], mpc_mask),
            "mpc_vx_max": float(np.max(mpc[mpc_mask, 1])),
            "mpc_yaw_mean": mean_valid(mpc[:, 2], mpc_mask),
            "mpc_body_height_mean": mean_valid(mpc[:, 3], mpc_mask),
            "mpc_clearance_mean": mean_valid(mpc[:, 4], mpc_mask),
            "mpc_enable_mean": mean_valid(mpc[:, 5], mpc_mask),
        })
    else:
        result.update({
            "mpc_vx_mean": float("nan"),
            "mpc_vx_max": float("nan"),
            "mpc_yaw_mean": float("nan"),
            "mpc_body_height_mean": float("nan"),
            "mpc_clearance_mean": float("nan"),
            "mpc_enable_mean": float("nan"),
        })

    # Existing analyzer convention:
    # proprio[:, 0:3] = xyz
    # proprio[:, 3:6] = rpy
    if proprio.ndim == 2 and proprio.shape[1] >= 6 and np.any(proprio_mask):
        i0, i1, first, last = first_last_valid_rows(proprio, proprio_mask)
        assert first is not None and last is not None

        xyz0 = first[:3]
        xyz1 = last[:3]
        rpy = proprio[:, 3:6]

        dx, dy, dz = (xyz1 - xyz0).tolist()
        min_z = min_valid(proprio[:, 2], proprio_mask)
        max_z = float(np.max(proprio[proprio_mask, 2]))
        max_abs_roll = max_abs_valid(rpy[:, 0], proprio_mask)
        max_abs_pitch = max_abs_valid(rpy[:, 1], proprio_mask)

        obs_vx = dx / duration if duration and math.isfinite(duration) and duration > 1e-9 else float("nan")
        obs_vy = dy / duration if duration and math.isfinite(duration) and duration > 1e-9 else float("nan")

        fall_like = bool(
            (math.isfinite(min_z) and min_z < min_z_fall_threshold)
            or (math.isfinite(max_abs_roll) and max_abs_roll > max_roll_fall_threshold)
            or (math.isfinite(max_abs_pitch) and max_abs_pitch > max_pitch_fall_threshold)
        )

        result.update({
            "start_x": float(xyz0[0]),
            "start_y": float(xyz0[1]),
            "start_z": float(xyz0[2]),
            "end_x": float(xyz1[0]),
            "end_y": float(xyz1[1]),
            "end_z": float(xyz1[2]),
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
            "min_z": min_z,
            "max_z": max_z,
            "max_abs_roll": max_abs_roll,
            "max_abs_pitch": max_abs_pitch,
            "observed_vx": float(obs_vx),
            "observed_vy": float(obs_vy),
            "fall_like": fall_like,
        })
    else:
        result.update({
            "start_x": float("nan"),
            "start_y": float("nan"),
            "start_z": float("nan"),
            "end_x": float("nan"),
            "end_y": float("nan"),
            "end_z": float("nan"),
            "dx": float("nan"),
            "dy": float("nan"),
            "dz": float("nan"),
            "min_z": float("nan"),
            "max_z": float("nan"),
            "max_abs_roll": float("nan"),
            "max_abs_pitch": float("nan"),
            "observed_vx": float("nan"),
            "observed_vy": float("nan"),
            "fall_like": True,
        })

    if objective.ndim == 2 and objective.shape[1] >= 3 and np.any(objective_mask):
        result.update({
            "beta_motion_mean": mean_valid(objective[:, 0], objective_mask),
            "beta_stability_mean": mean_valid(objective[:, 1], objective_mask),
            "beta_energy_mean": mean_valid(objective[:, 2], objective_mask),
        })
    else:
        result.update({
            "beta_motion_mean": float("nan"),
            "beta_stability_mean": float("nan"),
            "beta_energy_mean": float("nan"),
        })

    # Preference-learning friendly derived costs.
    result["abs_dx"] = abs(safe_float(result.get("dx")))
    result["abs_dy"] = abs(safe_float(result.get("dy")))
    result["z_drop"] = safe_float(result.get("start_z")) - safe_float(result.get("end_z"))
    result["orientation_peak"] = max(
        safe_float(result.get("max_abs_roll"), 0.0),
        safe_float(result.get("max_abs_pitch"), 0.0),
    )
    result["tracking_vx_error"] = abs(
        safe_float(result.get("observed_vx"), 0.0)
        - safe_float(result.get("mpc_vx_mean"), 0.0)
    )

    # Simple scalar score for quick ranking.
    # Lower cost is better.
    result["trajectory_cost_v0"] = (
        100.0 * float(result["fall_like"])
        + 5.0 * max(0.0, min_z_fall_threshold - safe_float(result.get("min_z"), 0.0))
        + 1.0 * safe_float(result.get("abs_dx"), 0.0)
        + 2.0 * safe_float(result.get("abs_dy"), 0.0)
        + 2.0 * max(0.0, safe_float(result.get("z_drop"), 0.0))
        + 0.5 * safe_float(result.get("orientation_peak"), 0.0)
    )

    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        help="Mission log .npz files or directories. If omitted, use data/mission_logs.",
    )
    ap.add_argument("--out-dir", default="data/rollout_metrics")
    ap.add_argument("--limit", type=int, default=0, help="Use only newest N logs after sorting by mtime.")
    ap.add_argument("--min-z-fall-threshold", type=float, default=0.18)
    ap.add_argument("--max-roll-fall-threshold", type=float, default=0.80)
    ap.add_argument("--max-pitch-fall-threshold", type=float, default=0.80)
    args = ap.parse_args()

    if args.paths:
        candidates: list[Path] = []
        for raw in args.paths:
            p = Path(raw)
            if p.is_dir():
                candidates.extend(sorted(p.glob("tracer_mission_log_*.npz")))
            else:
                candidates.append(p)
    else:
        candidates = sorted(Path("data/mission_logs").glob("tracer_mission_log_*.npz"))

    candidates = [p for p in candidates if p.exists() and p.suffix == ".npz"]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if args.limit > 0:
        candidates = candidates[: args.limit]
    candidates = list(reversed(candidates))

    rows = [
        extract_one(
            p,
            min_z_fall_threshold=args.min_z_fall_threshold,
            max_roll_fall_threshold=args.max_roll_fall_threshold,
            max_pitch_fall_threshold=args.max_pitch_fall_threshold,
        )
        for p in candidates
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "tracer_rollout_metrics_v0.jsonl"
    csv_path = out_dir / "tracer_rollout_metrics_v0.csv"

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)

    print(f"[TRACER] processed logs: {len(rows)}")
    print(f"[TRACER] wrote jsonl: {jsonl_path}")
    print(f"[TRACER] wrote csv:   {csv_path}")

    if rows:
        print("\n[TRACER] newest rows:")
        for row in rows[-min(10, len(rows)):]:
            print(
                f"{Path(row['npz_path']).name} "
                f"cost={row['trajectory_cost_v0']:.3f} "
                f"fall={row['fall_like']} "
                f"dx={row['dx']:.3f} dy={row['dy']:.3f} "
                f"min_z={row['min_z']:.3f} "
                f"vx_cmd={row['mpc_vx_mean']:.4f} "
                f"h={row['mpc_body_height_mean']:.3f} "
                f"clr={row['mpc_clearance_mean']:.3f}"
            )


if __name__ == "__main__":
    main()
