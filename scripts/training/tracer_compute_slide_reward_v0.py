#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
    except Exception:
        pass
    return default


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_exp_reward(x: float, scale: float) -> float:
    return float(math.exp(-scale * max(0.0, x)))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    priority = [
        "rank_by_slide_reward",
        "tag",
        "world",
        "terrain_key",
        "style",
        "primitive_family",
        "semantic_mode",
        "fall_like",
        "R_tracer_slide_v0",
        "R_weighted_raw",
        "R_motion",
        "R_stability",
        "R_energy_proxy",
        "R_aux",
        "beta_motion",
        "beta_stability",
        "beta_energy",
        "normalizer_N",
        "dx",
        "dy",
        "min_z",
        "max_abs_roll",
        "max_abs_pitch",
        "command_vx",
        "command_body_height",
        "command_clearance",
        "command_enable",
        "trajectory_cost_v0",
        "reward_mode",
        "note",
    ]

    fields = priority + sorted({k for r in rows for k in r.keys()} - set(priority))

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def infer_reward_mode(row: dict[str, Any]) -> str:
    family = str(row.get("primitive_family", "unknown"))
    vx = as_float(row.get("command_vx"), 0.0)

    if family in {"active_hold", "passive_or_disabled_hold"}:
        return "hold_or_recovery"
    if family == "negative_vx_backstep":
        return "backstep"
    if family == "forward_locomotion":
        return "locomotion"
    if abs(vx) <= 1e-4:
        return "hold_or_recovery"
    if vx < 0.0:
        return "backstep"
    return "locomotion"


def normalized_beta(row: dict[str, Any]) -> tuple[float, float, float]:
    # Prefer beta_* from metric/manifest if available.
    bv = as_float(row.get("beta_motion_mean"), 1.0 / 3.0)
    bs = as_float(row.get("beta_stability_mean"), 1.0 / 3.0)
    be = as_float(row.get("beta_energy_mean"), 1.0 / 3.0)

    s = bv + bs + be
    if s <= 1e-9:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

    return bv / s, bs / s, be / s


def compute_motion_reward(row: dict[str, Any], mode: str) -> tuple[float, dict[str, float]]:
    dx = as_float(row.get("dx"), 0.0)
    dy = as_float(row.get("dy"), 0.0)
    cmd_vx = as_float(row.get("command_vx"), as_float(row.get("mpc_vx_mean"), 0.0))
    observed_vx = as_float(row.get("observed_vx"), 0.0)

    drift_xy = math.sqrt(dx * dx + dy * dy)
    tracking_error = abs(observed_vx - cmd_vx)

    if mode == "hold_or_recovery":
        # For recovery/hold, the best motion is minimum drift.
        # Scale 3.0 means 0.33 m drift -> exp(-~1) ≈ 0.37.
        r = safe_exp_reward(drift_xy, scale=3.0)
    elif mode == "backstep":
        # For backstep, reward tracking negative commanded velocity,
        # but still penalize lateral drift.
        r_track = safe_exp_reward(tracking_error, scale=8.0)
        r_lateral = safe_exp_reward(abs(dy), scale=4.0)
        r = 0.7 * r_track + 0.3 * r_lateral
    else:
        # For normal locomotion, reward velocity tracking and discourage lateral drift.
        r_track = safe_exp_reward(tracking_error, scale=6.0)
        r_lateral = safe_exp_reward(abs(dy), scale=3.0)
        r = 0.75 * r_track + 0.25 * r_lateral

    return clamp01(r), {
        "motion_drift_xy": drift_xy,
        "motion_tracking_error": tracking_error,
    }


def compute_stability_reward(row: dict[str, Any]) -> tuple[float, dict[str, float]]:
    min_z = as_float(row.get("min_z"), 0.0)
    max_abs_roll = as_float(row.get("max_abs_roll"), 999.0)
    max_abs_pitch = as_float(row.get("max_abs_pitch"), 999.0)
    fall_like = as_bool(row.get("fall_like"))

    # These thresholds follow the current analyzer convention.
    z_fail = 0.18
    z_good = 0.28
    rpy_limit = 0.70

    z_score = clamp01((min_z - z_fail) / max(z_good - z_fail, 1e-9))
    roll_score = clamp01(1.0 - max_abs_roll / rpy_limit)
    pitch_score = clamp01(1.0 - max_abs_pitch / rpy_limit)

    r = 0.45 * z_score + 0.25 * roll_score + 0.30 * pitch_score

    if fall_like:
        r *= 0.10

    return clamp01(r), {
        "stability_z_score": z_score,
        "stability_roll_score": roll_score,
        "stability_pitch_score": pitch_score,
    }


def compute_energy_proxy(row: dict[str, Any]) -> tuple[float, dict[str, float]]:
    # Placeholder until Isaac Lab / low-level controller logs torque and joint power.
    cmd_vx = abs(as_float(row.get("command_vx"), 0.0))
    cmd_yaw = abs(as_float(row.get("command_yaw"), 0.0))
    body_h = as_float(row.get("command_body_height"), 0.305)
    clearance = as_float(row.get("command_clearance"), 0.055)
    enable = as_float(row.get("command_enable"), 1.0)

    # Penalize more aggressive commands, high body posture, and large clearance.
    # Disabled/passive hold receives a mild penalty because it is not a valid active controller.
    command_effort = (
        2.0 * cmd_vx
        + 0.5 * cmd_yaw
        + 3.0 * max(0.0, body_h - 0.305)
        + 4.0 * max(0.0, clearance - 0.055)
        + 0.15 * (1.0 if enable < 0.5 else 0.0)
    )

    r = safe_exp_reward(command_effort, scale=2.0)

    return clamp01(r), {
        "energy_proxy_effort": command_effort,
    }


def compute_aux_penalty(row: dict[str, Any]) -> tuple[float, dict[str, float]]:
    dx = as_float(row.get("dx"), 0.0)
    dy = as_float(row.get("dy"), 0.0)
    start_end_z_drop = as_float(row.get("z_drop"), 0.0)
    min_z = as_float(row.get("min_z"), 0.0)
    max_abs_roll = as_float(row.get("max_abs_roll"), 0.0)
    max_abs_pitch = as_float(row.get("max_abs_pitch"), 0.0)
    fall_like = as_bool(row.get("fall_like"))

    drift_xy = math.sqrt(dx * dx + dy * dy)

    fall_penalty = 1.0 if fall_like else 0.0
    low_z_penalty = clamp01((0.18 - min_z) / 0.18)
    z_drop_penalty = clamp01(max(0.0, start_end_z_drop) / 0.25)
    rpy_penalty = clamp01(max(max_abs_roll, max_abs_pitch) / 0.70)
    drift_penalty = clamp01(drift_xy / 1.0)

    r_aux = (
        1.00 * fall_penalty
        + 0.60 * low_z_penalty
        + 0.40 * z_drop_penalty
        + 0.30 * rpy_penalty
        + 0.20 * drift_penalty
    )

    return max(0.0, r_aux), {
        "aux_fall_penalty": fall_penalty,
        "aux_low_z_penalty": low_z_penalty,
        "aux_z_drop_penalty": z_drop_penalty,
        "aux_rpy_penalty": rpy_penalty,
        "aux_drift_penalty": drift_penalty,
    }


def compute_slide_reward(
    row: dict[str, Any],
    *,
    lambda_energy: float,
    aux_scale: float,
) -> dict[str, Any]:
    mode = infer_reward_mode(row)
    beta_motion, beta_stability, beta_energy = normalized_beta(row)

    r_motion, motion_parts = compute_motion_reward(row, mode)
    r_stability, stability_parts = compute_stability_reward(row)
    r_energy, energy_parts = compute_energy_proxy(row)
    r_aux, aux_parts = compute_aux_penalty(row)

    normalizer = beta_motion + beta_stability + beta_energy * lambda_energy
    if normalizer <= 1e-9:
        normalizer = 1.0

    r_weighted = (
        beta_motion * r_motion
        + beta_stability * r_stability
        + beta_energy * lambda_energy * r_energy
    ) / normalizer

    r_tracer = r_weighted * math.exp(-aux_scale * r_aux)

    out = dict(row)
    out.update({
        "reward_mode": mode,
        "beta_motion": beta_motion,
        "beta_stability": beta_stability,
        "beta_energy": beta_energy,
        "lambda_energy": lambda_energy,
        "aux_scale": aux_scale,
        "normalizer_N": normalizer,
        "R_motion": r_motion,
        "R_stability": r_stability,
        "R_energy_proxy": r_energy,
        "R_aux": r_aux,
        "R_weighted_raw": r_weighted,
        "R_tracer_slide_v0": r_tracer,
    })
    out.update(motion_parts)
    out.update(stability_parts)
    out.update(energy_parts)
    out.update(aux_parts)
    return out


def is_unknown_value(x: Any) -> bool:
    return str(x).strip().lower() in {"", "unknown", "none", "nan"}


def keep_row(
    row: dict[str, Any],
    *,
    require_known_world: bool,
    require_known_style: bool,
    drop_disabled: bool,
) -> bool:
    if require_known_world and is_unknown_value(row.get("world")):
        return False
    if require_known_style and is_unknown_value(row.get("style")):
        return False
    if drop_disabled and as_float(row.get("command_enable"), 1.0) < 0.5:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest-csv",
        default="data/rollout_manifests/tracer_rollout_manifest_v0.csv",
    )
    ap.add_argument("--out-dir", default="data/rollout_rewards")
    ap.add_argument("--lambda-energy", type=float, default=0.5)
    ap.add_argument("--aux-scale", type=float, default=1.5)
    ap.add_argument("--require-known-world", action="store_true")
    ap.add_argument("--require-known-style", action="store_true")
    ap.add_argument("--drop-disabled", action="store_true")
    args = ap.parse_args()

    rows_all = read_csv(Path(args.manifest_csv))
    rows = [
        r for r in rows_all
        if keep_row(
            r,
            require_known_world=args.require_known_world,
            require_known_style=args.require_known_style,
            drop_disabled=args.drop_disabled,
        )
    ]

    rewards = [
        compute_slide_reward(
            r,
            lambda_energy=args.lambda_energy,
            aux_scale=args.aux_scale,
        )
        for r in rows
    ]

    rewards_sorted = sorted(
        rewards,
        key=lambda r: as_float(r.get("R_tracer_slide_v0"), -1.0),
        reverse=True,
    )

    for i, r in enumerate(rewards_sorted, start=1):
        r["rank_by_slide_reward"] = i

    filtered = any([
        args.require_known_world,
        args.require_known_style,
        args.drop_disabled,
    ])
    suffix = "filtered" if filtered else "all"

    out_dir = Path(args.out_dir)
    out_csv = out_dir / f"tracer_slide_reward_v0_{suffix}.csv"
    out_jsonl = out_dir / f"tracer_slide_reward_v0_{suffix}.jsonl"

    write_csv(out_csv, rewards_sorted)
    write_jsonl(out_jsonl, rewards_sorted)

    print(f"[TRACER] input rows:    {len(rows_all)}")
    print(f"[TRACER] output rows:   {len(rows)}")
    print(f"[TRACER] wrote csv:     {out_csv}")
    print(f"[TRACER] wrote jsonl:   {out_jsonl}")

    print("\n[TRACER] slide reward ranking:")
    for r in rewards_sorted[:20]:
        print(
            f"{int(r['rank_by_slide_reward']):02d} "
            f"R={as_float(r['R_tracer_slide_v0']):.4f} "
            f"raw={as_float(r['R_weighted_raw']):.4f} "
            f"motion={as_float(r['R_motion']):.3f} "
            f"stab={as_float(r['R_stability']):.3f} "
            f"energy={as_float(r['R_energy_proxy']):.3f} "
            f"aux={as_float(r['R_aux']):.3f} "
            f"fall={r.get('fall_like')} "
            f"mode={r.get('reward_mode')} "
            f"style={r.get('style')}"
        )


if __name__ == "__main__":
    main()
