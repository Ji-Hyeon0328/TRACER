#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


FEATURES = [
    "mpc_vx",
    "mpc_yaw",
    "mpc_body_height",
    "mpc_clearance",
    "mpc_enable",
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "gate_level_code",
    "gate_action_code",
    "gate_would_override",
    "gate_control_risk",
    "gate_future_risk",
    "gate_fallen_prob",
    "gate_recovery_prob",
    "gate_sigma_mean",
    "gate_rho_norm",
    "gate_vx_scale",
    "gate_h_delta",
    "gate_clr_delta",
    "odom_vx",
]


def load_rows(path: Path):
    with path.open() as fobj:
        return list(csv.DictReader(fobj))


def load_summary_from_episode_csv(path: Path):
    # episodes/<id>.csv -> summaries/<id>.json
    parts = list(path.parts)
    if "episodes" not in parts:
        return {}
    i = parts.index("episodes")
    summary_path = Path(*parts[:i], "summaries", path.stem + ".json")
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text())
        except Exception:
            return {}
    return {}


def feature_vec(row):
    xs = []
    for k in FEATURES:
        v = f(row.get(k, 0.0), 0.0)

        # Basic robust clipping. This avoids huge values dominating RAM v1.
        if k.startswith("gate_"):
            v = clamp(v, -5.0, 5.0)
        elif k.startswith("mpc_"):
            v = clamp(v, -2.0, 2.0)
        elif k.startswith("beta_"):
            v = clamp(v, 0.0, 1.0)
        elif k.startswith("odom_"):
            v = clamp(v, -3.0, 3.0)

        xs.append(v)
    return xs


def build_samples(rows, summary, window, horizon, stride):
    samples = []

    n = len(rows)
    terrain = rows[0].get("terrain", "unknown") if rows else "unknown"
    policy_id = rows[0].get("policy_id", "unknown") if rows else "unknown"
    episode_id = rows[0].get("episode_id", "unknown") if rows else "unknown"

    success_proxy = bool(summary.get("success_proxy", False))
    episode_success = 1.0 if success_proxy else 0.0

    for end in range(window, max(window, n - horizon), stride):
        hist = rows[end - window:end]
        fut = rows[end:end + horizon]

        x = []
        for r in hist:
            x.extend(feature_vec(r))

        future_gate_level_max = max(f(r.get("gate_level_code", 0.0)) for r in fut) if fut else 0.0
        future_gate_action_max = max(f(r.get("gate_action_code", 0.0)) for r in fut) if fut else 0.0
        future_override_mean = sum(f(r.get("gate_would_override", 0.0)) for r in fut) / max(1, len(fut))
        future_vx_mean = sum(f(r.get("mpc_vx", 0.0)) for r in fut) / max(1, len(fut))
        future_enable_mean = sum(f(r.get("mpc_enable", 0.0)) for r in fut) / max(1, len(fut))

        # V1 labels: intervention/risk proxies, not raw RAM self-labels.
        y = {
            "future_gate_caution": 1.0 if future_gate_level_max >= 1.0 else 0.0,
            "future_gate_unstable": 1.0 if future_gate_level_max >= 2.0 else 0.0,
            "future_conservative_probe": 1.0 if future_gate_action_max >= 2.0 else 0.0,
            "future_override_mean": clamp(future_override_mean, 0.0, 1.0),
            "future_low_speed": 1.0 if future_vx_mean < 0.05 else 0.0,
            "future_disabled": 1.0 if future_enable_mean < 0.5 else 0.0,
            "episode_success": episode_success,
        }

        samples.append({
            "episode_id": episode_id,
            "terrain": terrain,
            "policy_id": policy_id,
            "t_end_index": end,
            "window": window,
            "horizon": horizon,
            "feature_names": FEATURES,
            "x": x,
            "y": y,
        })

    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-glob", default="data/rollout_dataset_v0/episodes/*.csv")
    ap.add_argument("--out", default="data/ram_window_dataset_v1/ram_windows_v1.jsonl")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-duration", type=float, default=20.0)
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.episodes_glob))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    used_files = 0
    by_terrain = {}

    with out.open("w") as fobj:
        for p in paths:
            rows = load_rows(p)
            if not rows:
                continue

            duration = f(rows[-1].get("elapsed", 0.0), 0.0)
            if duration < args.min_duration:
                continue

            summary = load_summary_from_episode_csv(p)
            samples = build_samples(rows, summary, args.window, args.horizon, args.stride)
            if not samples:
                continue

            used_files += 1
            terrain = samples[0]["terrain"]
            by_terrain[terrain] = by_terrain.get(terrain, 0) + len(samples)

            for s in samples:
                fobj.write(json.dumps(s) + "\n")
            total += len(samples)

    manifest = {
        "out": str(out),
        "episodes_glob": args.episodes_glob,
        "used_files": used_files,
        "n_samples": total,
        "window": args.window,
        "horizon": args.horizon,
        "stride": args.stride,
        "feature_dim_per_step": len(FEATURES),
        "input_dim": len(FEATURES) * args.window,
        "label_names": [
            "future_gate_caution",
            "future_gate_unstable",
            "future_conservative_probe",
            "future_override_mean",
            "future_low_speed",
            "future_disabled",
            "episode_success",
        ],
        "by_terrain": by_terrain,
    }

    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[TRACER] wrote RAM window dataset: {out}")
    print(f"[TRACER] manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
