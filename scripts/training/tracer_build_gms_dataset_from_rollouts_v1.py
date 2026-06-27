#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


GMS_LABELS = [
    "fast",
    "cautious_probe",
    "high_clearance_slow_probe",
    "conservative",
]


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def terrain_onehot(name: str):
    return [
        1.0 if name == "flat_normal" else 0.0,
        1.0 if name == "rough_mid" else 0.0,
        1.0 if name == "slope_5deg" else 0.0,
    ]


def infer_label(row):
    terrain = row.get("terrain", "unknown")
    vx = f(row.get("mpc_vx", 0.0))
    h = f(row.get("mpc_body_height", 0.30))
    clr = f(row.get("mpc_clearance", 0.03))
    gate_action = f(row.get("gate_action_code", 0.0))
    gate_level = f(row.get("gate_level_code", 0.0))

    if gate_action >= 2.0 or gate_level >= 2.0 or (vx <= 0.036 and h >= 0.35 and clr >= 0.11):
        return "conservative"

    if terrain == "flat_normal":
        return "fast"

    if terrain == "rough_mid":
        return "cautious_probe"

    if terrain == "slope_5deg":
        return "high_clearance_slow_probe"

    return "conservative"


def build_x(row):
    terrain = row.get("terrain", "unknown")

    return terrain_onehot(terrain) + [
        f(row.get("beta_motion", 0.0)),
        f(row.get("beta_stability", 0.0)),
        f(row.get("beta_energy", 0.0)),

        f(row.get("gate_level_code", 0.0)),
        f(row.get("gate_action_code", 0.0)),
        f(row.get("gate_would_override", 0.0)),
        f(row.get("gate_control_risk", 0.0)),
        f(row.get("gate_future_risk", 0.0)),
        f(row.get("gate_fallen_prob", 0.0)),
        f(row.get("gate_recovery_prob", 0.0)),
        f(row.get("gate_sigma_mean", 0.0)),
        f(row.get("gate_rho_norm", 0.0)),
        f(row.get("gate_vx_scale", 1.0)),
        f(row.get("gate_h_delta", 0.0)),
        f(row.get("gate_clr_delta", 0.0)),

        f(row.get("mpc_vx", 0.0)),
        f(row.get("mpc_body_height", 0.0)),
        f(row.get("mpc_clearance", 0.0)),
        f(row.get("mpc_enable", 0.0)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-glob", default="data/rollout_dataset_v0/episodes/*.csv")
    ap.add_argument("--out", default="data/gms_dataset_v1/gms_dataset_v1.jsonl")
    ap.add_argument("--min-duration", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.episodes_glob))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    used_files = 0
    by_label = {k: 0 for k in GMS_LABELS}
    by_terrain = {}

    with out.open("w") as fobj:
        for p in paths:
            with p.open() as ff:
                rows = list(csv.DictReader(ff))
            if not rows:
                continue

            duration = f(rows[-1].get("elapsed", 0.0))
            if duration < args.min_duration:
                continue

            used_files += 1

            for i, row in enumerate(rows):
                if i % args.stride != 0:
                    continue

                label = infer_label(row)
                if label not in GMS_LABELS:
                    continue

                terrain = row.get("terrain", "unknown")
                by_terrain[terrain] = by_terrain.get(terrain, 0) + 1
                by_label[label] = by_label.get(label, 0) + 1

                item = {
                    "episode_id": row.get("episode_id", p.stem),
                    "terrain": terrain,
                    "t_index": i,
                    "x": build_x(row),
                    "label": label,
                    "label_id": GMS_LABELS.index(label),
                    "label_names": GMS_LABELS,
                }
                fobj.write(json.dumps(item) + "\n")
                n += 1

    manifest = {
        "out": str(out),
        "used_files": used_files,
        "n_samples": n,
        "input_dim": 3 + 3 + 13 + 4,
        "label_names": GMS_LABELS,
        "by_label": by_label,
        "by_terrain": by_terrain,
    }

    mp = out.with_suffix(".manifest.json")
    mp.write_text(json.dumps(manifest, indent=2))

    print(f"[TRACER] wrote GMS dataset: {out}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
