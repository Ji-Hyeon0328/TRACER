#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def code(value: str, table: Dict[str, int], default: int = 0) -> float:
    if value is None:
        return float(default)
    return float(table.get(str(value), default))


GMS_CODE = {
    "fast": 0,
    "cautious_probe": 1,
    "high_clearance_slow_probe": 2,
    "conservative": 3,
    "disabled": 4,
    "unknown": 5,
}

RAM_LEVEL_CODE = {
    "unknown": 0,
    "stable": 1,
    "caution": 2,
    "unstable": 3,
}

GATE_ACTION_CODE = {
    "unknown": 0,
    "keep": 1,
    "would_conservative_probe": 2,
    "force_conservative_probe": 3,
    "disable": 4,
}


FEATURE_NAMES = [
    "final_vx",
    "final_yaw",
    "final_body_height",
    "final_clearance",
    "final_enable",
    "rule_gms_code",
    "learned_gms_code",
    "learned_gms_prob",
    "gms_disagree",
    "learned_beta_motion",
    "learned_beta_stability",
    "learned_beta_energy",
    "learned_ram_intervention_score",
    "learned_ram_future_override_mean",
    "rule_ram_level_code",
    "rule_gate_action_code",
    "rule_control_risk",
    "rule_fallen_prob_weak",
    "rule_sigma_mean",
]

LABEL_NAMES = [
    "future_intervention",
    "future_override_mean",
    "future_conservative",
    "future_low_speed",
    "future_gms_disagree",
    "episode_success",
]


def row_feature(r: dict) -> List[float]:
    final_vx = f(r.get("final_vx", r.get("rule_vx", 0.0)))
    final_yaw = f(r.get("final_yaw", r.get("rule_yaw", 0.0)))
    final_h = f(r.get("final_body_height", r.get("rule_body_height", 0.0)))
    final_clr = f(r.get("final_clearance", r.get("rule_clearance", 0.0)))
    final_enable = f(r.get("final_enable", r.get("rule_enable", 1.0)))

    return [
        final_vx,
        final_yaw,
        final_h,
        final_clr,
        final_enable,
        code(r.get("rule_gms_label", "unknown"), GMS_CODE, 5) / 5.0,
        code(r.get("learned_gms_label", "unknown"), GMS_CODE, 5) / 5.0,
        clip(f(r.get("learned_gms_prob", 0.0))),
        clip(f(r.get("gms_disagree", 0.0))),
        clip(f(r.get("learned_beta_motion", 0.0))),
        clip(f(r.get("learned_beta_stability", 0.0))),
        clip(f(r.get("learned_beta_energy", 0.0))),
        clip(f(r.get("learned_ram_intervention_score", 0.0))),
        clip(f(r.get("learned_ram_future_override_mean", 0.0))),
        code(r.get("rule_ram_level", "unknown"), RAM_LEVEL_CODE, 0) / 3.0,
        code(r.get("rule_ram_gate_action", "unknown"), GATE_ACTION_CODE, 0) / 4.0,
        clip(f(r.get("rule_control_risk", 0.0))),
        clip(f(r.get("rule_fallen_prob", 0.0))),
        clip(f(r.get("rule_sigma_mean", 0.0))),
    ]


def future_labels(rows: List[dict], start: int, horizon: int) -> List[float]:
    fut = rows[start:start + horizon]
    if not fut:
        fut = rows[start:start + 1]

    interventions = [clip(f(r.get("learned_ram_intervention_score", 0.0))) for r in fut]
    overrides = [clip(f(r.get("learned_ram_future_override_mean", 0.0))) for r in fut]
    conservative = [
        1.0 if r.get("rule_gms_label") == "conservative" or r.get("learned_gms_label") == "conservative" else 0.0
        for r in fut
    ]
    low_speed = [
        1.0 if f(r.get("final_vx", r.get("rule_vx", 0.0))) < 0.04 else 0.0
        for r in fut
    ]
    disagree = [clip(f(r.get("gms_disagree", 0.0))) for r in fut]
    errors = [1.0 if str(r.get("error", "")).strip() else 0.0 for r in fut]

    episode_success = 1.0 if max(errors) < 0.5 else 0.0

    return [
        1.0 if max(interventions) >= 0.5 else 0.0,
        sum(overrides) / max(1, len(overrides)),
        1.0 if max(conservative) >= 0.5 else 0.0,
        1.0 if max(low_speed) >= 0.5 else 0.0,
        1.0 if max(disagree) >= 0.5 else 0.0,
        episode_success,
    ]


def load_rows(path: Path) -> List[dict]:
    with path.open() as fobj:
        rows = list(csv.DictReader(fobj))
    rows = [r for r in rows if str(r.get("learned_ok", "1")).lower() not in {"false", "0"}]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/shadow_logs_v2_objective_v2/learned_stack_shadow_v2_*.csv")
    ap.add_argument("--out", default="data/ram_shadow_dataset_v2/ram_shadow_windows_v2.jsonl")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.glob))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    used_files = []

    for path in paths:
        rows = load_rows(path)
        if len(rows) < args.window + 1:
            continue

        terrain = rows[0].get("terrain", path.stem)
        used_files.append(str(path))

        feats = [row_feature(r) for r in rows]

        for end_idx in range(args.window, len(rows) - 1, args.stride):
            win = feats[end_idx - args.window:end_idx]
            y = future_labels(rows, end_idx, args.horizon)

            x_flat = []
            for ww in win:
                x_flat.extend(ww)

            samples.append({
                "source_csv": str(path),
                "terrain": terrain,
                "end_index": end_idx,
                "window": args.window,
                "horizon": args.horizon,
                "stride": args.stride,
                "x": x_flat,
                "y": y,
                "feature_names": FEATURE_NAMES,
                "label_names": LABEL_NAMES,
                "source": "ram_shadow_dataset_v2_from_supervised_stack_shadow_logs",
                "notes": "Warm-start RAM labels from shadow v2 teacher signals. Not true privileged physics labels yet.",
            })

    with out.open("w") as fobj:
        for s in samples:
            fobj.write(json.dumps(s) + "\n")

    by_terrain = {}
    for s in samples:
        by_terrain.setdefault(s["terrain"], 0)
        by_terrain[s["terrain"]] += 1

    manifest = {
        "version": "ram_shadow_dataset_v2",
        "n_samples": len(samples),
        "used_files": used_files,
        "window": args.window,
        "horizon": args.horizon,
        "stride": args.stride,
        "feature_dim_per_step": len(FEATURE_NAMES),
        "input_dim": args.window * len(FEATURE_NAMES),
        "output_dim": len(LABEL_NAMES),
        "feature_names": FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "by_terrain": by_terrain,
        "out": str(out),
    }

    manifest_path = out.parent / "manifest_v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("[TRACER] wrote RAM shadow dataset v2")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
