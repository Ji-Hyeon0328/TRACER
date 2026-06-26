#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

SEED_V1 = ROOT / "data/preference_datasets/tracer_objective_selector_seed_v1.csv"

OUT_DIR = ROOT / "data/preference_datasets"
OUT_CSV = OUT_DIR / "tracer_objective_selector_runtime_v0.csv"
OUT_JSONL = OUT_DIR / "tracer_objective_selector_runtime_v0.jsonl"
OUT_SUMMARY = OUT_DIR / "tracer_objective_selector_runtime_v0_summary.json"


def f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def b(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def terrain_family(terrain_key: str, world: str, style: str) -> str:
    text = f"{terrain_key} {world} {style}".lower()

    if "slippery" in text or "slip" in text or "micro_brake" in text:
        return "slippery"
    if "sponge" in text or "soft" in text:
        return "sponge"
    if "downhill" in text or "downslope" in text:
        return "downslope"
    if "rough" in text:
        return "rough"
    if "slope" in text:
        return "slope"
    if "flat" in text or "earth" in text:
        return "flat"
    if "backstep" in text:
        return "backstep"
    return "unknown"


def deploy_label(semantic_target: str) -> str:
    if semantic_target == "no_valid_forward_recovery_needed":
        return "do_not_deploy_forward"
    return "deploy_forward"


def confidence_weight(row: dict[str, Any]) -> float:
    role = str(row.get("training_role", ""))
    semantic = str(row.get("semantic_target", ""))

    if role == "positive":
        return 1.0
    if role == "calibration_warning":
        return 0.7
    if semantic == "no_valid_forward_recovery_needed":
        return 1.0
    return 0.5


def main() -> None:
    with SEED_V1.open(newline="") as fp:
        seed_rows = list(csv.DictReader(fp))

    rows: list[dict[str, Any]] = []

    for i, r in enumerate(seed_rows):
        terrain = str(r.get("terrain_key", "unknown"))
        world = str(r.get("world", "unknown"))
        semantic_prior = str(r.get("semantic_input", "unknown"))
        style_prior = str(r.get("style_input", "unknown"))
        semantic_target = str(r.get("semantic_target", "unknown"))

        fam = terrain_family(terrain, world, style_prior)

        out = {
            "runtime_sample_id": f"tracer_runtime_v0_{i:04d}",
            "source_sample_id": r.get("sample_id", ""),
            "source_matrix": r.get("source_matrix", ""),
            "source_path": r.get("source_path", ""),
            "case_id": r.get("case_id", ""),
            "terrain_key": terrain,
            "world": world,
            "terrain_family": fam,
            "semantic_prior": semantic_prior,
            "style_prior": style_prior,

            # Runtime-available state/proxy features.
            # These are aggregate proxies in the seed dataset; in runtime they
            # correspond to instantaneous or short-window RAM/gate readings.
            "ramgate_enabled": b(r.get("ramgate_enabled", False)),
            "ramgate_calibrated": b(r.get("ramgate_calibrated", False)),
            "ramgate_observed": b(r.get("ramgate_observed", False)),
            "ram_monitor_observed": b(r.get("ram_monitor_observed", False)),
            "ram_ctrl_risk": f(r.get("ram_ctrl_risk_mean", 0.0), 0.0),
            "ram_fallen_prob": f(r.get("ram_fallen_mean", 0.0), 0.0),
            "ram_recovery_prob": f(r.get("ram_recovery_mean", 0.0), 0.0),
            "gate_stable_prob": f(r.get("gate_stable_frac", 0.0), 0.0),
            "gate_caution_prob": f(r.get("gate_caution_frac", 0.0), 0.0),
            "gate_unstable_prob": f(r.get("gate_unstable_frac", 0.0), 0.0),
            "gate_override_prob": f(r.get("gate_override_frac", 0.0), 0.0),
            "gate_first_action": r.get("gate_first_action", ""),
            "gate_last_action": r.get("gate_last_action", ""),
            "calibration_first": r.get("calibration_first", ""),
            "calibration_last": r.get("calibration_last", ""),

            # Labels.
            "training_role": r.get("training_role", ""),
            "semantic_target": semantic_target,
            "deploy_label": deploy_label(semantic_target),
            "beta_v_target": f(r.get("beta_v_target", 0.0), 0.0),
            "beta_s_target": f(r.get("beta_s_target", 0.0), 0.0),
            "beta_e_target": f(r.get("beta_e_target", 0.0), 0.0),
            "sample_weight": confidence_weight(r),
            "note": r.get("note", ""),
        }
        rows.append(out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with OUT_JSONL.open("w") as fjsonl:
        for r in rows:
            fjsonl.write(json.dumps(r) + "\n")

    summary = {
        "num_samples": len(rows),
        "source": str(SEED_V1.relative_to(ROOT)),
        "outputs": {
            "csv": str(OUT_CSV.relative_to(ROOT)),
            "jsonl": str(OUT_JSONL.relative_to(ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(ROOT)),
        },
        "terrain_family_counts": dict(Counter(r["terrain_family"] for r in rows)),
        "semantic_prior_counts": dict(Counter(r["semantic_prior"] for r in rows)),
        "semantic_target_counts": dict(Counter(r["semantic_target"] for r in rows)),
        "deploy_label_counts": dict(Counter(r["deploy_label"] for r in rows)),
        "role_counts": dict(Counter(r["training_role"] for r in rows)),
        "removed_post_rollout_features": [
            "dx",
            "dy",
            "min_rel_z",
            "locomotion_valid",
            "fallen_relative",
            "strict_height_margin_valid",
            "mpc_vx_mean",
            "mpc_body_height_mean",
            "mpc_clearance_mean",
        ],
        "note": (
            "Runtime v0 removes post-rollout metrics from the seed dataset. "
            "RAM/gate fields are currently aggregate proxies and should map to "
            "instantaneous or short-window readings in runtime."
        ),
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    print("[TRACER] wrote", OUT_CSV)
    print("[TRACER] wrote", OUT_JSONL)
    print("[TRACER] wrote", OUT_SUMMARY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
