#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

MATRICES = [
    ("gms_only", ROOT / "data/rollout_metrics/feasible_closed_loop_gms_on_v0.json"),
    ("ramgate_raw", ROOT / "data/rollout_metrics/feasible_closed_loop_gms_ramgate_v0.json"),
    ("ramgate_calibrated", ROOT / "data/rollout_metrics/feasible_closed_loop_gms_ramgate_calibrated_v0.json"),
]

OUT_DIR = ROOT / "data/preference_datasets"
OUT_CSV = OUT_DIR / "tracer_objective_selector_seed_v0.csv"
OUT_JSONL = OUT_DIR / "tracer_objective_selector_seed_v0.jsonl"
OUT_PREF_JSONL = OUT_DIR / "tracer_objective_selector_pairwise_preferences_v0.jsonl"
OUT_SUMMARY = OUT_DIR / "tracer_objective_selector_seed_v0_summary.json"


def f(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def b(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def beta_target(row: dict[str, Any]) -> list[float]:
    """Heuristic seed target for Objective Selector β=[β_v, β_s, β_e].

    This is not the final learned selector. It creates a consistent bootstrap
    target from the current validated V0 runs.
    """
    semantic = str(row.get("semantic", ""))
    style = str(row.get("style", ""))
    locomotion_valid = b(row.get("locomotion_valid", False))
    strict_height = b(row.get("strict_height_margin_valid", row.get("locomotion_valid", False)))
    gate_unstable = f(row.get("gate_unstable_frac", 0.0), 0.0) or 0.0
    gate_override = f(row.get("gate_override_frac", 0.0), 0.0) or 0.0

    if not locomotion_valid or not strict_height:
        # Risk/calibration warning: emphasize safety.
        beta = [0.20, 0.65, 0.15]
    elif semantic == "validated_locomotion" and style == "fast":
        beta = [0.55, 0.25, 0.20]
    elif semantic == "cautious_probe":
        beta = [0.30, 0.50, 0.20]
    elif semantic == "high_clearance_slow_probe":
        beta = [0.25, 0.55, 0.20]
    else:
        beta = [0.34, 0.43, 0.23]

    # If the gate still sees unstable/override tendency, tilt toward safety.
    risk_tilt = max(0.0, min(0.3, 0.2 * gate_unstable + 0.1 * gate_override))
    beta[0] = max(0.05, beta[0] - risk_tilt)
    beta[1] = min(0.85, beta[1] + risk_tilt)

    s = sum(beta)
    return [x / s for x in beta]


def training_role(row: dict[str, Any]) -> str:
    locomotion_valid = b(row.get("locomotion_valid", False))
    strict_height = b(row.get("strict_height_margin_valid", row.get("locomotion_valid", False)))
    note = str(row.get("ramgate_calibration_note", row.get("calibration_note", "")))
    gate_unstable = f(row.get("gate_unstable_frac", 0.0), 0.0) or 0.0
    gate_override = f(row.get("gate_override_frac", 0.0), 0.0) or 0.0

    if not locomotion_valid or not strict_height:
        return "invalid_or_risky"
    if "false positive" in note.lower() or gate_unstable > 0.25 or gate_override > 0.25:
        return "calibration_warning"
    return "positive"


def semantic_target(row: dict[str, Any]) -> str:
    if training_role(row) == "invalid_or_risky":
        return "no_valid_forward_recovery_needed"
    return str(row.get("semantic", "unknown"))


def score(row: dict[str, Any]) -> float:
    dx = f(row.get("dx", 0.0), 0.0)
    min_rel_z = f(row.get("min_rel_z", 0.0), 0.0)
    gate_override = f(row.get("gate_override_frac", 0.0), 0.0) or 0.0
    gate_unstable = f(row.get("gate_unstable_frac", 0.0), 0.0) or 0.0

    s = 0.0
    s += 3.0 if b(row.get("locomotion_valid", False)) else -2.0
    s += 1.0 if b(row.get("no_fall_route_closed", row.get("locomotion_valid", False))) else -1.0
    s += 1.0 if b(row.get("strict_height_margin_valid", row.get("locomotion_valid", False))) else -1.0
    s += min(max(dx, 0.0), 1.0)
    s += min(max(min_rel_z - 0.18, 0.0), 0.2)
    s -= 0.75 * gate_override
    s -= 0.75 * gate_unstable
    return s


def read_matrix(name: str, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[TRACER][WARN] missing matrix: {path}")
        return []
    data = json.loads(path.read_text())
    out = []
    for row in data:
        r = dict(row)
        r["source_matrix"] = name
        r["source_path"] = str(path.relative_to(ROOT))
        out.append(r)
    return out


def main() -> None:
    rows: list[dict[str, Any]] = []
    for name, path in MATRICES:
        rows.extend(read_matrix(name, path))

    enriched: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        beta = beta_target(row)
        out = {
            "sample_id": f"tracer_v0_{i:04d}",
            "source_matrix": row.get("source_matrix", ""),
            "source_path": row.get("source_path", ""),
            "case_id": row.get("case_id", ""),
            "terrain_key": row.get("terrain_key", ""),
            "world": row.get("world", ""),
            "semantic_input": row.get("semantic", ""),
            "style_input": row.get("style", ""),
            "decision": row.get("decision", ""),
            "ramgate_enabled": b(row.get("ramgate_enabled", False)),
            "ramgate_calibrated": b(row.get("ramgate_calibrated", False)),
            "ramgate_observed": b(row.get("ramgate_observed", False)),
            "ram_monitor_observed": b(row.get("ram_monitor_observed", False)),
            "locomotion_valid": b(row.get("locomotion_valid", False)),
            "no_fall_route_closed": b(row.get("no_fall_route_closed", row.get("locomotion_valid", False))),
            "strict_height_margin_valid": b(row.get("strict_height_margin_valid", row.get("locomotion_valid", False))),
            "fallen_relative": b(row.get("fallen_relative", False)),
            "dx": f(row.get("dx", 0.0), 0.0),
            "dy": f(row.get("dy", 0.0), 0.0),
            "min_rel_z": f(row.get("min_rel_z", 0.0), 0.0),
            "mpc_vx_mean": f(row.get("mpc_vx_mean", 0.0), 0.0),
            "mpc_body_height_mean": f(row.get("mpc_body_height_mean", 0.0), 0.0),
            "mpc_clearance_mean": f(row.get("mpc_clearance_mean", 0.0), 0.0),
            "ram_ctrl_risk_mean": f(row.get("ram_ctrl_risk_mean", 0.0), 0.0),
            "ram_fallen_mean": f(row.get("ram_fallen_mean", 0.0), 0.0),
            "ram_recovery_mean": f(row.get("ram_recovery_mean", 0.0), 0.0),
            "gate_stable_frac": f(row.get("gate_stable_frac", 0.0), 0.0),
            "gate_caution_frac": f(row.get("gate_caution_frac", 0.0), 0.0),
            "gate_unstable_frac": f(row.get("gate_unstable_frac", 0.0), 0.0),
            "gate_override_frac": f(row.get("gate_override_frac", 0.0), 0.0),
            "gate_first_action": row.get("gate_first_action", ""),
            "gate_last_action": row.get("gate_last_action", ""),
            "calibration_first": row.get("calibration_first", ""),
            "calibration_last": row.get("calibration_last", ""),
            "training_role": training_role(row),
            "semantic_target": semantic_target(row),
            "beta_v_target": beta[0],
            "beta_s_target": beta[1],
            "beta_e_target": beta[2],
            "preference_score": score(row),
            "note": row.get("ramgate_calibration_note", row.get("calibration_note", "")),
        }
        enriched.append(out)

    # Pairwise preferences within the same terrain.
    prefs = []
    by_terrain: dict[str, list[dict[str, Any]]] = {}
    for r in enriched:
        by_terrain.setdefault(str(r["terrain_key"]), []).append(r)

    for terrain, items in by_terrain.items():
        for a in items:
            for c in items:
                if a["sample_id"] == c["sample_id"]:
                    continue
                if a["preference_score"] > c["preference_score"] + 0.5:
                    prefs.append(
                        {
                            "terrain_key": terrain,
                            "preferred_sample_id": a["sample_id"],
                            "rejected_sample_id": c["sample_id"],
                            "preferred_case_id": a["case_id"],
                            "rejected_case_id": c["case_id"],
                            "preferred_source": a["source_matrix"],
                            "rejected_source": c["source_matrix"],
                            "reason": (
                                "higher locomotion validity / height margin / lower RAM gate override"
                            ),
                            "score_preferred": a["preference_score"],
                            "score_rejected": c["preference_score"],
                        }
                    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(enriched[0].keys()))
        writer.writeheader()
        writer.writerows(enriched)

    with OUT_JSONL.open("w") as fjsonl:
        for r in enriched:
            fjsonl.write(json.dumps(r) + "\n")

    with OUT_PREF_JSONL.open("w") as fpref:
        for p in prefs:
            fpref.write(json.dumps(p) + "\n")

    summary = {
        "num_samples": len(enriched),
        "num_pairwise_preferences": len(prefs),
        "sources": {name: str(path.relative_to(ROOT)) for name, path in MATRICES},
        "outputs": {
            "csv": str(OUT_CSV.relative_to(ROOT)),
            "jsonl": str(OUT_JSONL.relative_to(ROOT)),
            "pairwise_jsonl": str(OUT_PREF_JSONL.relative_to(ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(ROOT)),
        },
        "role_counts": {},
        "semantic_target_counts": {},
    }

    for r in enriched:
        summary["role_counts"][r["training_role"]] = summary["role_counts"].get(r["training_role"], 0) + 1
        summary["semantic_target_counts"][r["semantic_target"]] = (
            summary["semantic_target_counts"].get(r["semantic_target"], 0) + 1
        )

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    print("[TRACER] wrote", OUT_CSV)
    print("[TRACER] wrote", OUT_JSONL)
    print("[TRACER] wrote", OUT_PREF_JSONL)
    print("[TRACER] wrote", OUT_SUMMARY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
