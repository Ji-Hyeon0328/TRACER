#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

V0_CSV = ROOT / "data/preference_datasets/tracer_objective_selector_seed_v0.csv"

NEGATIVE_SOURCES = [
    ROOT / "data/sanity_results/tracer_fusion_ram_gate_joined_gate_eval_closed_slippery_20260624_151729.csv",
    ROOT / "data/sanity_results/tracer_slippery_primitive_v1_compact_latest.csv",
    ROOT / "data/sanity_results/tracer_slippery_primitive_joined_slip_primitive_v1_20260623_211152.csv",
    ROOT / "data/sanity_results/tracer_slippery_micro_brake_v3_joined_slippery_micro_brake_v3_20260623_232039.csv",
    ROOT / "data/sanity_results/tracer_slippery_micro_brake_staged_v4_joined_slippery_micro_brake_staged_v4_fixed_20260623_234222.csv",
    ROOT / "data/sanity_results/tracer_micro_backstep_top2_repeat_joined_micro_backstep_top2_repeat_v1_20260623_230155.csv",
    ROOT / "data/sanity_results/tracer_true_backstep_top1_repeat_joined_true_backstep_top1_repeat_v0_20260623_223809.csv",
    ROOT / "data/sanity_results/tracer_slippery_active_hold_top2_joined_slip_active_hold_top2_v2_20260623_215051.csv",
]

SPONGE_EVIDENCE = ROOT / "data/rollout_metrics/gms_sponge_downslope_hold_evidence_v0.json"

OUT_DIR = ROOT / "data/preference_datasets"
OUT_CSV = OUT_DIR / "tracer_objective_selector_seed_v1.csv"
OUT_JSONL = OUT_DIR / "tracer_objective_selector_seed_v1.jsonl"
OUT_PREF_JSONL = OUT_DIR / "tracer_objective_selector_pairwise_preferences_v1.jsonl"
OUT_SUMMARY = OUT_DIR / "tracer_objective_selector_seed_v1_summary.json"


def f(x: Any, default: float = float("nan")) -> float:
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


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[TRACER][WARN] missing csv: {path}")
        return []
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def get(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def invalid_confidence(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []

    valid = get(row, "valid_locomotion_candidate", "valid_locomotion", default="")
    if str(valid).strip().lower() in {"false", "0", "no"}:
        reasons.append("valid_locomotion_false")

    fallen = get(row, "fallen_relative_z_based", "fallen", "fallen_relative", default="")
    if str(fallen).strip().lower() in {"true", "1", "yes"}:
        reasons.append("fallen_true")

    min_rel_z = f(get(row, "min_rel_z", "min_z", "min_base_z_world", default="nan"))
    if math.isfinite(min_rel_z) and min_rel_z < 0.18:
        reasons.append(f"low_height_min_rel_z={min_rel_z:.3f}")

    fallen_tail5 = f(get(row, "fallen_tail5_mean_gate", "fallen_tail5_mean", default="nan"))
    if math.isfinite(fallen_tail5) and fallen_tail5 >= 0.70:
        reasons.append(f"fallen_tail5_high={fallen_tail5:.3f}")

    ctrl_tail5 = f(get(row, "ctrl_ema_tail5_mean_gate", "ctrl_ema_tail5_mean", default="nan"))
    if math.isfinite(ctrl_tail5) and ctrl_tail5 >= 0.50:
        reasons.append(f"ctrl_ema_tail5_high={ctrl_tail5:.3f}")

    return bool(reasons), reasons


def negative_beta() -> list[float]:
    # safety-heavy target for invalid/risky cases
    return [0.10, 0.75, 0.15]


def normalize_base_row(row: dict[str, Any], sample_id: str) -> dict[str, Any]:
    out = dict(row)
    out["sample_id"] = sample_id
    return out


def make_negative_row(
    *,
    row: dict[str, Any],
    sample_id: str,
    source_path: Path,
    reasons: list[str],
) -> dict[str, Any]:
    beta = negative_beta()

    terrain = str(get(row, "terrain_key_fusion", "terrain_key", "terrain", default="unknown"))
    world = str(get(row, "world_name_fusion", "world_name", "world", default="unknown"))
    semantic_in = str(get(row, "semantic_mode", "semantic", "semantic_last", default="unknown"))
    style = str(get(row, "style", default="unknown"))
    decision = str(get(row, "decision", default=semantic_in))

    min_rel_z = f(get(row, "min_rel_z", "min_z", "min_base_z_world", default="nan"), 0.0)
    fallen = b(get(row, "fallen_relative_z_based", "fallen", "fallen_relative", default=False))
    valid = b(get(row, "valid_locomotion_candidate", "valid_locomotion", default=False))

    dx = f(get(row, "dx", "delta_x", "odom_delta_x", default="0.0"), 0.0)
    dy = f(get(row, "dy", "delta_y", "odom_delta_y", default="0.0"), 0.0)

    gate_unstable_proxy = 1.0 if any("fallen_tail5_high" in r or "ctrl_ema_tail5_high" in r for r in reasons) else 0.0

    return {
        "sample_id": sample_id,
        "source_matrix": "negative_v1",
        "source_path": str(source_path.relative_to(ROOT)),
        "case_id": str(get(row, "sanity_tag", "case_id", "tag", default=sample_id)),
        "terrain_key": terrain,
        "world": world,
        "semantic_input": semantic_in,
        "style_input": style,
        "decision": decision,
        "ramgate_enabled": True,
        "ramgate_calibrated": False,
        "ramgate_observed": True,
        "ram_monitor_observed": True,
        "locomotion_valid": False,
        "no_fall_route_closed": False if fallen else valid,
        "strict_height_margin_valid": False,
        "fallen_relative": fallen,
        "dx": dx,
        "dy": dy,
        "min_rel_z": min_rel_z,
        "mpc_vx_mean": f(get(row, "mpc_vx_mean", "vx_mean", default="0.0"), 0.0),
        "mpc_body_height_mean": f(get(row, "mpc_body_height_mean", "body_height_mean", default="0.0"), 0.0),
        "mpc_clearance_mean": f(get(row, "mpc_clearance_mean", "clearance_mean", default="0.0"), 0.0),
        "ram_ctrl_risk_mean": f(get(row, "ctrl_ema_tail5_mean_gate", "ctrl_ema_tail5_mean", default="0.0"), 0.0),
        "ram_fallen_mean": f(get(row, "fallen_tail5_mean_gate", "fallen_tail5_mean", "fallen_mean", default="0.0"), 0.0),
        "ram_recovery_mean": f(get(row, "recovery_mean", "recovery_tail5_mean", default="0.0"), 0.0),
        "gate_stable_frac": 0.0,
        "gate_caution_frac": 0.0,
        "gate_unstable_frac": gate_unstable_proxy,
        "gate_override_frac": 1.0,
        "gate_first_action": str(get(row, "action_first", default="invalid_or_risky")),
        "gate_last_action": str(get(row, "action_last", default="invalid_or_risky")),
        "calibration_first": "",
        "calibration_last": "",
        "training_role": "invalid_or_risky",
        "semantic_target": "no_valid_forward_recovery_needed",
        "beta_v_target": beta[0],
        "beta_s_target": beta[1],
        "beta_e_target": beta[2],
        "preference_score": -3.0 - 0.5 * len(reasons),
        "note": "high_confidence_negative: " + ";".join(reasons),
    }


def sponge_negative_rows(start_idx: int) -> list[dict[str, Any]]:
    if not SPONGE_EVIDENCE.exists():
        return []

    data = json.loads(SPONGE_EVIDENCE.read_text())
    cases = data.get("cases", []) if isinstance(data, dict) else []

    rows = []
    for c in cases:
        fm = c.get("finite_motion", {}) or {}
        sj = c.get("summary_json", {}) or {}
        min_z = f(fm.get("min_z", sj.get("min_rel_z", 999.0)))
        first_below = fm.get("first_z_below_0p18_index", None)

        if not (math.isfinite(min_z) and min_z < 0.18):
            continue

        beta = negative_beta()
        i = start_idx + len(rows)
        rows.append(
            {
                "sample_id": f"tracer_v1_neg_{i:04d}",
                "source_matrix": "sponge_downslope_evidence_v0",
                "source_path": str(SPONGE_EVIDENCE.relative_to(ROOT)),
                "case_id": c.get("name", f"sponge_downslope_case_{i}"),
                "terrain_key": data.get("terrain_name", "sponge_firm_downslope_5deg_forward"),
                "world": data.get("world", "tracer_sponge_firm_downslope_5deg"),
                "semantic_input": data.get("decision", "no_valid_forward_recovery_needed"),
                "style_input": "sponge_downslope_forward_probe",
                "decision": data.get("decision", "no_valid_forward_recovery_needed"),
                "ramgate_enabled": False,
                "ramgate_calibrated": False,
                "ramgate_observed": False,
                "ram_monitor_observed": False,
                "locomotion_valid": False,
                "no_fall_route_closed": False,
                "strict_height_margin_valid": False,
                "fallen_relative": False,
                "dx": f(fm.get("delta_x", 0.0), 0.0),
                "dy": f(fm.get("delta_y", 0.0), 0.0),
                "min_rel_z": min_z,
                "mpc_vx_mean": f(sj.get("mpc_vx_mean", 0.0), 0.0),
                "mpc_body_height_mean": f(sj.get("mpc_body_height_mean", 0.0), 0.0),
                "mpc_clearance_mean": f(sj.get("mpc_clearance_mean", 0.0), 0.0),
                "ram_ctrl_risk_mean": 0.0,
                "ram_fallen_mean": 0.0,
                "ram_recovery_mean": 0.0,
                "gate_stable_frac": 0.0,
                "gate_caution_frac": 0.0,
                "gate_unstable_frac": 1.0,
                "gate_override_frac": 1.0,
                "gate_first_action": "",
                "gate_last_action": "",
                "calibration_first": "",
                "calibration_last": "",
                "training_role": "invalid_or_risky",
                "semantic_target": "no_valid_forward_recovery_needed",
                "beta_v_target": beta[0],
                "beta_s_target": beta[1],
                "beta_e_target": beta[2],
                "preference_score": -3.5,
                "note": f"sponge_downslope_low_height min_z={min_z:.3f} first_below_0p18={first_below}",
            }
        )
    return rows


def make_pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefs = []
    by_terrain: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_terrain.setdefault(str(r["terrain_key"]), []).append(r)

    for terrain, items in by_terrain.items():
        for a in items:
            for c in items:
                if a["sample_id"] == c["sample_id"]:
                    continue
                if f(a.get("preference_score")) > f(c.get("preference_score")) + 0.75:
                    prefs.append(
                        {
                            "terrain_key": terrain,
                            "preferred_sample_id": a["sample_id"],
                            "rejected_sample_id": c["sample_id"],
                            "preferred_case_id": a["case_id"],
                            "rejected_case_id": c["case_id"],
                            "preferred_source": a["source_matrix"],
                            "rejected_source": c["source_matrix"],
                            "reason": "higher V0 preference score / lower risk",
                            "score_preferred": f(a["preference_score"]),
                            "score_rejected": f(c["preference_score"]),
                        }
                    )
    return prefs


def main() -> None:
    base_rows = read_csv(V0_CSV)
    rows: list[dict[str, Any]] = []

    for i, r in enumerate(base_rows):
        rows.append(normalize_base_row(r, f"tracer_v1_base_{i:04d}"))

    neg_count = 0
    seen_case_ids = {str(r.get("case_id", "")) for r in rows}

    for src in NEGATIVE_SOURCES:
        for r in read_csv(src):
            is_neg, reasons = invalid_confidence(r)
            if not is_neg:
                continue

            case_id = str(get(r, "sanity_tag", "case_id", "tag", default=""))
            # keep repeats if they have different sanity tags, but avoid exact duplicates
            if case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)

            sample_id = f"tracer_v1_neg_{neg_count:04d}"
            rows.append(make_negative_row(row=r, sample_id=sample_id, source_path=src, reasons=reasons))
            neg_count += 1

    sponge_rows = sponge_negative_rows(neg_count)
    rows.extend(sponge_rows)

    prefs = make_pairwise(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with OUT_JSONL.open("w") as fjsonl:
        for r in rows:
            fjsonl.write(json.dumps(r) + "\n")

    with OUT_PREF_JSONL.open("w") as fpref:
        for p in prefs:
            fpref.write(json.dumps(p) + "\n")

    summary: dict[str, Any] = {
        "num_samples": len(rows),
        "num_base_samples": len(base_rows),
        "num_negative_csv_samples": neg_count,
        "num_sponge_negative_samples": len(sponge_rows),
        "num_pairwise_preferences": len(prefs),
        "outputs": {
            "csv": str(OUT_CSV.relative_to(ROOT)),
            "jsonl": str(OUT_JSONL.relative_to(ROOT)),
            "pairwise_jsonl": str(OUT_PREF_JSONL.relative_to(ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(ROOT)),
        },
        "role_counts": {},
        "semantic_target_counts": {},
    }

    for r in rows:
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
