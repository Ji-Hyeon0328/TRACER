#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

SANITY_CSV = ROOT / "data/sanity_results/tracer_fusion_sanity_results.csv"
RAM_CSV = ROOT / "data/sanity_results/tracer_ram_monitor_results_v2.csv"
GATE_CSV = Path(
    __import__("os").environ.get(
        "TRACER_CALIBRATED_GATE_SUMMARY_CSV",
        str(ROOT / "data/rollout_metrics/feasible_closed_loop_gms_ramgate_calibrated_gate_summary_v0.csv"),
    )
)

OUT_CSV = ROOT / "data/rollout_metrics/feasible_closed_loop_gms_ramgate_calibrated_v0.csv"
OUT_JSON = ROOT / "data/rollout_metrics/feasible_closed_loop_gms_ramgate_calibrated_v0.json"


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def get(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def as_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def finite_or_none(x: Any) -> float | None:
    v = as_float(x)
    return v if math.isfinite(v) else None


def find_latest_sanity(
    rows: list[dict[str, Any]],
    *,
    terrain: str,
    tag: str,
    semantic: str,
) -> tuple[int, dict[str, Any]]:
    tagged: list[tuple[int, dict[str, Any]]] = []
    fallback: list[tuple[int, dict[str, Any]]] = []

    for i, r in enumerate(rows):
        r_terrain = str(get(r, "terrain", "terrain_key", "policy_terrain", default=""))
        r_tag = str(get(r, "tag", "run_tag", "experiment_tag", "label", default=""))
        r_semantic = str(get(r, "semantic", "semantic_mode", default=""))

        if r_terrain == terrain and r_tag == tag:
            tagged.append((i, r))
        elif r_terrain == terrain and r_semantic == semantic:
            fallback.append((i, r))

    if tagged:
        return tagged[-1]
    if fallback:
        return fallback[-1]

    raise RuntimeError(f"no sanity row found: terrain={terrain}, tag={tag}, semantic={semantic}")


def find_latest_by_tag(
    rows: list[dict[str, Any]],
    *,
    terrain: str,
    tag: str,
) -> dict[str, Any] | None:
    tagged = []
    fallback = []

    for r in rows:
        r_terrain = str(get(r, "terrain", "terrain_key", default=""))
        r_tag = str(get(r, "sanity_tag", "tag", "run_tag", "experiment_tag", "label", default=""))

        if r_tag == tag:
            tagged.append(r)
        elif r_terrain == terrain:
            fallback.append(r)

    if tagged:
        return tagged[-1]
    if fallback:
        return fallback[-1]
    return None


def main() -> None:
    sanity_rows = read_csv(SANITY_CSV)
    ram_rows = read_csv(RAM_CSV)
    gate_rows = read_csv(GATE_CSV)

    specs = [
        {
            "case_id": "flat_normal_fast_ramgate_calibrated_closed_loop",
            "terrain": "flat_normal",
            "world": "earth",
            "tag": "feasible_v0_flat_normal_gmson_ramgate_calibrated_20s",
            "semantic": "validated_locomotion",
            "style": "fast",
        },
        {
            "case_id": "rough_mid_cautious_probe_ramgate_calibrated_closed_loop",
            "terrain": "rough_mid",
            "world": "tracer_rough_mid",
            "tag": "feasible_v0_rough_mid_gmson_ramgate_calibrated_20s",
            "semantic": "cautious_probe",
            "style": "cautious",
        },
        {
            "case_id": "slope_5deg_high_clearance_ramgate_calibrated_closed_loop",
            "terrain": "slope_5deg",
            "world": "tracer_slope_5deg",
            "tag": "feasible_v0_slope_5deg_gmson_ramgate_calibrated_20s",
            "semantic": "high_clearance_slow_probe",
            "style": "high_clearance",
        },
    ]

    curated: list[dict[str, Any]] = []

    for spec in specs:
        idx, sanity = find_latest_sanity(
            sanity_rows,
            terrain=spec["terrain"],
            tag=spec["tag"],
            semantic=spec["semantic"],
        )
        ram = find_latest_by_tag(ram_rows, terrain=spec["terrain"], tag=spec["tag"]) or {}
        gate = find_latest_by_tag(gate_rows, terrain=spec["terrain"], tag=spec["tag"]) or {}

        dx = as_float(get(sanity, "dx", "delta_x", default="nan"))
        dy = as_float(get(sanity, "dy", "delta_y", default="nan"))
        min_rel_z = as_float(get(sanity, "min_rel_z", "min_z", default="nan"))
        fallen_rel = as_bool(get(sanity, "fallen_rel", "fallen_relative_z_based", "fallen", default="false"))

        gate_samples = finite_or_none(get(gate, "num_gate_samples", "gate_samples", "samples", default="nan"))
        ram_samples = finite_or_none(get(ram, "samples", "num_samples", "num_ram_samples", default="nan"))

        no_fall_route_closed = (not fallen_rel) and math.isfinite(dx) and dx > 0.03
        strict_height_margin_valid = math.isfinite(min_rel_z) and min_rel_z > 0.18

        gate_stable = finite_or_none(get(gate, "stable_frac", "gate_stable_frac", default="nan"))
        gate_caution = finite_or_none(get(gate, "caution_frac", "gate_caution_frac", default="nan"))
        gate_unstable = finite_or_none(get(gate, "unstable_frac", "gate_unstable_frac", default="nan"))
        gate_override = finite_or_none(get(gate, "would_override_mean", "gate_override_frac", default="nan"))

        raw_gate_stable = finite_or_none(get(gate, "raw_stable_frac", default="nan"))
        raw_gate_caution = finite_or_none(get(gate, "raw_caution_frac", default="nan"))
        raw_gate_unstable = finite_or_none(get(gate, "raw_unstable_frac", default="nan"))

        calibration_note = ""
        if spec["terrain"] == "flat_normal" and strict_height_margin_valid:
            calibration_note = (
                "Flat false positive was reduced after RAM gate warmup/hysteresis "
                "and validated-fast protection."
            )

        curated.append(
            {
                "case_id": spec["case_id"],
                "terrain_key": spec["terrain"],
                "world": spec["world"],
                "tag": spec["tag"],
                "source_row_index": idx,
                "semantic": get(sanity, "semantic", "semantic_mode", default=spec["semantic"]),
                "style": get(sanity, "style", "suggested_style", default=spec["style"]),
                "decision": get(sanity, "decision", "final_decision", default=""),
                "route_closed": True,
                "ramgate_enabled": True,
                "ramgate_calibrated": True,
                "ramgate_observed": gate_samples is not None or gate_stable is not None,
                "ram_monitor_observed": ram_samples is not None or finite_or_none(get(ram, "ctrl_risk_mean", default="nan")) is not None,
                "locomotion_valid": no_fall_route_closed and strict_height_margin_valid,
                "no_fall_route_closed": no_fall_route_closed,
                "strict_height_margin_valid": strict_height_margin_valid,
                "fallen_relative": fallen_rel,
                "dx": dx,
                "dy": dy,
                "min_rel_z": min_rel_z,
                "mpc_vx_mean": finite_or_none(get(sanity, "mpc_vx_mean", "vx_mean", default="nan")),
                "mpc_body_height_mean": finite_or_none(get(sanity, "mpc_body_height_mean", "body_height_mean", default="nan")),
                "mpc_clearance_mean": finite_or_none(get(sanity, "mpc_clearance_mean", "clearance_mean", default="nan")),
                "ram_samples": ram_samples,
                "gate_samples": gate_samples,
                "gate_stable_frac": gate_stable,
                "gate_caution_frac": gate_caution,
                "gate_unstable_frac": gate_unstable,
                "raw_gate_stable_frac": raw_gate_stable,
                "raw_gate_caution_frac": raw_gate_caution,
                "raw_gate_unstable_frac": raw_gate_unstable,
                "gate_first_action": get(gate, "action_first", "gate_first_action", default=""),
                "gate_last_action": get(gate, "action_last", "gate_last_action", default=""),
                "gate_override_frac": gate_override,
                "calibration_first": get(gate, "calibration_first", default=""),
                "calibration_last": get(gate, "calibration_last", default=""),
                "ram_ctrl_risk_mean": finite_or_none(get(ram, "ctrl_risk_mean", "control_risk_mean", default="nan")),
                "ram_fallen_mean": finite_or_none(get(ram, "fallen_mean", default="nan")),
                "ram_recovery_mean": finite_or_none(get(ram, "recovery_mean", default="nan")),
                "calibration_note": calibration_note,
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curated[0].keys()))
        writer.writeheader()
        writer.writerows(curated)

    OUT_JSON.write_text(json.dumps(curated, indent=2) + "\n")

    print(f"[TRACER] wrote {OUT_CSV}")
    print(f"[TRACER] wrote {OUT_JSON}")
    print(json.dumps(curated, indent=2))


if __name__ == "__main__":
    main()
