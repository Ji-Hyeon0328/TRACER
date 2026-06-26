#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

SANITY_CSV = ROOT / "data/sanity_results/tracer_fusion_sanity_results.csv"
RAM_CSV = ROOT / "data/sanity_results/tracer_ram_monitor_results_v2.csv"
GATE_CSV = ROOT / "data/sanity_results/tracer_ram_gate_monitor_results.csv"

OUT_CSV = ROOT / "data/rollout_metrics/feasible_closed_loop_gms_ramgate_v0.csv"
OUT_JSON = ROOT / "data/rollout_metrics/feasible_closed_loop_gms_ramgate_v0.json"


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def get(row: dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name])
    return default


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def as_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def find_latest(rows: list[dict[str, Any]], *, terrain: str, tag: str, semantic: str) -> tuple[int, dict[str, Any]]:
    tagged = []
    for i, r in enumerate(rows):
        r_terrain = get(r, "terrain", "terrain_key", "policy_terrain", default="")
        r_tag = get(r, "tag", "run_tag", "experiment_tag", "label", default="")
        r_semantic = get(r, "semantic", "semantic_mode", default="")
        if r_terrain == terrain and (tag in r_tag or r_tag == tag):
            tagged.append((i, r))
        elif r_terrain == terrain and r_semantic == semantic:
            tagged.append((i, r))
    if not tagged:
        raise RuntimeError(f"no row found: terrain={terrain}, tag={tag}, semantic={semantic}")
    return tagged[-1]


def find_latest_by_tag(rows: list[dict[str, Any]], *, tag: str, terrain: str) -> dict[str, Any] | None:
    out = []
    for r in rows:
        r_tag = get(r, "tag", "run_tag", "experiment_tag", "label", default="")
        r_terrain = get(r, "terrain", "terrain_key", default="")
        if r_tag == tag or tag in r_tag:
            out.append(r)
        elif r_terrain == terrain:
            out.append(r)
    return out[-1] if out else None


def main() -> None:
    sanity = read_csv(SANITY_CSV)
    ram = read_csv(RAM_CSV)
    gate = read_csv(GATE_CSV)

    specs = [
        {
            "case_id": "flat_normal_fast_ramgate_closed_loop",
            "terrain": "flat_normal",
            "world": "earth",
            "tag": "feasible_v0_flat_normal_gmson_ramgate_20s_server",
            "semantic": "validated_locomotion",
            "style": "fast",
        },
        {
            "case_id": "rough_mid_cautious_probe_ramgate_closed_loop",
            "terrain": "rough_mid",
            "world": "tracer_rough_mid",
            "tag": "feasible_v0_rough_mid_gmson_ramgate_20s_server",
            "semantic": "cautious_probe",
            "style": "cautious",
        },
        {
            "case_id": "slope_5deg_high_clearance_ramgate_closed_loop",
            "terrain": "slope_5deg",
            "world": "tracer_slope_5deg",
            "tag": "feasible_v0_slope_5deg_gmson_ramgate_20s_server",
            "semantic": "high_clearance_slow_probe",
            "style": "high_clearance",
        },
    ]

    curated = []
    for spec in specs:
        idx, row = find_latest(
            sanity,
            terrain=spec["terrain"],
            tag=spec["tag"],
            semantic=spec["semantic"],
        )

        ram_row = find_latest_by_tag(ram, tag=spec["tag"], terrain=spec["terrain"])
        gate_row = find_latest_by_tag(gate, tag=spec["tag"], terrain=spec["terrain"])

        dx = as_float(get(row, "dx", "delta_x", default="nan"))
        dy = as_float(get(row, "dy", "delta_y", default="nan"))
        min_rel_z = as_float(get(row, "min_rel_z", "min_z", default="nan"))
        fallen_rel = as_bool(get(row, "fallen_rel", "fallen_relative_z_based", "fallen", default="false"))

        gate_samples = as_float(get(gate_row or {}, "samples", "num_samples", default="0"), 0.0)
        ram_samples = as_float(get(ram_row or {}, "samples", "num_samples", default="0"), 0.0)

        curated.append(
            {
                "case_id": spec["case_id"],
                "terrain_key": spec["terrain"],
                "world": spec["world"],
                "tag": spec["tag"],
                "source_row_index": idx,
                "semantic": get(row, "semantic", "semantic_mode", default=spec["semantic"]),
                "style": get(row, "style", "suggested_style", default=spec["style"]),
                "decision": get(row, "decision", "final_decision", default=""),
                "route_closed": True,
                "ramgate_enabled": True,
                "ramgate_observed": gate_samples > 0,
                "ram_monitor_observed": ram_samples > 0,
                "locomotion_valid": (not fallen_rel) and min_rel_z > 0.18 and dx > 0.03,
                "fallen_relative": fallen_rel,
                "dx": dx,
                "dy": dy,
                "min_rel_z": min_rel_z,
                "mpc_vx_mean": as_float(get(row, "mpc_vx_mean", "vx_mean", default="nan")),
                "mpc_body_height_mean": as_float(get(row, "mpc_body_height_mean", "body_height_mean", default="nan")),
                "mpc_clearance_mean": as_float(get(row, "mpc_clearance_mean", "clearance_mean", default="nan")),
                "ram_samples": ram_samples,
                "gate_samples": gate_samples,
                "gate_stable_frac": as_float(get(gate_row or {}, "stable_frac", "stable", "level_stable_frac", default="nan")),
                "gate_caution_frac": as_float(get(gate_row or {}, "caution_frac", "caution", "level_caution_frac", default="nan")),
                "gate_unstable_frac": as_float(get(gate_row or {}, "unstable_frac", "unstable", "level_unstable_frac", default="nan")),
                "gate_first_action": get(gate_row or {}, "first_action", "action_first", default=""),
                "gate_last_action": get(gate_row or {}, "last_action", "action_last", default=""),
                "gate_override_frac": as_float(get(gate_row or {}, "would_override_frac", "override_frac", default="nan")),
                "ram_ctrl_risk_mean": as_float(get(ram_row or {}, "ctrl_risk_mean", "control_risk_mean", default="nan")),
                "ram_fallen_mean": as_float(get(ram_row or {}, "fallen_mean", default="nan")),
                "ram_recovery_mean": as_float(get(ram_row or {}, "recovery_mean", default="nan")),
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
