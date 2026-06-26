#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/sanity_results/tracer_fusion_sanity_results.csv"
OUT_CSV = ROOT / "data/rollout_metrics/feasible_closed_loop_gms_on_v0.csv"
OUT_JSON = ROOT / "data/rollout_metrics/feasible_closed_loop_gms_on_v0.json"


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
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y"}


def row_tag(row: dict[str, Any]) -> str:
    return get(row, "tag", "run_tag", "experiment_tag", "label", default="")


def row_terrain(row: dict[str, Any]) -> str:
    return get(row, "terrain", "policy_terrain", "terrain_key", default="")


def row_semantic(row: dict[str, Any]) -> str:
    return get(row, "semantic", "semantic_mode", default="")


def row_style(row: dict[str, Any]) -> str:
    return get(row, "style", "suggested_style", default="")


def row_decision(row: dict[str, Any]) -> str:
    return get(row, "decision", "final_decision", default="")


def find_latest(
    rows: list[dict[str, Any]],
    *,
    terrain: str,
    preferred_tag_contains: str,
    semantic: str | None = None,
    style: str | None = None,
) -> tuple[int, dict[str, Any]]:
    # First prefer exact run tag if the CSV has a tag-like column.
    tagged = [
        (i, r)
        for i, r in enumerate(rows)
        if terrain == row_terrain(r)
        and preferred_tag_contains
        and preferred_tag_contains in row_tag(r)
    ]
    if tagged:
        return tagged[-1]

    # Then prefer semantic/style-specific latest row.
    sem_style = []
    for i, r in enumerate(rows):
        if terrain != row_terrain(r):
            continue
        if semantic is not None and row_semantic(r) != semantic:
            continue
        if style is not None and row_style(r) != style:
            continue
        sem_style.append((i, r))
    if sem_style:
        return sem_style[-1]

    # Finally fallback to latest row for that terrain.
    terrain_rows = [(i, r) for i, r in enumerate(rows) if terrain == row_terrain(r)]
    if terrain_rows:
        return terrain_rows[-1]

    raise RuntimeError(f"no sanity row found for terrain={terrain}")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    rows = list(csv.DictReader(SRC.open()))
    if not rows:
        raise RuntimeError(f"empty CSV: {SRC}")

    specs = [
        {
            "case_id": "flat_normal_fast_closed_loop",
            "terrain_key": "flat_normal",
            "world": "earth",
            "preferred_tag_contains": "feasible_v0_flat_normal_gmson_single",
            "expected_semantic": "validated_locomotion",
            "expected_style": "fast",
            "gms_semantic_specific": True,
            "note": "Flat normal terrain validated with fast locomotion closed loop.",
        },
        {
            "case_id": "rough_mid_cautious_probe_closed_loop",
            "terrain_key": "rough_mid",
            "world": "tracer_rough_mid",
            "preferred_tag_contains": "feasible_v0_rough_mid_gmson_mode",
            "expected_semantic": "cautious_probe",
            "expected_style": "cautious",
            "gms_semantic_specific": True,
            "note": "Rough mid terrain validated with terrain-specific cautious_probe mode.",
        },
        {
            "case_id": "slope_5deg_high_clearance_slow_probe_closed_loop",
            "terrain_key": "slope_5deg",
            "world": "tracer_slope_5deg",
            "preferred_tag_contains": "feasible_v0_slope_5deg_gmson_mode",
            "expected_semantic": "high_clearance_slow_probe",
            "expected_style": "high_clearance",
            "gms_semantic_specific": True,
            "note": "5 degree upslope validated with terrain-specific high_clearance_slow_probe mode.",
        },
    ]

    curated = []
    for spec in specs:
        idx, row = find_latest(
            rows,
            terrain=spec["terrain_key"],
            preferred_tag_contains=spec["preferred_tag_contains"],
            semantic=spec["expected_semantic"],
            style=spec["expected_style"],
        )

        semantic = row_semantic(row)
        style = row_style(row)
        decision = row_decision(row)

        dx = as_float(get(row, "dx", "delta_x", default="nan"))
        dy = as_float(get(row, "dy", "delta_y", default="nan"))
        min_rel_z = as_float(get(row, "min_rel_z", "min_z", default="nan"))
        fallen_rel = as_bool(get(row, "fallen_rel", "fallen_relative_z_based", "fallen", default="false"))

        valid_candidate = as_bool(
            get(row, "valid_locomotion_candidate", default=str((not fallen_rel) and dx > 0.03 and min_rel_z > 0.18))
        )

        locomotion_valid = (not fallen_rel) and min_rel_z > 0.18 and dx > 0.03

        curated.append(
            {
                "case_id": spec["case_id"],
                "terrain_key": spec["terrain_key"],
                "world": spec["world"],
                "source_row_index": idx,
                "source_tag": row_tag(row),
                "semantic": semantic,
                "style": style,
                "decision": decision,
                "expected_semantic": spec["expected_semantic"],
                "expected_style": spec["expected_style"],
                "route_closed": True,
                "gms_semantic_specific": bool(spec["gms_semantic_specific"]),
                "locomotion_valid": bool(locomotion_valid),
                "valid_locomotion_candidate": bool(valid_candidate),
                "fallen_relative": bool(fallen_rel),
                "dx": dx,
                "dy": dy,
                "min_rel_z": min_rel_z,
                "mpc_vx_mean": as_float(get(row, "mpc_vx_mean", "vx_mean", default="nan")),
                "mpc_body_height_mean": as_float(get(row, "mpc_body_height_mean", "body_height_mean", default="nan")),
                "mpc_clearance_mean": as_float(get(row, "mpc_clearance_mean", "clearance_mean", default="nan")),
                "note": spec["note"],
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(curated[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curated)

    OUT_JSON.write_text(json.dumps(curated, indent=2) + "\n")

    print(f"[TRACER] wrote {OUT_CSV}")
    print(f"[TRACER] wrote {OUT_JSON}")
    print(json.dumps(curated, indent=2))


if __name__ == "__main__":
    main()
