#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write")

    fieldnames: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def get_col(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for n in names:
        if n in row:
            return to_float(row.get(n), default)
    return default


def classify(row: dict[str, Any]) -> str:
    stable = get_col(row, "stable_frac")
    zmin = get_col(row, "zmin_mean", "proprio_base_z_min")
    below22 = get_col(row, "below22_mean", "proprio_base_z_below_0p22_frac")
    margin = get_col(row, "target_height_margin_v2", default=999.0)
    score = get_col(row, "R_profile_v2_gated_mean", "R_gated_mean")

    # Use target-height-aware margin if present.
    has_margin = margin != 999.0

    if has_margin:
        if stable >= 0.99 and margin >= -0.02 and score >= 0.25:
            return "positive_teacher"
        if stable >= 0.50 and margin >= -0.08:
            return "borderline"
        return "risk_negative"

    # Fallback for old v0 scored csv.
    if stable >= 0.99 and zmin >= 0.22 and below22 <= 0.05 and score >= 0.25:
        return "positive_teacher"
    if stable >= 0.50:
        return "borderline"
    return "risk_negative"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data/phase_a_terrain_aware_bank_v0")
    args = ap.parse_args()

    merged: list[dict[str, Any]] = []

    for inp in args.inputs:
        p = Path(inp)
        rows = read_csv(p)
        for r in rows:
            rr = dict(r)
            rr["source_csv"] = str(p)
            rr["bank_label"] = classify(rr)

            # Normalize score field for ranker training.
            if "R_profile_v2_gated_mean" in rr:
                rr["R_train_score"] = rr["R_profile_v2_gated_mean"]
            else:
                rr["R_train_score"] = rr.get("R_gated_mean", "0.0")

            merged.append(rr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_csv = out_dir / "phase_a_terrain_aware_candidates_v0.csv"
    pos_csv = out_dir / "phase_a_terrain_aware_positive_v0.csv"
    border_csv = out_dir / "phase_a_terrain_aware_borderline_v0.csv"
    risk_csv = out_dir / "phase_a_terrain_aware_risk_negative_v0.csv"

    write_csv(all_csv, merged)
    write_csv(pos_csv, [r for r in merged if r["bank_label"] == "positive_teacher"])
    write_csv(border_csv, [r for r in merged if r["bank_label"] == "borderline"])
    write_csv(risk_csv, [r for r in merged if r["bank_label"] == "risk_negative"])

    counts: dict[str, int] = {}
    terrain_counts: dict[tuple[str, str], int] = {}

    for r in merged:
        label = r["bank_label"]
        terrain = str(r.get("terrain", "unknown"))
        counts[label] = counts.get(label, 0) + 1
        terrain_counts[(terrain, label)] = terrain_counts.get((terrain, label), 0) + 1

    summary = {
        "inputs": args.inputs,
        "num_rows": len(merged),
        "counts": counts,
        "terrain_counts": {f"{k[0]}::{k[1]}": v for k, v in terrain_counts.items()},
        "outputs": {
            "all_csv": str(all_csv),
            "positive_csv": str(pos_csv),
            "borderline_csv": str(border_csv),
            "risk_negative_csv": str(risk_csv),
        },
    }

    with (out_dir / "phase_a_terrain_aware_bank_summary_v0.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print()
    print("===== top positive/borderline by terrain =====")

    for terrain in sorted(set(str(r.get("terrain", "unknown")) for r in merged)):
        subset = [
            r for r in merged
            if str(r.get("terrain", "unknown")) == terrain
            and r["bank_label"] in {"positive_teacher", "borderline"}
        ]
        subset.sort(key=lambda r: to_float(r.get("R_train_score")), reverse=True)

        print()
        print(f"terrain={terrain}")
        for r in subset[:10]:
            print(
                f"  {r['bank_label']:16s} "
                f"{r.get('profile',''):18s} "
                f"{r.get('preset',''):36s} "
                f"score={to_float(r.get('R_train_score')):.4f} "
                f"stable={to_float(r.get('stable_frac')):.2f} "
                f"zmin={to_float(r.get('zmin_mean')):.3f} "
                f"margin={to_float(r.get('target_height_margin_v2'), 999.0):.3f} "
                f"vx={to_float(r.get('ref_vx')):.3f} "
                f"h={to_float(r.get('ref_body_height')):.3f} "
                f"c={to_float(r.get('ref_swing_clearance')):.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
