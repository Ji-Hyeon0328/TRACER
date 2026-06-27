#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


DEFAULT_TERRAINS = ["flat_normal", "rough_mid", "slope_5deg"]


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def newest_paths(pattern: str, n: int) -> list[Path]:
    paths = [Path(p) for p in Path(".").glob(pattern)]
    paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    return paths[-n:]


def safe_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value in ("", "nan", "None", None):
            return default
        return float(value)
    except Exception:
        return default


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(xs),
        "std": pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def simulate_shadow_gate(values: list[float], threshold: float, consecutive: int) -> dict:
    streak = 0
    trigger_count = 0
    active_rows = 0
    first_trigger_index = None
    trigger_indices = []

    for i, v in enumerate(values):
        if v >= threshold:
            streak += 1
        else:
            streak = 0

        if streak >= consecutive:
            active_rows += 1

        if streak == consecutive:
            trigger_count += 1
            trigger_indices.append(i)
            if first_trigger_index is None:
                first_trigger_index = i

    return {
        "threshold": threshold,
        "consecutive": consecutive,
        "trigger_count": trigger_count,
        "active_rows": active_rows,
        "active_rate": active_rows / max(1, len(values)),
        "first_trigger_index": first_trigger_index,
        "trigger_indices": trigger_indices[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--thresholds", type=str, default="0.2,0.4,0.6")
    ap.add_argument("--consecutive", type=str, default="1,3,5,8")
    ap.add_argument("--out-md", type=Path, default=Path("reports/ram_recovery_shadow_gate_v0_report.md"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/ram_recovery_shadow_gate_v0_report.json"))
    args = ap.parse_args()

    thresholds = parse_csv_floats(args.thresholds)
    consecutives = parse_csv_ints(args.consecutive)

    all_results = {}

    lines = []
    lines.append("# TRACER RAM Recovery Shadow Gate v0")
    lines.append("")
    lines.append("This report evaluates recovery-needed based shadow gating from recent sanitized beta-blend robust30 rollouts.")
    lines.append("It does not change the active command path.")
    lines.append("")

    for terrain in DEFAULT_TERRAINS:
        pattern = (
            f"data/rollout_dataset_v0/episodes/"
            f"{terrain}_theta_mlp_udp_v0_learned_stack_v3_beta_blend_robust30_{terrain}_r*.csv"
        )
        paths = newest_paths(pattern, args.limit)

        values = []
        vx_values = []
        proprio_mean_values = []
        fallen_values = []
        risk_values = []

        for p in paths:
            with p.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    values.append(safe_float(row, "ram_recovery_needed"))
                    vx_values.append(safe_float(row, "mpc_vx"))
                    proprio_mean_values.append(safe_float(row, "proprio_abs_mean"))
                    fallen_values.append(safe_float(row, "ram_run_fallen"))
                    risk_values.append(safe_float(row, "ram_future_risk"))

        terrain_result = {
            "files": [str(p) for p in paths],
            "n_rows": len(values),
            "ram_recovery_needed": stats(values),
            "mpc_vx": stats(vx_values),
            "proprio_abs_mean": stats(proprio_mean_values),
            "ram_run_fallen": stats(fallen_values),
            "ram_future_risk": stats(risk_values),
            "shadow_gate": [],
        }

        for th in thresholds:
            for cons in consecutives:
                terrain_result["shadow_gate"].append(
                    simulate_shadow_gate(values, th, cons)
                )

        all_results[terrain] = terrain_result

        lines.append(f"## {terrain}")
        lines.append("")
        lines.append("Files:")
        for p in paths:
            lines.append(f"- `{p.name}`")
        lines.append("")
        lines.append(
            f"- rows: {len(values)}"
        )
        lines.append(
            f"- recovery_needed mean/max: "
            f"{terrain_result['ram_recovery_needed']['mean']:.6f} / "
            f"{terrain_result['ram_recovery_needed']['max']:.6f}"
        )
        lines.append(
            f"- proprio_abs_mean mean/max: "
            f"{terrain_result['proprio_abs_mean']['mean']:.6f} / "
            f"{terrain_result['proprio_abs_mean']['max']:.6f}"
        )
        lines.append(
            f"- run_fallen mean: {terrain_result['ram_run_fallen']['mean']:.6f}"
        )
        lines.append(
            f"- future_risk mean: {terrain_result['ram_future_risk']['mean']:.6f}"
        )
        lines.append("")
        lines.append("| threshold | consecutive | triggers | active_rows | active_rate | first_trigger_index |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for item in terrain_result["shadow_gate"]:
            first_idx = item["first_trigger_index"]
            first_txt = "None" if first_idx is None else str(first_idx)
            lines.append(
                f"| {item['threshold']:.2f} | {item['consecutive']} | "
                f"{item['trigger_count']} | {item['active_rows']} | "
                f"{item['active_rate']:.6f} | {first_txt} |"
            )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `ram_recovery_needed` is the only reasonable recovery-gate candidate after feature sanitization.")
    lines.append("- `ram_run_fallen` and `ram_future_risk` remain high across terrains, so they should stay shadow-only.")
    lines.append("- A runtime recovery gate should require thresholding plus consecutive samples, not a single-frame trigger.")
    lines.append("- Recommended next runtime shadow setting: threshold=0.6, consecutive=3 or stricter.")
    lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    args.out_json.write_text(json.dumps(all_results, indent=2))

    print("\n".join(lines))
    print(f"[TRACER] wrote {args.out_md}")
    print(f"[TRACER] wrote {args.out_json}")


if __name__ == "__main__":
    main()
