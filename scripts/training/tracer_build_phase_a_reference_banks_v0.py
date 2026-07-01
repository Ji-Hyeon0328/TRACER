#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")


def classify(row: dict[str, Any]) -> str:
    valid_frac = f(row, "valid_frac")
    stable_frac = f(row, "stable_frac")
    failed_count = f(row, "failed_count")
    zmin = f(row, "zmin_mean")
    below22 = f(row, "below22_mean")
    roll = f(row, "rollmax_mean")
    pitch = f(row, "pitchmax_mean")
    rgated = f(row, "R_gated_mean")

    # Conservative positive teacher: safe enough to imitate.
    if (
        valid_frac >= 1.0
        and stable_frac >= 0.50
        and failed_count <= 0
        and zmin >= 0.145
        and below22 <= 0.60
        and roll <= 0.75
        and pitch <= 0.60
        and rgated > 0.10
    ):
        return "positive_teacher"

    # Risk/negative bank: useful for RAM/GMS/risk, not for direct BC imitation.
    if (
        valid_frac < 1.0
        or failed_count > 0
        or stable_frac <= 0.0
        or zmin < 0.12
        or below22 >= 0.65
        or roll >= 0.90
        or pitch >= 0.75
    ):
        return "risk_negative"

    return "borderline"


def item_from_row(row: dict[str, Any], label: str) -> dict[str, Any]:
    beta_raw = row.get("beta", "")
    try:
        beta = json.loads(beta_raw)
    except Exception:
        beta = beta_raw

    return {
        "bank_label": label,
        "profile": row.get("profile"),
        "beta": beta,
        "rank": int(float(row.get("rank", 999))),
        "terrain": row.get("terrain"),
        "preset": row.get("preset"),
        "action_phase_a": {
            "vx": f(row, "ref_vx"),
            "yaw_rate": f(row, "ref_yaw_rate"),
            "body_height": f(row, "ref_body_height"),
            "swing_clearance": f(row, "ref_swing_clearance"),
            "enable": 1.0,
        },
        "metrics": {
            "n": int(float(row.get("n", 0))),
            "valid_frac": f(row, "valid_frac"),
            "stable_frac": f(row, "stable_frac"),
            "good_count": int(float(row.get("good_count", 0))),
            "failed_count": int(float(row.get("failed_count", 0))),
            "R_gated_mean": f(row, "R_gated_mean"),
            "R_gated_std": f(row, "R_gated_std"),
            "R_raw_mean": f(row, "R_raw_mean"),
            "R_motion_mean": f(row, "R_motion_mean"),
            "R_stability_mean": f(row, "R_stability_mean"),
            "R_energy_mean": f(row, "R_energy_mean"),
            "R_aux_mean": f(row, "R_aux_mean"),
            "distance_mean": f(row, "distance_mean"),
            "zmin_mean": f(row, "zmin_mean"),
            "zmean_mean": f(row, "zmean_mean"),
            "below22_mean": f(row, "below22_mean"),
            "rollmax_mean": f(row, "rollmax_mean"),
            "pitchmax_mean": f(row, "pitchmax_mean"),
        },
        "task_type": "forward_velocity_tracking_phase_a",
        "source_run_dir": row.get("source_run_dir"),
        "note": (
            "Merged Phase-A reference-sweep sample. "
            "Use positive_teacher for BC/offline warm-start; "
            "use risk_negative/borderline for RAM/GMS/risk and preference learning."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-glob",
        default="data/rollouts/reference_sweep_v0/ref_sweep_v0_*/reference_sweep_profile_rescore_phase_a_v0.csv",
    )
    ap.add_argument("--out-dir", default="data/phase_a_reference_banks_v0")
    ap.add_argument("--profiles", default="balanced,motion,stability,energy,motion_extreme,stability_extreme,energy_extreme")
    args = ap.parse_args()

    profiles = {x.strip() for x in args.profiles.split(",") if x.strip()}
    paths = [Path(p) for p in sorted(glob.glob(args.run_glob))]
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.run_glob}")

    all_rows: list[dict[str, Any]] = []
    for path in paths:
        for r in read_rows(path):
            if r.get("profile") not in profiles:
                continue
            r = dict(r)
            r["profile_rescore_csv"] = str(path)
            label = classify(r)
            r["bank_label"] = label
            all_rows.append(r)

    # Sort by class, profile, then descending gated score.
    class_order = {"positive_teacher": 0, "borderline": 1, "risk_negative": 2}
    all_rows.sort(
        key=lambda r: (
            class_order.get(r["bank_label"], 9),
            r.get("profile", ""),
            -f(r, "R_gated_mean"),
            r.get("preset", ""),
        )
    )

    positives = [item_from_row(r, "positive_teacher") for r in all_rows if r["bank_label"] == "positive_teacher"]
    borderlines = [item_from_row(r, "borderline") for r in all_rows if r["bank_label"] == "borderline"]
    risks = [item_from_row(r, "risk_negative") for r in all_rows if r["bank_label"] == "risk_negative"]

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "phase_a_reference_candidates_merged_v0.csv", all_rows)
    write_jsonl(out_dir / "phase_a_positive_teacher_bank_v0.jsonl", positives)
    write_jsonl(out_dir / "phase_a_borderline_bank_v0.jsonl", borderlines)
    write_jsonl(out_dir / "phase_a_risk_negative_bank_v0.jsonl", risks)

    print(f"[TRACER] source files: {len(paths)}")
    for p in paths:
        print(f"  - {p}")
    print()
    print(f"[TRACER] wrote: {out_dir / 'phase_a_reference_candidates_merged_v0.csv'}")
    print(f"[TRACER] positive_teacher: {len(positives)}")
    print(f"[TRACER] borderline:       {len(borderlines)}")
    print(f"[TRACER] risk_negative:    {len(risks)}")
    print()

    print("===== positive teachers =====")
    for item in positives[:20]:
        m = item["metrics"]
        a = item["action_phase_a"]
        print(
            f"{item['profile']:18s} {item['preset']:32s} "
            f"vx={a['vx']:.3f} h={a['body_height']:.3f} c={a['swing_clearance']:.3f} "
            f"Rgated={m['R_gated_mean']:.4f} stable={m['stable_frac']:.2f} "
            f"zmin={m['zmin_mean']:.3f} below22={m['below22_mean']:.3f}"
        )

    print()
    print("===== strongest risk negatives =====")
    for item in risks[:20]:
        m = item["metrics"]
        a = item["action_phase_a"]
        print(
            f"{item['profile']:18s} {item['preset']:32s} "
            f"vx={a['vx']:.3f} h={a['body_height']:.3f} c={a['swing_clearance']:.3f} "
            f"Rgated={m['R_gated_mean']:.4f} stable={m['stable_frac']:.2f} "
            f"zmin={m['zmin_mean']:.3f} below22={m['below22_mean']:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
