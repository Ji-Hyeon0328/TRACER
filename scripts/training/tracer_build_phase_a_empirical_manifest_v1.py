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

    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def action_key(row: dict[str, Any], prefix: str = "a_hl") -> tuple:
    return (
        str(row.get("terrain", "")),
        round(to_float(row.get(f"{prefix}_vx")), 6),
        round(to_float(row.get(f"{prefix}_yaw_rate")), 6),
        round(to_float(row.get(f"{prefix}_body_height")), 6),
        round(to_float(row.get(f"{prefix}_swing_clearance")), 6),
    )


def aggregate_key(row: dict[str, Any]) -> tuple:
    return (
        str(row.get("terrain", "")),
        round(to_float(row.get("ref_vx")), 6),
        round(to_float(row.get("ref_yaw_rate")), 6),
        round(to_float(row.get("ref_body_height")), 6),
        round(to_float(row.get("ref_swing_clearance")), 6),
    )


def empirical_teacher_weight(
    label: str,
    stable_prob: float,
    lcb: float,
    unc: float,
    n_episodes: int,
) -> float:
    # Evidence-aware teacher weight.
    # Low-n apparent successes should not dominate better-supported candidates.
    evidence_factor = 0.55 + 0.45 * min(1.0, max(0.0, n_episodes / 8.0))
    lcb_factor = 0.50 + 0.50 * min(1.0, max(0.0, lcb / 0.55))

    if label == "robust_positive":
        base = 1.0
    elif label == "probabilistic_positive":
        base = max(0.25, min(0.85, 0.55 * stable_prob + 0.45 * lcb - 0.25 * unc))
    elif label == "borderline_high_variance":
        base = max(0.10, min(0.45, 0.35 * stable_prob + 0.20 * lcb - 0.30 * unc))
    else:
        base = max(0.05, min(0.20, 0.25 * stable_prob - 0.25 * unc))

    return max(0.03, min(1.0, base * evidence_factor * lcb_factor))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest-csv",
        default="data/phase_a_training_manifest_v0/phase_a_training_manifest_v0.csv",
    )
    ap.add_argument(
        "--aggregate-csv",
        default="data/phase_a_action_outcome_aggregates_v1_evidence/phase_a_action_outcome_aggregates_v1_evidence.csv",
    )
    ap.add_argument("--out-dir", default="data/phase_a_empirical_manifest_v1")
    args = ap.parse_args()

    manifest_rows = read_csv(Path(args.manifest_csv))
    aggregate_rows = read_csv(Path(args.aggregate_csv))

    agg_by_key = {aggregate_key(r): r for r in aggregate_rows}

    out_rows: list[dict[str, Any]] = []
    missing = 0

    for r in manifest_rows:
        key = action_key(r, prefix="a_hl")
        agg = agg_by_key.get(key)

        out = dict(r)

        if agg is None:
            missing += 1
            out.update({
                "empirical_available": 0,
                "empirical_outcome_label": "missing_aggregate",
                "empirical_n_episodes": 0,
                "empirical_stable_prob": "",
                "empirical_stable_prob_lcb95": "",
                "empirical_recovery_prob": "",
                "empirical_uncertainty": "",
                "empirical_teacher_weight": 0.0,
            })
        else:
            stable_prob = to_float(agg.get("stable_prob"))
            lcb = to_float(agg.get("stable_prob_lcb95"))
            unc = to_float(agg.get("outcome_uncertainty"))
            label = str(agg.get("outcome_label", ""))

            n_episodes = int(to_float(agg.get("n_episodes")))

            out.update({
                "empirical_available": 1,
                "empirical_outcome_label": label,
                "empirical_n_episodes": n_episodes,
                "empirical_stable_successes": int(to_float(agg.get("stable_successes"))),
                "empirical_stable_prob": stable_prob,
                "empirical_stable_prob_lcb95": lcb,
                "empirical_recovery_prob": 1.0 - stable_prob,
                "empirical_uncertainty": unc,
                "empirical_zmin_mean": to_float(agg.get("zmin_mean")),
                "empirical_zmin_std": to_float(agg.get("zmin_std")),
                "empirical_below22_mean": to_float(agg.get("below22_mean")),
                "empirical_below22_std": to_float(agg.get("below22_std")),
                "empirical_rollmax_mean": to_float(agg.get("rollmax_mean")),
                "empirical_pitchmax_mean": to_float(agg.get("pitchmax_mean")),
                "empirical_teacher_weight": empirical_teacher_weight(label, stable_prob, lcb, unc, n_episodes),
            })

            # RAM v1 targets should prefer empirical distributional targets.
            out["ram_recovery_needed_target_v1_empirical"] = 1.0 - stable_prob
            out["ram_uncertainty_proxy_target_v1_empirical"] = unc
            out["ram_stable_prob_lcb95_target_v1_empirical"] = lcb

            # Meta-planner teacher usability should also be empirical.
            out["meta_planner_usable_teacher_v1_empirical"] = 1.0 if (
                label in {"robust_positive", "probabilistic_positive"}
                and stable_prob >= 0.60
                and unc <= 0.75
            ) else 0.0

        out_rows.append(out)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "phase_a_empirical_manifest_v1.csv"
    summary_path = out_dir / "phase_a_empirical_manifest_summary_v1.json"

    write_csv(csv_path, out_rows)

    counts: dict[str, int] = {}
    terrain_counts: dict[str, int] = {}

    for r in out_rows:
        label = str(r.get("empirical_outcome_label", "unknown"))
        terrain = str(r.get("terrain", "unknown"))
        counts[label] = counts.get(label, 0) + 1
        terrain_counts[f"{terrain}::{label}"] = terrain_counts.get(f"{terrain}::{label}", 0) + 1

    summary = {
        "manifest_csv": args.manifest_csv,
        "aggregate_csv": args.aggregate_csv,
        "num_rows": len(out_rows),
        "missing_aggregate_rows": missing,
        "counts": counts,
        "terrain_counts": terrain_counts,
        "outputs": {
            "csv": str(csv_path),
            "summary": str(summary_path),
        },
        "note": (
            "Empirical Phase-A manifest. Joins single-run manifest rows with evidence-aware "
            "aggregate outcome statistics for RAM uncertainty/recovery targets and meta-planner weighting."
        ),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print()
    print("===== empirical manifest samples =====")
    for r in sorted(
        out_rows,
        key=lambda x: float(x.get("empirical_teacher_weight") or 0.0),
        reverse=True,
    )[:30]:
        print(
            f"terrain={r.get('terrain',''):20s} "
            f"label={r.get('empirical_outcome_label',''):25s} "
            f"profile={r.get('profile',''):18s} "
            f"w={to_float(r.get('empirical_teacher_weight')):.3f} "
            f"stable_p={to_float(r.get('empirical_stable_prob')):.2f} "
            f"lcb={to_float(r.get('empirical_stable_prob_lcb95')):.2f} "
            f"unc={to_float(r.get('empirical_uncertainty')):.2f} "
            f"a=[{to_float(r.get('a_hl_vx')):.3f},"
            f"{to_float(r.get('a_hl_body_height')):.3f},"
            f"{to_float(r.get('a_hl_swing_clearance')):.3f}]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
