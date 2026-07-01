#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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

    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def action_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("terrain", ""),
        row.get("preset", ""),
        round(to_float(row.get("a_hl_vx")), 6),
        round(to_float(row.get("a_hl_yaw_rate")), 6),
        round(to_float(row.get("a_hl_body_height")), 6),
        round(to_float(row.get("a_hl_swing_clearance")), 6),
    )


def context_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "terrain": row.get("terrain", ""),
        "actual_world_name": row.get("actual_world_name", ""),
        "a_hl_vx": to_float(row.get("a_hl_vx")),
        "a_hl_yaw_rate": to_float(row.get("a_hl_yaw_rate")),
        "a_hl_body_height": to_float(row.get("a_hl_body_height")),
        "a_hl_swing_clearance": to_float(row.get("a_hl_swing_clearance")),
        "stable_frac": to_float(row.get("stable_frac")),
        "zmin_mean": to_float(row.get("zmin_mean")),
        "below22_mean": to_float(row.get("below22_mean")),
        "distance_mean": to_float(row.get("distance_mean")),
        "rollmax_mean": to_float(row.get("rollmax_mean")),
        "pitchmax_mean": to_float(row.get("pitchmax_mean")),
        "ram_vertical_sink_target": to_float(row.get("ram_vertical_sink_target")),
        "ram_progress_mismatch_target": to_float(row.get("ram_progress_mismatch_target")),
        "ram_orientation_risk_target": to_float(row.get("ram_orientation_risk_target")),
        "ram_low_height_risk_target": to_float(row.get("ram_low_height_risk_target")),
        "ram_recovery_needed_target": to_float(row.get("ram_recovery_needed_target")),
        "ram_uncertainty_proxy_target": to_float(row.get("ram_uncertainty_proxy_target")),
        "manifest_label": row.get("manifest_label", ""),
    }


def beta_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": row.get("profile", ""),
        "beta_motion": to_float(row.get("beta_motion")),
        "beta_stability": to_float(row.get("beta_stability")),
        "beta_energy": to_float(row.get("beta_energy")),
        "score": to_float(row.get("R_score")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest-csv",
        default="data/phase_a_training_manifest_v0/phase_a_training_manifest_v0.csv",
    )
    ap.add_argument("--out-dir", default="data/phase_a_objective_preferences_v0")
    ap.add_argument("--score-margin", type=float, default=0.05)
    ap.add_argument("--include-risk", action="store_true")
    args = ap.parse_args()

    rows = read_csv(Path(args.manifest_csv))
    if not rows:
        raise RuntimeError(f"No rows found: {args.manifest_csv}")

    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        label = str(r.get("manifest_label", ""))
        if (not args.include_risk) and label == "risk_recovery":
            continue
        groups[action_key(r)].append(r)

    pairs: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []

    for key, grp in groups.items():
        # Need at least two objective profiles for same action/context.
        if len(grp) < 2:
            continue

        # Deduplicate profile, keep highest score if repeated.
        by_profile: dict[str, dict[str, Any]] = {}
        for r in grp:
            profile = str(r.get("profile", ""))
            if profile not in by_profile or to_float(r.get("R_score")) > to_float(by_profile[profile].get("R_score")):
                by_profile[profile] = r

        prof_rows = list(by_profile.values())
        if len(prof_rows) < 2:
            continue

        prof_rows.sort(key=lambda r: to_float(r.get("R_score")), reverse=True)
        best = prof_rows[0]
        rankings.append({
            **context_payload(best),
            "num_profiles": len(prof_rows),
            "best_profile": best.get("profile", ""),
            "best_beta_motion": to_float(best.get("beta_motion")),
            "best_beta_stability": to_float(best.get("beta_stability")),
            "best_beta_energy": to_float(best.get("beta_energy")),
            "best_score": to_float(best.get("R_score")),
            "ranked_profiles_json": json.dumps([beta_payload(r) for r in prof_rows]),
        })

        for i in range(len(prof_rows)):
            for j in range(i + 1, len(prof_rows)):
                a = prof_rows[i]
                b = prof_rows[j]
                sa = to_float(a.get("R_score"))
                sb = to_float(b.get("R_score"))
                diff = sa - sb

                if abs(diff) < args.score_margin:
                    continue

                # Since rows are sorted descending, a preferred over b.
                pair = {
                    **context_payload(a),
                    "preferred_profile": a.get("profile", ""),
                    "preferred_beta_motion": to_float(a.get("beta_motion")),
                    "preferred_beta_stability": to_float(a.get("beta_stability")),
                    "preferred_beta_energy": to_float(a.get("beta_energy")),
                    "preferred_score": sa,
                    "rejected_profile": b.get("profile", ""),
                    "rejected_beta_motion": to_float(b.get("beta_motion")),
                    "rejected_beta_stability": to_float(b.get("beta_stability")),
                    "rejected_beta_energy": to_float(b.get("beta_energy")),
                    "rejected_score": sb,
                    "score_diff": diff,
                    "pair_weight": min(1.0, max(0.1, diff)),
                }
                pairs.append(pair)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_csv = out_dir / "phase_a_objective_preference_pairs_v0.csv"
    pairs_jsonl = out_dir / "phase_a_objective_preference_pairs_v0.jsonl"
    ranking_csv = out_dir / "phase_a_objective_rankings_v0.csv"
    summary_path = out_dir / "phase_a_objective_preference_summary_v0.json"

    write_csv(pairs_csv, pairs)
    write_jsonl(pairs_jsonl, pairs)
    write_csv(ranking_csv, rankings)

    terrain_counts: dict[str, int] = {}
    preferred_counts: dict[str, int] = {}
    for p in pairs:
        terrain = str(p.get("terrain", "unknown"))
        pref = str(p.get("preferred_profile", "unknown"))
        terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
        preferred_counts[pref] = preferred_counts.get(pref, 0) + 1

    summary = {
        "manifest_csv": args.manifest_csv,
        "num_manifest_rows": len(rows),
        "num_groups": len(groups),
        "num_rankings": len(rankings),
        "num_pairs": len(pairs),
        "score_margin": args.score_margin,
        "include_risk": args.include_risk,
        "terrain_pair_counts": terrain_counts,
        "preferred_profile_counts": preferred_counts,
        "outputs": {
            "pairs_csv": str(pairs_csv),
            "pairs_jsonl": str(pairs_jsonl),
            "ranking_csv": str(ranking_csv),
            "summary": str(summary_path),
        },
        "note": (
            "Bootstrapped objective preference pairs for TRACER Objective Selector. "
            "Pairs compare beta profiles for the same terrain/action/context."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print()
    print("===== top pairs =====")
    for p in sorted(pairs, key=lambda x: float(x["score_diff"]), reverse=True)[:30]:
        print(
            f"terrain={p['terrain']:20s} "
            f"label={p['manifest_label']:18s} "
            f"a=[{p['a_hl_vx']:.3f},{p['a_hl_body_height']:.3f},{p['a_hl_swing_clearance']:.3f}] "
            f"pref={p['preferred_profile']:18s}({p['preferred_score']:.3f}) "
            f"> rej={p['rejected_profile']:18s}({p['rejected_score']:.3f}) "
            f"diff={p['score_diff']:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
