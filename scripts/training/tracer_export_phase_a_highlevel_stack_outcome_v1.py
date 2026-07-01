#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROFILE_TO_COL = {
    "balanced": "R_gated_balanced_mean",
    "motion": "R_gated_motion_mean",
    "stability": "R_gated_stability_mean",
    "energy": "R_gated_energy_mean",
    "motion_extreme": "R_gated_motion_extreme_mean",
    "stability_extreme": "R_gated_stability_extreme_mean",
    "energy_extreme": "R_gated_energy_extreme_mean",
}


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


def outcome_label_bonus(label: str) -> float:
    if label == "robust_positive":
        return 0.20
    if label == "probabilistic_positive":
        return 0.05
    if label == "borderline_high_variance":
        return -0.15
    if label == "risk_negative":
        return -0.50
    return 0.0


def score_candidate(
    row: dict[str, Any],
    objective_profile: str,
    *,
    uncertainty_penalty: float,
    below22_penalty: float,
    zmin_penalty: float,
) -> float:
    score_col = PROFILE_TO_COL.get(objective_profile, "R_gated_balanced_mean")
    base = to_float(row.get(score_col), to_float(row.get("R_gated_balanced_mean")))

    stable_p = to_float(row.get("stable_prob"))
    unc = to_float(row.get("outcome_uncertainty"))
    below22 = to_float(row.get("below22_mean"))
    zmin = to_float(row.get("zmin_mean"))
    label = str(row.get("outcome_label", ""))

    stability_gain = 0.50 + 0.50 * stable_p
    adjusted = base * stability_gain

    adjusted -= uncertainty_penalty * unc
    adjusted -= below22_penalty * below22
    adjusted -= zmin_penalty * max(0.0, 0.18 - zmin)
    adjusted += outcome_label_bonus(label)

    return adjusted


def select_rows(
    rows: list[dict[str, Any]],
    *,
    terrain: str,
    min_stable_prob: float,
    max_uncertainty: float,
    min_episodes: int,
    min_stable_lcb95: float,
    allow_labels: set[str],
) -> tuple[list[dict[str, Any]], str]:
    terrain_rows = [r for r in rows if str(r.get("terrain", "")) == terrain]
    if not terrain_rows:
        raise RuntimeError(f"No aggregate rows for terrain={terrain}")

    safe = []
    for r in terrain_rows:
        label = str(r.get("outcome_label", ""))
        stable_p = to_float(r.get("stable_prob"))
        unc = to_float(r.get("outcome_uncertainty"))
        n_episodes = int(to_float(r.get("n_episodes")))
        stable_lcb95 = to_float(r.get("stable_prob_lcb95"))

        if label not in allow_labels:
            continue
        if stable_p < min_stable_prob:
            continue
        if unc > max_uncertainty:
            continue
        if n_episodes < min_episodes:
            continue
        if stable_lcb95 < min_stable_lcb95:
            continue
        safe.append(r)

    if safe:
        return safe, "safe_filter_pass"

    # Fallback: do not hide the fact that we failed to find a safe aggregate candidate.
    # Use all terrain rows so the exported result is still inspectable.
    return terrain_rows, "fallback_no_safe_candidate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--objective-json", required=True)
    ap.add_argument(
        "--aggregate-csv",
        default="data/phase_a_action_outcome_aggregates_all_v0/phase_a_action_outcome_aggregates_v0.csv",
    )
    ap.add_argument("--min-stable-prob", type=float, default=0.80)
    ap.add_argument("--max-uncertainty", type=float, default=0.40)
    ap.add_argument("--min-episodes", type=int, default=1)
    ap.add_argument("--min-stable-lcb95", type=float, default=0.0)
    ap.add_argument(
        "--allow-labels",
        default="robust_positive,probabilistic_positive",
        help="Comma-separated aggregate outcome labels allowed by the safe filter.",
    )
    ap.add_argument("--uncertainty-penalty", type=float, default=0.40)
    ap.add_argument("--below22-penalty", type=float, default=0.60)
    ap.add_argument("--zmin-penalty", type=float, default=2.00)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--preset-name", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-rollout-yaml", required=True)
    args = ap.parse_args()

    objective = json.loads(Path(args.objective_json).read_text(encoding="utf-8"))
    selected_obj = objective["selected"]
    objective_profile = str(selected_obj.get("profile", "balanced"))

    rows = read_csv(Path(args.aggregate_csv))
    allow_labels = {x.strip() for x in args.allow_labels.split(",") if x.strip()}

    candidates, selection_status = select_rows(
        rows,
        terrain=args.terrain,
        min_stable_prob=args.min_stable_prob,
        max_uncertainty=args.max_uncertainty,
        min_episodes=args.min_episodes,
        min_stable_lcb95=args.min_stable_lcb95,
        allow_labels=allow_labels,
    )

    ranked = []
    for r in candidates:
        adjusted = score_candidate(
            r,
            objective_profile,
            uncertainty_penalty=args.uncertainty_penalty,
            below22_penalty=args.below22_penalty,
            zmin_penalty=args.zmin_penalty,
        )
        item = dict(r)
        item["objective_profile"] = objective_profile
        item["adjusted_score"] = adjusted
        ranked.append(item)

    ranked.sort(key=lambda r: float(r["adjusted_score"]), reverse=True)
    best = ranked[0]

    vx = to_float(best.get("ref_vx"))
    yaw = to_float(best.get("ref_yaw_rate"))
    h = to_float(best.get("ref_body_height"))
    c = to_float(best.get("ref_swing_clearance"))

    preset_name = args.preset_name or f"phase_a_stack_outcome_{args.terrain}_{objective_profile}_v1"

    result = {
        "terrain": args.terrain,
        "objective_json": args.objective_json,
        "objective_selected": selected_obj,
        "objective_profile": objective_profile,
        "aggregate_csv": args.aggregate_csv,
        "selection_status": selection_status,
        "safe_filter": {
            "min_stable_prob": args.min_stable_prob,
            "max_uncertainty": args.max_uncertainty,
            "min_episodes": args.min_episodes,
            "min_stable_lcb95": args.min_stable_lcb95,
            "allow_labels": sorted(allow_labels),
        },
        "selected_action": {
            "preset": preset_name,
            "vx": vx,
            "yaw_rate": yaw,
            "body_height": h,
            "swing_clearance": c,
            "enable": 1.0,
            "adjusted_score": to_float(best.get("adjusted_score")),
            "outcome_label": best.get("outcome_label", ""),
            "stable_prob": to_float(best.get("stable_prob")),
            "outcome_uncertainty": to_float(best.get("outcome_uncertainty")),
            "zmin_mean": to_float(best.get("zmin_mean")),
            "below22_mean": to_float(best.get("below22_mean")),
            "n_episodes": int(to_float(best.get("n_episodes"))),
            "source_presets_json": best.get("source_presets_json", "[]"),
            "source_runs_json": best.get("source_runs_json", "[]"),
        },
        "ranked_candidates": [
            {
                "adjusted_score": to_float(r.get("adjusted_score")),
                "outcome_label": r.get("outcome_label", ""),
                "stable_prob": to_float(r.get("stable_prob")),
                "outcome_uncertainty": to_float(r.get("outcome_uncertainty")),
                "zmin_mean": to_float(r.get("zmin_mean")),
                "below22_mean": to_float(r.get("below22_mean")),
                "n_episodes": int(to_float(r.get("n_episodes"))),
                "vx": to_float(r.get("ref_vx")),
                "yaw_rate": to_float(r.get("ref_yaw_rate")),
                "body_height": to_float(r.get("ref_body_height")),
                "swing_clearance": to_float(r.get("ref_swing_clearance")),
                "source_presets_json": r.get("source_presets_json", "[]"),
            }
            for r in ranked[: args.top_k]
        ],
        "note": (
            "Outcome-aware Phase-A high-level stack export. Uses aggregate stable probability "
            "and outcome uncertainty to avoid single-run high-variance teachers."
        ),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    yaml_text = f"""version: phase_a_highlevel_stack_outcome_v1
description: >
  Exported by guarded Objective Selector + aggregate outcome-aware action selection.
  This avoids high-variance single-run teacher actions.

terrains:
  - {args.terrain}

presets:
  - name: {preset_name}
    vx: {vx:.6f}
    yaw_rate: {yaw:.6f}
    body_height: {h:.6f}
    swing_clearance: {c:.6f}
    enable: 1.0
"""
    out_yaml = Path(args.out_rollout_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml_text, encoding="utf-8")

    print(f"[TRACER] terrain={args.terrain}")
    print(f"[TRACER] objective_profile={objective_profile}")
    print(f"[TRACER] selection_status={selection_status}")
    print(
        f"[TRACER] selected: {preset_name} "
        f"a=[{vx:.3f},{yaw:.3f},{h:.3f},{c:.3f}] "
        f"label={best.get('outcome_label','')} "
        f"stable_p={to_float(best.get('stable_prob')):.2f} "
        f"unc={to_float(best.get('outcome_uncertainty')):.2f} "
        f"z={to_float(best.get('zmin_mean')):.3f} "
        f"below22={to_float(best.get('below22_mean')):.3f} "
        f"score={to_float(best.get('adjusted_score')):.4f}"
    )

    print()
    print("===== ranked candidates =====")
    for i, r in enumerate(result["ranked_candidates"], start=1):
        print(
            f"{i:02d}. score={r['adjusted_score']:.4f} "
            f"label={r['outcome_label']:25s} "
            f"stable_p={r['stable_prob']:.2f} "
            f"unc={r['outcome_uncertainty']:.2f} "
            f"z={r['zmin_mean']:.3f} "
            f"below22={r['below22_mean']:.3f} "
            f"n={r['n_episodes']:3d} "
            f"a=[{r['vx']:.3f},{r['body_height']:.3f},{r['swing_clearance']:.3f}]"
        )

    print()
    print(f"[TRACER] wrote json: {out_json}")
    print(f"[TRACER] wrote yaml: {out_yaml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
