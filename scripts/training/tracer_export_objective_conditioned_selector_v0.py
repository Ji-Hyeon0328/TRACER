#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


OBJECTIVE_BETA = {
    "motion_objective": {
        "motion": 0.70,
        "stability": 0.20,
        "energy": 0.10,
    },
    "stability_objective": {
        "motion": 0.15,
        "stability": 0.75,
        "energy": 0.10,
    },
    "deploy_objective": {
        "motion": 0.34,
        "stability": 0.56,
        "energy": 0.10,
    },
}


STYLE_TO_SEMANTIC = {
    "fast": {
        "semantic_mode": "validated_locomotion",
        "suggested_style": "fast",
        "fused_mode": "locomotion",
    },
    "cautious": {
        "semantic_mode": "cautious_probe",
        "suggested_style": "cautious",
        "fused_mode": "locomotion",
    },
    "high_clearance": {
        "semantic_mode": "high_clearance_slow_probe",
        "suggested_style": "high_clearance",
        "fused_mode": "locomotion",
    },
    "conservative": {
        "semantic_mode": "cautious_locomotion",
        "suggested_style": "cautious",
        "fused_mode": "locomotion",
    },
}


DEFAULT_STYLE_BY_TERRAIN = {
    "flat_normal": "fast",
    "rough_mid": "cautious",
    "slope_5deg": "high_clearance",
}


def f(x, default=0.0):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def normalize_beta(beta):
    m = max(0.0, f(beta.get("motion", beta.get("beta_motion", 0.0))))
    s = max(0.0, f(beta.get("stability", beta.get("beta_stability", 0.0))))
    e = max(0.0, f(beta.get("energy", beta.get("beta_energy", 0.0))))
    z = m + s + e
    if z <= 1e-9:
        return {"motion": 0.34, "stability": 0.56, "energy": 0.10}
    return {"motion": m / z, "stability": s / z, "energy": e / z}


def beta_l2(a, b):
    aa = normalize_beta(a)
    bb = normalize_beta(b)
    return math.sqrt(
        (aa["motion"] - bb["motion"]) ** 2
        + (aa["stability"] - bb["stability"]) ** 2
        + (aa["energy"] - bb["energy"]) ** 2
    )


def nearest_objective_profile(beta):
    best_name = None
    best_dist = 1e9
    for name, prof in OBJECTIVE_BETA.items():
        d = beta_l2(beta, prof)
        if d < best_dist:
            best_name = name
            best_dist = d
    return best_name or "deploy_objective", best_dist


def vote_weight(row):
    # Confidence already includes margin softness.
    # Add a small floor so weak-but-valid pairs still contribute.
    conf = f(row.get("label_confidence"), 0.0)
    margin = f(row.get("margin"), 0.0)
    return max(0.05, conf) * max(0.01, margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs-csv",
        default="data/preference_datasets/tracer_objective_conditioned_style_pairs_3x3_v0.csv",
    )
    ap.add_argument(
        "--out-json",
        default="configs/highlevel_policy/tracer_objective_conditioned_selector_v0.json",
    )
    ap.add_argument(
        "--out-report",
        default="reports/tracer_objective_conditioned_selector_v0_report.csv",
    )
    args = ap.parse_args()

    rows = []
    with open(args.pairs_csv, newline="") as fp:
        for r in csv.DictReader(fp):
            rows.append(r)

    # Pairwise voting:
    # winner gets +weight, loser gets -weight under each terrain/objective.
    votes = defaultdict(lambda: defaultdict(float))
    evidence = defaultdict(list)

    for r in rows:
        terrain = r["terrain"]
        objective = r["objective_name"]
        winner = r["winner_style"]
        loser = r["loser_style"]
        key = (terrain, objective)
        w = vote_weight(r)

        votes[key][winner] += w
        votes[key][loser] -= w
        evidence[key].append(
            {
                "winner_style": winner,
                "loser_style": loser,
                "margin": f(r.get("margin")),
                "label_confidence": f(r.get("label_confidence")),
                "weight": w,
            }
        )

    terrain_objective_rules = {}
    report_rows = []

    terrains = sorted({r["terrain"] for r in rows})
    objectives = list(OBJECTIVE_BETA.keys())

    for terrain in terrains:
        terrain_objective_rules[terrain] = {}

        for objective in objectives:
            key = (terrain, objective)
            style_scores = dict(votes.get(key, {}))

            if not style_scores:
                # Missing/ambiguous objective for this terrain.
                fallback = DEFAULT_STYLE_BY_TERRAIN.get(terrain, "cautious")
                style_scores = {fallback: 0.0}
                reason = "fallback_no_pair_evidence"
                confidence = 0.0
            else:
                reason = "pairwise_vote"
                ordered_scores = sorted(style_scores.values(), reverse=True)
                if len(ordered_scores) >= 2:
                    confidence = max(0.0, ordered_scores[0] - ordered_scores[1])
                else:
                    confidence = max(0.0, ordered_scores[0])

            best_style = max(style_scores.items(), key=lambda kv: kv[1])[0]
            semantic = STYLE_TO_SEMANTIC.get(best_style, STYLE_TO_SEMANTIC["cautious"])

            rule = {
                "objective_name": objective,
                "beta_profile": OBJECTIVE_BETA[objective],
                "selected_style": best_style,
                "semantic_mode": semantic["semantic_mode"],
                "suggested_style": semantic["suggested_style"],
                "fused_mode": semantic["fused_mode"],
                "vote_scores": style_scores,
                "vote_confidence": confidence,
                "reason": reason,
                "evidence": evidence.get(key, []),
            }
            terrain_objective_rules[terrain][objective] = rule

            report_rows.append(
                {
                    "terrain": terrain,
                    "objective_name": objective,
                    "selected_style": best_style,
                    "semantic_mode": semantic["semantic_mode"],
                    "suggested_style": semantic["suggested_style"],
                    "vote_confidence": confidence,
                    "reason": reason,
                    "vote_scores_json": json.dumps(style_scores, sort_keys=True),
                }
            )

    selector = {
        "version": "tracer_objective_conditioned_selector_v0",
        "source_pairs_csv": args.pairs_csv,
        "selection_method": {
            "objective_profile_assignment": "nearest_beta_profile_l2",
            "style_selection": "pairwise_vote_per_terrain_objective",
            "runtime_note": (
                "Given terrain_key and beta, normalize beta, choose nearest objective profile, "
                "then return the selected semantic/style rule for that terrain/objective."
            ),
        },
        "objective_profiles": OBJECTIVE_BETA,
        "style_to_semantic": STYLE_TO_SEMANTIC,
        "default_style_by_terrain": DEFAULT_STYLE_BY_TERRAIN,
        "terrain_objective_rules": terrain_objective_rules,
    }

    out_json = Path(args.out_json)
    out_report = Path(args.out_report)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(selector, indent=2, sort_keys=True))

    fields = [
        "terrain",
        "objective_name",
        "selected_style",
        "semantic_mode",
        "suggested_style",
        "vote_confidence",
        "reason",
        "vote_scores_json",
    ]
    with out_report.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    print(f"[TRACER] read pairs={len(rows)}")
    print(f"[TRACER] wrote {out_json}")
    print(f"[TRACER] wrote {out_report}")
    print()

    for terrain in terrains:
        print(f"[{terrain}]")
        for objective in objectives:
            rule = terrain_objective_rules[terrain][objective]
            print(
                f"  {objective:20s} -> "
                f"{rule['selected_style']:15s} "
                f"semantic={rule['semantic_mode']:28s} "
                f"conf={rule['vote_confidence']:.3f} "
                f"reason={rule['reason']}"
            )

    print()
    print("[beta routing smoke]")
    smoke_betas = [
        {"motion": 0.70, "stability": 0.20, "energy": 0.10},
        {"motion": 0.15, "stability": 0.75, "energy": 0.10},
        {"motion": 0.34, "stability": 0.56, "energy": 0.10},
    ]
    for beta in smoke_betas:
        obj, dist = nearest_objective_profile(beta)
        print(f"  beta={beta} -> {obj} dist={dist:.3f}")


if __name__ == "__main__":
    main()
