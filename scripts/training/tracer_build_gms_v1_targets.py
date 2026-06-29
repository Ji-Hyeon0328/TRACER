#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


CAMPAIGN = "tracer_ram_gms_data_v2_core_hard"

IN_GMS = Path(f"reports/{CAMPAIGN}_gms_style_targets.csv")
OUT_CSV = Path("data/gms/tracer_gms_v1_risk_aware_targets.csv")
OUT_JSONL = Path("data/gms/tracer_gms_v1_risk_aware_targets.jsonl")
OUT_SUMMARY = Path("reports/tracer_gms_v1_risk_aware_targets_summary.json")


BETA_BY_MISSION = {
    "motion": (0.70, 0.20, 0.10),
    "deploy": (0.34, 0.56, 0.10),
    "stability": (0.15, 0.75, 0.10),
}

BETA_BY_RISK = {
    "low": None,
    "mid": (0.15, 0.75, 0.10),
    "high": (0.08, 0.82, 0.10),
}

RISK_VALUE = {
    "low": 0.05,
    "mid": 0.35,
    "high": 0.75,
}

RECOVERY_VALUE = {
    "low": 0.0,
    "mid": 0.0,
    "high": 1.0,
}


def safety_style_for_terrain(terrain: str, mission: str, fallback: str) -> str:
    t = terrain.lower()

    # For currently known deformable downslope, cautious was the best deploy/stability
    # label in v2 aggregate, while fast remained motion-best but unstable.
    if "sponge" in t:
        return "cautious"

    # Slippery/downhill should not use fast under mid/high risk.
    if "slippery" in t:
        return "high_clearance"

    # Rough/slope high risk: lift foot and slow down.
    if "rough" in t or "slope" in t:
        return "high_clearance"

    # Flat but high RAM risk: slow cautious probe is safer than fast.
    return "cautious"


def main() -> None:
    with IN_GMS.open(newline="") as fp:
        base_rows = list(csv.DictReader(fp))

    rows = []

    for base in base_rows:
        terrain = base["terrain"]

        empirical = {
            "motion": base["gms_motion_style"],
            "deploy": base["gms_deploy_style"],
            "stability": base["gms_stability_style"],
        }

        for mission in ["motion", "deploy", "stability"]:
            for risk_level in ["low", "mid", "high"]:
                beta = BETA_BY_RISK[risk_level] or BETA_BY_MISSION[mission]
                bm, bs, be = beta

                raw_style = empirical[mission]

                if risk_level == "low":
                    target_style = raw_style
                    source = "empirical_low_risk"
                else:
                    if raw_style == "fast":
                        target_style = safety_style_for_terrain(terrain, mission, raw_style)
                        source = f"risk_{risk_level}_override_fast"
                    else:
                        target_style = raw_style
                        source = f"risk_{risk_level}_keep_nonfast"

                rows.append({
                    "terrain": terrain,
                    "mission": mission,
                    "risk_level": risk_level,
                    "ram_risk": RISK_VALUE[risk_level],
                    "recovery_needed": RECOVERY_VALUE[risk_level],
                    "beta_motion": bm,
                    "beta_stability": bs,
                    "beta_energy": be,
                    "raw_empirical_style": raw_style,
                    "target_style": target_style,
                    "target_source": source,
                })

    fields = [
        "terrain",
        "mission",
        "risk_level",
        "ram_risk",
        "recovery_needed",
        "beta_motion",
        "beta_stability",
        "beta_energy",
        "raw_empirical_style",
        "target_style",
        "target_source",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with OUT_JSONL.open("w") as fp:
        for r in rows:
            fp.write(json.dumps(r) + "\n")

    counts = {}
    for r in rows:
        key = (r["terrain"], r["mission"], r["risk_level"], r["target_style"])
        counts[str(key)] = counts.get(str(key), 0) + 1

    summary = {
        "input": str(IN_GMS),
        "rows": len(rows),
        "terrains": sorted({r["terrain"] for r in rows}),
        "missions": ["motion", "deploy", "stability"],
        "risk_levels": ["low", "mid", "high"],
        "styles": sorted({r["target_style"] for r in rows}),
        "output_csv": str(OUT_CSV),
        "output_jsonl": str(OUT_JSONL),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    print("[TRACER] GMS v1 risk-aware targets")
    print(json.dumps(summary, indent=2))

    for r in rows:
        print(
            f"{r['terrain']:38s} "
            f"mission={r['mission']:9s} "
            f"risk={r['risk_level']:4s} "
            f"raw={r['raw_empirical_style']:14s} "
            f"target={r['target_style']:14s} "
            f"source={r['target_source']}"
        )


if __name__ == "__main__":
    main()
