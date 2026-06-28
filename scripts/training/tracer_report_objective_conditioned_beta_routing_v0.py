#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


def infer_style(vx: float, h: float, clr: float) -> str:
    if vx >= 0.20:
        return "fast"
    if h >= 0.33 or clr >= 0.08:
        return "high_clearance"
    if vx <= 0.08:
        return "cautious"
    return "unknown"


def main():
    files = sorted(Path("data/rollout_dataset_v0/summaries").glob("*objective_conditioned_beta_routing_v0*.json"))

    rows = []
    for p in files:
        d = json.loads(p.read_text())
        vx = float(d.get("mpc_vx_mean", 0.0))
        h = float(d.get("mpc_body_height_mean", 0.0))
        clr = float(d.get("mpc_clearance_mean", 0.0))
        terrain = d.get("terrain", "unknown")
        style = infer_style(vx, h, clr)

        # In this beta-routing run, runtime logs showed objective=deploy_objective
        # for flat_normal, rough_mid, and slope_5deg with override=<none>.
        objective = "deploy_objective"

        note = ""
        if terrain == "rough_mid" and style == "cautious":
            note = "selector selected high_clearance but confidence gate kept cautious"
        elif terrain == "flat_normal" and style == "cautious":
            note = "selector applied cautious for deploy objective"
        elif terrain == "slope_5deg" and style == "high_clearance":
            note = "selector applied or preserved high_clearance for deploy objective"

        rows.append(
            {
                "terrain": terrain,
                "policy_id": d.get("policy_id", ""),
                "inferred_objective": objective,
                "observed_style": style,
                "mpc_vx_mean": vx,
                "mpc_body_height_mean": h,
                "mpc_clearance_mean": clr,
                "success_proxy": d.get("success_proxy"),
                "distance_xy_proxy": d.get("distance_xy_proxy"),
                "fresh1": d.get("debug_fresh_rate_1p0s"),
                "fallen_p90": d.get("ram_run_fallen_p90"),
                "note": note,
                "summary_json": str(p),
            }
        )

    out_csv = Path("reports/tracer_objective_conditioned_beta_routing_v0.csv")
    out_json = Path("reports/tracer_objective_conditioned_beta_routing_v0_summary.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "terrain",
        "policy_id",
        "inferred_objective",
        "observed_style",
        "mpc_vx_mean",
        "mpc_body_height_mean",
        "mpc_clearance_mean",
        "success_proxy",
        "distance_xy_proxy",
        "fresh1",
        "fallen_p90",
        "note",
        "summary_json",
    ]

    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "n": len(rows),
        "objective_override": "none",
        "observed_runtime_objective": "deploy_objective",
        "note": (
            "Runtime logs showed override=<none> and objective=deploy_objective. "
            "This indicates the current beta profile routes to the deploy objective profile."
        ),
        "rows": rows,
    }
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"[TRACER] rows={len(rows)}")
    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_json}")
    print()
    for r in rows:
        print(
            f"{r['terrain']:12s} "
            f"objective={r['inferred_objective']:18s} "
            f"style={r['observed_style']:14s} "
            f"vx={float(r['mpc_vx_mean']):.3f} "
            f"h={float(r['mpc_body_height_mean']):.3f} "
            f"clr={float(r['mpc_clearance_mean']):.3f} "
            f"success={r['success_proxy']} "
            f"note={r['note']}"
        )


if __name__ == "__main__":
    main()
