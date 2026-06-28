#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED = {
    ("flat_normal", "motion_objective"): {
        "expected_style": "fast",
        "expected_vx": 0.280,
    },
    ("flat_normal", "deploy_objective"): {
        "expected_style": "cautious",
        "expected_vx": 0.055,
    },
    ("rough_mid", "motion_objective"): {
        "expected_style": "fast",
        "expected_vx": 0.280,
    },
    ("rough_mid", "deploy_objective"): {
        "expected_style": "high_clearance",
        "expected_vx": 0.045,
    },
    ("slope_5deg", "motion_objective"): {
        "expected_style": "fast",
        "expected_vx": 0.280,
    },
    ("slope_5deg", "deploy_objective"): {
        "expected_style": "high_clearance",
        "expected_vx": 0.045,
    },
}


def infer_objective(policy_id: str) -> str:
    if "motion_objective" in policy_id:
        return "motion_objective"
    if "stability_objective" in policy_id:
        return "stability_objective"
    if "deploy_objective" in policy_id:
        return "deploy_objective"
    return "unknown"


def infer_conf_mode(policy_id: str) -> str:
    if "force" in policy_id:
        return "force_min_conf_0p0"
    return "safe_min_conf_0p05"


def infer_observed_style(vx: float, h: float, clr: float) -> str:
    if vx >= 0.20:
        return "fast"
    if h >= 0.33 or clr >= 0.08:
        return "high_clearance"
    if vx <= 0.08:
        return "cautious"
    return "unknown"


def close(a: float, b: float, tol: float = 0.012) -> bool:
    return abs(float(a) - float(b)) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary-glob",
        default="data/rollout_dataset_v0/summaries/*objective_conditioned*matrix_v0*.json",
    )
    ap.add_argument(
        "--extra-glob",
        default="data/rollout_dataset_v0/summaries/*objective_conditioned_selector_apply_rough_deploy_force_v0*.json",
    )
    ap.add_argument(
        "--out-csv",
        default="reports/tracer_objective_conditioned_runtime_matrix_v0.csv",
    )
    ap.add_argument(
        "--out-json",
        default="reports/tracer_objective_conditioned_runtime_matrix_v0_summary.json",
    )
    args = ap.parse_args()

    paths = list(Path(".").glob(args.summary_glob))
    paths += list(Path(".").glob(args.extra_glob))
    paths = sorted(set(paths))

    rows = []
    for p in paths:
        d = json.loads(p.read_text())
        terrain = d.get("terrain", "unknown")
        policy_id = d.get("policy_id", "")
        objective = infer_objective(policy_id)
        conf_mode = infer_conf_mode(policy_id)

        vx = float(d.get("mpc_vx_mean", 0.0))
        h = float(d.get("mpc_body_height_mean", 0.0))
        clr = float(d.get("mpc_clearance_mean", 0.0))
        observed_style = infer_observed_style(vx, h, clr)

        exp = EXPECTED.get((terrain, objective), {})
        expected_style = exp.get("expected_style", "unknown")
        expected_vx = float(exp.get("expected_vx", -999.0))

        routing_ok = (
            expected_style != "unknown"
            and observed_style == expected_style
            and close(vx, expected_vx)
        )

        rows.append(
            {
                "terrain": terrain,
                "objective": objective,
                "confidence_mode": conf_mode,
                "policy_id": policy_id,
                "observed_style": observed_style,
                "expected_style": expected_style,
                "routing_ok": int(routing_ok),
                "mpc_vx_mean": vx,
                "expected_vx": expected_vx,
                "mpc_body_height_mean": h,
                "mpc_clearance_mean": clr,
                "success_proxy": d.get("success_proxy"),
                "distance_xy_proxy": d.get("distance_xy_proxy"),
                "fresh1": d.get("debug_fresh_rate_1p0s"),
                "fallen_p90": d.get("ram_run_fallen_p90"),
                "summary_json": str(p),
            }
        )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "terrain",
        "objective",
        "confidence_mode",
        "policy_id",
        "observed_style",
        "expected_style",
        "routing_ok",
        "mpc_vx_mean",
        "expected_vx",
        "mpc_body_height_mean",
        "mpc_clearance_mean",
        "success_proxy",
        "distance_xy_proxy",
        "fresh1",
        "fallen_p90",
        "summary_json",
    ]

    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n = len(rows)
    ok = sum(int(r["routing_ok"]) for r in rows)
    summary = {
        "n": n,
        "routing_ok": ok,
        "routing_rate": ok / n if n else 0.0,
        "rows": rows,
    }
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"[TRACER] rows={n} routing_ok={ok}/{n}")
    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_json}")
    print()

    for r in rows:
        print(
            f"{r['terrain']:12s} {r['objective']:18s} "
            f"{r['confidence_mode']:18s} "
            f"obs={r['observed_style']:14s} exp={r['expected_style']:14s} "
            f"ok={r['routing_ok']} "
            f"vx={float(r['mpc_vx_mean']):.3f} "
            f"h={float(r['mpc_body_height_mean']):.3f} "
            f"clr={float(r['mpc_clearance_mean']):.3f} "
            f"success={r['success_proxy']}"
        )


if __name__ == "__main__":
    main()
