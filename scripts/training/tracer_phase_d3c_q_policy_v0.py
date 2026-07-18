#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def parse_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in ["true", "1", "yes"]


def ffloat(row, key, default=0.0):
    v = row.get(key, "")
    if v in ["", "NA", "None", None]:
        return default
    return float(v)


def parse_schedule(schedule):
    chunks = [c.strip() for c in schedule.split(";") if c.strip()]
    p0 = [x.strip() for x in chunks[0].split(",")]
    p1 = [x.strip() for x in chunks[1].split(",")]
    return {
        "vx_start": float(p0[2]),
        "body_h": float(p0[3]),
        "clearance": float(p0[4]),
        "vx_tail": float(p1[2]),
    }


def component_scores(row, goal_x, lateral_bound, timeout_s):
    a = parse_schedule(row["schedule"])

    success = parse_bool(row.get("success", False))
    goal = parse_bool(row.get("goal", False))
    out_lane = parse_bool(row.get("out_lane", False))
    startup_failed = parse_bool(row.get("startup_failed", False))

    final_x = ffloat(row, "final_x", 0.0)
    max_abs_y = ffloat(row, "max_abs_y", 0.0)
    mean_abs_y = ffloat(row, "mean_abs_y", 0.0)

    time_raw = row.get("time", "")
    time_s = timeout_s if time_raw in ["", "NA", "None", None] else float(time_raw)

    progress = max(0.0, min(final_x / goal_x, 1.0))
    time_eff = max(0.0, 1.0 - min(time_s / timeout_s, 1.0))

    vx_start_norm = (a["vx_start"] - 0.195) / (0.225 - 0.195)
    vx_tail_norm = (a["vx_tail"] - 0.190) / (0.210 - 0.190)
    h_dev = (a["body_h"] - 0.320) / max(0.324 - 0.318, 1e-9)
    c_dev = (a["clearance"] - 0.045) / max(0.050 - 0.043, 1e-9)

    motion_score = (
        55.0 * progress
        + 35.0 * time_eff
        + 10.0 * vx_start_norm
        + 6.0 * vx_tail_norm
    )

    stability_score = (
        90.0 * float(not out_lane)
        - 40.0 * (max_abs_y / lateral_bound)
        - 25.0 * (mean_abs_y / lateral_bound)
    )

    energy_proxy = (
        0.40 * vx_start_norm ** 2
        + 0.35 * vx_tail_norm ** 2
        + 0.10 * h_dev ** 2
        + 0.15 * c_dev ** 2
    )
    energy_score = -30.0 * energy_proxy

    base_score = (
        100.0 * float(success)
        - 120.0 * float(out_lane)
        - 100.0 * float(startup_failed)
        - 40.0 * float((not goal) and (not startup_failed))
    )

    return {
        "base_score": base_score,
        "motion_score": motion_score,
        "stability_score": stability_score,
        "energy_score": energy_score,
        "success": float(success),
        "goal": float(goal),
        "out_lane": float(out_lane),
        "startup_failed": float(startup_failed),
        "time": None if time_raw in ["", "NA", "None", None] else time_s,
        "max_abs_y": max_abs_y,
        "mean_abs_y": mean_abs_y,
        "raw_score": ffloat(row, "score", 0.0),
        **a,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def cmd_build(args):
    rows = []
    with open(args.rollouts) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    groups = {}
    for r in rows:
        label = r["label"]
        groups.setdefault(label, {
            "label": label,
            "schedule": r["schedule"],
            "rows": [],
            "components": [],
        })
        groups[label]["rows"].append(r)
        groups[label]["components"].append(
            component_scores(r, args.goal_x, args.lateral_bound, args.timeout_s)
        )

    actions = []
    for label, g in groups.items():
        cs = g["components"]
        first = parse_schedule(g["schedule"])

        action = {
            "label": label,
            "schedule": g["schedule"],
            "n": len(cs),
            "vx_start": first["vx_start"],
            "vx_tail": first["vx_tail"],
            "body_h": first["body_h"],
            "clearance": first["clearance"],
            "base_score": mean([c["base_score"] for c in cs]),
            "motion_score": mean([c["motion_score"] for c in cs]),
            "stability_score": mean([c["stability_score"] for c in cs]),
            "energy_score": mean([c["energy_score"] for c in cs]),
            "success_rate": mean([c["success"] for c in cs]),
            "goal_rate": mean([c["goal"] for c in cs]),
            "out_lane_rate": mean([c["out_lane"] for c in cs]),
            "startup_failed_rate": mean([c["startup_failed"] for c in cs]),
            "time_mean": mean([c["time"] for c in cs]),
            "max_abs_y_mean": mean([c["max_abs_y"] for c in cs]),
            "mean_abs_y_mean": mean([c["mean_abs_y"] for c in cs]),
            "raw_score_mean": mean([c["raw_score"] for c in cs]),
        }
        actions.append(action)

    model = {
        "version": "tracer_phase_d3c_q_policy_v0",
        "rollouts": args.rollouts,
        "goal_x": args.goal_x,
        "lateral_bound": args.lateral_bound,
        "timeout_s": args.timeout_s,
        "actions": actions,
    }

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_model).write_text(json.dumps(model, indent=2))
    print(f"[TRACER:D3c-Q] wrote model: {args.out_model}")
    print("[TRACER:D3c-Q] actions:")
    for a in actions:
        print(f"  {a['label']}: n={a['n']} success={a['success_rate']:.3f} raw={a['raw_score_mean']:.3f}")


def q_value(a, beta):
    return (
        a["base_score"]
        + beta[0] * a["motion_score"]
        + beta[1] * a["stability_score"]
        + beta[2] * a["energy_score"]
    )


def cmd_predict(args):
    model = json.loads(Path(args.model).read_text())
    beta = args.beta

    ranked = []
    for a in model["actions"]:
        aa = dict(a)
        aa["q"] = q_value(a, beta)
        ranked.append(aa)

    if args.require_success_rate is not None:
        ranked = [a for a in ranked if a["success_rate"] >= args.require_success_rate]

    ranked.sort(key=lambda x: x["q"], reverse=True)

    best = ranked[0]
    print("[TRACER:D3c-Q] beta =", beta)
    print("[TRACER:D3c-Q] selected =", best["label"])
    print("[TRACER:D3c-Q] q =", best["q"])
    print("[TRACER:D3c-Q] schedule =", best["schedule"])

    print("\nrank,label,q,success_rate,raw_score,time,max_abs_y,mean_abs_y")
    for i, a in enumerate(ranked, 1):
        t = "NA" if math.isnan(a["time_mean"]) else f"{a['time_mean']:.3f}"
        print(
            f"{i},{a['label']},{a['q']:.3f},{a['success_rate']:.3f},"
            f"{a['raw_score_mean']:.3f},{t},{a['max_abs_y_mean']:.3f},{a['mean_abs_y_mean']:.3f}"
        )

    if args.out_bank:
        line_label = args.line_label or f"d3cq_{best['label']}"
        Path(args.out_bank).parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with open(args.out_bank, mode) as f:
            f.write(f"{line_label} {best['schedule']}\n")
        print(f"[TRACER:D3c-Q] wrote bank line: {args.out_bank}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build")
    p.add_argument("--rollouts", required=True)
    p.add_argument("--out-model", required=True)
    p.add_argument("--goal-x", type=float, default=8.0)
    p.add_argument("--lateral-bound", type=float, default=2.0)
    p.add_argument("--timeout-s", type=float, default=460.0)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--beta", nargs=3, type=float, required=True)
    p.add_argument("--require-success-rate", type=float, default=None)
    p.add_argument("--out-bank", default="")
    p.add_argument("--line-label", default="")
    p.add_argument("--append", action="store_true")
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
