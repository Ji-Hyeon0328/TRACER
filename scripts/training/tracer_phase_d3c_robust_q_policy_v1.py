#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path


def parse_bool(x):
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


def action_key(schedule):
    a = parse_schedule(schedule)
    return f"vs{a['vx_start']:.4f}_vt{a['vx_tail']:.4f}_h{a['body_h']:.4f}_c{a['clearance']:.4f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


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

    motion_score = 55.0 * progress + 35.0 * time_eff + 10.0 * vx_start_norm + 6.0 * vx_tail_norm

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
    }


def q_value(c, beta):
    return c["base_score"] + beta[0] * c["motion_score"] + beta[1] * c["stability_score"] + beta[2] * c["energy_score"]


def build(args):
    groups = {}

    for path in args.rollouts:
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = action_key(r["schedule"])
                a = parse_schedule(r["schedule"])
                groups.setdefault(key, {
                    "key": key,
                    "labels_seen": set(),
                    "schedule": r["schedule"],
                    "action": a,
                    "components": [],
                    "sources": [],
                })
                groups[key]["labels_seen"].add(r.get("label", ""))
                groups[key]["components"].append(
                    component_scores(r, args.goal_x, args.lateral_bound, args.timeout_s)
                )
                groups[key]["sources"].append(path)

    actions = []
    for key, g in groups.items():
        cs = g["components"]
        action = {
            "key": key,
            "labels_seen": sorted(list(g["labels_seen"])),
            "schedule": g["schedule"],
            "n": len(cs),
            **g["action"],
            "success_rate": mean([c["success"] for c in cs]),
            "goal_rate": mean([c["goal"] for c in cs]),
            "out_lane_rate": mean([c["out_lane"] for c in cs]),
            "startup_failed_rate": mean([c["startup_failed"] for c in cs]),
            "raw_score_mean": mean([c["raw_score"] for c in cs]),
            "raw_score_std": std([c["raw_score"] for c in cs]),
            "time_mean": mean([c["time"] for c in cs]),
            "max_abs_y_mean": mean([c["max_abs_y"] for c in cs]),
            "mean_abs_y_mean": mean([c["mean_abs_y"] for c in cs]),
            "components": cs,
        }
        actions.append(action)

    model = {
        "version": "tracer_phase_d3c_robust_q_policy_v1",
        "rollouts": args.rollouts,
        "goal_x": args.goal_x,
        "lateral_bound": args.lateral_bound,
        "timeout_s": args.timeout_s,
        "actions": actions,
    }

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_model).write_text(json.dumps(model, indent=2))

    print(f"[TRACER:D3c-Qv1] wrote {args.out_model}")
    for a in sorted(actions, key=lambda x: x["raw_score_mean"], reverse=True):
        print(
            f"{a['key']}: n={a['n']} success={a['success_rate']:.3f} "
            f"raw={a['raw_score_mean']:.3f} max_y={a['max_abs_y_mean']:.3f} labels={a['labels_seen']}"
        )


def predict(args):
    model = json.loads(Path(args.model).read_text())
    ranked = []

    for a in model["actions"]:
        qs = [q_value(c, args.beta) for c in a["components"]]
        q_mean = mean(qs)
        q_std = std(qs)

        robust_q = (
            q_mean
            - args.lambda_std * q_std
            - args.lambda_fail * (1.0 - a["success_rate"])
            - args.lambda_drift * max(0.0, a["max_abs_y_mean"] - args.drift_limit)
        )

        aa = dict(a)
        aa.pop("components", None)
        aa["q_mean"] = q_mean
        aa["q_std"] = q_std
        aa["robust_q"] = robust_q
        ranked.append(aa)

    if args.min_n is not None:
        ranked = [a for a in ranked if a["n"] >= args.min_n]
    if args.require_success_rate is not None:
        ranked = [a for a in ranked if a["success_rate"] >= args.require_success_rate]

    ranked.sort(key=lambda x: x["robust_q"], reverse=True)
    best = ranked[0]

    print("[TRACER:D3c-Qv1] beta =", args.beta)
    print("[TRACER:D3c-Qv1] selected =", best["key"])
    print("[TRACER:D3c-Qv1] labels_seen =", best["labels_seen"])
    print("[TRACER:D3c-Qv1] robust_q =", best["robust_q"])
    print("[TRACER:D3c-Qv1] schedule =", best["schedule"])

    print("\nrank,key,robust_q,q_mean,q_std,n,success,raw,time,max_y,mean_y,labels")
    for i, a in enumerate(ranked, 1):
        t = "NA" if math.isnan(a["time_mean"]) else f"{a['time_mean']:.3f}"
        print(
            f"{i},{a['key']},{a['robust_q']:.3f},{a['q_mean']:.3f},{a['q_std']:.3f},"
            f"{a['n']},{a['success_rate']:.3f},{a['raw_score_mean']:.3f},{t},"
            f"{a['max_abs_y_mean']:.3f},{a['mean_abs_y_mean']:.3f},{'|'.join(a['labels_seen'])}"
        )

    if args.out_bank:
        Path(args.out_bank).parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        line_label = args.line_label or f"d3cqv1_{best['key']}"
        with open(args.out_bank, mode) as f:
            f.write(f"{line_label} {best['schedule']}\n")
        print(f"[TRACER:D3c-Qv1] wrote bank line {args.out_bank}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build")
    p.add_argument("--rollouts", nargs="+", required=True)
    p.add_argument("--out-model", required=True)
    p.add_argument("--goal-x", type=float, default=8.0)
    p.add_argument("--lateral-bound", type=float, default=2.0)
    p.add_argument("--timeout-s", type=float, default=460.0)
    p.set_defaults(func=build)

    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--beta", nargs=3, type=float, required=True)
    p.add_argument("--min-n", type=int, default=3)
    p.add_argument("--require-success-rate", type=float, default=None)
    p.add_argument("--lambda-std", type=float, default=0.25)
    p.add_argument("--lambda-fail", type=float, default=80.0)
    p.add_argument("--lambda-drift", type=float, default=20.0)
    p.add_argument("--drift-limit", type=float, default=0.45)
    p.add_argument("--out-bank", default="")
    p.add_argument("--line-label", default="")
    p.add_argument("--append", action="store_true")
    p.set_defaults(func=predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
