#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from collections import defaultdict, Counter

def ffloat(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def iint(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def learned_dims(r):
    return f"{r.get('learned_vx')}/{r.get('learned_yaw')}/{r.get('learned_clearance')}"

def score_motion(r):
    success = iint(r.get("success_proxy"))
    goal = iint(r.get("goal_proxy"))
    out_lane = iint(r.get("out_lane_proxy"))
    return (
        2.0 * success
        + 2.0 * goal
        - 3.0 * out_lane
        + ffloat(r.get("motion_score_mean"))
        + 0.05 * ffloat(r.get("final_x"))
        - 0.15 * ffloat(r.get("path_mean_abs_y"))
    )

def score_stability(r):
    success = iint(r.get("success_proxy"))
    goal = iint(r.get("goal_proxy"))
    out_lane = iint(r.get("out_lane_proxy"))
    return (
        2.0 * success
        + 2.0 * goal
        - 5.0 * out_lane
        + 1.5 * ffloat(r.get("moving_accept_rate"))
        + ffloat(r.get("stability_score_mean"))
        - 1.0 * ffloat(r.get("path_max_abs_y"))
        - 0.5 * ffloat(r.get("path_mean_abs_y"))
        - 0.25 * abs(ffloat(r.get("final_y")))
    )

def score_energy(r):
    success = iint(r.get("success_proxy"))
    goal = iint(r.get("goal_proxy"))
    out_lane = iint(r.get("out_lane_proxy"))
    return (
        2.0 * success
        + 2.0 * goal
        - 3.0 * out_lane
        + ffloat(r.get("energy_score_mean"))
        - ffloat(r.get("effort_proxy_mean"))
        - 0.10 * ffloat(r.get("path_mean_abs_y"))
    )

def score_balanced(r):
    return 0.30 * score_motion(r) + 0.50 * score_stability(r) + 0.20 * score_energy(r)

SCORERS = {
    "motion": score_motion,
    "stability": score_stability,
    "energy": score_energy,
    "balanced": score_balanced,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--min-diff", type=float, default=0.03)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.metrics_csv)))
    for i, r in enumerate(rows):
        r["_idx"] = str(i)

    groups = defaultdict(list)
    for r in rows:
        key = (
            r.get("world", ""),
            round(ffloat(r.get("reset_y")), 2),
            round(ffloat(r.get("goal_x")), 2),
        )
        groups[key].append(r)

    pairs = []
    for key, group in groups.items():
        if len(group) < 2:
            continue

        for pref_type, scorer in SCORERS.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a = group[i]
                    b = group[j]
                    sa = scorer(a)
                    sb = scorer(b)
                    diff = abs(sa - sb)
                    if diff < args.min_diff:
                        continue

                    if sa >= sb:
                        pref, rej = a, b
                        sp, sr = sa, sb
                    else:
                        pref, rej = b, a
                        sp, sr = sb, sa

                    pairs.append({
                        "pref_type": pref_type,
                        "world": key[0],
                        "reset_y": key[1],
                        "goal_x": key[2],
                        "preferred_idx": pref["_idx"],
                        "rejected_idx": rej["_idx"],
                        "preferred_dims": learned_dims(pref),
                        "rejected_dims": learned_dims(rej),
                        "preferred_score": sp,
                        "rejected_score": sr,
                        "score_diff": sp - sr,

                        "preferred_success": pref.get("success_proxy"),
                        "rejected_success": rej.get("success_proxy"),
                        "preferred_goal": pref.get("goal_proxy"),
                        "rejected_goal": rej.get("goal_proxy"),

                        "preferred_final_y": pref.get("final_y"),
                        "rejected_final_y": rej.get("final_y"),
                        "preferred_path_max_abs_y": pref.get("path_max_abs_y"),
                        "rejected_path_max_abs_y": rej.get("path_max_abs_y"),
                        "preferred_path_mean_abs_y": pref.get("path_mean_abs_y"),
                        "rejected_path_mean_abs_y": rej.get("path_mean_abs_y"),
                        "preferred_moving_accept_rate": pref.get("moving_accept_rate"),
                        "rejected_moving_accept_rate": rej.get("moving_accept_rate"),

                        "preferred_beta_motion": pref.get("pred_beta_motion_mean"),
                        "preferred_beta_stability": pref.get("pred_beta_stability_mean"),
                        "preferred_beta_energy": pref.get("pred_beta_energy_mean"),
                        "rejected_beta_motion": rej.get("pred_beta_motion_mean"),
                        "rejected_beta_stability": rej.get("pred_beta_stability_mean"),
                        "rejected_beta_energy": rej.get("pred_beta_energy_mean"),

                        "preferred_manifest": pref.get("manifest"),
                        "rejected_manifest": rej.get("manifest"),
                        "preferred_d5_log_dir": pref.get("d5_log_dir"),
                        "rejected_d5_log_dir": rej.get("d5_log_dir"),
                    })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if pairs:
        fields = list(pairs[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for p in pairs:
                w.writerow(p)
    else:
        out_csv.write_text("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    c_type = Counter(p["pref_type"] for p in pairs)
    c_reset = Counter(str(p["reset_y"]) for p in pairs)
    c_dims = Counter((p["pref_type"], p["preferred_dims"], p["rejected_dims"]) for p in pairs)

    with open(out_md, "w") as f:
        f.write("# TRACER Phase-E2 Preference Pairs v0\n\n")
        f.write("This is a bootstrap preference-pair dataset built from rollout-level E1 metrics. It is not yet true IRL; it provides weak preference supervision for later Objective Selector training.\n\n")
        f.write(f"- metrics_csv: `{args.metrics_csv}`\n")
        f.write(f"- out_csv: `{out_csv}`\n")
        f.write(f"- num_pairs: `{len(pairs)}`\n")
        f.write(f"- min_diff: `{args.min_diff}`\n\n")

        f.write("## Pair counts by preference type\n\n")
        for k, v in sorted(c_type.items()):
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Pair counts by reset_y\n\n")
        for k, v in sorted(c_reset.items()):
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Top preferred/rejected dimension patterns\n\n")
        for (pref_type, pd, rd), v in c_dims.most_common(20):
            f.write(f"- {pref_type}: `{pd}` > `{rd}` : `{v}`\n")

        f.write("\n## Example pairs\n\n")
        f.write("| pref_type | reset_y | preferred | rejected | diff | pref max_y | rej max_y | pref accept | rej accept |\n")
        f.write("|---|---:|---|---|---:|---:|---:|---:|---:|\n")
        for p in sorted(pairs, key=lambda x: float(x["score_diff"]), reverse=True)[:20]:
            f.write(
                f"| {p['pref_type']} | {float(p['reset_y']):+.2f} | "
                f"{p['preferred_dims']} | {p['rejected_dims']} | "
                f"{float(p['score_diff']):.3f} | "
                f"{float(p['preferred_path_max_abs_y']):.3f} | "
                f"{float(p['rejected_path_max_abs_y']):.3f} | "
                f"{float(p['preferred_moving_accept_rate']):.3f} | "
                f"{float(p['rejected_moving_accept_rate']):.3f} |\n"
            )

    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_md}")
    print(f"[TRACER] num_pairs={len(pairs)}")
    print("[TRACER] counts_by_type:", dict(c_type))
    print("[TRACER] counts_by_reset_y:", dict(c_reset))

if __name__ == "__main__":
    main()
