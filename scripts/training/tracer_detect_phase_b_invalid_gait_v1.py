#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict


LEGS = ["FL", "FR", "RL", "RR"]
JOINTS = ["hip", "thigh", "calf"]


def ff(x, default=float("nan")):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def mean(xs, default=float("nan")):
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else default


def safe_min(xs, default=float("nan")):
    xs = [x for x in xs if math.isfinite(x)]
    return min(xs) if xs else default


def safe_max(xs, default=float("nan")):
    xs = [x for x in xs if math.isfinite(x)]
    return max(xs) if xs else default


def variance(xs, default=0.0):
    xs = [x for x in xs if math.isfinite(x)]
    if len(xs) < 2:
        return default
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def load_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def infer_world_action_episode(csv_path):
    # Expected: root/world/action/epXXX/phase_b_episode.csv
    ep_dir = csv_path.parent
    action = ep_dir.parent.name if ep_dir.parent else "unknown"
    world = ep_dir.parent.parent.name if ep_dir.parent and ep_dir.parent.parent else "unknown"
    episode = ep_dir.name
    return world, action, episode


def leg_motion(rows, leg):
    vals = []
    for r in rows:
        for j in JOINTS:
            vals.append(ff(r.get(f"dq_{leg}_{j}")))
    vals = [abs(x) for x in vals if math.isfinite(x)]
    return mean(vals, 0.0)


def leg_q_abs_mean(rows, leg):
    vals = []
    for r in rows:
        for j in JOINTS:
            vals.append(abs(ff(r.get(f"q_{leg}_{j}"))))
    return mean(vals, 0.0)


def leg_q_var(rows, leg):
    vals = []
    for r in rows:
        for j in JOINTS:
            vals.append(ff(r.get(f"q_{leg}_{j}")))
    return variance(vals, 0.0)


def contact_mean(rows, leg):
    return mean([ff(r.get(f"contact_{leg}")) for r in rows], 0.0)


def force_mean(rows, leg):
    return mean([ff(r.get(f"force_{leg}")) for r in rows], 0.0)


def frac_condition(rows, fn):
    if not rows:
        return 0.0
    return sum(1 for r in rows if fn(r)) / len(rows)


def detect_one(csv_path):
    rows = load_csv(csv_path)
    world, action, episode = infer_world_action_episode(csv_path)
    summary_path = csv_path.parent / "phase_b_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}

    has_state = all(c in rows[0] for c in [
        "base_z", "base_roll", "base_pitch",
        "q_FL_hip", "q_FR_hip", "q_RL_hip", "q_RR_hip",
        "dq_FL_hip", "dq_FR_hip", "dq_RL_hip", "dq_RR_hip",
        "contact_FL", "contact_FR", "contact_RL", "contact_RR",
    ]) if rows else False

    if not rows or not has_state:
        return {
            "schema": "phase_b_invalid_gait_detection_v1",
            "csv_path": str(csv_path),
            "world": world,
            "action": action,
            "episode": episode,
            "has_state": False,
            "invalid_gait": None,
            "reasons": ["missing_state_columns"],
        }

    base_zs = [ff(r.get("base_z")) for r in rows]
    rolls = [abs(ff(r.get("base_roll"))) for r in rows]
    pitches = [abs(ff(r.get("base_pitch"))) for r in rows]

    contacts = {leg: contact_mean(rows, leg) for leg in LEGS}
    forces = {leg: force_mean(rows, leg) for leg in LEGS}
    motions = {leg: leg_motion(rows, leg) for leg in LEGS}
    q_abs = {leg: leg_q_abs_mean(rows, leg) for leg in LEGS}
    q_var = {leg: leg_q_var(rows, leg) for leg in LEGS}

    diag_a_motion = motions["FL"] + motions["RR"]
    diag_b_motion = motions["FR"] + motions["RL"]
    diag_motion_ratio = abs(diag_a_motion - diag_b_motion) / max(diag_a_motion + diag_b_motion, 1e-6)

    diag_a_contact = contacts["FL"] + contacts["RR"]
    diag_b_contact = contacts["FR"] + contacts["RL"]
    diag_contact_ratio = abs(diag_a_contact - diag_b_contact) / max(diag_a_contact + diag_b_contact, 1e-6)

    min_motion_leg = min(motions, key=motions.get)
    max_motion_leg = max(motions, key=motions.get)
    motion_spread_ratio = (motions[max_motion_leg] - motions[min_motion_leg]) / max(motions[max_motion_leg], 1e-6)

    min_contact_leg = min(contacts, key=contacts.get)
    max_contact_leg = max(contacts, key=contacts.get)
    contact_spread_ratio = contacts[max_contact_leg] - contacts[min_contact_leg]

    # Folded posture proxy: calf too tucked or thigh/calf extreme for sustained time.
    # These thresholds are intentionally broad; they should be calibrated after inspecting v1 data.
    folded_frac = {
        leg: frac_condition(
            rows,
            lambda r, leg=leg: (
                abs(ff(r.get(f"q_{leg}_calf"))) > 2.2
                or abs(ff(r.get(f"q_{leg}_thigh"))) > 2.4
            ),
        )
        for leg in LEGS
    }

    low_base_frac = frac_condition(rows, lambda r: ff(r.get("base_z")) < 0.20)
    high_roll_pitch_frac = frac_condition(
        rows,
        lambda r: abs(ff(r.get("base_roll"))) > 0.60 or abs(ff(r.get("base_pitch"))) > 0.60,
    )

    reasons = []

    mean_base_z = mean(base_zs, 999.0)
    min_base_z = safe_min(base_zs, 999.0)
    max_abs_roll = safe_max(rolls, 0.0)
    max_abs_pitch = safe_max(pitches, 0.0)

    if mean_base_z < 0.22 or min_base_z < 0.16:
        reasons.append("low_base_height")

    if max_abs_roll > 0.80 or max_abs_pitch > 0.80:
        reasons.append("large_roll_or_pitch")

    if diag_motion_ratio > 0.55:
        reasons.append("diagonal_motion_imbalance")

    if diag_contact_ratio > 0.55:
        reasons.append("diagonal_contact_imbalance")

    if motion_spread_ratio > 0.75:
        reasons.append("per_leg_motion_collapse")

    if contact_spread_ratio > 0.65:
        reasons.append("contact_asymmetry")

    folded_legs = [leg for leg, frac in folded_frac.items() if frac > 0.50]
    if folded_legs:
        reasons.append("folded_joint_posture:" + ",".join(folded_legs))

    # Suspicious metric-level proxy remains auxiliary, not direct invalid gait.
    reached = bool(summary.get("reached_stop_distance", False))
    final_dist = ff(summary.get("final_rel_dist"), 999.0)
    min_dist = ff(summary.get("min_rel_dist"), 999.0)
    progress = ff(summary.get("progress_initial_minus_min"), 0.0)
    if (not reached) and final_dist < 1.2 and progress < 0.25:
        reasons.append("metric_suspicious_low_progress_stable")

    invalid = len(reasons) > 0

    return {
        "schema": "phase_b_invalid_gait_detection_v1",
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "world": world,
        "action": action,
        "episode": episode,
        "has_state": True,
        "invalid_gait": invalid,
        "reasons": reasons,
        "features": {
            "num_rows": len(rows),
            "mean_base_z": mean(base_zs),
            "min_base_z": safe_min(base_zs),
            "max_abs_roll": safe_max(rolls, 0.0),
            "max_abs_pitch": safe_max(pitches, 0.0),
            "low_base_frac": low_base_frac,
            "high_roll_pitch_frac": high_roll_pitch_frac,
            "contacts": contacts,
            "forces": forces,
            "motions": motions,
            "q_abs": q_abs,
            "q_var": q_var,
            "folded_frac": folded_frac,
            "diag_motion_ratio": diag_motion_ratio,
            "diag_contact_ratio": diag_contact_ratio,
            "motion_spread_ratio": motion_spread_ratio,
            "contact_spread_ratio": contact_spread_ratio,
            "min_motion_leg": min_motion_leg,
            "max_motion_leg": max_motion_leg,
            "min_contact_leg": min_contact_leg,
            "max_contact_leg": max_contact_leg,
            "reached_stop_distance": reached,
            "final_rel_dist": final_dist,
            "min_rel_dist": min_dist,
            "progress_initial_minus_min": progress,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    csvs = sorted(root.rglob("phase_b_episode.csv"))

    rows = [detect_one(p) for p in csvs]

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    by_action = defaultdict(lambda: {"n": 0, "invalid": 0, "missing_state": 0, "reasons": defaultdict(int)})
    by_world = defaultdict(lambda: {"n": 0, "invalid": 0, "missing_state": 0})

    for r in rows:
        key = f"{r.get('world')}::{r.get('action')}"
        w = r.get("world")
        by_action[key]["n"] += 1
        by_world[w]["n"] += 1

        if not r.get("has_state"):
            by_action[key]["missing_state"] += 1
            by_world[w]["missing_state"] += 1
            continue

        if r.get("invalid_gait"):
            by_action[key]["invalid"] += 1
            by_world[w]["invalid"] += 1
            for reason in r.get("reasons", []):
                by_action[key]["reasons"][reason] += 1

    report = {
        "schema": "phase_b_invalid_gait_report_v1",
        "root": args.root,
        "num_episodes": len(rows),
        "num_invalid": sum(1 for r in rows if r.get("invalid_gait") is True),
        "num_missing_state": sum(1 for r in rows if not r.get("has_state")),
        "by_world": {},
        "by_action": {},
    }

    for k, v in sorted(by_world.items()):
        n = max(1, v["n"])
        report["by_world"][k] = {
            "n": v["n"],
            "invalid": v["invalid"],
            "missing_state": v["missing_state"],
            "invalid_rate": v["invalid"] / n,
            "missing_state_rate": v["missing_state"] / n,
        }

    for k, v in sorted(by_action.items()):
        n = max(1, v["n"])
        report["by_action"][k] = {
            "n": v["n"],
            "invalid": v["invalid"],
            "missing_state": v["missing_state"],
            "invalid_rate": v["invalid"] / n,
            "missing_state_rate": v["missing_state"] / n,
            "top_reasons": sorted(v["reasons"].items(), key=lambda x: (-x[1], x[0]))[:8],
        }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    lines = []
    lines.append("# Phase-B Invalid Gait Detection v1")
    lines.append("")
    lines.append(f"- Root: `{args.root}`")
    lines.append(f"- Episodes: `{report['num_episodes']}`")
    lines.append(f"- Invalid: `{report['num_invalid']}`")
    lines.append(f"- Missing state: `{report['num_missing_state']}`")
    lines.append("")
    lines.append("## By world")
    lines.append("")
    lines.append("| world | n | invalid_rate | missing_state_rate |")
    lines.append("|---|---:|---:|---:|")
    for k, v in report["by_world"].items():
        lines.append(f"| {k} | {v['n']} | {v['invalid_rate']:.3f} | {v['missing_state_rate']:.3f} |")

    lines.append("")
    lines.append("## By action")
    lines.append("")
    lines.append("| world::action | n | invalid_rate | missing_state_rate | top reasons |")
    lines.append("|---|---:|---:|---:|---|")
    for k, v in report["by_action"].items():
        reasons = ", ".join(f"{name}:{cnt}" for name, cnt in v["top_reasons"])
        lines.append(
            f"| {k} | {v['n']} | {v['invalid_rate']:.3f} | {v['missing_state_rate']:.3f} | {reasons} |"
        )

    md = "\n".join(lines) + "\n"
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md)
    print(md)
    print("[wrote]", args.out_jsonl)
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
