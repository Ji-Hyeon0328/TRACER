#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def load_table_entry(table_json, world_name):
    table = json.load(open(table_json))
    return table.get("table", {}).get(world_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-name", default=os.environ.get("TRACER_PHASE_B_WORLD", "earth"))
    ap.add_argument("--table-json", default="configs/phase_b_theta_lite_robust_teacher_table_v0/current_table.json")
    ap.add_argument("--runner", default="scripts/runtime/tracer_run_phase_b_goal_episode_v0.sh")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--sample-hz", type=float, default=float(os.environ.get("TRACER_PHASE_B_SAMPLE_HZ", "20.0")))
    ap.add_argument("--record-duration", type=float, default=0.0)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root or f"artifacts/phase_b_theta_lite_guarded_episode_v0_{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    entry = load_table_entry(args.table_json, args.world_name)

    if entry is None:
        gate = {
            "schema": "phase_b_theta_lite_guarded_gate_v0",
            "world_name": args.world_name,
            "found_in_table": False,
            "allow_normal_walk": False,
            "normal_walk_blocked": True,
            "requires_more_theta_search": True,
            "requires_alternative_primitive": False,
            "reason": "world_not_found_in_robust_theta_lite_table",
        }
    else:
        gate = {
            "schema": "phase_b_theta_lite_guarded_gate_v0",
            "world_name": args.world_name,
            "found_in_table": True,
            "selection_status": entry.get("selection_status"),
            "selected_profile_name": entry.get("selected_profile_name"),
            "theta_action": entry.get("theta_action"),
            "allow_normal_walk": bool(entry.get("trusted_normal_walk", False)),
            "normal_walk_blocked": bool(entry.get("normal_walk_blocked", True)),
            "requires_more_theta_search": bool(entry.get("requires_more_theta_search", False)),
            "requires_alternative_primitive": bool(entry.get("requires_alternative_primitive", False)),
            "selected_profile_summary": entry.get("selected_profile_summary"),
        }

    gate_path = out_root / "runtime_gate_v0.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    if gate["normal_walk_blocked"]:
        episode_dir = out_root / "episode"
        episode_dir.mkdir(parents=True, exist_ok=True)

        blocked_summary = {
            "schema": "phase_b_theta_lite_guarded_blocked_summary_v0",
            "world_name": args.world_name,
            "run_dir": str(out_root),
            "runtime_gate_json": str(gate_path),
            "rollout_executed": False,
            "allow_normal_walk": False,
            "normal_walk_blocked": True,
            "requires_more_theta_search": gate.get("requires_more_theta_search", False),
            "requires_alternative_primitive": gate.get("requires_alternative_primitive", False),
            "selection_status": gate.get("selection_status"),
            "selected_profile_name": gate.get("selected_profile_name"),
            "theta_action": gate.get("theta_action"),
            "reason": (
                "normal walking blocked by robust theta-lite table; "
                "skip A1-QP-MPC diagonal gait rollout"
            ),
        }

        (episode_dir / "phase_b_summary.json").write_text(
            json.dumps(blocked_summary, indent=2, sort_keys=True) + "\n"
        )
        (out_root / "guarded_run_result_v0.json").write_text(
            json.dumps(blocked_summary, indent=2, sort_keys=True) + "\n"
        )

        print(json.dumps(blocked_summary, indent=2, sort_keys=True))
        return 0

    theta = gate["theta_action"]
    if not theta:
        raise RuntimeError("normal walk allowed but theta_action is missing")

    env = os.environ.copy()
    env.update({
        "TRACER_PHASE_B_WORLD": args.world_name,
        "TRACER_PHASE_B_RUN_DIR": str(out_root),
        "TRACER_PHASE_B_SAMPLE_HZ": str(args.sample_hz),

        "TRACER_PHASE_B_VX_FAR": str(theta["vx_far"]),
        "TRACER_PHASE_B_VX_NEAR": str(theta["vx_near"]),
        "TRACER_PHASE_B_GOAL_SLOW_DISTANCE": str(theta["goal_slow_distance"]),
        "TRACER_PHASE_B_GOAL_STOP_DISTANCE": str(theta["goal_stop_distance"]),
        "TRACER_PHASE_B_BODY_HEIGHT": str(theta["body_height"]),
        "TRACER_PHASE_B_SWING_CLEARANCE": str(theta["swing_clearance"]),

        "TRACER_PHASE_B_THETA_LITE_GUARD_JSON": str(gate_path),
        "TRACER_PHASE_B_THETA_LITE_PROFILE_NAME": str(gate.get("selected_profile_name", "")),
    })

    if args.record_duration > 0:
        env["TRACER_PHASE_B_RECORD_DURATION"] = str(args.record_duration)

    runner = Path(args.runner)
    if not runner.is_file():
        raise FileNotFoundError(f"runner not found: {runner}")

    print(json.dumps({
        "schema": "phase_b_theta_lite_guarded_run_start_v0",
        "world_name": args.world_name,
        "run_dir": str(out_root),
        "selected_profile_name": gate.get("selected_profile_name"),
        "theta_action": theta,
        "runner": str(runner),
    }, indent=2, sort_keys=True))

    subprocess.check_call(["bash", str(runner)], env=env)

    # Normalize the summary location. Some Phase-B runners write directly to
    # run_dir/phase_b_summary.json, while blocked guarded episodes use
    # run_dir/episode/phase_b_summary.json. Keep a canonical episode path for
    # downstream dataset/scorer tools.
    episode_dir = out_root / "episode"
    episode_dir.mkdir(parents=True, exist_ok=True)

    summary_candidates = [
        out_root / "episode" / "phase_b_summary.json",
        out_root / "phase_b_summary.json",
    ]
    summary_candidates.extend(sorted(out_root.glob("**/phase_b_summary.json")))

    summary_src = None
    for c in summary_candidates:
        if c.is_file():
            summary_src = c
            break

    canonical_summary = episode_dir / "phase_b_summary.json"
    if summary_src is not None and summary_src.resolve() != canonical_summary.resolve():
        canonical_summary.write_text(summary_src.read_text())

    result = {
        "schema": "phase_b_theta_lite_guarded_run_result_v0",
        "world_name": args.world_name,
        "run_dir": str(out_root),
        "runtime_gate_json": str(gate_path),
        "rollout_executed": True,
        "allow_normal_walk": True,
        "normal_walk_blocked": False,
        "selected_profile_name": gate.get("selected_profile_name"),
        "theta_action": theta,
        "summary_json": str(canonical_summary) if canonical_summary.is_file() else None,
        "summary_source_json": str(summary_src) if summary_src is not None else None,
    }
    (out_root / "guarded_run_result_v0.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
