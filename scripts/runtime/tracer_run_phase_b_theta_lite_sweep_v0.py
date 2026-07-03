#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run(cmd, env):
    print()
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase_b_theta_lite_sweep_v0/theta_lite_sweep_v0.json")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--world-filter", default="")
    ap.add_argument("--profile-filter", default="")
    ap.add_argument("--runner", default="scripts/runtime/tracer_run_phase_b_goal_episode_v0.sh")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    worlds = cfg["worlds"]
    profiles = cfg["profiles"]

    if args.world_filter:
        keys = {x.strip() for x in args.world_filter.split(",") if x.strip()}
        worlds = [w for w in worlds if w["name"] in keys or w.get("label") in keys]

    if args.profile_filter:
        keys = {x.strip() for x in args.profile_filter.split(",") if x.strip()}
        profiles = [p for p in profiles if p["name"] in keys or p.get("family") in keys]

    if args.smoke:
        # Keep smoke small and representative.
        worlds = worlds[:2]
        profiles = [profiles[0], profiles[3]]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root or f"artifacts/phase_b_theta_lite_sweep_v0_{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "phase_b_theta_lite_sweep_manifest_v0",
        "config": args.config,
        "out_root": str(out_root),
        "sample_hz": args.sample_hz,
        "repeat": args.repeat,
        "smoke": args.smoke,
        "runs": [],
    }

    runner = Path(args.runner)
    if not runner.is_file():
        raise FileNotFoundError(f"runner not found: {runner}")

    for rep in range(args.repeat):
        for w in worlds:
            for p in profiles:
                run_name = f"{w['label']}__{p['name']}__rep{rep:02d}"
                run_dir = out_root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                profile_json = run_dir / "theta_lite_profile_v0.json"
                profile_record = {
                    "schema": "phase_b_theta_lite_profile_v0",
                    "world": w,
                    "profile": p,
                    "repeat_index": rep,
                }
                profile_json.write_text(json.dumps(profile_record, indent=2, sort_keys=True) + "\n")

                env = os.environ.copy()
                env.update({
                    "TRACER_PHASE_B_WORLD": str(w["name"]),
                    "TRACER_PHASE_B_RECORD_DURATION": str(w.get("duration", 45.0)),
                    "TRACER_PHASE_B_SAMPLE_HZ": str(args.sample_hz),
                    "TRACER_PHASE_B_RUN_DIR": str(run_dir),

                    "TRACER_PHASE_B_GOAL_DISTANCE_AHEAD": "0.5",
                    "TRACER_PHASE_B_GOAL_STOP_DISTANCE": str(p["goal_stop_distance"]),
                    "TRACER_PHASE_B_GOAL_SLOW_DISTANCE": str(p["goal_slow_distance"]),

                    "TRACER_PHASE_B_VX_FAR": str(p["vx_far"]),
                    "TRACER_PHASE_B_VX_NEAR": str(p["vx_near"]),
                    "TRACER_PHASE_B_BODY_HEIGHT": str(p["body_height"]),
                    "TRACER_PHASE_B_SWING_CLEARANCE": str(p["swing_clearance"]),

                    "TRACER_PHASE_B_THETA_LITE_PROFILE_JSON": str(profile_json),
                    "TRACER_PHASE_B_THETA_LITE_PROFILE_NAME": str(p["name"]),
                    "TRACER_PHASE_B_THETA_LITE_WORLD_LABEL": str(w.get("label", w["name"])),
                })

                status = "ok"
                error = ""
                try:
                    run(["bash", str(runner)], env)
                except subprocess.CalledProcessError as e:
                    status = "failed"
                    error = str(e)

                manifest["runs"].append({
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "world_name": w["name"],
                    "world_label": w.get("label"),
                    "profile_name": p["name"],
                    "profile_family": p.get("family"),
                    "repeat_index": rep,
                    "status": status,
                    "error": error,
                    "theta": {
                        "vx_far": p["vx_far"],
                        "vx_near": p["vx_near"],
                        "goal_slow_distance": p["goal_slow_distance"],
                        "goal_stop_distance": p["goal_stop_distance"],
                        "body_height": p["body_height"],
                        "swing_clearance": p["swing_clearance"],
                    },
                })

                (out_root / "manifest_v0.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )

    print()
    print(json.dumps({
        "out_root": str(out_root),
        "num_runs": len(manifest["runs"]),
        "manifest": str(out_root / "manifest_v0.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
