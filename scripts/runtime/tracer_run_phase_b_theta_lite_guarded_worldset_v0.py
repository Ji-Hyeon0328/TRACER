#!/usr/bin/env python3

import argparse
import json
import subprocess
import time
from pathlib import Path


DEFAULT_WORLDS = [
    "earth",
    "tracer_sponge_firm_flat",
    "tracer_sponge_firm_slope_5deg",
    "tracer_sponge_firm_downslope_5deg",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default=",".join(DEFAULT_WORLDS))
    ap.add_argument("--out-root", default="")
    ap.add_argument("--record-duration", type=float, default=30.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--runner", default="scripts/runtime/tracer_run_phase_b_theta_lite_guarded_episode_v0.py")
    args = ap.parse_args()

    worlds = [w.strip() for w in args.worlds.split(",") if w.strip()]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root or f"artifacts/phase_b_theta_lite_guarded_worldset_v0_{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "phase_b_theta_lite_guarded_worldset_manifest_v0",
        "out_root": str(out_root),
        "worlds": worlds,
        "record_duration": args.record_duration,
        "sample_hz": args.sample_hz,
        "runs": [],
    }

    for w in worlds:
        run_dir = out_root / w
        cmd = [
            "python3",
            args.runner,
            "--world-name", w,
            "--out-root", str(run_dir),
            "--record-duration", str(args.record_duration),
            "--sample-hz", str(args.sample_hz),
        ]

        print()
        print("[guarded world]", w)
        print("[run]", " ".join(cmd))

        status = "ok"
        error = ""
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            status = "failed"
            error = str(e)

        result_path = run_dir / "guarded_run_result_v0.json"
        gate_path = run_dir / "runtime_gate_v0.json"
        summary_path = run_dir / "episode" / "phase_b_summary.json"

        run_record = {
            "world_name": w,
            "run_dir": str(run_dir),
            "status": status,
            "error": error,
            "guarded_result_json": str(result_path) if result_path.is_file() else None,
            "runtime_gate_json": str(gate_path) if gate_path.is_file() else None,
            "summary_json": str(summary_path) if summary_path.is_file() else None,
        }

        if result_path.is_file():
            try:
                r = json.load(open(result_path))
                run_record.update({
                    "rollout_executed": r.get("rollout_executed"),
                    "normal_walk_blocked": r.get("normal_walk_blocked"),
                    "allow_normal_walk": r.get("allow_normal_walk"),
                    "requires_alternative_primitive": r.get("requires_alternative_primitive"),
                    "requires_more_theta_search": r.get("requires_more_theta_search"),
                    "selection_status": r.get("selection_status"),
                    "selected_profile_name": r.get("selected_profile_name"),
                })
            except Exception as ex:
                run_record["result_read_error"] = str(ex)

        manifest["runs"].append(run_record)
        (out_root / "manifest_v0.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    print()
    print(json.dumps({
        "out_root": str(out_root),
        "manifest": str(out_root / "manifest_v0.json"),
        "num_runs": len(manifest["runs"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
