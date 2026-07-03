#!/usr/bin/env python3

import argparse
import json
import random
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
    ap.add_argument("--episodes-per-world", type=int, default=3)
    ap.add_argument("--out-root", default="")
    ap.add_argument("--policy-json", default="configs/phase_b_theta_lite_rl_v0/current_bandit_policy.json")
    ap.add_argument("--record-duration", type=float, default=35.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--epsilon", type=float, default=0.20)
    ap.add_argument("--ucb-c", type=float, default=1.0)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    worlds = [w.strip() for w in args.worlds.split(",") if w.strip()]
    jobs = []
    for ep in range(args.episodes_per_world):
        for w in worlds:
            jobs.append((ep, w))

    if args.shuffle:
        random.shuffle(jobs)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root or f"artifacts/phase_b_theta_lite_bandit_worldset_v0_{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "phase_b_theta_lite_bandit_worldset_manifest_v0",
        "out_root": str(out_root),
        "policy_json": args.policy_json,
        "worlds": worlds,
        "episodes_per_world": args.episodes_per_world,
        "record_duration": args.record_duration,
        "sample_hz": args.sample_hz,
        "epsilon": args.epsilon,
        "ucb_c": args.ucb_c,
        "runs": [],
    }

    for idx, (ep, w) in enumerate(jobs):
        run_dir = out_root / f"{idx:03d}_{w}_ep{ep:02d}"
        cmd = [
            "python3",
            "scripts/runtime/tracer_run_phase_b_theta_lite_bandit_episode_v0.py",
            "--world-name", w,
            "--out-root", str(run_dir),
            "--policy-json", args.policy_json,
            "--record-duration", str(args.record_duration),
            "--sample-hz", str(args.sample_hz),
            "--epsilon", str(args.epsilon),
            "--ucb-c", str(args.ucb_c),
        ]

        print()
        print("[bandit worldset]", idx, w, "ep", ep)
        print("[run]", " ".join(cmd))

        status = "ok"
        error = ""
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            status = "failed"
            error = str(e)

        result_path = run_dir / "bandit_episode_result_v0.json"
        run_record = {
            "idx": idx,
            "episode_index_for_world": ep,
            "world_name": w,
            "run_dir": str(run_dir),
            "status": status,
            "error": error,
            "result_json": str(result_path) if result_path.is_file() else None,
        }

        if result_path.is_file():
            try:
                r = json.load(open(result_path))
                run_record.update({
                    "selected_action_name": r.get("selected_action_name"),
                    "semantic": r.get("semantic"),
                    "reward": r.get("reward"),
                    "summary_json": r.get("summary_json"),
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
        "policy_json": args.policy_json,
        "num_runs": len(manifest["runs"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
