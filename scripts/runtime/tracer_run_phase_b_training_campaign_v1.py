#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_jsonl(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def build_action_bank(teacher_json):
    t = load_json(teacher_json)
    bank = {}

    def add_action(name, theta):
        if not name or not theta:
            return
        theta = dict(theta)
        theta.setdefault("name", name)
        bank[name] = theta

    for _, obj in t.get("worlds", {}).items():
        for key in ["ranked_actions", "ppo_action_prior", "actions"]:
            for a in obj.get(key, []) or []:
                name = a.get("action_name") or a.get("name")
                theta = a.get("theta_action") or a.get("theta")
                if theta is None and name:
                    maybe = {
                        k: a[k]
                        for k in [
                            "vx_far",
                            "vx_near",
                            "goal_slow_distance",
                            "goal_stop_distance",
                            "body_height",
                            "swing_clearance",
                        ]
                        if k in a
                    }
                    theta = maybe if maybe else None
                add_action(name, theta)

        if "top_action" in obj:
            a = obj["top_action"]
            add_action(a.get("action_name") or a.get("name"), a.get("theta_action") or a.get("theta"))

    if not bank:
        raise RuntimeError(f"No actions found in teacher json: {teacher_json}")

    return bank


def run_cmd(cmd, timeout):
    print("[run]", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    print(proc.stdout, flush=True)
    return proc.returncode, proc.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase_b_training_pipeline_v1/training_campaign_v1.json")
    ap.add_argument("--out-root", default="artifacts/phase_b_training_campaign_v1")
    ap.add_argument("--world-filter", default="")
    ap.add_argument("--action-filter", default="")
    ap.add_argument("--repeats-override", type=int, default=None)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_json(args.config)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    teacher_json = cfg["teacher_json"]
    bank = build_action_bank(teacher_json)

    worlds = cfg["worlds"]
    actions = cfg["actions"]
    repeats = args.repeats_override if args.repeats_override is not None else int(cfg["repeats_per_world_action"])

    if args.world_filter:
        allowed = set(x.strip() for x in args.world_filter.split(",") if x.strip())
        worlds = [w for w in worlds if w in allowed]

    if args.action_filter:
        allowed = set(x.strip() for x in args.action_filter.split(",") if x.strip())
        actions = [a for a in actions if a in allowed]

    plan = []
    for world in worlds:
        for action in actions:
            if action not in bank:
                print(f"[skip] action not in bank: {action}", flush=True)
                continue
            for ep in range(repeats):
                run_dir = out_root / world / action / f"ep{ep:03d}"
                result_path = run_dir / "ppo_policy_episode_result_v0.json"
                plan.append({
                    "world": world,
                    "action": action,
                    "episode_index": ep,
                    "run_dir": str(run_dir),
                    "result_path": str(result_path),
                    "exists": result_path.exists(),
                })

    if args.max_episodes is not None:
        plan = plan[:args.max_episodes]

    manifest = {
        "schema": "phase_b_training_campaign_manifest_v1",
        "config": cfg,
        "out_root": str(out_root),
        "num_planned": len(plan),
        "plan": plan,
    }
    save_json(out_root / "campaign_manifest_v1.json", manifest)

    print(json.dumps({
        "out_root": str(out_root),
        "num_planned": len(plan),
        "dry_run": args.dry_run,
        "resume": args.resume
    }, indent=2), flush=True)

    if args.dry_run:
        return

    progress_path = out_root / "campaign_progress_v1.jsonl"

    for i, item in enumerate(plan):
        world = item["world"]
        action = item["action"]
        run_dir = Path(item["run_dir"])
        result_path = Path(item["result_path"])

        if args.resume and result_path.exists():
            print(f"[resume skip] {world} {action} ep={item['episode_index']} {result_path}", flush=True)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        start_t = time.time()
        cmd = [
            "python3",
            "scripts/runtime/tracer_run_phase_b_fixed_theta_episode_v0.py",
            "--world-name", world,
            "--action-name", action,
            "--out-root", str(run_dir),
            "--teacher-json", teacher_json,
            "--record-duration", str(cfg["record_duration"]),
            "--sample-hz", str(cfg["sample_hz"]),
            "--runner-timeout", str(cfg["runner_timeout"]),
        ]

        status = "ok"
        error = ""

        try:
            rc, output = run_cmd(cmd, timeout=float(cfg["runner_timeout"]) + 60.0)
            if rc != 0:
                status = "failed"
                error = output[-4000:]
        except subprocess.TimeoutExpired as e:
            status = "timeout"
            error = str(e)

        elapsed = time.time() - start_t

        progress = {
            "schema": "phase_b_training_campaign_progress_row_v1",
            "index": i,
            "world": world,
            "action": action,
            "episode_index": item["episode_index"],
            "run_dir": str(run_dir),
            "result_path": str(result_path),
            "status": status,
            "elapsed_sec": elapsed,
            "error": error,
        }
        append_jsonl(progress_path, progress)
        print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    main()
