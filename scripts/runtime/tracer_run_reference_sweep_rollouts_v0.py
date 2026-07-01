#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("TRACER_ROOT", str(Path.home() / "Tracer" / "TRACER")))


def sh(cmd: str, *, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    print(f"\n[CMD] {cmd}", flush=True)
    return subprocess.run(
        cmd,
        shell=True,
        cwd=str(ROOT),
        executable="/bin/bash",
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=check,
        timeout=timeout,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_simple_reference_yaml(path: Path) -> dict[str, Any]:
    """
    Tiny parser for configs/rollout/tracer_reference_sweep_presets_v0.yaml.
    This intentionally supports only the simple structure used by this config,
    so the runner does not depend on PyYAML being installed in /usr/bin/python3.
    """
    text = read_text(path)
    terrains: list[str] = []
    presets: list[dict[str, Any]] = []

    section = None
    cur: dict[str, Any] | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()

        if not s or s.startswith("#") or s.startswith(">"):
            continue

        if s == "terrains:":
            section = "terrains"
            continue
        if s == "presets:":
            section = "presets"
            if cur:
                presets.append(cur)
                cur = None
            continue

        if section == "terrains":
            m = re.match(r"^-\s+(.+)$", s)
            if m:
                terrains.append(m.group(1).strip().strip('"').strip("'"))
            continue

        if section == "presets":
            m_name = re.match(r"^-\s+name:\s*(.+)$", s)
            if m_name:
                if cur:
                    presets.append(cur)
                cur = {"name": m_name.group(1).strip().strip('"').strip("'")}
                continue

            m_kv = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.+)$", s)
            if m_kv and cur is not None:
                k = m_kv.group(1)
                vraw = m_kv.group(2).strip().strip('"').strip("'")
                try:
                    v: Any = float(vraw)
                except Exception:
                    v = vraw
                cur[k] = v

    if cur:
        presets.append(cur)

    if not terrains:
        terrains = ["flat_normal"]
    if not presets:
        raise RuntimeError(f"No presets parsed from {path}")

    required = ["name", "vx", "yaw_rate", "body_height", "swing_clearance", "enable"]
    for p in presets:
        missing = [k for k in required if k not in p]
        if missing:
            raise RuntimeError(f"Preset missing fields {missing}: {p}")

    return {"terrains": terrains, "presets": presets, "raw_text": text}


def gazebo_physics(action: str, gazebo_container: str) -> None:
    assert action in {"pause", "unpause"}
    service = f"/gazebo/{action}_physics"
    sh(
        f"""docker exec {gazebo_container} bash --noprofile --norc -lc '
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash 2>/dev/null || true
timeout 5 rosservice call {service} "{{}}" >/tmp/tracer_{action}_physics.log 2>&1
'""",
        check=False,
    )



def run_capture(cmd: list[str], *, timeout: float = 5.0) -> str:
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return cp.stdout or ""
    except Exception as e:
        return f"[capture_error] {type(e).__name__}: {e}"


def load_terrain_world_map(path: Path) -> dict[str, str]:
    """
    Tiny parser for configs/terrain_dataset/terrain_set_v0.yaml.
    Only extracts:
      - name: <terrain>
        world_name: <world>
    so we do not depend on PyYAML in /usr/bin/python3.
    """
    if not path.exists():
        return {}

    mapping: dict[str, str] = {}
    cur_name: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        m_name = re.match(r"^-\s+name:\s*(.+)$", s)
        if m_name:
            cur_name = m_name.group(1).strip().strip('"').strip("'")
            continue

        m_world = re.match(r"^world_name:\s*(.+)$", s)
        if m_world and cur_name:
            mapping[cur_name] = m_world.group(1).strip().strip('"').strip("'")

    return mapping


def probe_gazebo_world(gazebo_container: str) -> dict[str, Any]:
    """
    Probe the currently running Gazebo world from the container.
    This is intentionally best-effort and never raises, because rollout should
    still produce diagnostic rows even when Gazebo is unhealthy.
    """
    proc_out = run_capture(
        [
            "docker",
            "exec",
            gazebo_container,
            "bash",
            "--noprofile",
            "--norc",
            "-lc",
            "pgrep -af 'gzserver|roslaunch unitree_gazebo normal.launch' || true",
        ],
        timeout=5.0,
    )

    world_file = ""
    world_name = ""

    m_world = re.search(r"(\S+\.world)", proc_out)
    if m_world:
        world_file = m_world.group(1)
        world_name = Path(world_file).stem
    else:
        m_wname = re.search(r"wname:=([A-Za-z0-9_]+)", proc_out)
        if m_wname:
            world_name = m_wname.group(1)

    world_props = run_capture(
        [
            "docker",
            "exec",
            gazebo_container,
            "bash",
            "--noprofile",
            "--norc",
            "-lc",
            """set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash 2>/dev/null || true
timeout 5 rosservice call /gazebo/get_world_properties "{}" || true
""",
        ],
        timeout=8.0,
    )

    model_names: list[str] = []
    m_models = re.search(r"model_names:\s*\[([^\]]*)\]", world_props)
    if m_models:
        model_names = [
            x.strip().strip("'").strip('"')
            for x in m_models.group(1).split(",")
            if x.strip()
        ]

    return {
        "actual_world_process": proc_out.strip(),
        "actual_world_file": world_file,
        "actual_world_name": world_name,
        "actual_gazebo_model_names": model_names,
        "actual_world_probe_raw": world_props.strip(),
    }


def check_world_match(
    terrain: str,
    expected_world: str,
    probe: dict[str, Any],
) -> tuple[bool, str]:
    actual_world = str(probe.get("actual_world_name", "") or "")

    if not expected_world:
        return True, f"no expected_world mapping for terrain={terrain}"

    if not actual_world:
        return False, f"expected_world={expected_world}, but actual_world could not be detected"

    if actual_world == expected_world:
        return True, f"expected_world={expected_world} matches actual_world={actual_world}"

    return False, f"expected_world={expected_world}, actual_world={actual_world}"


def kill_conflicting_publishers() -> None:
    patterns = [
        "tracer_fusion_policy_mpc_ref_node.py",
        "tracer_objective_selector_stub_node.py",
        "tracer_learned_high_level_policy_udp_client_v1_node.py",
        "tracer_learned_high_level_policy_udp_client_node.py",
        "tracer_high_level_controller_stub_node.py",
        "tracer_runtime_validated_policy_node_v0.py",
        "tracer_safe_bank_policy_node_v0.py",
        "tracer_objective_selector_beta_node_v0.py",
        "ros2 topic pub.*/tracer/mpc_reference",
    ]
    for pat in patterns:
        sh(f"pkill -9 -f '{pat}' 2>/dev/null || true", check=False)


def start_reference_publisher(ref: list[float], hz: float, log_path: Path) -> subprocess.Popen:
    data = ", ".join(f"{x:.6g}" for x in ref)
    cmd = f"""
set +u
source /opt/ros/humble/setup.bash
if [ -f "{ROOT}/ros2_ws/install/setup.bash" ]; then
  source "{ROOT}/ros2_ws/install/setup.bash"
fi
exec ros2 topic pub /tracer/mpc_reference std_msgs/msg/Float64MultiArray "{{data: [{data}]}}" -r {hz}
"""
    log_f = open(log_path, "w", encoding="utf-8")
    print(f"[TRACER] starting reference publisher ref=[{data}] hz={hz} log={log_path}", flush=True)
    return subprocess.Popen(
        ["/bin/bash", "-lc", cmd],
        cwd=str(ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop_process_tree(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        time.sleep(0.8)
    except Exception:
        pass
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def run_recorder(
    terrain: str,
    policy_id: str,
    episode_id: str,
    duration: float,
    sample_hz: float,
    out_dir: Path,
) -> int:
    timeout_sec = max(10.0, duration + 8.0)
    env = os.environ.copy()
    env.update(
        {
            "TRACER_TERRAIN": terrain,
            "TRACER_POLICY_ID": policy_id,
            "TRACER_EPISODE_ID": episode_id,
            "TRACER_ROLLOUT_DURATION_SEC": str(duration),
            "TRACER_ROLLOUT_SAMPLE_HZ": str(sample_hz),
            "TRACER_ROLLOUT_OUT_DIR": str(out_dir),
        }
    )
    cmd = [
        "timeout",
        "--kill-after=2s",
        f"{timeout_sec}s",
        "/usr/bin/python3",
        "scripts/runtime/tracer_record_rollout_episode_v0.py",
    ]
    print(f"[TRACER] recorder start episode={episode_id} timeout={timeout_sec:.1f}s", flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return int(p.returncode)


def load_summary(out_dir: Path, episode_id: str) -> dict[str, Any]:
    path = out_dir / "summaries" / f"{episode_id}.json"
    if not path.exists():
        return {"episode_id": episode_id, "summary_missing": 1}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data["summary_missing"] = 0
            return data
    except Exception as e:
        return {"episode_id": episode_id, "summary_missing": 1, "summary_error": str(e)}
    return {"episode_id": episode_id, "summary_missing": 1}


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rollout/tracer_reference_sweep_presets_v0.yaml")
    ap.add_argument("--terrains", default="", help="Override terrain list, space/comma separated")
    ap.add_argument("--duration-sec", type=float, default=float(os.environ.get("TRACER_REF_SWEEP_DURATION_SEC", "5.0")))
    ap.add_argument("--sample-hz", type=float, default=float(os.environ.get("TRACER_REF_SWEEP_SAMPLE_HZ", "5.0")))
    ap.add_argument("--publish-hz", type=float, default=float(os.environ.get("TRACER_REF_SWEEP_PUBLISH_HZ", "20.0")))
    ap.add_argument("--repeats", type=int, default=int(os.environ.get("TRACER_REF_SWEEP_REPEATS", "1")))
    ap.add_argument("--out-dir", default=os.environ.get("TRACER_REF_SWEEP_OUT_DIR", "data/rollouts/reference_sweep_v0"))
    ap.add_argument("--skip-lite-start", action="store_true")
    ap.add_argument("--skip-reset", action="store_true")
    ap.add_argument("--gazebo-container", default=os.environ.get("TRACER_GAZEBO_CONTAINER", "a1_unitree_gazebo_docker"))
    ap.add_argument("--terrain-set", default=os.environ.get("TRACER_TERRAIN_SET", "configs/terrain_dataset/terrain_set_v0.yaml"))
    ap.add_argument("--strict-world-check", action="store_true")
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    out_root = ROOT / args.out_dir
    run_id = datetime.now().strftime("ref_sweep_v0_%Y%m%d_%H%M%S")
    out_dir = out_root / run_id
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (out_dir / "summaries").mkdir(parents=True, exist_ok=True)

    cfg = parse_simple_reference_yaml(cfg_path)
    terrains = cfg["terrains"]
    if args.terrains.strip():
        terrains = [x for x in re.split(r"[,\s]+", args.terrains.strip()) if x]
    presets = cfg["presets"]
    terrain_world_map = load_terrain_world_map(ROOT / args.terrain_set)

    shutil.copy2(cfg_path, out_dir / "reference_sweep_config.yaml")

    print("[TRACER] reference sweep v0")
    print(f"[TRACER] out_dir:      {out_dir}")
    print(f"[TRACER] terrains:     {terrains}")
    print(f"[TRACER] presets:      {[p['name'] for p in presets]}")
    print(f"[TRACER] repeats:      {args.repeats}")
    print(f"[TRACER] duration_sec: {args.duration_sec}")
    print(f"[TRACER] sample_hz:    {args.sample_hz}")
    print(f"[TRACER] publish_hz:   {args.publish_hz}")
    print(f"[TRACER] terrain_set:  {args.terrain_set}")
    print(f"[TRACER] strict_world_check: {args.strict_world_check}")

    if not args.skip_lite_start:
        sh("scripts/runtime/tracer_start_data_collection_lite.sh", check=True)

    all_rows: list[dict[str, Any]] = []

    for terrain in terrains:
        for preset in presets:
            for rep in range(1, args.repeats + 1):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                preset_name = str(preset["name"])
                policy_id = f"reference_sweep_v0_{preset_name}"
                episode_id = f"{terrain}_{preset_name}_r{rep}_{ts}"

                ref = [
                    0.0,
                    float(preset["vx"]),
                    float(preset["yaw_rate"]),
                    float(preset["body_height"]),
                    float(preset["swing_clearance"]),
                    float(preset["enable"]),
                ]

                print("\n" + "=" * 80)
                print(f"[TRACER] episode={episode_id}")
                print(f"[TRACER] terrain={terrain} preset={preset_name} ref={ref}")

                expected_world = terrain_world_map.get(terrain, "")
                world_probe = probe_gazebo_world(args.gazebo_container)
                world_check_ok, world_check_note = check_world_match(
                    terrain=terrain,
                    expected_world=expected_world,
                    probe=world_probe,
                )

                print(
                    f"[TRACER] world_check ok={world_check_ok} "
                    f"terrain={terrain} expected={expected_world or '<unknown>'} "
                    f"actual={world_probe.get('actual_world_name', '') or '<unknown>'}"
                )
                print(f"[TRACER] world_check note: {world_check_note}")

                if args.strict_world_check and not world_check_ok:
                    raise RuntimeError(
                        f"World check failed for terrain={terrain}: {world_check_note}"
                    )

                kill_conflicting_publishers()

                # Give ROS2 CLI/DDS discovery a moment to drop stale publishers.
                time.sleep(1.0)
                sh(
                    """set +u
source /opt/ros/humble/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then source ros2_ws/install/setup.bash; fi
echo '=== pre-publisher /tracer/mpc_reference topic info ==='
ros2 topic info -v /tracer/mpc_reference || true
""",
                    check=False,
                )

                gazebo_physics("pause", args.gazebo_container)

                if not args.skip_reset:
                    sh("scripts/runtime/tracer_force_reset_gazebo_a1_pose.sh", check=True)

                sh("scripts/runtime/tracer_ensure_mpc_ref_bridge.sh", check=True)
                sh("scripts/runtime/tracer_ensure_ros2_mpc_ref_udp_sender.sh", check=True)

                pub_log = logs_dir / f"{episode_id}_mpc_ref_pub.log"
                pub_proc = start_reference_publisher(ref, args.publish_hz, pub_log)

                time.sleep(1.0)

                sh(
                    """set +u
source /opt/ros/humble/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then source ros2_ws/install/setup.bash; fi
echo '=== ROS2 /tracer/mpc_reference ==='
timeout 3 ros2 topic echo --once /tracer/mpc_reference || true
echo '=== ROS2 topic info ==='
ros2 topic info -v /tracer/mpc_reference || true
""",
                    check=False,
                )

                gazebo_physics("unpause", args.gazebo_container)
                rec_status = run_recorder(
                    terrain=terrain,
                    policy_id=policy_id,
                    episode_id=episode_id,
                    duration=args.duration_sec,
                    sample_hz=args.sample_hz,
                    out_dir=out_dir,
                )
                gazebo_physics("pause", args.gazebo_container)
                stop_process_tree(pub_proc)

                summary = load_summary(out_dir, episode_id)
                row = {
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "terrain": terrain,
                    "declared_terrain": terrain,
                    "expected_world_name": expected_world,
                    "actual_world_name": world_probe.get("actual_world_name", ""),
                    "actual_world_file": world_probe.get("actual_world_file", ""),
                    "actual_world_process": world_probe.get("actual_world_process", ""),
                    "actual_gazebo_model_names": world_probe.get("actual_gazebo_model_names", []),
                    "world_check_ok": world_check_ok,
                    "world_check_note": world_check_note,
                    "preset": preset_name,
                    "repeat": rep,
                    "ref_counter": ref[0],
                    "ref_vx": ref[1],
                    "ref_yaw_rate": ref[2],
                    "ref_body_height": ref[3],
                    "ref_swing_clearance": ref[4],
                    "ref_enable": ref[5],
                    "rec_status": rec_status,
                }
                row.update(summary)
                all_rows.append(flatten_for_csv(row))

                print("[TRACER] episode summary:")
                print(json.dumps(summary, indent=2, ensure_ascii=False))

                if rec_status != 0:
                    print(f"[TRACER][WARN] recorder status={rec_status}; continuing sweep")

    csv_path = out_dir / "reference_sweep_summary.csv"
    keys: list[str] = []
    for row in all_rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for row in all_rows:
            wr.writerow(row)

    print("\n[TRACER] reference sweep done")
    print(f"[TRACER] summary csv: {csv_path}")
    print(f"[TRACER] out_dir:     {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
