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
sys.path.insert(0, str(ROOT))

from tracer_core.highlevel.decoder_mapper import decode_meta_gait_to_low_level_ref
from tracer_core.highlevel.theta_direct_mapper import (
    meta_gait_from_normalized_theta,
    normalized_theta_to_physical,
)


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


def parse_float_list(s: str) -> list[float]:
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise RuntimeError(f"Expected list syntax, got: {s}")
    inner = s[1:-1].strip()
    if not inner:
        return []
    return [float(x.strip()) for x in inner.split(",")]


def parse_simple_theta_yaml(path: Path) -> dict[str, Any]:
    """
    Tiny parser for configs/rollout/tracer_theta_sweep_presets_v0.yaml.
    This intentionally avoids PyYAML because this runner may be executed with
    a runtime Python that does not have extra packages installed.
    """
    text = read_text(path)

    terrains: list[str] = []
    presets: list[dict[str, Any]] = []
    beta: dict[str, float] = {}

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
        if s == "beta:":
            section = "beta"
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

        if section == "beta":
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.+)$", s)
            if m:
                beta[m.group(1)] = float(m.group(2).strip().strip('"').strip("'"))
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
                if k == "theta":
                    cur[k] = parse_float_list(vraw)
                else:
                    try:
                        cur[k] = float(vraw)
                    except Exception:
                        cur[k] = vraw

    if cur:
        presets.append(cur)

    if not terrains:
        terrains = ["flat_normal"]

    if not beta:
        beta = {"motion": 1.0 / 3.0, "stability": 1.0 / 3.0, "energy": 1.0 / 3.0}

    if not presets:
        raise RuntimeError(f"No theta presets parsed from {path}")

    for p in presets:
        if "name" not in p:
            raise RuntimeError(f"Preset missing name: {p}")
        if "theta" not in p:
            raise RuntimeError(f"Preset missing theta: {p}")
        if len(p["theta"]) != 8:
            raise RuntimeError(f"theta must have length 8: {p}")

    return {
        "terrains": terrains,
        "beta": beta,
        "presets": presets,
        "raw_text": text,
    }


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


def kill_conflicting_publishers() -> None:
    patterns = [
        "tracer_fusion_policy_mpc_ref_node.py",
        "tracer_objective_selector_stub_node.py",
        "tracer_learned_high_level_policy_udp_client_v1_node.py",
        "tracer_learned_high_level_policy_udp_client_node.py",
        "tracer_high_level_controller_stub_node.py",
        "tracer_record_rollout_episode_v0.py",
        "tracer_rollout_episode_recorder_v0",
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
    print(f"[TRACER] starting theta-decoded ref publisher ref=[{data}] hz={hz} log={log_path}", flush=True)
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
    ap.add_argument("--config", default="configs/rollout/tracer_theta_sweep_presets_v0.yaml")
    ap.add_argument("--terrains", default="", help="Override terrain list, space/comma separated")
    ap.add_argument("--duration-sec", type=float, default=float(os.environ.get("TRACER_THETA_SWEEP_DURATION_SEC", "5.0")))
    ap.add_argument("--sample-hz", type=float, default=float(os.environ.get("TRACER_THETA_SWEEP_SAMPLE_HZ", "5.0")))
    ap.add_argument("--publish-hz", type=float, default=float(os.environ.get("TRACER_THETA_SWEEP_PUBLISH_HZ", "20.0")))
    ap.add_argument("--repeats", type=int, default=int(os.environ.get("TRACER_THETA_SWEEP_REPEATS", "1")))
    ap.add_argument("--out-dir", default=os.environ.get("TRACER_THETA_SWEEP_OUT_DIR", "data/rollouts/theta_sweep_v0"))
    ap.add_argument("--skip-lite-start", action="store_true")
    ap.add_argument("--skip-reset", action="store_true")
    ap.add_argument("--gazebo-container", default=os.environ.get("TRACER_GAZEBO_CONTAINER", "a1_unitree_gazebo_docker"))
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    out_root = ROOT / args.out_dir
    run_id = datetime.now().strftime("theta_sweep_v0_%Y%m%d_%H%M%S")
    out_dir = out_root / run_id
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (out_dir / "summaries").mkdir(parents=True, exist_ok=True)

    cfg = parse_simple_theta_yaml(cfg_path)
    terrains = cfg["terrains"]
    if args.terrains.strip():
        terrains = [x for x in re.split(r"[,\s]+", args.terrains.strip()) if x]

    beta = cfg["beta"]
    presets = cfg["presets"]

    shutil.copy2(cfg_path, out_dir / "theta_sweep_config.yaml")

    print("[TRACER] theta sweep v0")
    print(f"[TRACER] out_dir:      {out_dir}")
    print(f"[TRACER] terrains:     {terrains}")
    print(f"[TRACER] beta:         {beta}")
    print(f"[TRACER] presets:      {[p['name'] for p in presets]}")
    print(f"[TRACER] repeats:      {args.repeats}")
    print(f"[TRACER] duration_sec: {args.duration_sec}")
    print(f"[TRACER] sample_hz:    {args.sample_hz}")
    print(f"[TRACER] publish_hz:   {args.publish_hz}")

    if not args.skip_lite_start:
        sh("scripts/runtime/tracer_start_data_collection_lite.sh", check=True)

    all_rows: list[dict[str, Any]] = []

    for terrain in terrains:
        for preset in presets:
            for rep in range(1, args.repeats + 1):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                preset_name = str(preset["name"])
                theta_norm = [float(x) for x in preset["theta"]]

                meta = meta_gait_from_normalized_theta(theta_norm, beta=beta)
                ref_obj = decode_meta_gait_to_low_level_ref(meta)
                ref = ref_obj.to_mpc_reference_array(counter=0.0)
                theta_physical = normalized_theta_to_physical(theta_norm)

                policy_id = f"theta_sweep_v0_{preset_name}"
                episode_id = f"{terrain}_{preset_name}_r{rep}_{ts}"

                print("\n" + "=" * 80)
                print(f"[TRACER] episode={episode_id}")
                print(f"[TRACER] terrain={terrain} preset={preset_name}")
                print(f"[TRACER] theta_norm={theta_norm}")
                print(f"[TRACER] theta_physical={json.dumps(theta_physical, sort_keys=True)}")
                print(f"[TRACER] decoded_ref={ref}")

                kill_conflicting_publishers()

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

                # Give ROS2 graph discovery time to drop recorder subscriptions.
                sh("pkill -9 -f tracer_record_rollout_episode_v0.py 2>/dev/null || true", check=False)
                sh("pkill -9 -f tracer_rollout_episode_recorder_v0 2>/dev/null || true", check=False)
                time.sleep(1.0)

                summary = load_summary(out_dir, episode_id)

                row = {
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "terrain": terrain,
                    "preset": preset_name,
                    "repeat": rep,
                    "theta_norm": theta_norm,
                    "theta_physical": theta_physical,
                    "theta_source_mode": meta.source_mode,
                    "theta_source_reason": meta.source_reason,
                    "theta_vx": meta.vx,
                    "theta_yaw_rate": meta.yaw_rate,
                    "theta_body_height": meta.body_height,
                    "theta_swing_clearance": meta.swing_clearance,
                    "theta_enable": meta.enable,
                    "theta_gait_period": meta.gait_period,
                    "theta_duty_factor": meta.duty_factor,
                    "theta_step_length": meta.step_length,
                    "theta_stance_width": meta.stance_width,
                    "theta_impedance_scale": meta.impedance_scale,
                    "theta_residual_gain_scale": meta.residual_gain_scale,
                    "theta_risk_scale": meta.risk_scale,
                    "beta_motion": float(beta.get("motion", 1.0 / 3.0)),
                    "beta_stability": float(beta.get("stability", 1.0 / 3.0)),
                    "beta_energy": float(beta.get("energy", 1.0 / 3.0)),
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

    csv_path = out_dir / "theta_sweep_summary.csv"
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

    print("\n[TRACER] theta sweep done")
    print(f"[TRACER] summary csv: {csv_path}")
    print(f"[TRACER] out_dir:     {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
