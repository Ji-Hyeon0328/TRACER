#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

POLICY_TERRAIN="${1:?usage: $0 <policy_terrain_key> <gazebo_world_name> [duration_sec] [tag]}"
WORLD="${2:?usage: $0 <policy_terrain_key> <gazebo_world_name> [duration_sec] [tag]}"
DURATION="${3:-10}"
TAG="${4:-fusion_${POLICY_TERRAIN}}"

echo "============================================================"
echo "[TRACER] fusion policy sanity run"
echo "============================================================"
echo "[TRACER] policy terrain: $POLICY_TERRAIN"
echo "[TRACER] gazebo world:   $WORLD"
echo "[TRACER] duration:       $DURATION"
echo "[TRACER] tag:            $TAG"
echo "============================================================"

WORLD_FILE_CHECK_CMD="
if [ -f /root/unitree_ws/src/unitree_ros/unitree_gazebo/worlds/${WORLD}.world ]; then
  echo FOUND
else
  echo MISSING
fi
"

FOUND="$(sudo docker exec a1_unitree_gazebo_docker bash -lc "$WORLD_FILE_CHECK_CMD" | tail -n 1 || true)"
if [ "$FOUND" != "FOUND" ]; then
  echo "[TRACER][ERROR] world file not found: ${WORLD}.world"
  echo "[TRACER] available matching worlds:"
  sudo docker exec a1_unitree_gazebo_docker bash -lc "
    ls /root/unitree_ws/src/unitree_ros/unitree_gazebo/worlds | grep -E 'slippery|sponge|slope|earth|rough' | sort
  " || true
  exit 1
fi

echo
echo "========== stop previous logger =========="
"$ROOT/scripts/runtime/tracer_stop_mission_logger.sh" || true

echo
echo "========== launch world =========="
"$ROOT/scripts/runtime/tracer_launch_gazebo_world_clean.sh" "$WORLD"

echo
echo "========== wait for Gazebo services/model settle =========="
sudo docker exec a1_unitree_gazebo_docker bash -lc '
set +u
source /opt/ros/melodic/setup.bash

echo "[TRACER] waiting for /gazebo/get_world_properties..."
for i in $(seq 1 20); do
  if timeout 3 rosservice call /gazebo/get_world_properties "{}" >/tmp/tracer_world_props_wait.log 2>&1; then
    echo "[TRACER] get_world_properties OK"
    break
  fi
  echo "[TRACER] waiting world service... $i"
  sleep 1
done

echo "[TRACER] waiting for model a1_gazebo..."
for i in $(seq 1 20); do
  if timeout 3 rosservice call /gazebo/get_model_state "{model_name: a1_gazebo, relative_entity_name: world}" >/tmp/tracer_a1_state_wait.log 2>&1; then
    if grep -q "success: True" /tmp/tracer_a1_state_wait.log; then
      echo "[TRACER] a1_gazebo model state OK"
      break
    fi
  fi
  echo "[TRACER] waiting a1_gazebo... $i"
  sleep 1
done

echo "[TRACER] extra settle sleep 3s"
sleep 3

echo "[TRACER] pause physics before reset"
timeout 5 rosservice call /gazebo/pause_physics "{}" >/tmp/tracer_pause_before_reset.log 2>&1 || cat /tmp/tracer_pause_before_reset.log || true
'

echo
echo "========== reset/controller/qwerty =========="
"$ROOT/scripts/runtime/tracer_robust_reset_to_qwer_state.sh"
"$ROOT/scripts/runtime/tracer_start_a1_qpmc_controller_only.sh"
"$ROOT/scripts/runtime/tracer_validate_a1_ready_standing_pose.sh"
"$ROOT/scripts/runtime/tracer_start_qwerty_state.sh"

echo
echo "========== fusion overlay =========="
"$ROOT/scripts/runtime/tracer_start_fusion_policy_overlay_from_qwerty.sh" "$POLICY_TERRAIN" 180

echo
echo "========== ROS1 command check =========="
sudo docker exec -i a1_cpp_ctrl_docker bash -lc '
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic echo -n 1 /tracer/mpc_reference
'

echo
echo "========== ROS2 beta check =========="
set +u
source /opt/ros/humble/setup.bash
set -u
ros2 topic echo /tracer/objective_weights --once || true

echo
echo "========== optional online RAM monitor =========="
if [ "${TRACER_ENABLE_RAM_MONITOR:-0}" = "1" ]; then
  echo "[TRACER] online RAM monitor enabled"
  "$ROOT/scripts/runtime/tracer_start_odom_bridge_only.sh"

  export TRACER_RAM_REQUIRE_ODOM="${TRACER_RAM_REQUIRE_ODOM:-1}"
  export TRACER_RAM_CLIENT_LOG="${TRACER_RAM_CLIENT_LOG:-/tmp/tracer_online_ram_${TAG}.log}"

  "$ROOT/scripts/runtime/tracer_start_online_ram_client_v0.sh"

  if [ "${TRACER_ENABLE_RAM_GATE_MONITOR:-0}" = "1" ]; then
    echo "[TRACER] RAM gate monitor enabled"
    export TRACER_GATE_POLICY_TERRAIN="$POLICY_TERRAIN"
    export TRACER_GATE_POLICY_JSON="${TRACER_GATE_POLICY_JSON:-$TRACER_FUSION_POLICY_JSON}"
    export TRACER_RAM_GATE_MONITOR_LOG="${TRACER_RAM_GATE_MONITOR_LOG:-/tmp/tracer_ram_gate_monitor_${TAG}.log}"
    "$ROOT/scripts/runtime/tracer_start_ram_gate_monitor_v0.sh" "$POLICY_TERRAIN"
  else
    echo "[TRACER] RAM gate monitor disabled"
  fi
else
  echo "[TRACER] online RAM monitor disabled"
fi

echo
echo "========== logger + physics pulse =========="
"$ROOT/scripts/runtime/tracer_start_mission_logger.sh" "$TAG"
"$ROOT/scripts/runtime/tracer_pulse_gazebo_physics.sh" "$DURATION"
"$ROOT/scripts/runtime/tracer_stop_mission_logger.sh"

echo
echo "========== latest finite metric =========="
python3 - <<'PY'
from pathlib import Path
import json
import numpy as np

logs = sorted(Path("data/mission_logs").glob("tracer_mission_log_*.npz"))
summaries = sorted(Path("data/mission_logs").glob("tracer_mission_summary_*.json"))

print("latest npz:", logs[-1] if logs else None)
print("latest summary:", summaries[-1] if summaries else None)

if summaries:
    print("\n========== summary ==========")
    print(Path(summaries[-1]).read_text())

if not logs:
    raise SystemExit

d = np.load(logs[-1], allow_pickle=True)
pro = d["proprio"]
mpc = d["mpc_reference"]

valid_p = np.isfinite(pro).all(axis=1)
valid_m = np.isfinite(mpc).all(axis=1)

print("\n========== valid samples ==========")
print("proprio:", int(valid_p.sum()), "/", len(valid_p))
print("mpc:", int(valid_m.sum()), "/", len(valid_m))

pv = pro[valid_p]
mv = mpc[valid_m]

if len(pv) > 1:
    x0, y0, z0 = pv[0,1], pv[0,2], pv[0,3]
    x1, y1, z1 = pv[-1,1], pv[-1,2], pv[-1,3]

    print("\n========== finite motion ==========")
    print("start xyz:", float(x0), float(y0), float(z0))
    print("end   xyz:", float(x1), float(y1), float(z1))
    print("delta x:", float(x1 - x0))
    print("delta y:", float(y1 - y0))
    print("min z:", float(pv[:,3].min()))
    print("max |roll|:", float(np.abs(pv[:,4]).max()))
    print("max |pitch|:", float(np.abs(pv[:,5]).max()))

if len(mv) > 1:
    print("\n========== finite mpc reference ==========")
    print("first:", mv[0].tolist())
    print("last: ", mv[-1].tolist())
    print("enable fraction:", float(np.mean(mv[:,5] > 0.5)))
    print("vx mean:", float(np.mean(mv[:,1])))
    print("body height mean:", float(np.mean(mv[:,3])))
    print("clearance mean:", float(np.mean(mv[:,4])))
PY

echo
echo "========== latest terrain-relative metric =========="
TRACER_WORLD_NAME="$WORLD" "$ROOT/scripts/runtime/tracer_analyze_latest_mission_relative_height.py" || true

echo
echo "========== append sanity result CSV =========="
TRACER_TERRAIN_KEY="$POLICY_TERRAIN" \
TRACER_WORLD_NAME="$WORLD" \
TRACER_SANITY_TAG="$TAG" \
TRACER_FUSION_POLICY_JSON="${TRACER_FUSION_POLICY_JSON:-$ROOT/configs/highlevel_policy/tracer_fusion_policy_v0.json}" \
"$ROOT/scripts/runtime/tracer_append_latest_sanity_result.py" || true

echo
echo "========== optional RAM monitor summary =========="
if [ "${TRACER_ENABLE_RAM_MONITOR:-0}" = "1" ]; then
  TRACER_TERRAIN_KEY="$POLICY_TERRAIN" \
  TRACER_WORLD_NAME="$WORLD" \
  TRACER_SANITY_TAG="$TAG" \
  TRACER_RAM_CLIENT_LOG="${TRACER_RAM_CLIENT_LOG:-/tmp/tracer_online_ram_${TAG}.log}" \
  "$ROOT/scripts/runtime/tracer_summarize_ram_client_log.py" || true
else
  echo "[TRACER] online RAM monitor summary skipped"
fi

echo
echo "========== optional RAM gate monitor summary =========="
if [ "${TRACER_ENABLE_RAM_GATE_MONITOR:-0}" = "1" ]; then
  TRACER_TERRAIN_KEY="$POLICY_TERRAIN" \
  TRACER_WORLD_NAME="$WORLD" \
  TRACER_SANITY_TAG="$TAG" \
  TRACER_RAM_GATE_MONITOR_LOG="${TRACER_RAM_GATE_MONITOR_LOG:-/tmp/tracer_ram_gate_monitor_${TAG}.log}" \
  "$ROOT/scripts/runtime/tracer_summarize_ram_gate_log.py" || true
else
  echo "[TRACER] RAM gate monitor summary skipped"
fi

echo
echo "[TRACER] fusion sanity run done"
