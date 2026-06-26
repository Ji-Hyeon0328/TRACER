#!/usr/bin/env bash
set -euo pipefail

TERRAIN_NAME="${1:?usage: $0 TERRAIN_NAME [DURATION_SEC]}"
DURATION="${2:-10}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BODY_H="${TRACER_ZERO_HOLD_BODY_HEIGHT:-0.325}"
CLR="${TRACER_ZERO_HOLD_CLEARANCE:-0.04}"
TOPIC="${TRACER_MPC_REF_TOPIC:-/tracer/mpc_reference}"

TAG="zero_vx_hold_${TERRAIN_NAME}"

echo "[TRACER] zero-vx hold test"
echo "[TRACER] terrain: $TERRAIN_NAME"
echo "[TRACER] duration: $DURATION"
echo "[TRACER] ref: [0.0, 0.0, 0.0, $BODY_H, $CLR, 1.0]"

echo
echo "========== cleanup =========="
pkill -f tracer_publish_gms_lowlevel_ref_ros2.py 2>/dev/null || true
pkill -9 -f tracer_learned_high_level_policy_udp_client_v1_node.py 2>/dev/null || true
pkill -f "ros2 topic pub $TOPIC" 2>/dev/null || true
kill "$(cat /tmp/tracer_zero_vx_hold_pub.pid 2>/dev/null)" 2>/dev/null || true

echo
echo "========== reset/controller/qwerty =========="
scripts/runtime/tracer_robust_reset_to_qwer_state.sh
scripts/runtime/tracer_start_a1_qpmc_controller_only.sh
scripts/runtime/tracer_validate_a1_ready_standing_pose.sh
scripts/runtime/tracer_start_qwerty_state.sh

echo
echo "========== keep bridge, remove learned publisher =========="
sleep 1
pkill -9 -f tracer_learned_high_level_policy_udp_client_v1_node.py 2>/dev/null || true
scripts/runtime/tracer_ensure_mpc_ref_bridge.sh

echo
echo "========== start zero-vx ROS2 publisher =========="
# ROS setup scripts may reference unset variables internally, so disable nounset while sourcing.
set +u
source /opt/ros/humble/setup.bash
set -u
ros2 topic pub "$TOPIC" std_msgs/msg/Float64MultiArray \
"{data: [0.0, 0.0, 0.0, ${BODY_H}, ${CLR}, 1.0]}" \
-r 10 > /tmp/tracer_zero_vx_hold_pub.log 2>&1 &

echo $! > /tmp/tracer_zero_vx_hold_pub.pid
sleep 1

echo
echo "========== ROS2 topic check =========="
ros2 topic info "$TOPIC" -v || true
timeout 5 ros2 topic echo "$TOPIC" --once || true

echo
echo "========== ROS1 topic check =========="
sudo docker exec -i a1_cpp_ctrl_docker bash -lc "
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
timeout 8 rostopic echo -n 1 $TOPIC
"

echo
echo "========== mission =========="
scripts/runtime/tracer_start_mission_logger.sh "$TAG"
scripts/runtime/tracer_pulse_gazebo_physics.sh "$DURATION"
scripts/runtime/tracer_stop_mission_logger.sh

kill "$(cat /tmp/tracer_zero_vx_hold_pub.pid 2>/dev/null)" 2>/dev/null || true

echo
echo "========== finite-only analysis =========="
python3 - <<'PY'
from pathlib import Path
import numpy as np
import json

logs = sorted(Path("data/mission_logs").glob("tracer_mission_log_*.npz"))
summaries = sorted(Path("data/mission_logs").glob("tracer_mission_summary_*.json"))

log = logs[-1]
summary = summaries[-1]

print("log:", log)
print("summary:", summary)
print(Path(summary).read_text())

d = np.load(log, allow_pickle=True)
p = d["proprio"]
m = d["mpc_reference"]

valid_p = np.isfinite(p).all(axis=1)
valid_m = np.isfinite(m).all(axis=1)

print("\nvalid proprio:", int(valid_p.sum()), "/", len(valid_p))
print("valid mpc:", int(valid_m.sum()), "/", len(valid_m))

result = {
    "log": str(log),
    "summary": str(summary),
}

if valid_p.any():
    pp = p[valid_p]
    x0, y0, z0 = pp[0,1], pp[0,2], pp[0,3]
    x1, y1, z1 = pp[-1,1], pp[-1,2], pp[-1,3]
    fall_idx = np.where(pp[:,3] < 0.18)[0]

    print("\nfinite-only motion:")
    print("first finite xyz:", float(x0), float(y0), float(z0))
    print("last  finite xyz:", float(x1), float(y1), float(z1))
    print("delta x:", float(x1 - x0))
    print("delta y:", float(y1 - y0))
    print("min z:", float(pp[:,3].min()))
    print("max z:", float(pp[:,3].max()))
    print("max |roll|:", float(np.nanmax(np.abs(pp[:,4]))))
    print("max |pitch|:", float(np.nanmax(np.abs(pp[:,5]))))
    print("first z < 0.18 index:", int(fall_idx[0]) if len(fall_idx) else None)

if valid_m.any():
    mm = m[valid_m]
    print("\nmpc reference:")
    print("first finite:", mm[0].tolist())
    print("last  finite:", mm[-1].tolist())
    print("mean:", np.mean(mm, axis=0).tolist())
    print("min:", np.min(mm, axis=0).tolist())
    print("max:", np.max(mm, axis=0).tolist())
PY
