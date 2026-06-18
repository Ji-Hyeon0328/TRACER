#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"

LABEL="${1:-${TRACER_OBJECTIVE_LABEL:-balanced}}"
WEIGHTS="${2:-${TRACER_OBJECTIVE_WEIGHTS:-0.3333333333333333,0.3333333333333333,0.3333333333333333}}"
PUB_SEC="${TRACER_OBJECTIVE_PUB_SEC:-2.0}"
RATE="${TRACER_OBJECTIVE_PUB_RATE:-10}"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

case "$LABEL" in
  balanced) MODE=0 ;;
  motion) MODE=1 ;;
  stability) MODE=2 ;;
  energy) MODE=3 ;;
  *) MODE=-1 ;;
esac

echo "[TRACER] set objective"
echo "[TRACER] label:   $LABEL"
echo "[TRACER] weights: $WEIGHTS"
echo "[TRACER] mode:    $MODE"
echo "[TRACER] publish: ${PUB_SEC}s at ${RATE}Hz"

timeout 8 /usr/bin/python3 - "$LABEL" "$WEIGHTS" "$MODE" "$PUB_SEC" "$RATE" <<'PY'
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32

label = sys.argv[1]
weights = [float(x) for x in sys.argv[2].split(",")]
mode = int(sys.argv[3])
pub_sec = float(sys.argv[4])
rate = float(sys.argv[5])

rclpy.init()
node = Node("tracer_objective_weight_setter_once")

pub_weights = node.create_publisher(Float64MultiArray, "/tracer/objective_weights_cmd", 10)
pub_mode = node.create_publisher(Int32, "/tracer/objective_mode", 10)

# Give DDS discovery a short moment.
end_discovery = time.time() + 0.5
while time.time() < end_discovery and rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.05)

period = 1.0 / max(rate, 1e-6)
end = time.time() + pub_sec

count = 0
while time.time() < end and rclpy.ok():
    if mode >= 0:
        m = Int32()
        m.data = mode
        pub_mode.publish(m)

    w = Float64MultiArray()
    w.data = weights
    pub_weights.publish(w)

    rclpy.spin_once(node, timeout_sec=0.0)
    count += 1
    time.sleep(period)

print(f"[TRACER] objective setter published {count} messages: label={label}, weights={weights}, mode={mode}")

node.destroy_node()
rclpy.shutdown()
PY

echo
echo "[TRACER] current /tracer/objective_weights snapshot:"
timeout 4 ros2 topic echo /tracer/objective_weights --once || true

echo
echo "[TRACER] objective setter done."
