#!/usr/bin/env bash
set -eo pipefail

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

STYLE_NAME="${TRACER_STYLE_NAME:-nominal}"
VX="${TRACER_STYLE_VX:-0.21}"
YAW_RATE="${TRACER_STYLE_YAW_RATE:-0.0}"
BODY_HEIGHT="${TRACER_STYLE_BODY_HEIGHT:-0.30}"
CLEARANCE="${TRACER_STYLE_CLEARANCE:-0.035}"
ENABLE="${TRACER_STYLE_ENABLE:-1.0}"
HZ="${TRACER_STYLE_HZ:-20}"
DURATION="${TRACER_STYLE_DURATION_SEC:-30}"

echo "[TRACER] publish ROS1 style MPC reference"
echo "[TRACER] style:        ${STYLE_NAME}"
echo "[TRACER] vx:           ${VX}"
echo "[TRACER] yaw_rate:     ${YAW_RATE}"
echo "[TRACER] body_height:  ${BODY_HEIGHT}"
echo "[TRACER] clearance:    ${CLEARANCE}"
echo "[TRACER] enable:       ${ENABLE}"
echo "[TRACER] hz:           ${HZ}"
echo "[TRACER] duration:     ${DURATION}s"

sudo docker exec \
  -e TRACER_STYLE_NAME="$STYLE_NAME" \
  -e TRACER_STYLE_VX="$VX" \
  -e TRACER_STYLE_YAW_RATE="$YAW_RATE" \
  -e TRACER_STYLE_BODY_HEIGHT="$BODY_HEIGHT" \
  -e TRACER_STYLE_CLEARANCE="$CLEARANCE" \
  -e TRACER_STYLE_ENABLE="$ENABLE" \
  -e TRACER_STYLE_HZ="$HZ" \
  -e TRACER_STYLE_DURATION_SEC="$DURATION" \
  "$CTRL_CONTAINER" \
  bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash

python - <<PY
import os
import time
import rospy
from std_msgs.msg import Float64MultiArray

style_name = os.environ.get("TRACER_STYLE_NAME", "nominal")
vx = float(os.environ.get("TRACER_STYLE_VX", "0.21"))
yaw_rate = float(os.environ.get("TRACER_STYLE_YAW_RATE", "0.0"))
body_height = float(os.environ.get("TRACER_STYLE_BODY_HEIGHT", "0.30"))
clearance = float(os.environ.get("TRACER_STYLE_CLEARANCE", "0.035"))
enable = float(os.environ.get("TRACER_STYLE_ENABLE", "1.0"))
hz = float(os.environ.get("TRACER_STYLE_HZ", "20"))
duration = float(os.environ.get("TRACER_STYLE_DURATION_SEC", "30"))

rospy.init_node("tracer_style_mpc_ref_ros1_publisher", anonymous=True)
pub = rospy.Publisher("/tracer/mpc_reference", Float64MultiArray, queue_size=10)

time.sleep(0.5)

rate = rospy.Rate(hz)
t0 = time.time()
counter = 0

print("[TRACER] ROS1 style publisher started: style=%s vx=%.3f yaw=%.3f h=%.3f clear=%.3f enable=%.1f duration=%.1f" %
      (style_name, vx, yaw_rate, body_height, clearance, enable, duration))

while not rospy.is_shutdown() and (time.time() - t0) < duration:
    msg = Float64MultiArray()
    msg.data = [float(counter), vx, yaw_rate, body_height, clearance, enable]
    pub.publish(msg)
    counter += 1
    rate.sleep()

print("[TRACER] ROS1 style publisher finished: sent=%d" % counter)
PY
'
