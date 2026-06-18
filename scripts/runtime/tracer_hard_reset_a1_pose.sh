#!/usr/bin/env bash
set -eo pipefail

GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"

MODEL_NAME="${TRACER_GAZEBO_MODEL_NAME:-a1_gazebo}"
RESET_X="${TRACER_RESET_X:-0.0}"
RESET_Y="${TRACER_RESET_Y:-0.0}"
RESET_Z="${TRACER_RESET_Z:-0.325}"
RESET_ROLL="${TRACER_RESET_ROLL:-0.0}"
RESET_PITCH="${TRACER_RESET_PITCH:-0.0}"
RESET_YAW="${TRACER_RESET_YAW:-0.0}"

echo "[TRACER] hard reset A1 Gazebo model pose"
echo "[TRACER] model: $MODEL_NAME"
echo "[TRACER] pose:  x=$RESET_X y=$RESET_Y z=$RESET_Z rpy=($RESET_ROLL,$RESET_PITCH,$RESET_YAW)"

sudo docker exec \
  -e MODEL_NAME="$MODEL_NAME" \
  -e RESET_X="$RESET_X" \
  -e RESET_Y="$RESET_Y" \
  -e RESET_Z="$RESET_Z" \
  -e RESET_ROLL="$RESET_ROLL" \
  -e RESET_PITCH="$RESET_PITCH" \
  -e RESET_YAW="$RESET_YAW" \
  "$GAZEBO_CONTAINER" \
  bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash

python - <<PY
import math
import os
import rospy
import time
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist


def quat_from_rpy(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


model_name = os.environ.get("MODEL_NAME", "a1_gazebo")
x = float(os.environ.get("RESET_X", "0.0"))
y = float(os.environ.get("RESET_Y", "0.0"))
z = float(os.environ.get("RESET_Z", "0.325"))
roll = float(os.environ.get("RESET_ROLL", "0.0"))
pitch = float(os.environ.get("RESET_PITCH", "0.0"))
yaw = float(os.environ.get("RESET_YAW", "0.0"))

rospy.init_node("tracer_hard_reset_a1_pose", anonymous=True, disable_signals=True)

rospy.wait_for_service("/gazebo/pause_physics", timeout=5.0)
rospy.wait_for_service("/gazebo/set_model_state", timeout=5.0)

from std_srvs.srv import Empty
pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

pause()

state = ModelState()
state.model_name = model_name
state.reference_frame = "world"
state.pose.position.x = x
state.pose.position.y = y
state.pose.position.z = z
qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)
state.pose.orientation.x = qx
state.pose.orientation.y = qy
state.pose.orientation.z = qz
state.pose.orientation.w = qw
state.twist.linear.x = 0.0
state.twist.linear.y = 0.0
state.twist.linear.z = 0.0
state.twist.angular.x = 0.0
state.twist.angular.y = 0.0
state.twist.angular.z = 0.0

# Call multiple times because some Gazebo plugins can update state immediately after reset.
for i in range(5):
    resp = set_state(state)
    if not resp.success:
        print("[TRACER] set_model_state failed:", resp.status_message)
    time.sleep(0.05)

print("[TRACER] hard reset set_model_state done")
PY
'
