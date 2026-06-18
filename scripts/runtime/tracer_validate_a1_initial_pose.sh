#!/usr/bin/env bash
set -eo pipefail

GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"

MODEL_NAME="${TRACER_GAZEBO_MODEL_NAME:-a1_gazebo}"
MIN_Z="${TRACER_INIT_MIN_Z:-0.22}"
MAX_ABS_ROLL="${TRACER_INIT_MAX_ABS_ROLL:-0.45}"
MAX_ABS_PITCH="${TRACER_INIT_MAX_ABS_PITCH:-0.45}"

sudo docker exec \
  -e MODEL_NAME="$MODEL_NAME" \
  -e MIN_Z="$MIN_Z" \
  -e MAX_ABS_ROLL="$MAX_ABS_ROLL" \
  -e MAX_ABS_PITCH="$MAX_ABS_PITCH" \
  "$GAZEBO_CONTAINER" \
  bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash

python - <<PY
import math
import os
import rospy
from gazebo_msgs.srv import GetModelState

model_name = os.environ.get("MODEL_NAME", "a1_gazebo")
min_z = float(os.environ.get("MIN_Z", "0.22"))
max_abs_roll = float(os.environ.get("MAX_ABS_ROLL", "0.45"))
max_abs_pitch = float(os.environ.get("MAX_ABS_PITCH", "0.45"))

def rpy_from_quat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w*y - z*x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

rospy.init_node("tracer_validate_a1_initial_pose", anonymous=True, disable_signals=True)
rospy.wait_for_service("/gazebo/get_model_state", timeout=5.0)
get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

resp = get_state(model_name, "world")
if not resp.success:
    print("[ERROR] get_model_state failed:", resp.status_message)
    raise SystemExit(20)

p = resp.pose.position
q = resp.pose.orientation
roll, pitch, yaw = rpy_from_quat(q)

print("[TRACER] initial pose validation:")
print("  xyz=(%.4f, %.4f, %.4f)" % (p.x, p.y, p.z))
print("  rpy=(%.4f, %.4f, %.4f)" % (roll, pitch, yaw))

ok = True
if p.z < min_z:
    print("[ERROR] base z too low: %.4f < %.4f" % (p.z, min_z))
    ok = False
if abs(roll) > max_abs_roll:
    print("[ERROR] abs roll too large: %.4f > %.4f" % (abs(roll), max_abs_roll))
    ok = False
if abs(pitch) > max_abs_pitch:
    print("[ERROR] abs pitch too large: %.4f > %.4f" % (abs(pitch), max_abs_pitch))
    ok = False

if not ok:
    raise SystemExit(30)

print("[TRACER] initial pose validation passed")
PY
'
