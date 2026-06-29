#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${TRACER_CTRL_CONTAINER:-a1_cpp_ctrl_docker}"
Z="${TRACER_RESET_Z:-0.36}"
THIGH="${TRACER_RESET_THIGH:-0.80}"
CALF="${TRACER_RESET_CALF:--1.60}"
HIP="${TRACER_RESET_HIP:-0.00}"

docker exec \
  -e TRACER_RESET_Z_INNER="$Z" \
  -e TRACER_RESET_HIP_INNER="$HIP" \
  -e TRACER_RESET_THIGH_INNER="$THIGH" \
  -e TRACER_RESET_CALF_INNER="$CALF" \
  "$CONTAINER" bash -lc '
source /opt/ros/melodic/setup.bash
python - <<'"'"'PY'"'"'
import os
import rospy

from gazebo_msgs.srv import (
    GetWorldProperties,
    GetModelProperties,
    SetModelState,
    SetModelConfiguration,
)
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist

z = float(os.environ.get("TRACER_RESET_Z_INNER", "0.36"))
hip = float(os.environ.get("TRACER_RESET_HIP_INNER", "0.0"))
thigh = float(os.environ.get("TRACER_RESET_THIGH_INNER", "0.80"))
calf = float(os.environ.get("TRACER_RESET_CALF_INNER", "-1.60"))

rospy.init_node("tracer_force_reset_a1_pose_once", anonymous=True)

rospy.wait_for_service("/gazebo/pause_physics", timeout=5.0)
try:
    pause = rospy.ServiceProxy("/gazebo/pause_physics", __import__("std_srvs.srv").srv.Empty)
    pause()
except Exception as e:
    print("[TRACER][WARN] pause failed:", e)

rospy.wait_for_service("/gazebo/get_world_properties", timeout=5.0)
get_world = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
world = get_world()

names = list(world.model_names)
print("[TRACER] model names:", names)

candidates = [
    n for n in names
    if ("a1" in n.lower())
    or ("unitree" in n.lower())
    or ("go1" in n.lower())
    or ("robot" in n.lower())
]

if not candidates:
    raise RuntimeError("Could not find A1-like robot model in Gazebo model list")

model_name = candidates[0]
print("[TRACER] selected model:", model_name)

rospy.wait_for_service("/gazebo/get_model_properties", timeout=5.0)
get_model = rospy.ServiceProxy("/gazebo/get_model_properties", GetModelProperties)
props = get_model(model_name)

joint_names = list(props.joint_names)
print("[TRACER] joint names:", joint_names)

joint_positions = []
selected_joint_names = []

for j in joint_names:
    lj = j.lower()

    # Skip fixed/internal joints if any.
    if "fixed" in lj:
        continue

    if "calf" in lj or "shank" in lj or "knee" in lj:
        q = calf
    elif "thigh" in lj or "upper" in lj:
        q = thigh
    elif "hip" in lj or "abad" in lj:
        q = hip
    else:
        # Unknown joints are left at 0.0 rather than omitted.
        q = 0.0

    selected_joint_names.append(j)
    joint_positions.append(q)

print("[TRACER] reset joint names:", selected_joint_names)
print("[TRACER] reset joint positions:", joint_positions)

if selected_joint_names:
    rospy.wait_for_service("/gazebo/set_model_configuration", timeout=5.0)
    set_cfg = rospy.ServiceProxy("/gazebo/set_model_configuration", SetModelConfiguration)
    resp_cfg = set_cfg(
        model_name=model_name,
        urdf_param_name="robot_description",
        joint_names=selected_joint_names,
        joint_positions=joint_positions,
    )
    print("[TRACER] set_model_configuration:", resp_cfg.success, resp_cfg.status_message)

state = ModelState()
state.model_name = model_name
state.reference_frame = "world"
state.pose.position.x = 0.0
state.pose.position.y = 0.0
state.pose.position.z = z
state.pose.orientation.x = 0.0
state.pose.orientation.y = 0.0
state.pose.orientation.z = 0.0
state.pose.orientation.w = 1.0
state.twist = Twist()

rospy.wait_for_service("/gazebo/set_model_state", timeout=5.0)
set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
resp_state = set_state(state)
print("[TRACER] set_model_state:", resp_state.success, resp_state.status_message)
PY
'
