#!/usr/bin/env bash
set -euo pipefail

GAZEBO_CONTAINER="${GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"

TRACER_RESET_X="${TRACER_RESET_X:-0.0}"
TRACER_RESET_Y="${TRACER_RESET_Y:-0.0}"
TRACER_RESET_REL_Z="${TRACER_RESET_REL_Z:-0.34}"
TRACER_GROUND_THICKNESS="${TRACER_GROUND_THICKNESS:-0.10}"

sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc "
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash

echo '[TRACER] pause physics'
rosservice call /gazebo/pause_physics '{}' >/dev/null || true

TRACER_RESET_X='$TRACER_RESET_X' \
TRACER_RESET_Y='$TRACER_RESET_Y' \
TRACER_RESET_REL_Z='$TRACER_RESET_REL_Z' \
TRACER_GROUND_THICKNESS='$TRACER_GROUND_THICKNESS' \
python - <<'PY'
import os
import math
import rospy
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

rospy.init_node('tracer_slope_relative_reset_a1_pose', anonymous=True)

rospy.wait_for_service('/gazebo/get_model_state', timeout=5.0)
rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)

get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

ground = get_state('tracer_ground', 'world')
if not ground.success:
    raise RuntimeError('tracer_ground model not found. This world is not compatible with slope-relative reset.')

gx = ground.pose.position.x
gz = ground.pose.position.z
qy = ground.pose.orientation.y
qw = ground.pose.orientation.w

pitch = 2.0 * math.atan2(qy, qw)

x = float(os.environ.get('TRACER_RESET_X', '0.0'))
y = float(os.environ.get('TRACER_RESET_Y', '0.0'))
target_rel_z = float(os.environ.get('TRACER_RESET_REL_Z', '0.34'))
thickness = float(os.environ.get('TRACER_GROUND_THICKNESS', '0.10'))

half_thickness = 0.5 * thickness

# Top surface height of pitched ground slab.
z_top = gz - math.tan(pitch) * (x - gx) + half_thickness / max(1e-6, math.cos(pitch))
z = z_top + target_rel_z

state = ModelState()
state.model_name = 'a1_gazebo'
state.reference_frame = 'world'

state.pose.position.x = x
state.pose.position.y = y
state.pose.position.z = z

# Start level; QP-MPC/stand controller will settle on terrain.
state.pose.orientation.x = 0.0
state.pose.orientation.y = 0.0
state.pose.orientation.z = 0.0
state.pose.orientation.w = 1.0

state.twist.linear.x = 0.0
state.twist.linear.y = 0.0
state.twist.linear.z = 0.0
state.twist.angular.x = 0.0
state.twist.angular.y = 0.0
state.twist.angular.z = 0.0

resp = set_state(state)

print('[TRACER] slope-relative reset success:', resp.success)
print('[TRACER] status:', resp.status_message)
print('[TRACER] ground_pitch_deg:', math.degrees(pitch))
print('[TRACER] reset_x:', x)
print('[TRACER] ground_top_z:', z_top)
print('[TRACER] robot_z:', z)
print('[TRACER] target_rel_z:', target_rel_z)
PY
"
