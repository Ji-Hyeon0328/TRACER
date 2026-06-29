#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${TRACER_CTRL_CONTAINER:-a1_cpp_ctrl_docker}"
Z="${TRACER_RESET_Z:-0.36}"

docker exec "$CONTAINER" bash -lc "
source /opt/ros/melodic/setup.bash
python - <<'PY'
import os
import rospy
from gazebo_msgs.srv import GetWorldProperties, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist

z = float(os.environ.get('TRACER_RESET_Z_INNER', '$Z'))

rospy.init_node('tracer_force_reset_a1_pose_once', anonymous=True)

rospy.wait_for_service('/gazebo/get_world_properties', timeout=5.0)
get_world = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
world = get_world()

names = list(world.model_names)
print('[TRACER] model names:', names)

candidates = [
    n for n in names
    if ('a1' in n.lower())
    or ('unitree' in n.lower())
    or ('go1' in n.lower())
    or ('robot' in n.lower())
]

if not candidates:
    raise RuntimeError('Could not find A1-like robot model in Gazebo model list')

model_name = candidates[0]
print('[TRACER] selected model:', model_name)

state = ModelState()
state.model_name = model_name
state.reference_frame = 'world'
state.pose.position.x = 0.0
state.pose.position.y = 0.0
state.pose.position.z = z
state.pose.orientation.x = 0.0
state.pose.orientation.y = 0.0
state.pose.orientation.z = 0.0
state.pose.orientation.w = 1.0
state.twist = Twist()

rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
resp = set_state(state)
print('[TRACER] set_model_state:', resp.success, resp.status_message)
PY
"
