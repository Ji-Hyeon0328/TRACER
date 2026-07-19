#!/usr/bin/env bash
set -eo pipefail

GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"
MODEL_NAME="${TRACER_GAZEBO_MODEL_NAME:-a1_gazebo}"
RESET_Y="${TRACER_RESET_Y_OFFSET:-0.0}"

echo "[TRACER] apply post-reset y offset"
echo "  gazebo_container=$GAZEBO_CONTAINER"
echo "  model_name=$MODEL_NAME"
echo "  reset_y=$RESET_Y"

docker exec \
  -e TRACER_GAZEBO_MODEL_NAME="$MODEL_NAME" \
  -e TRACER_RESET_Y_OFFSET="$RESET_Y" \
  "$GAZEBO_CONTAINER" bash --noprofile --norc -lc '
set -eo pipefail
source /opt/ros/melodic/setup.bash

python - <<'"'"'PY'"'"'
import os
import rospy
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist

model = os.environ.get("TRACER_GAZEBO_MODEL_NAME", "a1_gazebo")
reset_y = float(os.environ.get("TRACER_RESET_Y_OFFSET", "0.0"))

rospy.init_node("tracer_apply_reset_y_offset_v0", anonymous=True)

rospy.wait_for_service("/gazebo/get_model_state", timeout=5.0)
rospy.wait_for_service("/gazebo/set_model_state", timeout=5.0)

get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

resp = get_state(model, "world")
if not resp.success:
    raise RuntimeError("get_model_state failed: %s" % resp.status_message)

state = ModelState()
state.model_name = model
state.reference_frame = "world"
state.pose = resp.pose
state.pose.position.y = reset_y

# Zero twist after teleporting the model laterally.
state.twist = Twist()

resp2 = set_state(state)
print("[TRACER] set_model_state:", resp2.success, resp2.status_message)
if not resp2.success:
    raise RuntimeError("set_model_state failed: %s" % resp2.status_message)

resp3 = get_state(model, "world")
print(
    "[TRACER] state after y-offset: "
    "x=%.4f y=%.4f z=%.4f" %
    (resp3.pose.position.x, resp3.pose.position.y, resp3.pose.position.z)
)
PY
'
