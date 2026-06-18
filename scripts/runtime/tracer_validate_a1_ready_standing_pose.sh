#!/usr/bin/env bash
set -eo pipefail

GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"
MODEL_NAME="${TRACER_GAZEBO_MODEL_NAME:-a1_gazebo}"

Z_MIN="${TRACER_READY_Z_MIN:-0.285}"
Z_MAX="${TRACER_READY_Z_MAX:-0.380}"
RP_MAX="${TRACER_READY_RP_MAX:-0.15}"
LIN_MAX="${TRACER_READY_LINVEL_MAX:-0.25}"
ANG_MAX="${TRACER_READY_ANGVEL_MAX:-0.50}"
XY_MAX="${TRACER_READY_XY_MAX:-0.25}"

sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc "
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash

python - <<'PY'
import math
import sys
import rospy
import tf.transformations as tft
from gazebo_msgs.srv import GetModelState

model_name = '${MODEL_NAME}'

z_min = float('${Z_MIN}')
z_max = float('${Z_MAX}')
rp_max = float('${RP_MAX}')
lin_max = float('${LIN_MAX}')
ang_max = float('${ANG_MAX}')
xy_max = float('${XY_MAX}')

rospy.init_node('tracer_validate_a1_ready_standing_pose', anonymous=True, disable_signals=True)

try:
    rospy.wait_for_service('/gazebo/get_model_state', timeout=3.0)
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    res = get_state(model_name, 'world')
except Exception as e:
    print('[ERROR] failed to call /gazebo/get_model_state: {}'.format(e))
    sys.exit(2)

if not res.success:
    print('[ERROR] get_model_state failed: {}'.format(res.status_message))
    sys.exit(3)

p = res.pose.position
q = res.pose.orientation
tw = res.twist

roll, pitch, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])

lin = math.sqrt(tw.linear.x**2 + tw.linear.y**2 + tw.linear.z**2)
ang = math.sqrt(tw.angular.x**2 + tw.angular.y**2 + tw.angular.z**2)

print('[TRACER] ready-standing validation:')
print('  xyz=({:.4f}, {:.4f}, {:.4f})'.format(p.x, p.y, p.z))
print('  rpy=({:.4f}, {:.4f}, {:.4f})'.format(roll, pitch, yaw))
print('  lin_vel_norm={:.4f}'.format(lin))
print('  ang_vel_norm={:.4f}'.format(ang))

ok = True

if not (z_min <= p.z <= z_max):
    print('[ERROR] base height out of ready-standing range [{:.3f}, {:.3f}]'.format(z_min, z_max))
    ok = False

if abs(roll) > rp_max:
    print('[ERROR] roll too large: {:.4f} > {:.4f}'.format(abs(roll), rp_max))
    ok = False

if abs(pitch) > rp_max:
    print('[ERROR] pitch too large: {:.4f} > {:.4f}'.format(abs(pitch), rp_max))
    ok = False

if lin > lin_max:
    print('[ERROR] base linear velocity too large: {:.4f} > {:.4f}'.format(lin, lin_max))
    ok = False

if ang > ang_max:
    print('[ERROR] base angular velocity too large: {:.4f} > {:.4f}'.format(ang, ang_max))
    ok = False

if abs(p.x) > xy_max or abs(p.y) > xy_max:
    print('[ERROR] base drifted too far from reset origin: x={:.4f}, y={:.4f}, limit={:.4f}'.format(p.x, p.y, xy_max))
    ok = False

if not ok:
    sys.exit(10)

print('[TRACER] ready-standing validation passed')
PY
"
