#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"
CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

SERVO_SEC="${TRACER_SERVO_SEC:-3.0}"
KINETIC_SEC="${TRACER_KINETIC_SEC:-3.0}"
AFTER_PAUSE_SEC="${TRACER_AFTER_PAUSE_SEC:-0.5}"

echo "[TRACER] reset to qwer state by standing-pose sequence"
echo "[TRACER] gazebo/reset container: $GAZEBO_CONTAINER"
echo "[TRACER] controller container:    $CTRL_CONTAINER"
echo "[TRACER] servo:   rosrun unitree_controller unitree_servo for ${SERVO_SEC}s"
echo "[TRACER] kinetic: rosrun unitree_controller unitree_move_kinetic for ${KINETIC_SEC}s before pause"

echo
echo "[TRACER] stopping previous host runtime nodes..."
pkill -9 -f tracer_udp_odom_to_ros2_node.py || true
pkill -9 -f tracer_global_goal_to_relative_goal_node.py || true
pkill -9 -f tracer_global_goal_to_relative_goal_v1_node.py || true
pkill -9 -f tracer_waypoint_manager_node.py || true
pkill -9 -f tracer_waypoint_manager_v1_node.py || true
pkill -9 -f tracer_waypoints_ahead_publisher_node.py || true

pkill -9 -f tracer_udp_proprio_to_ros2_node.py || true
pkill -9 -f tracer_ros2_mpc_ref_udp_sender.py || true
pkill -9 -f tracer_learned_encoder_udp_client_node.py || true
pkill -9 -f tracer_objective_selector_stub_node.py || true
pkill -9 -f tracer_learned_high_level_policy_udp_client_v1_node.py || true
pkill -9 -f tracer_proprio_encoder_udp_server_v0.py || true
pkill -9 -f tracer_high_level_policy_udp_server_v1.py || true

echo
echo "[TRACER] cleaning previous reset helper processes in gazebo container..."
sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc '
pkill -9 -f unitree_servo || true
pkill -9 -f unitree_move_kinetic || true
' || true

echo
echo "[TRACER] cleaning previous ROS1 bridge processes in controller container..."
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
pkill -9 -f tracer_udp_to_ros1_mpc_ref.py || true
pkill -9 -f tracer_ros1_proprio_udp_sender.py || true
pkill -9 -f tracer_ros1_odom_udp_sender.py || true
' || true

echo
echo "[TRACER] running standing-pose reset sequence in gazebo container..."
sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc "
set +e

source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash

echo '[TRACER: gazebo container] check unitree_controller'
rospack find unitree_controller

echo '[TRACER: gazebo container] unpause physics'
rosservice call /gazebo/unpause_physics '{}' >/dev/null 2>&1 || true

sleep 0.5

echo '[TRACER: gazebo container] start servo: rosrun unitree_controller unitree_servo'
rosrun unitree_controller unitree_servo &
SERVO_PID=\$!

sleep ${SERVO_SEC}

echo '[TRACER: gazebo container] Ctrl-C servo'
kill -INT \$SERVO_PID 2>/dev/null || true
wait \$SERVO_PID 2>/dev/null || true

sleep 0.5

echo '[TRACER: gazebo container] start kinetic: rosrun unitree_controller unitree_move_kinetic'
rosrun unitree_controller unitree_move_kinetic &
KINETIC_PID=\$!

sleep ${KINETIC_SEC}

echo '[TRACER: gazebo container] pause physics while kinetic is still running'
rosservice call /gazebo/pause_physics '{}' >/dev/null 2>&1 || true

sleep ${AFTER_PAUSE_SEC}

echo '[TRACER: gazebo container] Ctrl-C kinetic after pause'
kill -INT \$KINETIC_PID 2>/dev/null || true
wait \$KINETIC_PID 2>/dev/null || true

echo '[TRACER: gazebo container] standing-pose reset sequence done'
"

echo
echo "[TRACER] starting ROS1 MPC reference receiver in controller container..."
sudo docker exec -d "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rosrun a1_cpp tracer_udp_to_ros1_mpc_ref.py > /tmp/tracer_udp_to_ros1_mpc_ref.log 2>&1
'

echo
echo "[TRACER] reset/qwer state done."
echo "[TRACER] expected:"
echo "  - Gazebo robot is standing near zero point"
echo "  - Gazebo physics is paused"
echo "  - tracer_udp_to_ros1_mpc_ref.py is running in controller container"
echo
echo "[TRACER] next:"
echo "  scripts/runtime/tracer_precheck_reset_state.sh"
echo "  scripts/runtime/tracer_start_qwerty_state.sh"
