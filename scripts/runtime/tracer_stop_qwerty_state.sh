#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

echo "[TRACER] stopping qwerty state processes"

echo "[TRACER] stopping ROS1 container qwerty processes..."
sudo docker exec a1_cpp_ctrl_docker bash -lc '
pkill -9 -f tracer_ros1_proprio_udp_sender.py || true
pkill -9 -f tracer_udp_to_ros1_mpc_ref.py || true
' || true

echo "[TRACER] stopping host qwerty processes..."
pkill -9 -f tracer_udp_proprio_to_ros2_node.py || true
pkill -9 -f tracer_ros2_mpc_ref_udp_sender.py || true
pkill -9 -f tracer_learned_encoder_udp_client_node.py || true
pkill -9 -f tracer_objective_selector_stub_node.py || true
pkill -9 -f tracer_learned_high_level_policy_udp_client_v1_node.py || true
pkill -9 -f tracer_proprio_encoder_udp_server_v0.py || true
pkill -9 -f tracer_high_level_policy_udp_server_v1.py || true

echo "[TRACER] qwerty state stopped."
echo "[TRACER] Gazebo / A1-QP-MPC solver were not stopped."
