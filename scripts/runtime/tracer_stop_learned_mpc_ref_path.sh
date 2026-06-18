#!/usr/bin/env bash
set -eo pipefail

echo "[TRACER] stop learned high-level MPC reference path"

echo
echo "[TRACER] killing host ROS2 MPC ref sender and learned policy client..."
pkill -f "tracer_ros2_mpc_ref_udp_sender.py" || true
pkill -f "tracer_learned_high_level_policy_udp_client_v1_node.py" || true

echo
echo "[TRACER] killing learned high-level policy UDP server..."
pkill -f "tracer_high_level_policy_udp_server_v1.py" || true

sleep 1

echo
echo "[TRACER] remaining related processes:"
pgrep -af "tracer_ros2_mpc_ref_udp_sender.py|tracer_learned_high_level_policy_udp_client_v1_node.py|tracer_high_level_policy_udp_server_v1.py" || true

echo
echo "[TRACER] done."
