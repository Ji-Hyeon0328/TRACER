#!/usr/bin/env bash
set -eo pipefail

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

echo "[TRACER] stop data_collection_lite"

echo
echo "========== stop host lite/learned/reference processes =========="
pkill -9 -f tracer_udp_proprio_to_ros2_node.py 2>/dev/null || true
pkill -9 -f tracer_udp_odom_to_ros2_node.py 2>/dev/null || true
pkill -9 -f tracer_ros2_mpc_ref_udp_sender.py 2>/dev/null || true
pkill -9 -f tracer_record_rollout_episode_v0.py 2>/dev/null || true
pkill -9 -f tracer_run_reference_sweep_rollouts_v0.py 2>/dev/null || true

pkill -9 -f tracer_fusion_policy_mpc_ref_node.py 2>/dev/null || true
pkill -9 -f tracer_objective_selector_stub_node.py 2>/dev/null || true
pkill -9 -f tracer_learned_high_level_policy_udp_client_v1_node.py 2>/dev/null || true
pkill -9 -f tracer_learned_high_level_policy_udp_client_node.py 2>/dev/null || true
pkill -9 -f "ros2 topic pub.*/tracer/mpc_reference" 2>/dev/null || true

echo
echo "========== stop ROS1 lite bridge processes =========="
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
pkill -9 -f tracer_ros1_proprio_udp_sender.py || true
pkill -9 -f tracer_ros1_odom_udp_sender.py || true
pkill -9 -f tracer_udp_to_ros1_mpc_ref.py || true
' || true

echo
echo "========== remaining process check =========="
pgrep -af "tracer_udp_proprio_to_ros2_node.py|tracer_udp_odom_to_ros2_node.py|tracer_ros2_mpc_ref_udp_sender.py|tracer_fusion_policy_mpc_ref_node.py|tracer_objective_selector_stub_node.py|tracer_learned_high_level_policy_udp_client|ros2 topic pub.*/tracer/mpc_reference" || true

sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
pgrep -af "tracer_ros1_proprio_udp_sender.py|tracer_ros1_odom_udp_sender.py|tracer_udp_to_ros1_mpc_ref.py" || true
' || true

echo
echo "[TRACER] data_collection_lite stopped."
