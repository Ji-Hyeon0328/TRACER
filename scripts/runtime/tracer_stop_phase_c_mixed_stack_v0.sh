#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
set -u

echo "[TRACER] stop Phase-C mixed stack v0"

echo
echo "========== stop ROS2/host processes =========="
pkill -f tracer_ros2_mpc_ref_udp_sender.py || true
pkill -f tracer_udp_odom_to_ros2_node.py || true
pkill -f tracer_mixed_terrain_context_provider_v0.py || true
pkill -f tracer_phase_c_oracle_schedule_mpc_ref_node_v0.py || true
pkill -f tracer_phase_c_fixed_mpc_ref_node_v0.py || true
pkill -f tracer_phase_c_beta_schedule_mpc_ref_node_v0.py || true
pkill -f "ros2 topic pub.*tracer/mpc_reference" || true

echo
echo "========== stop ROS1/container bridge + qwer init processes =========="
timeout 8s docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
pkill -f tracer_udp_to_ros1_mpc_ref.py || true
pkill -f tracer_ros1_model_states_odom_udp_sender.py || true
pkill -f tracer_ros1_odom_udp_sender.py || true
pkill -f unitree_servo || true
pkill -f unitree_move_kinetic || true
' || true

echo
echo "========== pause Gazebo physics if available =========="
timeout 8s docker exec a1_unitree_gazebo_docker bash --noprofile --norc -lc '
set +u
source /opt/ros/melodic/setup.bash 2>/dev/null || true
source /root/unitree_ws/devel/setup.bash 2>/dev/null || true
rosservice call /gazebo/pause_physics "{}" >/tmp/tracer_phase_c_pause.log 2>&1 || true
' || true

echo
echo "[TRACER] Phase-C mixed stack stopped."

# Phase-D segment-action cleanup
pkill -f "tracer_phase_d_segment_action_mpc_ref_node_v0.py" 2>/dev/null || true

# Phase-D4 context-meta selector cleanup
pkill -f "tracer_phase_d4_context_meta_selector_node_v0.py" 2>/dev/null || true


# Phase-D5 shadow nodes
pkill -f "tracer_phase_d5_shadow_inputs_node_v0.py" 2>/dev/null || true
pkill -f "tracer_phase_d5_policy_input_logger_v0.py" 2>/dev/null || true

# Phase-D4/D5 host runtime nodes
pkill -f "tracer_phase_d4_context_meta_selector_node_v0.py" 2>/dev/null || true
pkill -f "tracer_phase_d5_shadow_inputs_node_v0.py" 2>/dev/null || true
pkill -f "tracer_phase_d5_policy_input_logger_v0.py" 2>/dev/null || true
pkill -f "tracer_udp_odom_to_ros2_node.py" 2>/dev/null || true
pkill -f "tracer_ros2_mpc_ref_udp_sender.py" 2>/dev/null || true

# Phase-D5 learned selector shadow node
pkill -f "tracer_phase_d5_learned_selector_shadow_node_v0.py" 2>/dev/null || true

# Phase-D5 gated selector dry-run node
pkill -f "tracer_phase_d5_gated_selector_dryrun_node_v0.py" 2>/dev/null || true

# Phase-D6 MLP selector shadow node
pkill -f "tracer_phase_d6_mlp_selector_shadow_node_v0.py" 2>/dev/null || true
