#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-reports/phase_f/phase_f0_signal_audit_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

echo "[TRACER] Phase-F0 runtime signal audit"
echo "[TRACER] OUT_DIR=$OUT_DIR"

{
  echo "# TRACER Phase-F0 Runtime Signal Audit v0"
  echo
  echo "- created_at: $(date -Is)"
  echo "- out_dir: \`$OUT_DIR\`"
  echo
} > "$OUT_DIR/summary.md"

echo
echo "===== ROS2 topics ====="
ros2 topic list -t | tee "$OUT_DIR/ros2_topics_typed.txt" || true

echo
echo "===== ROS2 nodes ====="
ros2 node list | tee "$OUT_DIR/ros2_nodes.txt" || true

echo
echo "===== ROS2 topic hz samples ====="
{
  for topic in \
    /tracer/robot_odom_flat \
    /tracer/terrain_context_label \
    /tracer/terrain_context_flat \
    /tracer/objective_beta \
    /tracer/objective_beta_static \
    /tracer/ram_mismatch \
    /tracer/empirical_mpc_reference \
    /tracer/mlp_selector_shadow_ref \
    /tracer/mpc_reference
  do
    echo
    echo "============================================================"
    echo "$topic"
    echo "============================================================"
    timeout 5s ros2 topic hz "$topic" || true
  done
} | tee "$OUT_DIR/ros2_topic_hz_samples.txt"

echo
echo "===== ROS2 one-message samples ====="
{
  for topic in \
    /tracer/robot_odom_flat \
    /tracer/terrain_context_label \
    /tracer/terrain_context_flat \
    /tracer/objective_beta \
    /tracer/objective_beta_static \
    /tracer/ram_mismatch \
    /tracer/empirical_mpc_reference \
    /tracer/mlp_selector_shadow_ref \
    /tracer/mpc_reference
  do
    echo
    echo "============================================================"
    echo "$topic"
    echo "============================================================"
    timeout 4s ros2 topic echo --once "$topic" || true
  done
} | tee "$OUT_DIR/ros2_topic_echo_samples.txt"

echo
echo "===== ROS1 topics inside gazebo container ====="
docker exec a1_unitree_gazebo_docker bash -lc '
  source /opt/ros/melodic/setup.bash
  rostopic list
' | tee "$OUT_DIR/ros1_gazebo_topics.txt" || true

echo
echo "===== ROS1 topics inside controller container ====="
docker exec a1_cpp_ctrl_docker bash -lc '
  source /opt/ros/melodic/setup.bash 2>/dev/null || source /opt/ros/noetic/setup.bash 2>/dev/null || true
  rostopic list
' | tee "$OUT_DIR/ros1_controller_topics.txt" || true

echo
echo "===== ROS1 typed topics inside gazebo container ====="
docker exec a1_unitree_gazebo_docker bash -lc '
  source /opt/ros/melodic/setup.bash
  for t in $(rostopic list); do
    ty=$(rostopic type "$t" 2>/dev/null || true)
    echo "$t [$ty]"
  done
' | tee "$OUT_DIR/ros1_gazebo_topics_typed.txt" || true

echo
echo "===== Candidate ROS1 samples ====="
docker exec a1_unitree_gazebo_docker bash -lc '
  source /opt/ros/melodic/setup.bash
  for t in \
    /gazebo/model_states \
    /trunk_imu \
    /a1_gazebo/FR_hip_controller/state \
    /a1_gazebo/FR_thigh_controller/state \
    /a1_gazebo/FR_calf_controller/state \
    /a1_gazebo/FL_hip_controller/state \
    /a1_gazebo/FL_thigh_controller/state \
    /a1_gazebo/FL_calf_controller/state \
    /a1_gazebo/RR_hip_controller/state \
    /a1_gazebo/RR_thigh_controller/state \
    /a1_gazebo/RR_calf_controller/state \
    /a1_gazebo/RL_hip_controller/state \
    /a1_gazebo/RL_thigh_controller/state \
    /a1_gazebo/RL_calf_controller/state \
    /a1_gazebo/joint_states
  do
    echo
    echo "============================================================"
    echo "$t"
    echo "============================================================"
    timeout 4s rostopic echo -n 1 "$t" || true
  done
' | tee "$OUT_DIR/ros1_candidate_echo_samples.txt" || true

{
  echo
  echo "## Files"
  echo
  for f in "$OUT_DIR"/*; do
    echo "- \`$f\`"
  done
  echo
  echo "## Next"
  echo
  echo "Use this audit to decide which topics can support:"
  echo
  echo "- true energy proxy: joint effort × joint velocity"
  echo "- RAM mismatch: commanded vs actual base velocity / yaw / lateral drift"
  echo "- contact/slip proxy: foot contact or GRF if available"
} >> "$OUT_DIR/summary.md"

echo
echo "[TRACER] wrote $OUT_DIR"
cat "$OUT_DIR/summary.md"
