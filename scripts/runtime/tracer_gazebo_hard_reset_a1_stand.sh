#!/usr/bin/env bash
set -euo pipefail

MODEL="${TRACER_GAZEBO_MODEL_NAME:-a1_gazebo}"
Z="${TRACER_GAZEBO_RESET_Z:-0.32}"

# Conservative nominal standing configuration for Unitree A1-like URDF.
# Order follows /gazebo/get_model_properties for a1_gazebo:
# FL, FR, RL, RR. Adjust if model properties change.
JOINT_NAMES="${TRACER_GAZEBO_JOINT_NAMES:-FL_hip_joint,FL_thigh_joint,FL_calf_joint,FR_hip_joint,FR_thigh_joint,FR_calf_joint,RL_hip_joint,RL_thigh_joint,RL_calf_joint,RR_hip_joint,RR_thigh_joint,RR_calf_joint}"
JOINT_POSITIONS="${TRACER_GAZEBO_JOINT_POSITIONS:-0.0,0.8,-1.5,0.0,0.8,-1.5,0.0,0.8,-1.5,0.0,0.8,-1.5}"

echo "[TRACER] hard reset Gazebo model to stand"
echo "[TRACER] model=${MODEL}"
echo "[TRACER] z=${Z}"
echo "[TRACER] joint_names=${JOINT_NAMES}"
echo "[TRACER] joint_positions=${JOINT_POSITIONS}"

docker exec a1_unitree_gazebo_docker bash --noprofile --norc -lc "
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash 2>/dev/null || true

echo '[TRACER] pause physics'
rosservice call /gazebo/pause_physics '{}' >/dev/null 2>&1 || true

echo '[TRACER] reset base pose'
rosservice call /gazebo/set_model_state \"{
  model_state: {
    model_name: '${MODEL}',
    pose: {
      position: {x: 0.0, y: 0.0, z: ${Z}},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    },
    twist: {
      linear: {x: 0.0, y: 0.0, z: 0.0},
      angular: {x: 0.0, y: 0.0, z: 0.0}
    },
    reference_frame: 'world'
  }
}\"

echo '[TRACER] reset joint configuration'
rosservice call /gazebo/set_model_configuration \"{
  model_name: '${MODEL}',
  urdf_param_name: 'robot_description',
  joint_names: [${JOINT_NAMES}],
  joint_positions: [${JOINT_POSITIONS}]
}\"

echo '[TRACER] state after reset'
rosservice call /gazebo/get_model_state \"{model_name: '${MODEL}', relative_entity_name: 'world'}\" || true
"
