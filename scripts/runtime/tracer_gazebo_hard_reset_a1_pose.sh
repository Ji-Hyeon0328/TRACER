#!/usr/bin/env bash
set -euo pipefail

MODEL="${TRACER_GAZEBO_MODEL_NAME:-a1_gazebo}"
Z="${TRACER_GAZEBO_RESET_Z:-0.42}"

echo "[TRACER] hard reset Gazebo model pose"
echo "[TRACER] model=${MODEL}"
echo "[TRACER] z=${Z}"

docker exec a1_unitree_gazebo_docker bash --noprofile --norc -lc "
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash 2>/dev/null || true

echo '[TRACER] pause physics'
rosservice call /gazebo/pause_physics '{}' >/dev/null 2>&1 || true

echo '[TRACER] set model state'
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
}\" || true

echo '[TRACER] state after set_model_state'
rosservice call /gazebo/get_model_state \"{model_name: '${MODEL}', relative_entity_name: 'world'}\" || true
"
