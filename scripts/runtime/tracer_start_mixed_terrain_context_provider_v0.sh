#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

set +u
if command -v use_ros2 >/dev/null 2>&1; then
  use_ros2
else
  source /opt/ros/humble/setup.bash
fi
set -u

echo "[TRACER] start mixed terrain context provider v0"
echo "[TRACER] python: /usr/bin/python3"
echo "[TRACER] odom:   /tracer/robot_odom_flat"
echo "[TRACER] out:    /tracer/terrain_context_flat"
echo "[TRACER] label:  /tracer/terrain_context_label"

/usr/bin/python3 scripts/runtime/tracer_mixed_terrain_context_provider_v0.py
