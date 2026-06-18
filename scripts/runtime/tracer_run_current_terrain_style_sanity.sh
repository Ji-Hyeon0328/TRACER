#!/usr/bin/env bash
set -eo pipefail

TERRAIN_NAME="${TRACER_TERRAIN_NAME:-unknown}"
WORLD_NAME="${TRACER_WORLD_NAME:-unknown}"
STYLES_CSV="${TRACER_STYLES:-nominal,fast,cautious}"
WAYPOINT_DISTANCES="${TRACER_WAYPOINT_DISTANCES:-0.6,1.2,1.8}"
TIMEOUT_SEC="${TRACER_MISSION_TIMEOUT_SEC:-60}"

IFS=',' read -ra STYLES <<< "$STYLES_CSV"

echo "============================================================"
echo "[TRACER] Current terrain style sanity"
echo "============================================================"
echo "[TRACER] terrain:   $TERRAIN_NAME"
echo "[TRACER] world:     $WORLD_NAME"
echo "[TRACER] styles:    $STYLES_CSV"
echo "[TRACER] distances: $WAYPOINT_DISTANCES"
echo "[TRACER] timeout:   $TIMEOUT_SEC"

echo
echo "[TRACER] current Gazebo world:"
sudo docker exec "${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/get_world_properties "{}" | head -20
'

for STYLE in "${STYLES[@]}"; do
  STYLE="$(echo "$STYLE" | xargs)"
  if [ -z "$STYLE" ]; then
    continue
  fi

  echo
  echo "============================================================"
  echo "[TRACER] Run terrain=$TERRAIN_NAME style=$STYLE"
  echo "============================================================"

  scripts/runtime/tracer_cleanup_stale_style_publishers.sh || true
  scripts/runtime/tracer_restart_a1_qpmc_controller_only.sh

  TRACER_TERRAIN_NAME="$TERRAIN_NAME" \
  TRACER_WORLD_NAME="$WORLD_NAME" \
  TRACER_STYLE_NAME="$STYLE" \
  TRACER_WAYPOINT_DISTANCES="$WAYPOINT_DISTANCES" \
  TRACER_MISSION_TIMEOUT_SEC="$TIMEOUT_SEC" \
  scripts/runtime/tracer_run_style_preset_mission.sh || true

  scripts/runtime/tracer_cleanup_stale_style_publishers.sh || true

  python3 scripts/runtime/tracer_summarize_style_missions.py \
    --root data/style_missions || true
done

echo
echo "============================================================"
echo "[TRACER] Current terrain style sanity finished"
echo "============================================================"
