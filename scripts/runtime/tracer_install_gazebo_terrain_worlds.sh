#!/usr/bin/env bash
set -eo pipefail

GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"
SRC_DIR="${TRACER_TERRAIN_WORLD_SRC_DIR:-generated/gazebo_worlds}"
DST_DIR="/root/unitree_ws/src/unitree_ros/unitree_gazebo/worlds"

echo "[TRACER] install Gazebo terrain worlds"
echo "[TRACER] container: $GAZEBO_CONTAINER"
echo "[TRACER] src:       $SRC_DIR"
echo "[TRACER] dst:       $DST_DIR"

if [ ! -d "$SRC_DIR" ]; then
  echo "[ERROR] source directory not found: $SRC_DIR"
  exit 1
fi

shopt -s nullglob
worlds=("$SRC_DIR"/*.world)
if [ "${#worlds[@]}" -eq 0 ]; then
  echo "[ERROR] no .world files found in $SRC_DIR"
  exit 1
fi

for w in "${worlds[@]}"; do
  echo "[TRACER] docker cp $w -> $DST_DIR/"
  sudo docker cp "$w" "$GAZEBO_CONTAINER:$DST_DIR/"
done

echo
echo "[TRACER] installed worlds:"
sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc "
find $DST_DIR -maxdepth 1 -type f -name 'tracer_*.world' -printf '%f\n' | sort
"
