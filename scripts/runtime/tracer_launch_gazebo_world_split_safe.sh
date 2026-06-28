#!/usr/bin/env bash
set -euo pipefail

WORLD="${1:-${TRACER_GAZEBO_WORLD:-stairs_single}}"
GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"
RESTART_CONTAINER="${TRACER_GAZEBO_RESTART_CONTAINER:-1}"

echo "[TRACER] clean launch Gazebo world"
echo "[TRACER] container:          $GAZEBO_CONTAINER"
echo "[TRACER] world:              $WORLD"
echo "[TRACER] restart_container:  $RESTART_CONTAINER"

if [ "$RESTART_CONTAINER" = "1" ]; then
  echo "[TRACER] docker restart Gazebo container for clean world state"
  docker restart "$GAZEBO_CONTAINER" >/dev/null
  sleep 4
else
  echo "[TRACER] kill old Gazebo/Unitree launch processes"
  docker exec "$GAZEBO_CONTAINER" bash -lc '
    set +e
    ps -eo pid,comm,args | awk '"'"'
      $2=="gzserver" ||
      $2=="gzclient" ||
      $0 ~ /controller_spawner/ ||
      $0 ~ /robot_state_publisher/ ||
      $0 ~ /gazebo_low_state_pub/ ||
      $0 ~ /unitree_gazebo/ ||
      $0 ~ /normal.launch/ {
        print $1
      }
    '"'"' | sort -u | xargs -r kill -9 || true
    sleep 2
  ' || true
fi

echo "[TRACER] start Gazebo world=$WORLD"
docker exec -d "$GAZEBO_CONTAINER" bash -lc "
  source /opt/ros/melodic/setup.bash
  source /root/unitree_ws/devel/setup.bash

  roslaunch unitree_gazebo normal.launch rname:=a1 wname:=${WORLD} \
    > /tmp/tracer_gazebo_${WORLD}.log 2>&1
"

echo "[TRACER] wait for Gazebo startup"
sleep 12

echo "[TRACER] Gazebo process check"
docker exec "$GAZEBO_CONTAINER" bash -lc '
  ps -eo pid,comm,args | awk '"'"'
    $2=="gzserver" ||
    $2=="gzclient" ||
    $0 ~ /unitree_gazebo/ ||
    $0 ~ /gazebo_low_state_pub/ ||
    $0 ~ /robot_state_publisher/ ||
    $0 ~ /controller_spawner/ {
      print
    }
  '"'"' || true
'

echo "[TRACER] active world evidence"
docker exec "$GAZEBO_CONTAINER" bash -lc "
  ps -eo pid,comm,args | grep -E 'gzserver .*${WORLD}\\.world' | grep -v grep || true
"

echo "[TRACER] Gazebo launch log tail"
docker exec "$GAZEBO_CONTAINER" bash -lc "tail -100 /tmp/tracer_gazebo_${WORLD}.log || true"

echo "[TRACER] Gazebo world launch done"
