#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
LOG="${TRACER_RAM_CLIENT_LOG:-/tmp/tracer_online_ram_udp_client_v0.log}"

echo "[TRACER] starting online RAM UDP client"
echo "[TRACER] root: $ROOT"
echo "[TRACER] log:  $LOG"

pkill -f "tracer_online_ram_udp_client_node.py" 2>/dev/null || true

set +u
source /opt/ros/humble/setup.bash
set -u

nohup /usr/bin/python3 \
  "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_online_ram_udp_client_node.py" \
  > "$LOG" 2>&1 &

PID=$!
echo "[TRACER] online RAM client pid=$PID"

sleep 1

echo "[TRACER] process check:"
pgrep -af "tracer_online_ram_udp_client_node.py" || true

echo "[TRACER] log tail:"
tail -n 20 "$LOG" || true
