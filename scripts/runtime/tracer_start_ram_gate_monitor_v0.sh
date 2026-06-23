#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-${TRACER_GATE_POLICY_TERRAIN:-flat_normal}}"

export TRACER_GATE_POLICY_TERRAIN="$TERRAIN"
export TRACER_GATE_POLICY_JSON="${TRACER_GATE_POLICY_JSON:-$ROOT/configs/highlevel_policy/tracer_fusion_policy_v1.json}"

LOG="${TRACER_RAM_GATE_MONITOR_LOG:-/tmp/tracer_ram_gate_monitor_${TERRAIN}.log}"

echo "[TRACER] starting RAM gate monitor"
echo "[TRACER] root:    $ROOT"
echo "[TRACER] terrain: $TRACER_GATE_POLICY_TERRAIN"
echo "[TRACER] policy:  $TRACER_GATE_POLICY_JSON"
echo "[TRACER] log:     $LOG"

pkill -f "tracer_ram_gate_monitor_node.py" 2>/dev/null || true
sleep 0.5

set +u
source /opt/ros/humble/setup.bash
set -u

nohup /usr/bin/python3 \
  "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_ram_gate_monitor_node.py" \
  > "$LOG" 2>&1 &

PID=$!
echo "[TRACER] RAM gate monitor pid=$PID"

sleep 1

echo "[TRACER] process check:"
pgrep -af "tracer_ram_gate_monitor_node.py" || true

echo "[TRACER] log tail:"
tail -n 30 "$LOG" || true
