#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"

OUTPUT_DIR="${TRACER_MISSION_LOG_DIR:-$ROOT/data/mission_logs}"
LOG_HZ="${TRACER_MISSION_LOG_HZ:-20.0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/mission_logger_${RUN_ID}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

echo "[TRACER] starting mission logger"
echo "[TRACER] output dir: $OUTPUT_DIR"
echo "[TRACER] log dir:    $LOG_DIR"
echo "[TRACER] log hz:     $LOG_HZ"

pkill -9 -f tracer_mission_logger_node.py || true

nohup /usr/bin/python3 "$WS/src/tracer_a1_qpmc_adapter/scripts/tracer_mission_logger_node.py" \
  --ros-args \
  -p output_dir:="$OUTPUT_DIR" \
  -p log_hz:="$LOG_HZ" \
  > "$LOG_DIR/mission_logger.log" 2>&1 &

PID=$!
echo "$PID" > "$ROOT/logs/latest_mission_logger.pid"
ln -sfn "$LOG_DIR" "$ROOT/logs/latest_mission_logger"

echo "[TRACER] mission logger pid: $PID"
echo "[TRACER] mission logger stdout: $LOG_DIR/mission_logger.log"
