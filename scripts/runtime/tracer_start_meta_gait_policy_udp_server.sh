#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
HOST="${TRACER_META_GAIT_POLICY_UDP_HOST:-127.0.0.1}"
PORT="${TRACER_META_GAIT_POLICY_UDP_PORT:-50310}"
KIND="${TRACER_META_GAIT_POLICY_KIND:-learned}"
MODEL="${TRACER_META_GAIT_POLICY_MODEL:-$ROOT/artifacts/meta_gait_policy_v0/model.pt}"
PYTHON_BIN="${TRACER_META_GAIT_POLICY_PYTHON:-$HOME/anaconda3/envs/env_isaaclab/bin/python}"
LOG="${TRACER_META_GAIT_POLICY_LOG:-/tmp/tracer_meta_gait_policy_udp_server.log}"
PIDFILE="${TRACER_META_GAIT_POLICY_PIDFILE:-/tmp/tracer_meta_gait_policy_udp_server.pid}"

echo "[TRACER] starting meta-gait policy UDP server"
echo "[TRACER] root:   $ROOT"
echo "[TRACER] bind:   $HOST:$PORT"
echo "[TRACER] kind:   $KIND"
echo "[TRACER] model:  ${MODEL:-<none>}"
echo "[TRACER] python: $PYTHON_BIN"
echo "[TRACER] log:    $LOG"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[TRACER][ERROR] python not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [ "$KIND" = "learned" ] || [ "$KIND" = "torch" ]; then
  if [ ! -f "$MODEL" ]; then
    echo "[TRACER][ERROR] learned meta-gait model not found: $MODEL" >&2
    echo "[TRACER][HINT] run scripts/training/tracer_train_meta_gait_policy_v0.py first" >&2
    exit 1
  fi
fi

if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE" || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[TRACER] stopping previous meta-gait UDP server pid=$OLD_PID"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 0.5
  fi
fi

pkill -f "tracer_meta_gait_policy_udp_server_v0.py" 2>/dev/null || true

cd "$ROOT"

TRACER_META_GAIT_POLICY_KIND="$KIND" \
TRACER_META_GAIT_POLICY_MODEL="$MODEL" \
TRACER_META_GAIT_POLICY_UDP_HOST="$HOST" \
TRACER_META_GAIT_POLICY_UDP_PORT="$PORT" \
PYTHONPATH="$ROOT" \
"$PYTHON_BIN" "$ROOT/scripts/runtime/tracer_meta_gait_policy_udp_server_v0.py" \
  > "$LOG" 2>&1 &

PID="$!"
echo "$PID" > "$PIDFILE"

echo "[TRACER] meta-gait UDP server started pid=$PID"
sleep 1.0
tail -n 20 "$LOG" || true
