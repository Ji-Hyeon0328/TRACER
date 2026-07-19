#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

D5_LOG_DIR="${TRACER_PHASE_D5_LOG_DIR:-logs/phase_d7_objective_shadow_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$D5_LOG_DIR"
export TRACER_PHASE_D5_LOG_DIR="$D5_LOG_DIR"

echo "[TRACER] starting base D5/D6 stack for D7 objective shadow"
echo "[TRACER] D5_LOG_DIR=$D5_LOG_DIR"

scripts/runtime/tracer_start_phase_d5_shadow_stack_v0.sh

echo
echo "[TRACER] starting D7 objective selector shadow node"

export TRACER_PHASE_D7_OBJECTIVE_MODEL_JSON="${TRACER_PHASE_D7_OBJECTIVE_MODEL_JSON:-models/phase_d7/d7_objective_selector_ridge_v0.json}"
export TRACER_PHASE_D7_SHADOW_LOG_CSV="${TRACER_PHASE_D7_SHADOW_LOG_CSV:-$D5_LOG_DIR/d7_objective_selector_shadow_v0.csv}"
export TRACER_PHASE_D7_REF_TOPIC="${TRACER_PHASE_D7_REF_TOPIC:-/tracer/empirical_mpc_reference}"
export TRACER_PHASE_D7_ACTUAL_BETA_TOPIC="${TRACER_PHASE_D7_ACTUAL_BETA_TOPIC:-/tracer/objective_beta}"
export TRACER_PHASE_D7_OBJECTIVE_BETA_TOPIC="${TRACER_PHASE_D7_OBJECTIVE_BETA_TOPIC:-/tracer/objective_beta_shadow}"
export TRACER_PHASE_D7_GOAL_X="${TRACER_PHASE_D7_GOAL_X:-${TRACER_PHASE_D4_GOAL_X:-8.0}}"
export TRACER_PHASE_D7_LATERAL_BOUND="${TRACER_PHASE_D7_LATERAL_BOUND:-${TRACER_PHASE_D4_LATERAL_BOUND:-2.0}}"
export TRACER_PHASE_D7_STOP_MARGIN="${TRACER_PHASE_D7_STOP_MARGIN:-${TRACER_PHASE_D4_STOP_MARGIN:--0.05}}"

set +u
source /opt/ros/humble/setup.bash
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi
set -u

/usr/bin/python3 scripts/runtime/tracer_phase_d7_objective_selector_shadow_node_v0.py \
  > "$D5_LOG_DIR/d7_objective_selector_shadow_node_v0.log" 2>&1 &

echo $! > "$D5_LOG_DIR/d7_objective_selector_shadow_node_v0.pid"

echo "[TRACER] D7 objective shadow pid=$(cat "$D5_LOG_DIR/d7_objective_selector_shadow_node_v0.pid")"
echo "[TRACER] D7 objective shadow csv=$TRACER_PHASE_D7_SHADOW_LOG_CSV"
