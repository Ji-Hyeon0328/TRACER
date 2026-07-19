#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

POLICY_JSON="${TRACER_PHASE_D4_POLICY_JSON:-models/phase_d4/d4_context_meta_policy_rough_clearance_holdvx0025_v0.json}"

TS="$(date +%Y%m%d_%H%M%S)"
D5_LOG_DIR="${TRACER_PHASE_D5_LOG_DIR:-logs/phase_d5_shadow_${TS}}"
mkdir -p "$D5_LOG_DIR"

echo "[TRACER] starting base D4 policy stack first"
echo "  POLICY_JSON=$POLICY_JSON"
echo "  D5_LOG_DIR=$D5_LOG_DIR"

TRACER_PHASE_D4_POLICY_JSON="$POLICY_JSON" \
bash scripts/runtime/tracer_start_phase_d4_context_meta_stack_v0.sh

echo "[TRACER] launching D5 shadow input publisher/logger"

pkill -f "tracer_phase_d5_shadow_inputs_node_v0.py" 2>/dev/null || true
pkill -f "tracer_phase_d5_policy_input_logger_v0.py" 2>/dev/null || true

nohup /usr/bin/python3 scripts/runtime/tracer_phase_d5_shadow_inputs_node_v0.py \
  > "$D5_LOG_DIR/shadow_inputs_node.log" 2>&1 &

TRACER_PHASE_D5_INPUT_LOG="$D5_LOG_DIR/policy_inputs_v0.csv" \
nohup /usr/bin/python3 scripts/runtime/tracer_phase_d5_policy_input_logger_v0.py \
  > "$D5_LOG_DIR/policy_input_logger.log" 2>&1 &


LEARNED_MODEL_JSON="${TRACER_PHASE_D5_LEARNED_MODEL_JSON:-}"
if [ -n "$LEARNED_MODEL_JSON" ]; then
  echo "[TRACER] launching D5 learned selector shadow node"
  pkill -f "tracer_phase_d5_learned_selector_shadow_node_v0.py" 2>/dev/null || true

  TRACER_PHASE_D5_LEARNED_MODEL_JSON="$LEARNED_MODEL_JSON" \
  TRACER_PHASE_D5_LEARNED_LOG="$D5_LOG_DIR/learned_selector_shadow_v0.csv" \
  nohup /usr/bin/python3 scripts/runtime/tracer_phase_d5_learned_selector_shadow_node_v0.py \
    > "$D5_LOG_DIR/learned_selector_shadow_node.log" 2>&1 &

  echo "  learned_model=$LEARNED_MODEL_JSON"
  echo "  learned_log=$D5_LOG_DIR/learned_selector_shadow_v0.csv"
  echo "  learned_node_log=$D5_LOG_DIR/learned_selector_shadow_node.log"
fi


if [ "${TRACER_PHASE_D5_ENABLE_GATE_DRYRUN:-0}" = "1" ]; then
  echo "[TRACER] launching D5 gated selector dry-run node"
  pkill -f "tracer_phase_d5_gated_selector_dryrun_node_v0.py" 2>/dev/null || true

  TRACER_PHASE_D5_GATE_LOG="$D5_LOG_DIR/gated_selector_dryrun_v0.csv" \
  nohup /usr/bin/python3 scripts/runtime/tracer_phase_d5_gated_selector_dryrun_node_v0.py \
    > "$D5_LOG_DIR/gated_selector_dryrun_node.log" 2>&1 &

  echo "  gate_log=$D5_LOG_DIR/gated_selector_dryrun_v0.csv"
  echo "  gate_node_log=$D5_LOG_DIR/gated_selector_dryrun_node.log"
fi

echo "[TRACER] D5 shadow nodes launched"
echo "  input_log=$D5_LOG_DIR/policy_inputs_v0.csv"
echo "  shadow_log=$D5_LOG_DIR/shadow_inputs_node.log"
echo "  logger_log=$D5_LOG_DIR/policy_input_logger.log"
