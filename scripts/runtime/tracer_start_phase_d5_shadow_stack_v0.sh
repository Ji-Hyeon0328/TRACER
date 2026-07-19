#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

POLICY_JSON="${TRACER_PHASE_D4_POLICY_JSON:-models/phase_d4/d4_context_meta_policy_rough_clearance_holdvx0025_v0.json}"

TS="$(date +%Y%m%d_%H%M%S)"
D5_LOG_DIR="${TRACER_PHASE_D5_LOG_DIR:-logs/phase_d5_shadow_${TS}}"
mkdir -p "$D5_LOG_DIR"

CONTROL_MODE="${TRACER_PHASE_D5_CONTROL_MODE:-shadow}"
EMPIRICAL_REF_TOPIC="${TRACER_PHASE_D5_EMPIRICAL_REF_TOPIC:-/tracer/empirical_mpc_reference}"
GATED_CONTROL_TOPIC="${TRACER_PHASE_D5_GATED_CONTROL_TOPIC:-/tracer/mpc_reference}"

D6_USE_MLP_GATE="${TRACER_PHASE_D6_USE_MLP_GATE:-0}"
D6_MLP_GATE_TOPIC="${TRACER_PHASE_D6_MLP_GATE_TOPIC:-/tracer/mlp_selector_shadow_ref}"

echo "[TRACER] starting base D4 policy stack first"
echo "  POLICY_JSON=$POLICY_JSON"
echo "  D5_LOG_DIR=$D5_LOG_DIR"
echo "  CONTROL_MODE=$CONTROL_MODE"

if [ "$CONTROL_MODE" = "gated" ]; then
  echo "[TRACER] gated control mode: D4 empirical selector publishes to $EMPIRICAL_REF_TOPIC"
  TRACER_PHASE_D4_POLICY_JSON="$POLICY_JSON" \
  TRACER_REF_TOPIC="$EMPIRICAL_REF_TOPIC" \
  bash scripts/runtime/tracer_start_phase_d4_context_meta_stack_v0.sh
else
  TRACER_PHASE_D4_POLICY_JSON="$POLICY_JSON" \
  bash scripts/runtime/tracer_start_phase_d4_context_meta_stack_v0.sh
fi

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

  if [ "$CONTROL_MODE" = "gated" ]; then
    LEARNED_ACTUAL_REF_TOPIC="$EMPIRICAL_REF_TOPIC"
  else
    LEARNED_ACTUAL_REF_TOPIC="${TRACER_REF_TOPIC:-/tracer/mpc_reference}"
  fi

  TRACER_PHASE_D5_LEARNED_MODEL_JSON="$LEARNED_MODEL_JSON" \
  TRACER_PHASE_D5_LEARNED_LOG="$D5_LOG_DIR/learned_selector_shadow_v0.csv" \
  TRACER_REF_TOPIC="$LEARNED_ACTUAL_REF_TOPIC" \
  nohup /usr/bin/python3 scripts/runtime/tracer_phase_d5_learned_selector_shadow_node_v0.py \
    > "$D5_LOG_DIR/learned_selector_shadow_node.log" 2>&1 &

  echo "  learned_model=$LEARNED_MODEL_JSON"
  echo "  learned_log=$D5_LOG_DIR/learned_selector_shadow_v0.csv"
  echo "  learned_node_log=$D5_LOG_DIR/learned_selector_shadow_node.log"
fi


if [ "${TRACER_PHASE_D5_ENABLE_GATE_DRYRUN:-0}" = "1" ] || [ "$CONTROL_MODE" = "gated" ]; then
  if [ "$CONTROL_MODE" = "gated" ]; then
    GATE_EMPIRICAL_TOPIC="$EMPIRICAL_REF_TOPIC"
    GATE_OUTPUT_TOPIC="$GATED_CONTROL_TOPIC"
    echo "[TRACER] launching D5 gated selector CONTROL node"
  else
    GATE_EMPIRICAL_TOPIC="${TRACER_REF_TOPIC:-/tracer/mpc_reference}"
    GATE_OUTPUT_TOPIC="${TRACER_PHASE_D5_GATED_SHADOW_TOPIC:-/tracer/gated_selector_dryrun_ref}"
    echo "[TRACER] launching D5 gated selector dry-run node"
  fi

  pkill -f "tracer_phase_d5_gated_selector_dryrun_node_v0.py" 2>/dev/null || true

  if [ "$D6_USE_MLP_GATE" = "1" ]; then
    GATE_LEARNED_TOPIC="$D6_MLP_GATE_TOPIC"
    GATE_USE_BODY_H="${TRACER_PHASE_D5_GATE_USE_LEARNED_BODY_H:-0}"
    GATE_USE_ENABLE="${TRACER_PHASE_D5_GATE_USE_LEARNED_ENABLE:-0}"
  else
    GATE_LEARNED_TOPIC="${TRACER_PHASE_D5_LEARNED_SHADOW_TOPIC:-/tracer/learned_selector_shadow_ref}"
    GATE_USE_BODY_H="${TRACER_PHASE_D5_GATE_USE_LEARNED_BODY_H:-1}"
    GATE_USE_ENABLE="${TRACER_PHASE_D5_GATE_USE_LEARNED_ENABLE:-1}"
  fi

  TRACER_PHASE_D5_GATE_LOG="$D5_LOG_DIR/gated_selector_dryrun_v0.csv" \
  TRACER_PHASE_D5_EMPIRICAL_REF_TOPIC="$GATE_EMPIRICAL_TOPIC" \
  TRACER_PHASE_D5_LEARNED_SHADOW_TOPIC="$GATE_LEARNED_TOPIC" \
  TRACER_PHASE_D5_GATED_SHADOW_TOPIC="$GATE_OUTPUT_TOPIC" \
  TRACER_PHASE_D5_GATE_USE_LEARNED_VX="${TRACER_PHASE_D5_GATE_USE_LEARNED_VX:-1}" \
  TRACER_PHASE_D5_GATE_USE_LEARNED_YAW="${TRACER_PHASE_D5_GATE_USE_LEARNED_YAW:-1}" \
  TRACER_PHASE_D5_GATE_USE_LEARNED_BODY_H="$GATE_USE_BODY_H" \
  TRACER_PHASE_D5_GATE_USE_LEARNED_CLEARANCE="${TRACER_PHASE_D5_GATE_USE_LEARNED_CLEARANCE:-1}" \
  TRACER_PHASE_D5_GATE_USE_LEARNED_ENABLE="$GATE_USE_ENABLE" \
  nohup /usr/bin/python3 scripts/runtime/tracer_phase_d5_gated_selector_dryrun_node_v0.py \
    > "$D5_LOG_DIR/gated_selector_dryrun_node.log" 2>&1 &

  echo "  gate_empirical_topic=$GATE_EMPIRICAL_TOPIC"
  echo "  gate_learned_topic=$GATE_LEARNED_TOPIC"
  echo "  gate_output_topic=$GATE_OUTPUT_TOPIC"
  echo "  gate_log=$D5_LOG_DIR/gated_selector_dryrun_v0.csv"
  echo "  gate_node_log=$D5_LOG_DIR/gated_selector_dryrun_node.log"
fi


D6_MLP_MODEL_PT="${TRACER_PHASE_D6_MLP_MODEL_PT:-}"
if [ -n "$D6_MLP_MODEL_PT" ]; then
  echo "[TRACER] launching D6 MLP selector shadow node"
  pkill -f "tracer_phase_d6_mlp_selector_shadow_node_v0.py" 2>/dev/null || true

  if [ "$CONTROL_MODE" = "gated" ]; then
    D6_MLP_ACTUAL_REF_TOPIC="$EMPIRICAL_REF_TOPIC"
  else
    D6_MLP_ACTUAL_REF_TOPIC="${TRACER_REF_TOPIC:-/tracer/mpc_reference}"
  fi

  D6_TORCH_SITE="${TRACER_PHASE_D6_TORCH_SITE_PACKAGES:-/home/kraken/miniconda3/envs/tracer_train/lib/python3.10/site-packages}"

  TRACER_PHASE_D6_TORCH_SITE_PACKAGES="$D6_TORCH_SITE" \
  TRACER_PHASE_D6_MLP_MODEL_PT="$D6_MLP_MODEL_PT" \
  TRACER_PHASE_D6_MLP_LOG="$D5_LOG_DIR/mlp_selector_shadow_v0.csv" \
  TRACER_PHASE_D6_MLP_ACTUAL_REF_TOPIC="$D6_MLP_ACTUAL_REF_TOPIC" \
  PYTHONPATH="$D6_TORCH_SITE:${PYTHONPATH:-}" \
  nohup /usr/bin/python3 scripts/runtime/tracer_phase_d6_mlp_selector_shadow_node_v0.py \
    > "$D5_LOG_DIR/mlp_selector_shadow_node.log" 2>&1 &

  echo "  mlp_model=$D6_MLP_MODEL_PT"
  echo "  mlp_actual_ref_topic=$D6_MLP_ACTUAL_REF_TOPIC"
  echo "  mlp_log=$D5_LOG_DIR/mlp_selector_shadow_v0.csv"
  echo "  mlp_node_log=$D5_LOG_DIR/mlp_selector_shadow_node.log"
fi

echo "[TRACER] D5 shadow nodes launched"
echo "  input_log=$D5_LOG_DIR/policy_inputs_v0.csv"
echo "  shadow_log=$D5_LOG_DIR/shadow_inputs_node.log"
echo "  logger_log=$D5_LOG_DIR/policy_input_logger.log"
