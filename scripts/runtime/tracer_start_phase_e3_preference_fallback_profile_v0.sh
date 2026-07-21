#!/usr/bin/env bash
set -euo pipefail

RESET_Y="${TRACER_RESET_Y_OFFSET:-0.0}"
NEG_THRESHOLD="${TRACER_PHASE_E3_NEG_RESET_THRESHOLD:--0.15}"

MODE="$(python3 - <<PY
reset_y = float("${RESET_Y}")
thr = float("${NEG_THRESHOLD}")
print("negative_lateral_fallback" if reset_y < thr else "nominal_full_learned")
PY
)"

echo "[TRACER] Phase-E3 preference fallback profile v0"
echo "  TRACER_RESET_Y_OFFSET=${RESET_Y}"
echo "  TRACER_PHASE_E3_NEG_RESET_THRESHOLD=${NEG_THRESHOLD}"
echo "  selected_mode=${MODE}"

if [ "$MODE" = "negative_lateral_fallback" ]; then
  export TRACER_PHASE_D5_GATE_USE_LEARNED_VX=0
  export TRACER_PHASE_D5_GATE_USE_LEARNED_YAW=0
  export TRACER_PHASE_D5_GATE_USE_LEARNED_BODY_H=0
  export TRACER_PHASE_D5_GATE_USE_LEARNED_CLEARANCE=1
  export TRACER_PHASE_D5_GATE_USE_LEARNED_ENABLE=0
else
  export TRACER_PHASE_D5_GATE_USE_LEARNED_VX=1
  export TRACER_PHASE_D5_GATE_USE_LEARNED_YAW=1
  export TRACER_PHASE_D5_GATE_USE_LEARNED_BODY_H=0
  export TRACER_PHASE_D5_GATE_USE_LEARNED_CLEARANCE=1
  export TRACER_PHASE_D5_GATE_USE_LEARNED_ENABLE=0
fi

echo "  learned_dims:"
echo "    vx=${TRACER_PHASE_D5_GATE_USE_LEARNED_VX}"
echo "    yaw=${TRACER_PHASE_D5_GATE_USE_LEARNED_YAW}"
echo "    body_h=${TRACER_PHASE_D5_GATE_USE_LEARNED_BODY_H}"
echo "    clearance=${TRACER_PHASE_D5_GATE_USE_LEARNED_CLEARANCE}"
echo "    enable=${TRACER_PHASE_D5_GATE_USE_LEARNED_ENABLE}"

if [ "${TRACER_PHASE_E3_PROFILE_PRINT_ONLY:-0}" = "1" ]; then
  echo "[TRACER] print-only mode; not launching stack"
  exit 0
fi

exec bash scripts/runtime/tracer_start_phase_d7_objective_active_beta_stack_v0.sh
