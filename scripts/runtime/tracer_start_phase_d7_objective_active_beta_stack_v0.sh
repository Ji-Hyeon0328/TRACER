#!/usr/bin/env bash
set -euo pipefail

# Phase-D7 active beta topic integration.
#
# Goal:
#   D7 publishes live beta to /tracer/objective_beta.
#
# Safety:
#   D5 static beta is moved to /tracer/objective_beta_static.
#   D6 MLP still consumes static beta, so /tracer/mpc_reference control remains protected.

export TRACER_PHASE_D5_BETA_TOPIC="${TRACER_PHASE_D5_BETA_TOPIC:-/tracer/objective_beta_static}"
export TRACER_PHASE_D6_BETA_TOPIC="${TRACER_PHASE_D6_BETA_TOPIC:-/tracer/objective_beta_static}"

export TRACER_PHASE_D7_ACTUAL_BETA_TOPIC="${TRACER_PHASE_D7_ACTUAL_BETA_TOPIC:-/tracer/objective_beta_static}"
export TRACER_PHASE_D7_OBJECTIVE_BETA_TOPIC="${TRACER_PHASE_D7_OBJECTIVE_BETA_TOPIC:-/tracer/objective_beta}"

echo "[TRACER] Phase-D7 active beta stack wrapper"
echo "  TRACER_PHASE_D5_BETA_TOPIC=${TRACER_PHASE_D5_BETA_TOPIC}"
echo "  TRACER_PHASE_D6_BETA_TOPIC=${TRACER_PHASE_D6_BETA_TOPIC}"
echo "  TRACER_PHASE_D7_ACTUAL_BETA_TOPIC=${TRACER_PHASE_D7_ACTUAL_BETA_TOPIC}"
echo "  TRACER_PHASE_D7_OBJECTIVE_BETA_TOPIC=${TRACER_PHASE_D7_OBJECTIVE_BETA_TOPIC}"

exec bash scripts/runtime/tracer_start_phase_d7_objective_shadow_stack_v0.sh
