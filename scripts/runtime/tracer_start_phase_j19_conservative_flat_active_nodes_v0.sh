#!/usr/bin/env bash
set -u

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

LOG_ROOT="${TRACER_PHASE_J19_LOG_ROOT:-logs/phase_j/j19_conservative_flat_active_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_ROOT"/{j3,j15b,j4,j7}

echo "$LOG_ROOT" | tee /tmp/tracer_j19_latest_log_root.txt

echo "[TRACER:J19] Conservative flat-only active profile"
echo "[TRACER:J19] LOG_ROOT=$LOG_ROOT"
echo "[TRACER:J19] output=/tracer/mpc_reference"
echo "[TRACER:J19] allowed_contexts=flat"
echo

pkill -f "tracer_phase_j3_meta_action_shadow_node_v0.py" 2>/dev/null || true
pkill -f "tracer_phase_j15b_flat_noop_override_node_v0.py" 2>/dev/null || true
pkill -f "tracer_phase_j4_meta_action_ref_projection_shadow_node_v0.py" 2>/dev/null || true
pkill -f "tracer_phase_j7_guarded_meta_action_ref_gate_node_v0.py" 2>/dev/null || true
sleep 1

TRACER_PHASE_J3_LOG_DIR="$LOG_ROOT/j3" \
TRACER_PHASE_J3_POLICY_JSON=models/phase_j/j2_meta_action_tabular_policy_v0.json \
TRACER_PHASE_J3_BANK_JSON=configs/phase_j/meta_action_bank_v0.json \
TRACER_PHASE_J3_BETA_TOPIC=/tracer/objective_beta \
TRACER_PHASE_J3_RAM_TOPIC=/tracer/ram_mismatch \
python3 scripts/runtime/tracer_phase_j3_meta_action_shadow_node_v0.py \
  > "$LOG_ROOT/j3/node.log" 2>&1 &
echo $! > "$LOG_ROOT/j3.pid"

TRACER_PHASE_J15B_LOG_DIR="$LOG_ROOT/j15b" \
TRACER_PHASE_J15B_INPUT_THETA_TOPIC=/tracer/meta_action_theta_shadow \
TRACER_PHASE_J15B_OUTPUT_THETA_TOPIC=/tracer/meta_action_theta_j19_flat_noop_shadow \
python3 scripts/runtime/tracer_phase_j15b_flat_noop_override_node_v0.py \
  > "$LOG_ROOT/j15b/node.log" 2>&1 &
echo $! > "$LOG_ROOT/j15b.pid"

TRACER_PHASE_J4_LOG_DIR="$LOG_ROOT/j4" \
TRACER_PHASE_J4_EMPIRICAL_REF_TOPIC=/tracer/empirical_mpc_reference \
TRACER_PHASE_J4_THETA_TOPIC=/tracer/meta_action_theta_j19_flat_noop_shadow \
TRACER_PHASE_J4_PROJECTED_REF_TOPIC=/tracer/meta_action_projected_ref_shadow \
python3 scripts/runtime/tracer_phase_j4_meta_action_ref_projection_shadow_node_v0.py \
  > "$LOG_ROOT/j4/node.log" 2>&1 &
echo $! > "$LOG_ROOT/j4.pid"

TRACER_PHASE_J7_LOG_DIR="$LOG_ROOT/j7" \
TRACER_PHASE_J7_OUTPUT_REF_TOPIC=/tracer/mpc_reference \
TRACER_PHASE_J7_EMPIRICAL_REF_TOPIC=/tracer/empirical_mpc_reference \
TRACER_PHASE_J7_PROJECTED_REF_TOPIC=/tracer/meta_action_projected_ref_shadow \
TRACER_PHASE_J7_THETA_TOPIC=/tracer/meta_action_theta_j19_flat_noop_shadow \
TRACER_PHASE_J7_ALLOWED_CONTEXTS=flat \
TRACER_PHASE_J7_ACTIVE_FAILSAFE_HOLD=1 \
TRACER_PHASE_J7_MAX_INPUT_AGE_S=0.75 \
python3 scripts/runtime/tracer_phase_j7_guarded_meta_action_ref_gate_node_v0.py \
  > "$LOG_ROOT/j7/node.log" 2>&1 &
echo $! > "$LOG_ROOT/j7.pid"

sleep 3

echo "[TRACER:J19] process check"
pgrep -af "tracer_phase_j3|tracer_phase_j15b|tracer_phase_j4|tracer_phase_j7" || true

echo
echo "[TRACER:J19] topic check"
ros2 topic info /tracer/mpc_reference -v | sed -n '1,80p' || true

echo
echo "[TRACER:J19] started."
echo "[TRACER:J19] IMPORTANT: when running D4/D5 rollout, divert D5 output:"
echo "  TRACER_PHASE_D5_GATED_CONTROL_TOPIC=/tracer/d5_gated_ref_shadow_for_j19"
