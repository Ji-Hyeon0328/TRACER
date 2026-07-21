#!/usr/bin/env bash
set -euo pipefail

TAG="${TRACER_PHASE_F3_TAG:-clean}"
RESET_Y="${TRACER_RESET_Y_OFFSET:-0.00}"
F1_DURATION="${TRACER_PHASE_F3_F1_DURATION:-170}"
F1_HZ="${TRACER_PHASE_F3_F1_HZ:-50}"
TIMEOUT_S="${TRACER_PHASE_F3_TIMEOUT_S:-240}"
HOLD_OBS_S="${TRACER_PHASE_F3_HOLD_OBS_S:-30}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="logs/phase_f/f3_${TAG}_${STAMP}_rollout.log"
TRUE_CSV="datasets/phase_f/f3_${TAG}_${STAMP}_true_metrics.csv"
SUMMARY_CSV="datasets/phase_f/f3_${TAG}_${STAMP}_true_metric_summary.csv"
SUMMARY_MD="reports/phase_f/f3_${TAG}_${STAMP}_true_metric_summary.md"

mkdir -p logs/phase_f datasets/phase_f reports/phase_f

echo "[TRACER] Phase-F3 true metric rollout with F1 logger v0"
echo "  TAG=${TAG}"
echo "  RESET_Y=${RESET_Y}"
echo "  F1_DURATION=${F1_DURATION}"
echo "  F1_HZ=${F1_HZ}"
echo "  TIMEOUT_S=${TIMEOUT_S}"
echo "  TRUE_CSV=${TRUE_CSV}"
echo "  RUN_LOG=${RUN_LOG}"

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"

(
  set +u

  # Self-contained ROS2 activation.
  # Do not rely on interactive-shell helper functions such as use_ros2.
  if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
  fi

  if [ -f ros2_ws/install/setup.bash ]; then
    source ros2_ws/install/setup.bash
  fi

  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-98}"
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

  sudo -v

  TRACER_D4_START_SCRIPT=scripts/runtime/tracer_start_phase_e3_preference_fallback_profile_v0.sh \
  TRACER_RESET_Y_OFFSET="${RESET_Y}" \
  TRACER_PHASE_E3_NEG_RESET_THRESHOLD=-0.15 \
  TRACER_PHASE_D5_CONTROL_MODE=gated \
  TRACER_PHASE_D6_USE_MLP_GATE=1 \
  TRACER_PHASE_D6_TORCH_SITE_PACKAGES="$CONDA_BASE/envs/tracer_train/lib/python3.10/site-packages" \
  TRACER_PHASE_D7_OBJECTIVE_MODEL_JSON=models/phase_d7/d7_objective_selector_ridge_runtime_aligned_v1.json \
  TRACER_PHASE_D7_BETA_FLOOR=0.05 \
  TRACER_PHASE_D7_PRIOR_BLEND=0.20 \
  TRACER_PHASE_D5_BETA_TOPIC=/tracer/objective_beta_static \
  TRACER_PHASE_D6_BETA_TOPIC=/tracer/objective_beta \
  TRACER_PHASE_D7_ACTUAL_BETA_TOPIC=/tracer/objective_beta_static \
  TRACER_PHASE_D7_OBJECTIVE_BETA_TOPIC=/tracer/objective_beta \
  TRACER_PHASE_D4_POLICY_JSON=models/phase_d5/d5_empirical_policy_from_shadow_dataset_20260718_215059_v0.json \
  TRACER_PHASE_D6_MLP_MODEL_PT=models/phase_d6/d6_mlp_selector_d7_beta_conditioned_v0.pt \
  TRACER_PHASE_D5_EMPIRICAL_REF_TOPIC=/tracer/empirical_mpc_reference \
  TRACER_PHASE_D5_GATED_CONTROL_TOPIC=/tracer/mpc_reference \
  TRACER_D4_REPEAT_WORLD=tracer_mixed_stress_course_v5_lowfric_from_solid \
  TRACER_D4_REPEAT_GOAL_X=8.0 \
  TRACER_D4_REPEAT_LATERAL_BOUND=2.0 \
  TRACER_D4_REPEAT_N=1 \
  TRACER_D4_REPEAT_TIMEOUT_S="${TIMEOUT_S}" \
  TRACER_D4_HOLD_OBS_S="${HOLD_OBS_S}" \
  TRACER_D4_REPEAT_OPEN_GUI=0 \
  TRACER_PHASE_D4_STOP_MARGIN=-0.05 \
  bash scripts/runtime/tracer_run_phase_d4_context_meta_repeat_v0.sh
) 2>&1 | tee "${RUN_LOG}" &

RUN_PID=$!

echo "[TRACER] waiting until rollout reaches t=10s..."
for i in $(seq 1 180); do
  if grep -q "\[TRACER\] t=10s state" "${RUN_LOG}" 2>/dev/null; then
    echo "[TRACER] detected t=10s state; starting F1 logger"
    break
  fi
  if ! kill -0 "${RUN_PID}" 2>/dev/null; then
    echo "[TRACER][ERROR] rollout process exited before t=10s" >&2
    wait "${RUN_PID}" || true
    exit 4
  fi
  sleep 1
done

bash scripts/training/tracer_collect_phase_f1_ros1_true_metrics_v0.sh \
  --out-csv "${TRUE_CSV}" \
  --duration "${F1_DURATION}" \
  --hz "${F1_HZ}"

echo "[TRACER] waiting for rollout process to finish..."
wait "${RUN_PID}" || true

MANIFEST_ABS="$(grep -Eo '/home/kraken/Tracer/TRACER/reports/phase_d4_context_meta_repeat_[0-9]+_manifest.tsv' "${RUN_LOG}" | tail -1 || true)"
if [ -z "${MANIFEST_ABS}" ]; then
  MANIFEST_REL="$(ls -t reports/phase_d4_context_meta_repeat_*_manifest.tsv | head -1)"
else
  MANIFEST_REL="${MANIFEST_ABS#/home/kraken/Tracer/TRACER/}"
fi

echo "[TRACER] MANIFEST=${MANIFEST_REL}"
echo "[TRACER] TRUE_CSV=${TRUE_CSV}"

python3 scripts/training/tracer_build_phase_f2_true_metric_summary_v0.py \
  --true-csv "${TRUE_CSV}" \
  --manifest "${MANIFEST_REL}" \
  --out-csv "${SUMMARY_CSV}" \
  --out-md "${SUMMARY_MD}"

echo
echo "[TRACER] Phase-F3 output files:"
echo "  ${TRUE_CSV}"
echo "  ${SUMMARY_CSV}"
echo "  ${SUMMARY_MD}"
echo "  ${RUN_LOG}"
