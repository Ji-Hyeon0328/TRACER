#!/usr/bin/env bash
set -euo pipefail

cd ~/Tracer/TRACER

echo "[TRACER] objective-gate long-prestand eval v0"

scripts/runtime/tracer_ensure_mpc_ref_bridge.sh
scripts/runtime/tracer_ensure_ros2_mpc_ref_udp_sender.sh

TRACER_RUNTIME_SELECTOR_PROFILE=balanced \
TRACER_RUNTIME_SELECTOR_PATH=configs/runtime/tracer_runtime_validated_theta_selector_v0.json \
TRACER_RESET_ROS2_DAEMON=1 \
scripts/runtime/tracer_ensure_runtime_validated_policy_node_v0.sh

TRACER_OBJECTIVE_PROFILE=energy_extreme \
TRACER_OBJECTIVE_HZ=5.0 \
TRACER_RESET_ROS2_DAEMON=0 \
scripts/runtime/tracer_ensure_objective_selector_beta_node_v0.sh

set +u
source /opt/ros/humble/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then source ros2_ws/install/setup.bash; fi
set -u

/usr/bin/python3 scripts/runtime/tracer_run_safe_bank_runtime_rollouts_v0.py \
  --profile energy_extreme \
  --duration-sec 3.0 \
  --sample-hz 10.0 \
  --repeats 10 \
  --control-gazebo \
  --stand-reset-gazebo \
  --reset-model-name a1_gazebo \
  --reset-z 0.32 \
  --reset-settle-sec 1.0 \
  --prestand-sec 4.0 \
  --metric-warmup-sec 0.2 \
  --disable-beta-pub

LATEST=$(ls -td data/rollouts/safe_bank_runtime_v0/safe_bank_runtime_v0_* | head -1)

echo
echo "===== LATEST ====="
echo "$LATEST"

python3 scripts/training/tracer_score_theta_sweep_v0.py \
  --summary-csv "$LATEST/theta_sweep_summary.csv"

python3 scripts/training/tracer_rescore_theta_profiles_v1_safety_gate.py \
  --scored-csv "$LATEST/theta_sweep_scored_v0.csv"

echo
echo "===== SUMMARY ====="
cat "$LATEST/theta_sweep_summary.csv"
