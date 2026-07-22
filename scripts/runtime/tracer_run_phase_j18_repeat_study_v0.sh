#!/usr/bin/env bash
set -u

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

N="${TRACER_J18_N_PER_MODE:-6}"
MODES="${TRACER_J18_MODES:-baseline,flat_noop,upslope_noop}"
LOG_ROOT="${TRACER_J18_LOG_ROOT:-logs/phase_j/j18_repeat_study_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_CSV="$LOG_ROOT/j18_repeat_study_summary_v0.csv"
FINAL_MD="reports/phase_j/j18_repeat_study_summary_v0.md"

mkdir -p "$LOG_ROOT" reports/phase_j

echo "[J18] ROOT=$ROOT"
echo "[J18] N_PER_MODE=$N"
echo "[J18] MODES=$MODES"
echo "[J18] LOG_ROOT=$LOG_ROOT"
echo "[J18] SUMMARY_CSV=$SUMMARY_CSV"
echo

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[J18:ERROR] ros2 not found. Run: set +u; use_ros2"
  exit 2
fi

if [[ ! -x scripts/runtime/tracer_phase_j15b_flat_noop_override_node_v0.py ]]; then
  echo "[J18:ERROR] missing executable: scripts/runtime/tracer_phase_j15b_flat_noop_override_node_v0.py"
  exit 3
fi

if [[ ! -x scripts/runtime/tracer_phase_j17_upslope_noop_override_node_v0.py ]]; then
  echo "[J18:ERROR] missing executable: scripts/runtime/tracer_phase_j17_upslope_noop_override_node_v0.py"
  exit 4
fi

echo "mode,idx,goal_reached,final_x,final_y,max_x,max_abs_y,mean_abs_y,rows,valid_xy,j7_rows,j7_accepted,j7_accept_rate,j7_projected,j7_empirical,j7_failsafe,manifest,d5_log_dir,j7_csv" > "$SUMMARY_CSV"

SUDO_KEEP_PID=""
if sudo -v; then
  (
    while true; do
      sudo -n true 2>/dev/null || exit 0
      sleep 60
    done
  ) &
  SUDO_KEEP_PID="$!"
fi

kill_active_nodes() {
  pkill -f "tracer_phase_j3_meta_action_shadow_node_v0.py" 2>/dev/null || true
  pkill -f "tracer_phase_j15b_flat_noop_override_node_v0.py" 2>/dev/null || true
  pkill -f "tracer_phase_j17_upslope_noop_override_node_v0.py" 2>/dev/null || true
  pkill -f "tracer_phase_j4_meta_action_ref_projection_shadow_node_v0.py" 2>/dev/null || true
  pkill -f "tracer_phase_j7_guarded_meta_action_ref_gate_node_v0.py" 2>/dev/null || true
}

cleanup_runtime() {
  kill_active_nodes
  bash scripts/runtime/tracer_stop_phase_c_mixed_stack_v0.sh >/dev/null 2>&1 || true
  sleep 2
}

final_cleanup() {
  kill_active_nodes
  if [[ -n "${SUDO_KEEP_PID:-}" ]]; then
    kill "$SUDO_KEEP_PID" 2>/dev/null || true
  fi
}
trap final_cleanup EXIT

start_active_nodes() {
  local mode="$1"
  local idx="$2"
  local run_dir="$LOG_ROOT/${idx}_${mode}"

  mkdir -p "$run_dir/j3" "$run_dir/j4" "$run_dir/j7" "$run_dir/j15b" "$run_dir/j17"

  echo "[J18][$mode][$idx] start active nodes"

  TRACER_PHASE_J3_LOG_DIR="$run_dir/j3" \
  TRACER_PHASE_J3_POLICY_JSON=models/phase_j/j2_meta_action_tabular_policy_v0.json \
  TRACER_PHASE_J3_BANK_JSON=configs/phase_j/meta_action_bank_v0.json \
  TRACER_PHASE_J3_BETA_TOPIC=/tracer/objective_beta \
  TRACER_PHASE_J3_RAM_TOPIC=/tracer/ram_mismatch \
  python3 scripts/runtime/tracer_phase_j3_meta_action_shadow_node_v0.py \
    > "$run_dir/j3/node.log" 2>&1 &
  echo $! >> "$run_dir/pids.txt"

  local theta_topic="/tracer/meta_action_theta_shadow"
  local allowed=""

  if [[ "$mode" == "flat_noop" ]]; then
    TRACER_PHASE_J15B_LOG_DIR="$run_dir/j15b" \
    TRACER_PHASE_J15B_INPUT_THETA_TOPIC=/tracer/meta_action_theta_shadow \
    TRACER_PHASE_J15B_OUTPUT_THETA_TOPIC=/tracer/meta_action_theta_j15b_flat_noop_shadow \
    python3 scripts/runtime/tracer_phase_j15b_flat_noop_override_node_v0.py \
      > "$run_dir/j15b/node.log" 2>&1 &
    echo $! >> "$run_dir/pids.txt"

    theta_topic="/tracer/meta_action_theta_j15b_flat_noop_shadow"
    allowed="flat"
  elif [[ "$mode" == "upslope_noop" ]]; then
    TRACER_PHASE_J17_LOG_DIR="$run_dir/j17" \
    TRACER_PHASE_J17_INPUT_THETA_TOPIC=/tracer/meta_action_theta_shadow \
    TRACER_PHASE_J17_OUTPUT_THETA_TOPIC=/tracer/meta_action_theta_j17_upslope_noop_shadow \
    python3 scripts/runtime/tracer_phase_j17_upslope_noop_override_node_v0.py \
      > "$run_dir/j17/node.log" 2>&1 &
    echo $! >> "$run_dir/pids.txt"

    theta_topic="/tracer/meta_action_theta_j17_upslope_noop_shadow"
    allowed="upslope"
  else
    echo "[J18:ERROR] unknown active mode: $mode"
    exit 5
  fi

  TRACER_PHASE_J4_LOG_DIR="$run_dir/j4" \
  TRACER_PHASE_J4_EMPIRICAL_REF_TOPIC=/tracer/empirical_mpc_reference \
  TRACER_PHASE_J4_THETA_TOPIC="$theta_topic" \
  TRACER_PHASE_J4_PROJECTED_REF_TOPIC=/tracer/meta_action_projected_ref_shadow \
  python3 scripts/runtime/tracer_phase_j4_meta_action_ref_projection_shadow_node_v0.py \
    > "$run_dir/j4/node.log" 2>&1 &
  echo $! >> "$run_dir/pids.txt"

  TRACER_PHASE_J7_LOG_DIR="$run_dir/j7" \
  TRACER_PHASE_J7_OUTPUT_REF_TOPIC=/tracer/mpc_reference \
  TRACER_PHASE_J7_EMPIRICAL_REF_TOPIC=/tracer/empirical_mpc_reference \
  TRACER_PHASE_J7_PROJECTED_REF_TOPIC=/tracer/meta_action_projected_ref_shadow \
  TRACER_PHASE_J7_THETA_TOPIC="$theta_topic" \
  TRACER_PHASE_J7_ALLOWED_CONTEXTS="$allowed" \
  TRACER_PHASE_J7_ACTIVE_FAILSAFE_HOLD=1 \
  TRACER_PHASE_J7_MAX_INPUT_AGE_S=0.75 \
  python3 scripts/runtime/tracer_phase_j7_guarded_meta_action_ref_gate_node_v0.py \
    > "$run_dir/j7/node.log" 2>&1 &
  echo $! >> "$run_dir/pids.txt"

  sleep 3
}

summarize_run() {
  local mode="$1"
  local idx="$2"
  local manifest="$3"
  local d5_log_dir="$4"
  local j7_csv="$5"

  python3 - "$mode" "$idx" "$manifest" "$d5_log_dir" "$j7_csv" "$SUMMARY_CSV" <<'PY'
import csv, math, sys
from collections import Counter

mode, idx, manifest, d5_log_dir, j7_csv, summary_csv = sys.argv[1:]

def ff(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

policy_csv = f"{d5_log_dir}/policy_inputs_v0.csv"
rows = list(csv.DictReader(open(policy_csv)))
valid = [r for r in rows if ff(r.get("x")) is not None and ff(r.get("y")) is not None]
xs = [ff(r["x"]) for r in valid]
ys = [ff(r["y"]) for r in valid]

final_x = xs[-1] if xs else float("nan")
final_y = ys[-1] if ys else float("nan")
max_x = max(xs) if xs else float("nan")
max_abs_y = max(abs(y) for y in ys) if ys else float("nan")
mean_abs_y = sum(abs(y) for y in ys) / len(ys) if ys else float("nan")
goal = bool(xs and max_x >= 8.0)

j7_rows = j7_accepted = j7_projected = j7_empirical = j7_failsafe = 0
if j7_csv and j7_csv != "NA":
    try:
        jrows = list(csv.DictReader(open(j7_csv)))
        j7_rows = len(jrows)
        j7_accepted = sum(1 for r in jrows if r.get("accepted") in ("1", "True", "true") or r.get("decision_reason") == "accepted_guarded" or r.get("reason") == "accepted_guarded")
        src = Counter(r.get("output_source", r.get("source", "")) for r in jrows)
        j7_projected = src.get("projected", 0)
        j7_empirical = src.get("empirical", 0)
        j7_failsafe = src.get("failsafe_hold", 0)
    except Exception:
        pass

j7_accept_rate = (j7_accepted / j7_rows) if j7_rows else 0.0

with open(summary_csv, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        mode, idx, goal,
        f"{final_x:.6f}", f"{final_y:.6f}", f"{max_x:.6f}",
        f"{max_abs_y:.6f}", f"{mean_abs_y:.6f}",
        len(rows), len(valid),
        j7_rows, j7_accepted, f"{j7_accept_rate:.6f}",
        j7_projected, j7_empirical, j7_failsafe,
        manifest, d5_log_dir, j7_csv,
    ])

print(f"[J18][{mode}][{idx}] goal={goal} final_x={final_x:.3f} final_y={final_y:.3f} max_abs_y={max_abs_y:.3f} mean_abs_y={mean_abs_y:.3f} j7_accept={j7_accept_rate:.3f}")
PY
}

run_one() {
  local mode="$1"
  local idx="$2"
  local run_dir="$LOG_ROOT/${idx}_${mode}"
  mkdir -p "$run_dir"

  echo
  echo "============================================================"
  echo "[J18] run idx=$idx mode=$mode"
  echo "============================================================"

  cleanup_runtime

  local d5_out="/tracer/mpc_reference"
  local j7_csv="NA"

  if [[ "$mode" == "flat_noop" || "$mode" == "upslope_noop" ]]; then
    start_active_nodes "$mode" "$idx"
    d5_out="/tracer/d5_gated_ref_shadow_for_j18_${mode}_${idx}"
    j7_csv="$run_dir/j7/guarded_meta_action_ref_gate_j7_v0.csv"
  fi

  local conda_base
  conda_base="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"

  TRACER_D4_START_SCRIPT=scripts/runtime/tracer_start_phase_e3_preference_fallback_profile_v0.sh \
  TRACER_RESET_Y_OFFSET=0.00 \
  TRACER_PHASE_E3_NEG_RESET_THRESHOLD=-0.15 \
  TRACER_PHASE_D5_CONTROL_MODE=gated \
  TRACER_PHASE_D6_USE_MLP_GATE=1 \
  TRACER_PHASE_D6_TORCH_SITE_PACKAGES="$conda_base/envs/tracer_train/lib/python3.10/site-packages" \
  TRACER_PHASE_D7_OBJECTIVE_MODEL_JSON=models/phase_d7/d7_objective_selector_ridge_runtime_aligned_v1.json \
  TRACER_PHASE_D7_BETA_FLOOR=0.05 \
  TRACER_PHASE_D7_PRIOR_BLEND=0.20 \
  TRACER_PHASE_D5_BETA_TOPIC=/tracer/objective_beta_static \
  TRACER_PHASE_D6_BETA_TOPIC=/tracer/objective_beta \
  TRACER_PHASE_D7_ACTUAL_BETA_TOPIC=/tracer/objective_beta_static \
  TRACER_PHASE_D7_OBJECTIVE_BETA_TOPIC=/tracer/objective_beta \
  TRACER_REF_TOPIC=/tracer/empirical_mpc_reference \
  TRACER_PHASE_D5_EMPIRICAL_REF_TOPIC=/tracer/empirical_mpc_reference \
  TRACER_PHASE_D5_GATED_CONTROL_TOPIC="$d5_out" \
  TRACER_PHASE_D4_POLICY_JSON=models/phase_d5/d5_empirical_policy_from_shadow_dataset_20260718_215059_v0.json \
  TRACER_PHASE_D6_MLP_MODEL_PT=models/phase_d6/d6_mlp_selector_d7_beta_conditioned_v0.pt \
  TRACER_D4_REPEAT_WORLD=tracer_mixed_stress_course_v5_lowfric_from_solid \
  TRACER_D4_REPEAT_GOAL_X=8.0 \
  TRACER_D4_REPEAT_LATERAL_BOUND=2.0 \
  TRACER_D4_REPEAT_N=1 \
  TRACER_D4_REPEAT_TIMEOUT_S=220 \
  TRACER_D4_HOLD_OBS_S=30 \
  TRACER_D4_REPEAT_OPEN_GUI=0 \
  TRACER_PHASE_D4_STOP_MARGIN=-0.05 \
  bash scripts/runtime/tracer_run_phase_d4_context_meta_repeat_v0.sh \
    > "$run_dir/d4_rollout.log" 2>&1

  local manifest
  manifest="$(ls -t reports/phase_d4_context_meta_repeat_*_manifest.tsv | head -1)"
  echo "$manifest" > "$run_dir/manifest.txt"

  local d5_log_dir
  d5_log_dir="$(awk -F'\t' 'NR==2{print $8}' "$manifest")"
  echo "$d5_log_dir" > "$run_dir/d5_log_dir.txt"

  if [[ -f "$j7_csv" ]]; then
    python3 scripts/training/tracer_check_phase_j7_guarded_ref_gate_log_v0.py \
      --csv "$j7_csv" \
      --out-md "reports/phase_j/j18_${mode}_${idx}_j7_check_v0.md" \
      > "$run_dir/j7_check_stdout.log" 2>&1 || true
  fi

  summarize_run "$mode" "$idx" "$manifest" "$d5_log_dir" "$j7_csv"

  kill_active_nodes
  sleep 2
}

IFS=',' read -r -a MODE_ARR <<< "$MODES"

for idx in $(seq 1 "$N"); do
  for mode in "${MODE_ARR[@]}"; do
    run_one "$mode" "$idx"
  done
done

cleanup_runtime

python3 - "$SUMMARY_CSV" "$FINAL_MD" "$LOG_ROOT" <<'PY'
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path

summary_csv, final_md, log_root = sys.argv[1:]
rows = list(csv.DictReader(open(summary_csv)))

groups = defaultdict(list)
for r in rows:
    groups[r["mode"]].append(r)

lines = []
lines.append("# TRACER Phase-J18 Repeat Study Summary v0")
lines.append("")
lines.append(f"- log root: `{log_root}`")
lines.append(f"- summary csv: `{summary_csv}`")
lines.append("")
lines.append("## Group summary")
lines.append("")
lines.append("| mode | n | goals | max_abs_y mean | max_abs_y min | max_abs_y max | mean_abs_y mean |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")

for mode, rs in groups.items():
    max_abs = [float(r["max_abs_y"]) for r in rs]
    mean_abs = [float(r["mean_abs_y"]) for r in rs]
    goals = sum(1 for r in rs if r["goal_reached"] == "True")
    lines.append(
        f"| {mode} | {len(rs)} | {goals} | "
        f"{statistics.mean(max_abs):.6f} | {min(max_abs):.6f} | {max(max_abs):.6f} | "
        f"{statistics.mean(mean_abs):.6f} |"
    )

lines.append("")
lines.append("## Per-run results")
lines.append("")
lines.append("| mode | idx | goal | final_x | final_y | max_abs_y | mean_abs_y | j7_accept_rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append(
        f"| {r['mode']} | {r['idx']} | {r['goal_reached']} | "
        f"{r['final_x']} | {r['final_y']} | {r['max_abs_y']} | {r['mean_abs_y']} | {r['j7_accept_rate']} |"
    )

lines.append("")
lines.append("## Interpretation guide")
lines.append("")
lines.append("- If baseline stays low while flat_noop stays low, flat active routing is acceptable only for no-op/conservative commands.")
lines.append("- If upslope_noop remains higher than baseline, upslope active routing or segment timing should remain frozen.")
lines.append("- If baseline itself varies widely, active comparisons need larger N-repeat statistics before changing the action bank.")

Path(final_md).write_text("\n".join(lines) + "\n")
print(f"[J18] wrote {final_md}")
print("\n".join(lines))
PY

echo
echo "[J18] done"
echo "[J18] summary csv: $SUMMARY_CSV"
echo "[J18] final md: $FINAL_MD"
