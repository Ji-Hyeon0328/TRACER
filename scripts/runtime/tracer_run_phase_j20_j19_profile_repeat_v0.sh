#!/usr/bin/env bash
set -u

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

N="${TRACER_J20_N_PER_MODE:-12}"
MODES="${TRACER_J20_MODES:-baseline,j19_profile}"
LOG_ROOT="${TRACER_J20_LOG_ROOT:-logs/phase_j/j20_j19_profile_repeat_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_CSV="$LOG_ROOT/j20_j19_profile_repeat_summary_v0.csv"
FINAL_MD="reports/phase_j/j20_j19_profile_repeat_summary_v0.md"

mkdir -p "$LOG_ROOT" reports/phase_j

echo "[J20] ROOT=$ROOT"
echo "[J20] N_PER_MODE=$N"
echo "[J20] MODES=$MODES"
echo "[J20] LOG_ROOT=$LOG_ROOT"
echo "[J20] SUMMARY_CSV=$SUMMARY_CSV"
echo

echo "mode,idx,goal_reached,final_x,final_y,max_x,max_abs_y,mean_abs_y,rows,valid_xy,j7_rows,j7_accepted,j7_accept_rate,j7_projected,j7_empirical,j7_failsafe,flat_rows,flat_accepted,upslope_rows,upslope_accepted,rough_rows,rough_accepted,downslope_rows,downslope_accepted,manifest,d5_log_dir,j7_csv" > "$SUMMARY_CSV"

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

start_j19_profile() {
  local idx="$1"
  local run_dir="$LOG_ROOT/${idx}_j19_profile"
  local node_dir="$run_dir/j19_nodes"
  mkdir -p "$run_dir"

  echo "[J20][j19_profile][$idx] start J19 profile nodes"

  TRACER_PHASE_J19_LOG_ROOT="$node_dir" \
    bash scripts/runtime/tracer_start_phase_j19_conservative_flat_active_nodes_v0.sh \
    > "$run_dir/j19_start_stdout.log" 2>&1

  echo "$node_dir"
}

parse_and_append() {
  local mode="$1"
  local idx="$2"
  local manifest="$3"
  local d5_log_dir="$4"
  local j7_csv="$5"
  local j7_report="$6"

  python3 - "$mode" "$idx" "$manifest" "$d5_log_dir" "$j7_csv" "$j7_report" "$SUMMARY_CSV" <<'PY'
import csv, math, re, sys
from pathlib import Path

mode, idx, manifest, d5_log_dir, j7_csv, j7_report, summary_csv = sys.argv[1:]

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

j = {
    "j7_rows": 0, "j7_accepted": 0, "j7_accept_rate": 0.0,
    "j7_projected": 0, "j7_empirical": 0, "j7_failsafe": 0,
    "flat_rows": 0, "flat_accepted": 0,
    "upslope_rows": 0, "upslope_accepted": 0,
    "rough_rows": 0, "rough_accepted": 0,
    "downslope_rows": 0, "downslope_accepted": 0,
}

if j7_report != "NA" and Path(j7_report).exists():
    txt = Path(j7_report).read_text()

    m = re.search(r"- rows: `?([0-9]+)`?", txt)
    if m: j["j7_rows"] = int(m.group(1))

    m = re.search(r"- accepted rows: `?([0-9]+)`?", txt)
    if m: j["j7_accepted"] = int(m.group(1))

    m = re.search(r"- accept rate: `?([0-9.]+)`?", txt)
    if m: j["j7_accept_rate"] = float(m.group(1))

    for src, key in [("projected", "j7_projected"), ("empirical", "j7_empirical"), ("failsafe_hold", "j7_failsafe")]:
        m = re.search(rf"\| {src} \| ([0-9]+) \|", txt)
        if m: j[key] = int(m.group(1))

    for ctx in ["flat", "upslope", "rough", "downslope"]:
        m = re.search(rf"\| {ctx} \| ([0-9]+) \| ([0-9]+) \| ([0-9.]+) \|", txt)
        if m:
            j[f"{ctx}_rows"] = int(m.group(1))
            j[f"{ctx}_accepted"] = int(m.group(2))

with open(summary_csv, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        mode, idx, goal,
        f"{final_x:.6f}", f"{final_y:.6f}", f"{max_x:.6f}",
        f"{max_abs_y:.6f}", f"{mean_abs_y:.6f}",
        len(rows), len(valid),
        j["j7_rows"], j["j7_accepted"], f"{j['j7_accept_rate']:.6f}",
        j["j7_projected"], j["j7_empirical"], j["j7_failsafe"],
        j["flat_rows"], j["flat_accepted"],
        j["upslope_rows"], j["upslope_accepted"],
        j["rough_rows"], j["rough_accepted"],
        j["downslope_rows"], j["downslope_accepted"],
        manifest, d5_log_dir, j7_csv,
    ])

print(f"[J20][{mode}][{idx}] goal={goal} final_x={final_x:.3f} final_y={final_y:.3f} max_abs_y={max_abs_y:.3f} mean_abs_y={mean_abs_y:.3f} j7_accept={j['j7_accept_rate']:.3f}")
PY
}

run_one() {
  local mode="$1"
  local idx="$2"
  local run_dir="$LOG_ROOT/${idx}_${mode}"
  mkdir -p "$run_dir"

  echo
  echo "============================================================"
  echo "[J20] run idx=$idx mode=$mode"
  echo "============================================================"

  cleanup_runtime

  local d5_out="/tracer/mpc_reference"
  local j7_csv="NA"
  local j7_report="NA"

  if [[ "$mode" == "j19_profile" ]]; then
    local node_dir
    node_dir="$(start_j19_profile "$idx")"
    d5_out="/tracer/d5_gated_ref_shadow_for_j20_j19_${idx}"
    j7_csv="$node_dir/j7/guarded_meta_action_ref_gate_j7_v0.csv"
    j7_report="reports/phase_j/j20_j19_profile_${idx}_j7_check_v0.md"
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

  if [[ "$mode" == "j19_profile" && -f "$j7_csv" ]]; then
    python3 scripts/training/tracer_check_phase_j7_guarded_ref_gate_log_v0.py \
      --csv "$j7_csv" \
      --out-md "$j7_report" \
      > "$run_dir/j7_check_stdout.log" 2>&1 || true
  fi

  parse_and_append "$mode" "$idx" "$manifest" "$d5_log_dir" "$j7_csv" "$j7_report"

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
lines.append("# TRACER Phase-J20 J19-Profile Repeat Summary v0")
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
lines.append("| mode | idx | goal | final_x | final_y | max_abs_y | mean_abs_y | j7 accepted | flat acc | upslope acc | projected | empirical | failsafe |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in rows:
    flat_acc = "NA"
    if int(r["flat_rows"]) > 0:
        flat_acc = f"{r['flat_accepted']}/{r['flat_rows']}"
    upslope_acc = "NA"
    if int(r["upslope_rows"]) > 0:
        upslope_acc = f"{r['upslope_accepted']}/{r['upslope_rows']}"

    lines.append(
        f"| {r['mode']} | {r['idx']} | {r['goal_reached']} | {r['final_x']} | {r['final_y']} | "
        f"{r['max_abs_y']} | {r['mean_abs_y']} | {r['j7_accepted']} | {flat_acc} | {upslope_acc} | "
        f"{r['j7_projected']} | {r['j7_empirical']} | {r['j7_failsafe']} |"
    )

lines.append("")
lines.append("## Interpretation guide")
lines.append("")
lines.append("- If j19_profile stays close to or below baseline, the J19 smoke was a bad single run and flat no-op profile can remain the conservative candidate.")
lines.append("- If j19_profile repeatedly exceeds baseline, the deployable profile has a hidden issue and should not be promoted.")
lines.append("- If both baseline and j19_profile have high variance, larger N or per-segment diagnostics are needed before action-bank changes.")

Path(final_md).write_text("\n".join(lines) + "\n")
print(f"[J20] wrote {final_md}")
print("\n".join(lines))
PY

echo
echo "[J20] done"
echo "[J20] summary csv: $SUMMARY_CSV"
echo "[J20] final md: $FINAL_MD"
