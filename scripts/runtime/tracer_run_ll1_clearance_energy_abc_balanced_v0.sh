#!/usr/bin/env bash
set -Eeo pipefail
set +u

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_ll1_docker}"
START_SCRIPT="${TRACER_ABC_ENERGY_START_SCRIPT:-/tmp/tracer_start_phase_d4_context_meta_ll1_residual_energy_v3.sh}"
ENERGY_LOGGER="${TRACER_ENERGY_LOGGER_HOST:-/tmp/tracer_ros1_joint_energy_proxy_logger_v0.py}"
WORLD="${TRACER_ABC_ENERGY_WORLD:-tracer_mixed_stress_course_v5_lowfric_from_solid}"
TIMEOUT_S="${TRACER_ABC_ENERGY_TIMEOUT_S:-220}"
HOLD_OBS_S="${TRACER_ABC_ENERGY_HOLD_OBS_S:-5}"
OPEN_FIRST_GUI="${TRACER_ABC_ENERGY_OPEN_FIRST_GUI:-0}"
START_INDEX="${TRACER_ABC_ENERGY_START_INDEX:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${TRACER_ABC_ENERGY_RUN_ROOT:-$ROOT/logs/revalidation_clearance_energy_abc_${TS}}"
MANIFEST="$RUN_ROOT/manifest.tsv"
VALIDATION_DIR="$RUN_ROOT/energy_validation"

NEUTRAL_TABLE='flat:0.210,0.320,0.045;start_flat:0.210,0.320,0.045;upslope:0.210,0.320,0.045;rough:0.2050,0.320,0.045;downslope:0.2025,0.320,0.045;goal_flat:0.2025,0.320,0.045;unknown:0.2025,0.320,0.045'
ROUGH_ONLY_TABLE='flat:0.210,0.320,0.045;start_flat:0.210,0.320,0.045;upslope:0.210,0.320,0.045;rough:0.2050,0.320,0.055;downslope:0.2025,0.320,0.045;goal_flat:0.2025,0.320,0.045;unknown:0.2025,0.320,0.045'
FIXED_HIGH_TABLE='flat:0.210,0.320,0.055;start_flat:0.210,0.320,0.055;upslope:0.210,0.320,0.055;rough:0.2050,0.320,0.055;downslope:0.2025,0.320,0.055;goal_flat:0.2025,0.320,0.055;unknown:0.2025,0.320,0.055'

mkdir -p "$RUN_ROOT" "$VALIDATION_DIR"

if [[ ! -s "$MANIFEST" ]]; then
  printf '%s\n' $'run_index\tblock\tcondition\tcondition_name\trc\tlog_dir\tconsole_log\tenergy_validation\tstart_time\tend_time' > "$MANIFEST"
fi

kill_energy_logger() {
  docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
    set +e
    pkill -INT -f "[t]racer_ros1_joint_energy_proxy_logger_v0.py" || true
  ' >/dev/null 2>&1 || true
}
trap kill_energy_logger EXIT

latest_energy_dir() {
  python3 - "$ROOT/logs" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
items = [
    p.parent for p in root.glob("phase_d4_context_meta_*/ros1_joint_energy_proxy.csv")
    if p.is_file()
]
if not items:
    raise SystemExit("no phase_d4 energy directory found")
print(max(items, key=lambda d: (d / "ros1_joint_energy_proxy.csv").stat().st_mtime_ns))
PY
}

condition_name() {
  case "$1" in
    A) printf '%s' neutral ;;
    B) printf '%s' rough_only_high ;;
    C) printf '%s' fixed_high ;;
    *) return 2 ;;
  esac
}

condition_table() {
  case "$1" in
    A) printf '%s' "$NEUTRAL_TABLE" ;;
    B) printf '%s' "$ROUGH_ONLY_TABLE" ;;
    C) printf '%s' "$FIXED_HIGH_TABLE" ;;
    *) return 2 ;;
  esac
}

validate_energy_csv() {
  local csv_path="$1"
  local condition="$2"
  local output_path="$3"

  python3 - "$csv_path" "$condition" > "$output_path" <<'PY'
import csv
import math
import statistics
import sys

path, condition = sys.argv[1], sys.argv[2]

def ff(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None

with open(path, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

print("path:", path)
print("condition:", condition)
print("rows:", len(rows))
if len(rows) < 500:
    raise SystemExit("[FAIL] fewer than 500 samples")

valid = [r for r in rows if r.get("valid") == "1"]
valid_fraction = len(valid) / float(len(rows))
print("valid_rows:", len(valid))
print("valid_fraction:", valid_fraction)
if valid_fraction < 0.95:
    raise SystemExit("[FAIL] valid fraction below 0.95")

times = [ff(r.get("sim_time")) for r in valid]
times = [x for x in times if x is not None]
positive_dt = [b-a for a,b in zip(times, times[1:]) if 0.0 < b-a <= 0.20]
backward = sum(b + 1e-12 < a for a,b in zip(times, times[1:]))
if len(times) < 2 or not positive_dt:
    raise SystemExit("[FAIL] insufficient finite simulation timestamps")
span = max(times) - min(times)
hz = 1.0 / statistics.median(positive_dt)
print("sim_span:", span)
print("effective_sim_hz:", hz)
print("backward_timestamp_count:", backward)
if backward:
    raise SystemExit("[FAIL] simulation time moved backward")
if span < 30.0:
    raise SystemExit("[FAIL] simulated duration below 30 s")
if not 40.0 <= hz <= 60.0:
    raise SystemExit("[FAIL] logger rate is not near 50 Hz")

tau_cols = [k for k in rows[0] if k.endswith("_tau_cmd")]
dq_cols = [k for k in rows[0] if k.endswith("_dq")]
power_cols = [k for k in rows[0] if k.endswith("_power_cmd")]
print("tau_columns:", len(tau_cols))
print("dq_columns:", len(dq_cols))
print("power_columns:", len(power_cols))
if (len(tau_cols), len(dq_cols), len(power_cols)) != (12, 12, 12):
    raise SystemExit("[FAIL] expected 12 torque, dq, and power columns")

by_context = {}
for row in valid:
    c = row.get("context", "unknown")
    q = ff(row.get("clearance_cmd"))
    if q is not None:
        by_context.setdefault(c, []).append(q)

print("contexts:", sorted(by_context))

required_contexts = {
    "flat",
    "upslope",
    "rough",
    "downslope",
    "goal_flat",
}

missing_contexts = sorted(
    required_contexts - set(by_context)
)

print("missing_contexts:", missing_contexts)

if missing_contexts:
    raise SystemExit(
        "[FAIL] incomplete course context coverage: "
        + ",".join(missing_contexts)
    )
for c in sorted(by_context):
    vals = by_context[c]
    f45 = sum(abs(x-0.045) <= 1e-6 for x in vals) / float(len(vals))
    f55 = sum(abs(x-0.055) <= 1e-6 for x in vals) / float(len(vals))
    print("context=%s n=%d frac_045=%.6f frac_055=%.6f" % (c, len(vals), f45, f55))

active = ("flat", "upslope", "rough", "downslope")
if condition == "A":
    vals = [x for c in active for x in by_context.get(c, [])]
    if not vals or sum(abs(x-0.045) <= 1e-6 for x in vals) / float(len(vals)) < 0.99:
        raise SystemExit("[FAIL] neutral clearance coverage below 99%")
elif condition == "B":
    rough = by_context.get("rough", [])
    other = [x for c in ("flat", "upslope", "downslope") for x in by_context.get(c, [])]
    if not rough or not other:
        raise SystemExit("[FAIL] missing rough or nonrough samples")
    rough_high = sum(abs(x-0.055) <= 1e-6 for x in rough) / float(len(rough))
    other_neutral = sum(abs(x-0.045) <= 1e-6 for x in other) / float(len(other))
    print("rough_high_fraction:", rough_high)
    print("nonrough_neutral_fraction:", other_neutral)
    if rough_high < 0.95 or other_neutral < 0.95:
        raise SystemExit("[FAIL] rough-only schedule coverage mismatch")
elif condition == "C":
    vals = [x for c in active for x in by_context.get(c, [])]
    if not vals or sum(abs(x-0.055) <= 1e-6 for x in vals) / float(len(vals)) < 0.99:
        raise SystemExit("[FAIL] fixed-high clearance coverage below 99%")
else:
    raise SystemExit("[FAIL] unknown condition")

metrics = {}
for name in (
    "power_abs_W_cmd", "power_positive_W_cmd", "work_abs_J_cmd",
    "work_positive_J_cmd", "tau_sq_integral", "dq_sq_integral"
):
    vals = [ff(r.get(name)) for r in valid]
    vals = [x for x in vals if x is not None]
    if not vals:
        raise SystemExit("[FAIL] no finite values for " + name)
    metrics[name] = vals

print("power_abs_median:", statistics.median(metrics["power_abs_W_cmd"]))
print("power_abs_max:", max(metrics["power_abs_W_cmd"]))
print("final_work_abs:", metrics["work_abs_J_cmd"][-1])
print("final_work_positive:", metrics["work_positive_J_cmd"][-1])
print("final_tau_sq_integral:", metrics["tau_sq_integral"][-1])
print("final_dq_sq_integral:", metrics["dq_sq_integral"][-1])
if max(metrics["power_abs_W_cmd"]) <= 1e-3 or metrics["work_abs_J_cmd"][-1] <= 1e-3:
    raise SystemExit("[FAIL] energy proxy remained zero")

for name in ("work_abs_J_cmd", "work_positive_J_cmd", "tau_sq_integral", "dq_sq_integral"):
    vals = metrics[name]
    decreases = sum(b + 1e-6 < a for a,b in zip(vals, vals[1:]))
    print(name + "_decrease_count:", decreases)
    if decreases:
        raise SystemExit("[FAIL] cumulative metric decreased: " + name)

joint_age = [ff(r.get("max_joint_age_s")) for r in valid]
joint_age = [x for x in joint_age if x is not None]
ref_age = [ff(r.get("ref_age_s")) for r in valid]
ref_age = [x for x in ref_age if x is not None]
print("max_joint_age:", max(joint_age))
print("max_ref_age:", max(ref_age))
print("[PASS] full-course energy log")
PY
}

echo "===== PREFLIGHT ====="
for path in \
  "$START_SCRIPT" \
  "$ENERGY_LOGGER" \
  "$ROOT/scripts/runtime/tracer_run_phase_d4_context_meta_repeat_v0.sh"
do
  test -s "$path" || { echo "[ERROR] missing: $path"; exit 1; }
done
bash -n "$START_SCRIPT"
python3 -m py_compile "$ENERGY_LOGGER"
docker inspect "$CTRL_CONTAINER" >/dev/null

grep -nA3 '"%.9f" % now' "$ENERGY_LOGGER" || {
  echo "[ERROR] timestamp precision patch missing from energy logger"
  exit 1
}

echo "ROOT=$ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "WORLD=$WORLD"
echo "TIMEOUT_S=$TIMEOUT_S"
echo "HOLD_OBS_S=$HOLD_OBS_S"
echo "START_INDEX=$START_INDEX"

schedule=(
  "1 A"
  "1 B"
  "1 C"
  "2 C"
  "2 A"
  "2 B"
  "3 B"
  "3 C"
  "3 A"
)

run_index=0
for item in "${schedule[@]}"; do
  read -r block condition <<< "$item"
  run_index=$((run_index + 1))

  if (( run_index < START_INDEX )); then
    echo "[TRACER] resume skip run=$run_index"
    continue
  fi

  name="$(condition_name "$condition")"
  table="$(condition_table "$condition")"
  console_log="$RUN_ROOT/run_${run_index}_block_${block}_${condition}_${name}.console.log"
  validation_log="$VALIDATION_DIR/run_${run_index}_${condition}_${name}.txt"

  open_gui=0
  if [[ "$run_index" -eq 1 ]]; then
    open_gui="$OPEN_FIRST_GUI"
  fi

  before_latest="$(latest_energy_dir 2>/dev/null || true)"
  start_time="$(date --iso-8601=seconds)"

  echo
  echo "============================================================"
  echo "RUN $run_index / 9  block=$block  condition=$condition ($name)"
  echo "============================================================"

  set +e
  (
    cd "$ROOT"
    TRACER_CTRL_CONTAINER="$CTRL_CONTAINER" \
    TRACER_A1_CONTAINER="$CTRL_CONTAINER" \
    TRACER_D4_START_SCRIPT="$START_SCRIPT" \
    TRACER_ENERGY_LOGGER_HOST="$ENERGY_LOGGER" \
    TRACER_D4_REPEAT_WORLD="$WORLD" \
    TRACER_D4_REPEAT_GOAL_X=8.0 \
    TRACER_D4_REPEAT_N=1 \
    TRACER_D4_REPEAT_TIMEOUT_S="$TIMEOUT_S" \
    TRACER_D4_HOLD_OBS_S="$HOLD_OBS_S" \
    TRACER_D4_REPEAT_SLEEP_S=2 \
    TRACER_D4_REPEAT_OPEN_GUI="$open_gui" \
    TRACER_PHASE_D4_CONTEXT_ACTION_TABLE="$table" \
    bash scripts/runtime/tracer_run_phase_d4_context_meta_repeat_v0.sh
  ) 2>&1 | tee "$console_log"
  rc="${PIPESTATUS[0]}"
  set -e

  kill_energy_logger
  sleep 1

  log_dir="$(latest_energy_dir)"
  if [[ -n "$before_latest" && "$log_dir" == "$before_latest" ]]; then
    echo "[ERROR] no new energy run directory detected"
    exit 3
  fi

  energy_csv="$log_dir/ros1_joint_energy_proxy.csv"
  test -s "$energy_csv" || { echo "[ERROR] missing energy CSV: $energy_csv"; exit 3; }

  set +e
  validate_energy_csv "$energy_csv" "$condition" "$validation_log"
  validation_rc=$?
  set -e
  cat "$validation_log"
  if [[ "$validation_rc" -ne 0 ]]; then
    echo "[ERROR] energy validation failed for run=$run_index"
    exit 4
  fi

  end_time="$(date --iso-8601=seconds)"
  cat > "$log_dir/revalidation_energy_condition.txt" <<META
run_index=$run_index
block=$block
condition=$condition
condition_name=$name
action_table=$table
world=$WORLD
timeout_s=$TIMEOUT_S
hold_observation_s=$HOLD_OBS_S
controller_container=$CTRL_CONTAINER
start_script=$START_SCRIPT
energy_logger=$ENERGY_LOGGER
run_rc=$rc
energy_validation_rc=$validation_rc
META

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_index" "$block" "$condition" "$name" "$rc" "$log_dir" \
    "$console_log" "$validation_log" "$start_time" "$end_time" >> "$MANIFEST"

  echo "[TRACER] completed run=$run_index rc=$rc"
  echo "[TRACER] log_dir=$log_dir"
  echo "[TRACER] energy_validation=PASS"
  sleep 2
done

kill_energy_logger
trap - EXIT

echo
echo "===== ENERGY A/B/C CAMPAIGN COMPLETE ====="
echo "RUN_ROOT=$RUN_ROOT"
echo "MANIFEST=$MANIFEST"
column -t -s $'\t' "$MANIFEST" || cat "$MANIFEST"

ARCHIVE="$ROOT/logs/$(basename "$RUN_ROOT")_full.tar.gz"
python3 - "$ROOT" "$RUN_ROOT" "$MANIFEST" "$ARCHIVE" <<'PY'
from pathlib import Path
import csv
import sys
import tarfile

root = Path(sys.argv[1]).resolve()
run_root = Path(sys.argv[2]).resolve()
manifest = Path(sys.argv[3]).resolve()
archive = Path(sys.argv[4]).resolve()

items = [(run_root, run_root.relative_to(root))]
with manifest.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        path = Path(row["log_dir"]).resolve()
        items.append((path, path.relative_to(root)))

with tarfile.open(str(archive), "w:gz") as tar:
    for path, arcname in items:
        tar.add(str(path), arcname=str(arcname))

print(archive)
PY

ls -lh "$ARCHIVE"
sha256sum "$ARCHIVE"
echo "archive=$ARCHIVE"
