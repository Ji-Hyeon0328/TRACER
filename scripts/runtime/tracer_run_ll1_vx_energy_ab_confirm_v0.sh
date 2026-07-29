#!/usr/bin/env bash
set -Eeo pipefail
set +u

# VX campaign DDS isolation.
# The workstation previously exported a CycloneDDS URI tied to a
# disconnected Ethernet interface. Force local DDS for this campaign.
unset CYCLONEDDS_URI
export ROS_LOCALHOST_ONLY="${TRACER_VX_ROS_LOCALHOST_ONLY:-1}"
export RMW_IMPLEMENTATION="${TRACER_VX_RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_ll1_docker}"

START_SCRIPT="${TRACER_VX_ABC_START_SCRIPT:-/tmp/tracer_start_phase_d4_context_meta_ll1_residual_energy_v3.sh}"

ENERGY_LOGGER="${TRACER_VX_ENERGY_LOGGER_HOST:-/tmp/tracer_ros1_joint_energy_vx_logger_v0.py}"

WORLD="${TRACER_VX_ABC_WORLD:-tracer_mixed_stress_course_v5_lowfric_from_solid}"

TIMEOUT_S="${TRACER_VX_ABC_TIMEOUT_S:-220}"
HOLD_OBS_S="${TRACER_VX_ABC_HOLD_OBS_S:-5}"
OPEN_FIRST_GUI="${TRACER_VX_ABC_OPEN_FIRST_GUI:-0}"

START_INDEX="${TRACER_VX_ABC_START_INDEX:-1}"
END_INDEX="${TRACER_VX_ABC_END_INDEX:-4}"

TS="$(date +%Y%m%d_%H%M%S)"

RUN_ROOT="${TRACER_VX_ABC_RUN_ROOT:-$ROOT/logs/revalidation_vx_energy_ab_confirm_${TS}}"

MANIFEST="$RUN_ROOT/manifest.tsv"
VALIDATION_DIR="$RUN_ROOT/energy_validation"

FIXED_LOW_TABLE='flat:0.090,0.320,0.045;start_flat:0.090,0.320,0.045;upslope:0.090,0.320,0.045;rough:0.090,0.320,0.045;downslope:0.090,0.320,0.045;goal_flat:0.090,0.320,0.045;unknown:0.090,0.320,0.045'

TERRAIN_SCHEDULED_TABLE='flat:0.115,0.320,0.045;start_flat:0.115,0.320,0.045;upslope:0.090,0.320,0.045;rough:0.090,0.320,0.045;downslope:0.090,0.320,0.045;goal_flat:0.090,0.320,0.045;unknown:0.090,0.320,0.045'

FIXED_HIGH_TABLE='flat:0.115,0.320,0.045;start_flat:0.115,0.320,0.045;upslope:0.115,0.320,0.045;rough:0.115,0.320,0.045;downslope:0.115,0.320,0.045;goal_flat:0.115,0.320,0.045;unknown:0.115,0.320,0.045'

mkdir -p \
  "$RUN_ROOT" \
  "$VALIDATION_DIR"

if [[ ! -s "$MANIFEST" ]]; then
  printf '%s\n' \
    $'run_index\tblock\tcondition\tcondition_name\trc\tlog_dir\tconsole_log\tenergy_validation\tstart_time\tend_time' \
    > "$MANIFEST"
fi


kill_energy_logger() {
  docker exec \
    "$CTRL_CONTAINER" \
    bash --noprofile --norc -lc '
      set +e

      pkill -INT -f \
        "[t]racer_ros1_joint_energy_proxy_logger_v0.py" \
        || true
    ' \
    >/dev/null 2>&1 \
    || true
}


trap kill_energy_logger EXIT


latest_energy_dir() {
  python3 - "$ROOT/logs" <<'PY_LATEST'
from pathlib import Path
import sys


root = Path(sys.argv[1])

items = [
    path.parent
    for path in root.glob(
        "phase_d4_context_meta_*/ros1_joint_energy_proxy.csv"
    )
    if path.is_file()
]

if not items:
    raise SystemExit(
        "no phase_d4 energy directory found"
    )

print(
    max(
        items,
        key=lambda directory: (
            directory
            / "ros1_joint_energy_proxy.csv"
        ).stat().st_mtime_ns,
    )
)
PY_LATEST
}


condition_name() {
  case "$1" in
    A)
      printf '%s' fixed_low
      ;;
    B)
      printf '%s' terrain_scheduled
      ;;
    C)
      printf '%s' fixed_high
      ;;
    *)
      return 2
      ;;
  esac
}


condition_table() {
  case "$1" in
    A)
      printf '%s' "$FIXED_LOW_TABLE"
      ;;
    B)
      printf '%s' "$TERRAIN_SCHEDULED_TABLE"
      ;;
    C)
      printf '%s' "$FIXED_HIGH_TABLE"
      ;;
    *)
      return 2
      ;;
  esac
}


validate_energy_csv() {
  local csv_path="$1"
  local condition="$2"
  local output_path="$3"

  python3 - \
    "$csv_path" \
    "$condition" \
    > "$output_path" \
    <<'PY_VALIDATE'
import csv
import math
import statistics
import sys


path, condition = (
    sys.argv[1],
    sys.argv[2],
)


def ff(value):
    try:
        value = float(value)

        return (
            value
            if math.isfinite(value)
            else None
        )

    except Exception:
        return None


with open(
    path,
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    rows = list(
        csv.DictReader(stream)
    )


print("path:", path)
print("condition:", condition)
print("rows:", len(rows))


if len(rows) < 500:
    raise SystemExit(
        "[FAIL] fewer than 500 samples"
    )


valid = [
    row
    for row in rows
    if row.get("valid") == "1"
]

valid_fraction = (
    len(valid)
    / float(len(rows))
)

print("valid_rows:", len(valid))
print("valid_fraction:", valid_fraction)


if valid_fraction < 0.95:
    raise SystemExit(
        "[FAIL] valid fraction below 0.95"
    )


times = [
    ff(row.get("sim_time"))
    for row in valid
]

times = [
    value
    for value in times
    if value is not None
]

positive_dt = [
    current - previous
    for previous, current in zip(
        times,
        times[1:],
    )
    if (
        0.0
        < current - previous
        <= 0.20
    )
]

backward = sum(
    current + 1e-12 < previous
    for previous, current in zip(
        times,
        times[1:],
    )
)


if (
    len(times) < 2
    or not positive_dt
):
    raise SystemExit(
        "[FAIL] insufficient finite "
        "simulation timestamps"
    )


span = max(times) - min(times)

hz = (
    1.0
    / statistics.median(
        positive_dt
    )
)

print("sim_span:", span)
print("effective_sim_hz:", hz)
print(
    "backward_timestamp_count:",
    backward,
)


if backward:
    raise SystemExit(
        "[FAIL] simulation time moved backward"
    )


if span < 30.0:
    raise SystemExit(
        "[FAIL] simulated duration below 30 s"
    )


if not 40.0 <= hz <= 60.0:
    raise SystemExit(
        "[FAIL] logger rate is not near 50 Hz"
    )


tau_columns = [
    key
    for key in rows[0]
    if key.endswith("_tau_cmd")
]

dq_columns = [
    key
    for key in rows[0]
    if key.endswith("_dq")
]

power_columns = [
    key
    for key in rows[0]
    if key.endswith("_power_cmd")
]


print(
    "tau_columns:",
    len(tau_columns),
)

print(
    "dq_columns:",
    len(dq_columns),
)

print(
    "power_columns:",
    len(power_columns),
)


if (
    len(tau_columns),
    len(dq_columns),
    len(power_columns),
) != (12, 12, 12):
    raise SystemExit(
        "[FAIL] expected 12 torque, "
        "dq, and power columns"
    )


required_state_fields = (
    "z",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "vx_world",
    "vy_world",
    "vx_body",
    "vy_body",
)


for field in required_state_fields:
    values = [
        ff(row.get(field))
        for row in valid
    ]

    finite_fraction = (
        sum(
            value is not None
            for value in values
        )
        / float(len(values))
    )

    print(
        field + "_finite_fraction:",
        finite_fraction,
    )

    if finite_fraction < 0.99:
        raise SystemExit(
            "[FAIL] insufficient finite "
            "state data for " + field
        )


model_age = [
    ff(row.get("model_age_s"))
    for row in valid
]

model_age = [
    value
    for value in model_age
    if value is not None
]

if not model_age:
    raise SystemExit(
        "[FAIL] no finite model age"
    )

print(
    "max_model_age:",
    max(model_age),
)


by_context = {}

for row in valid:
    context = row.get(
        "context",
        "unknown",
    )

    vx = ff(
        row.get("vx_cmd")
    )

    if vx is not None:
        by_context.setdefault(
            context,
            [],
        ).append(vx)


print(
    "contexts:",
    sorted(by_context),
)


required_contexts = {
    "flat",
    "upslope",
    "rough",
    "downslope",
    "goal_flat",
}


missing_contexts = sorted(
    required_contexts
    - set(by_context)
)

print(
    "missing_contexts:",
    missing_contexts,
)


if missing_contexts:
    raise SystemExit(
        "[FAIL] incomplete course context "
        "coverage: "
        + ",".join(missing_contexts)
    )


for context in sorted(by_context):
    values = by_context[context]

    fraction_090 = (
        sum(
            abs(value - 0.090)
            <= 1e-6
            for value in values
        )
        / float(len(values))
    )

    fraction_115 = (
        sum(
            abs(value - 0.115)
            <= 1e-6
            for value in values
        )
        / float(len(values))
    )

    fraction_hold = (
        sum(
            abs(value - 0.025)
            <= 1e-6
            for value in values
        )
        / float(len(values))
    )

    print(
        (
            "context=%s n=%d "
            "frac_090=%.6f "
            "frac_115=%.6f "
            "frac_hold025=%.6f"
        )
        % (
            context,
            len(values),
            fraction_090,
            fraction_115,
            fraction_hold,
        )
    )


active = (
    "flat",
    "upslope",
    "rough",
    "downslope",
)


if condition == "A":
    values = [
        value
        for context in active
        for value in by_context.get(
            context,
            [],
        )
    ]

    fraction = (
        sum(
            abs(value - 0.090)
            <= 1e-6
            for value in values
        )
        / float(len(values))
        if values
        else 0.0
    )

    print(
        "fixed_low_fraction:",
        fraction,
    )

    if fraction < 0.99:
        raise SystemExit(
            "[FAIL] fixed-low VX "
            "coverage below 99%"
        )

elif condition == "B":
    flat = by_context.get(
        "flat",
        [],
    )

    terrain = [
        value
        for context in (
            "upslope",
            "rough",
            "downslope",
        )
        for value in by_context.get(
            context,
            [],
        )
    ]

    if (
        not flat
        or not terrain
    ):
        raise SystemExit(
            "[FAIL] missing flat or "
            "difficult-terrain samples"
        )

    flat_high = (
        sum(
            abs(value - 0.115)
            <= 1e-6
            for value in flat
        )
        / float(len(flat))
    )

    terrain_low = (
        sum(
            abs(value - 0.090)
            <= 1e-6
            for value in terrain
        )
        / float(len(terrain))
    )

    print(
        "flat_high_fraction:",
        flat_high,
    )

    print(
        "terrain_low_fraction:",
        terrain_low,
    )

    if (
        flat_high < 0.95
        or terrain_low < 0.95
    ):
        raise SystemExit(
            "[FAIL] terrain-scheduled "
            "VX coverage mismatch"
        )

elif condition == "C":
    values = [
        value
        for context in active
        for value in by_context.get(
            context,
            [],
        )
    ]

    fraction = (
        sum(
            abs(value - 0.115)
            <= 1e-6
            for value in values
        )
        / float(len(values))
        if values
        else 0.0
    )

    print(
        "fixed_high_fraction:",
        fraction,
    )

    if fraction < 0.99:
        raise SystemExit(
            "[FAIL] fixed-high VX "
            "coverage below 99%"
        )

else:
    raise SystemExit(
        "[FAIL] unknown condition"
    )


active_rows = [
    row
    for row in valid
    if row.get("context")
    in active
]


for field, target in (
    (
        "body_height_cmd",
        0.320,
    ),
    (
        "clearance_cmd",
        0.045,
    ),
):
    values = [
        ff(row.get(field))
        for row in active_rows
    ]

    values = [
        value
        for value in values
        if value is not None
    ]

    fraction = (
        sum(
            abs(value - target)
            <= 1e-6
            for value in values
        )
        / float(len(values))
        if values
        else 0.0
    )

    print(
        field + "_fixed_fraction:",
        fraction,
    )

    if fraction < 0.99:
        raise SystemExit(
            "[FAIL] fixed parameter "
            "coverage mismatch: " + field
        )


realized_vx = [
    ff(row.get("vx_body"))
    for row in active_rows
]

realized_vx = [
    value
    for value in realized_vx
    if value is not None
]

if (
    not realized_vx
    or max(
        abs(value)
        for value in realized_vx
    ) <= 1e-3
):
    raise SystemExit(
        "[FAIL] realized body VX "
        "remained zero"
    )


print(
    "median_realized_vx_body:",
    statistics.median(
        realized_vx
    ),
)


metrics = {}

for name in (
    "power_abs_W_cmd",
    "power_positive_W_cmd",
    "work_abs_J_cmd",
    "work_positive_J_cmd",
    "tau_sq_integral",
    "dq_sq_integral",
):
    values = [
        ff(row.get(name))
        for row in valid
    ]

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        raise SystemExit(
            "[FAIL] no finite values for "
            + name
        )

    metrics[name] = values


print(
    "power_abs_median:",
    statistics.median(
        metrics["power_abs_W_cmd"]
    ),
)

print(
    "power_abs_max:",
    max(
        metrics["power_abs_W_cmd"]
    ),
)

print(
    "final_work_abs:",
    metrics["work_abs_J_cmd"][-1],
)

print(
    "final_work_positive:",
    metrics[
        "work_positive_J_cmd"
    ][-1],
)

print(
    "final_tau_sq_integral:",
    metrics[
        "tau_sq_integral"
    ][-1],
)

print(
    "final_dq_sq_integral:",
    metrics[
        "dq_sq_integral"
    ][-1],
)


if (
    max(
        metrics["power_abs_W_cmd"]
    ) <= 1e-3
    or metrics[
        "work_abs_J_cmd"
    ][-1] <= 1e-3
):
    raise SystemExit(
        "[FAIL] energy proxy remained zero"
    )


for name in (
    "work_abs_J_cmd",
    "work_positive_J_cmd",
    "tau_sq_integral",
    "dq_sq_integral",
):
    values = metrics[name]

    decreases = sum(
        current + 1e-6 < previous
        for previous, current in zip(
            values,
            values[1:],
        )
    )

    print(
        name + "_decrease_count:",
        decreases,
    )

    if decreases:
        raise SystemExit(
            "[FAIL] cumulative metric "
            "decreased: " + name
        )


joint_age = [
    ff(row.get("max_joint_age_s"))
    for row in valid
]

joint_age = [
    value
    for value in joint_age
    if value is not None
]

ref_age = [
    ff(row.get("ref_age_s"))
    for row in valid
]

ref_age = [
    value
    for value in ref_age
    if value is not None
]


if not joint_age or not ref_age:
    raise SystemExit(
        "[FAIL] missing age diagnostics"
    )


print(
    "max_joint_age:",
    max(joint_age),
)

print(
    "max_ref_age:",
    max(ref_age),
)

print(
    "[PASS] full-course VX energy log"
)
PY_VALIDATE
}


echo "===== PREFLIGHT ====="

for path in \
  "$START_SCRIPT" \
  "$ENERGY_LOGGER" \
  "$ROOT/scripts/runtime/tracer_run_phase_d4_context_meta_repeat_v0.sh"
do
  test -s "$path" || {
    echo "[ERROR] missing: $path"
    exit 1
  }
done

bash -n "$START_SCRIPT"

python3 -m py_compile \
  "$ENERGY_LOGGER"

docker inspect \
  "$CTRL_CONTAINER" \
  >/dev/null

grep -nA3 \
  '"%.9f" % now' \
  "$ENERGY_LOGGER" \
  || {
    echo "[ERROR] timestamp precision missing"
    exit 1
  }

echo "ROOT=$ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "WORLD=$WORLD"
echo "TIMEOUT_S=$TIMEOUT_S"
echo "HOLD_OBS_S=$HOLD_OBS_S"
echo "START_INDEX=$START_INDEX"
echo "END_INDEX=$END_INDEX"

schedule=(
  "4 A"
  "4 B"
  "5 B"
  "5 A"
)

run_index=0

for item in "${schedule[@]}"; do
  read -r block condition <<< "$item"

  run_index=$((run_index + 1))

  if (( run_index < START_INDEX )); then
    echo "[TRACER] resume skip run=$run_index"
    continue
  fi

  if (( run_index > END_INDEX )); then
    echo "[TRACER] end-index stop before run=$run_index"
    break
  fi

  name="$(condition_name "$condition")"
  table="$(condition_table "$condition")"

  console_log="$RUN_ROOT/run_${run_index}_block_${block}_${condition}_${name}.console.log"

  validation_log="$VALIDATION_DIR/run_${run_index}_${condition}_${name}.txt"

  open_gui=0

  if [[ "$run_index" -eq 1 ]]; then
    open_gui="$OPEN_FIRST_GUI"
  fi

  before_latest="$(
    latest_energy_dir \
      2>/dev/null \
      || true
  )"

  start_time="$(
    date --iso-8601=seconds
  )"

  echo
  echo "============================================================"
  echo "RUN $run_index / 4 block=$block condition=$condition ($name)"
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
    bash \
      scripts/runtime/tracer_run_phase_d4_context_meta_repeat_v0.sh
  ) 2>&1 \
    | tee "$console_log"

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

  test -s "$energy_csv" || {
    echo "[ERROR] missing energy CSV: $energy_csv"
    exit 3
  }

  set +e

  validate_energy_csv \
    "$energy_csv" \
    "$condition" \
    "$validation_log"

  validation_rc=$?

  set -e

  cat "$validation_log"

  if [[ "$rc" -ne 0 ]]; then
    echo "[ERROR] rollout returned rc=$rc for run=$run_index"
    exit 4
  fi

  if [[ "$validation_rc" -ne 0 ]]; then
    echo "[ERROR] energy validation failed for run=$run_index"
    exit 5
  fi

  end_time="$(
    date --iso-8601=seconds
  )"

  cat > \
    "$log_dir/revalidation_vx_energy_condition.txt" \
    <<META
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
routing_mode=direct_phase_d4_selector_to_mpc_reference
META

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_index" \
    "$block" \
    "$condition" \
    "$name" \
    "$rc" \
    "$log_dir" \
    "$console_log" \
    "$validation_log" \
    "$start_time" \
    "$end_time" \
    >> "$MANIFEST"

  echo "[TRACER] completed run=$run_index rc=$rc"
  echo "[TRACER] log_dir=$log_dir"
  echo "[TRACER] energy_validation=PASS"

  sleep 2
done

kill_energy_logger
trap - EXIT

echo
echo "===== VX ENERGY A/B CONFIRMATORY EXTENSION COMPLETE ====="
echo "RUN_ROOT=$RUN_ROOT"
echo "MANIFEST=$MANIFEST"

column \
  -t \
  -s $'\t' \
  "$MANIFEST" \
  || cat "$MANIFEST"

ARCHIVE="$ROOT/logs/$(basename "$RUN_ROOT")_full.tar.gz"

python3 - \
  "$ROOT" \
  "$RUN_ROOT" \
  "$MANIFEST" \
  "$ARCHIVE" \
  <<'PY_ARCHIVE'
from pathlib import Path
import csv
import sys
import tarfile


root = Path(sys.argv[1]).resolve()
run_root = Path(sys.argv[2]).resolve()
manifest = Path(sys.argv[3]).resolve()
archive = Path(sys.argv[4]).resolve()

items = [
    (
        run_root,
        run_root.relative_to(root),
    )
]

with manifest.open(
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    for row in csv.DictReader(
        stream,
        delimiter="\t",
    ):
        path = Path(
            row["log_dir"]
        ).resolve()

        items.append(
            (
                path,
                path.relative_to(root),
            )
        )

with tarfile.open(
    str(archive),
    "w:gz",
) as tar:
    for path, archive_name in items:
        tar.add(
            str(path),
            arcname=str(
                archive_name
            ),
        )
PY_ARCHIVE

sha256sum "$ARCHIVE"

echo "[PASS] A/B confirmatory extension runs complete"
