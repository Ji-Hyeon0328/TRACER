#!/usr/bin/env bash
set -Eeo pipefail

LL1_CONTAINER="a1_cpp_ctrl_ll1_docker"
GAZEBO_CONTAINER="a1_unitree_gazebo_docker"

RUN_TAG="tracer_ll1_realized_authority_ab_$(date +%Y%m%d_%H%M%S)"
HOST_OUT="/tmp/${RUN_TAG}"
REMOTE_OUT="/tmp/${RUN_TAG}"

mkdir -p "$HOST_OUT"

docker exec \
  "$LL1_CONTAINER" \
  mkdir -p "$REMOTE_OUT"

echo "host output:      $HOST_OUT"
echo "container output: $REMOTE_OUT"

ROOT="$(
  git rev-parse --show-toplevel
)"

EXPECTED_ROOT="$HOME/Tracer/TRACER-lowcontroller-audit"

ROOT_REAL="$(
  realpath -e "$ROOT"
)"

EXPECTED_ROOT_REAL="$(
  realpath -e "$EXPECTED_ROOT"
)"

BRANCH="$(
  git branch --show-current
)"

echo "worktree path:      $ROOT"
echo "worktree canonical: $ROOT_REAL"
echo "expected canonical: $EXPECTED_ROOT_REAL"
echo "branch:             $BRANCH"

if [[ "$ROOT_REAL" != "$EXPECTED_ROOT_REAL" ]]; then
  echo "[ERROR] wrong worktree"
  exit 1
fi

if [[ "$BRANCH" != "TRACER-lowcontroller" ]]; then
  echo "[ERROR] wrong branch: $BRANCH"
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "[ERROR] audit worktree is not clean"
  git status --short
  exit 3
fi


stop_ll1() {
  docker exec -i \
    "$LL1_CONTAINER" \
    bash -s <<'EOS'
set +e

source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash

rosservice call \
  /gazebo/pause_physics \
  >/dev/null 2>&1 \
  || true

pattern="/usr/bin/python /opt/ros/melodic/bin/roslaunch a1_cpp a1_ctrl.launch type:=gazebo solver_type:=mpc"

mapfile -t launchers < <(
  pgrep -f -x "$pattern" || true
)

for pid in "${launchers[@]}"; do
  kill -INT "$pid" \
    >/dev/null 2>&1 \
    || true
done

for _ in $(seq 1 40); do
  if ! pgrep -x gazebo_a1_ctrl >/dev/null; then
    break
  fi

  sleep 0.25
done

rosparam set \
  /tracer_enable_swing_apex_residual \
  false

rosparam set \
  /tracer_clearance_cmd_neutral \
  0.045

rosparam set \
  /tracer_swing_apex_delta_min \
  -0.010

rosparam set \
  /tracer_swing_apex_delta_max \
  0.020

if pgrep -x gazebo_a1_ctrl >/dev/null; then
  echo "[ERROR] gazebo_a1_ctrl remains alive"
  pgrep -ax gazebo_a1_ctrl
  exit 1
fi
EOS
}


reset_standing_pose() {
  echo
  echo "===== CANONICAL STANDING RESET ====="

  docker exec -i \
    "$GAZEBO_CONTAINER" \
    bash -s <<'EOS'
set -Eeo pipefail

source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash

cleanup() {
  set +e

  rosservice call \
    /gazebo/pause_physics \
    >/dev/null 2>&1 \
    || true

  pkill -INT -x unitree_servo \
    >/dev/null 2>&1 \
    || true

  pkill -INT -x unitree_move_kinetic \
    >/dev/null 2>&1 \
    || true
}

trap cleanup EXIT

pkill -9 -x unitree_servo \
  >/dev/null 2>&1 \
  || true

pkill -9 -x unitree_move_kinetic \
  >/dev/null 2>&1 \
  || true

rosservice call \
  /gazebo/pause_physics \
  >/dev/null

python - <<'PY'
from __future__ import print_function

import time

import rospy

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


rospy.init_node(
    "tracer_ll1_realized_authority_reset",
    anonymous=True,
    disable_signals=True,
)

rospy.wait_for_service(
    "/gazebo/set_model_state",
    timeout=5.0,
)

set_model_state = rospy.ServiceProxy(
    "/gazebo/set_model_state",
    SetModelState,
)

state = ModelState()
state.model_name = "a1_gazebo"
state.reference_frame = "world"

state.pose.position.x = 0.0
state.pose.position.y = 0.0
state.pose.position.z = 0.325

state.pose.orientation.x = 0.0
state.pose.orientation.y = 0.0
state.pose.orientation.z = 0.0
state.pose.orientation.w = 1.0

state.twist.linear.x = 0.0
state.twist.linear.y = 0.0
state.twist.linear.z = 0.0

state.twist.angular.x = 0.0
state.twist.angular.y = 0.0
state.twist.angular.z = 0.0

success_count = 0

for attempt in range(5):
    response = set_model_state(state)

    print(
        "set_model_state {}/5: success={}".format(
            attempt + 1,
            response.success,
        )
    )

    if response.success:
        success_count += 1

    time.sleep(0.1)

if success_count != 5:
    raise RuntimeError(
        "only {}/5 reset calls succeeded".format(
            success_count
        )
    )
PY

rosparam set /robot_name a1

rosservice call \
  /gazebo/unpause_physics \
  >/dev/null

rosrun \
  unitree_controller \
  unitree_servo \
  > /tmp/tracer_realized_ab_servo.log \
  2>&1 &

servo_pid=$!

sleep 3

kill -INT "$servo_pid" \
  2>/dev/null \
  || true

wait "$servo_pid" \
  2>/dev/null \
  || true

rosrun \
  unitree_controller \
  unitree_move_kinetic \
  > /tmp/tracer_realized_ab_kinetic.log \
  2>&1 &

kinetic_pid=$!

sleep 3

rosservice call \
  /gazebo/pause_physics \
  >/dev/null

kill -INT "$kinetic_pid" \
  2>/dev/null \
  || true

wait "$kinetic_pid" \
  2>/dev/null \
  || true

trap - EXIT

echo "[OK] canonical standing pose restored"
echo "[OK] physics paused"
EOS
}


start_ll1() {
  local trial_name="$1"
  local runtime_log="/tmp/${RUN_TAG}_${trial_name}.log"

  docker exec -i \
    "$LL1_CONTAINER" \
    bash -s <<'EOS'
set -eo pipefail

source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash

rosservice call \
  /gazebo/pause_physics \
  >/dev/null

if pgrep -x gazebo_a1_ctrl >/dev/null; then
  echo "[ERROR] controller already running"
  pgrep -ax gazebo_a1_ctrl
  exit 1
fi

rosparam set \
  /tracer_enable_swing_apex_residual \
  true

rosparam set \
  /tracer_clearance_cmd_neutral \
  0.045

rosparam set \
  /tracer_swing_apex_delta_min \
  -0.010

rosparam set \
  /tracer_swing_apex_delta_max \
  0.020
EOS

  docker exec -d \
    -e TRACER_RUNTIME_LOG="$runtime_log" \
    "$LL1_CONTAINER" \
    bash -lc '
      source /opt/ros/melodic/setup.bash
      source /root/unitree_ws/devel/setup.bash
      source /root/A1_ctrl_ws/devel/setup.bash

      exec roslaunch \
        a1_cpp \
        a1_ctrl.launch \
        type:=gazebo \
        solver_type:=mpc \
        > "$TRACER_RUNTIME_LOG" \
        2>&1
    '

  local controller_count=0

  for _ in $(seq 1 60); do
    controller_count="$(
      docker exec \
        "$LL1_CONTAINER" \
        bash -lc '
          pgrep -cx gazebo_a1_ctrl || true
        ' \
        | tail -1
    )"

    if [[ "$controller_count" -eq 1 ]]; then
      break
    fi

    sleep 0.25
  done

  if [[ "$controller_count" -ne 1 ]]; then
    echo "[ERROR] controller failed to start"

    docker exec \
      "$LL1_CONTAINER" \
      tail -120 "$runtime_log" \
      || true

    exit 4
  fi

  echo "[OK] LL1 controller started"
}


run_trial() {
  local trial_name="$1"
  local command="$2"
  local expected_delta="$3"

  local remote_csv="${REMOTE_OUT}/${trial_name}.csv"
  local host_csv="${HOST_OUT}/${trial_name}.csv"

  echo
  echo "============================================================"
  echo "TRIAL: $trial_name"
  echo "command: $command"
  echo "expected delta: $expected_delta"
  echo "============================================================"

  reset_standing_pose
  start_ll1 "$trial_name"

  docker exec -i \
    -e TRACER_COMMAND="$command" \
    -e TRACER_EXPECTED_DELTA="$expected_delta" \
    -e TRACER_OUTPUT_CSV="$remote_csv" \
    "$LL1_CONTAINER" \
    bash -s <<'EOS'
set -eo pipefail

source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash

python - <<'PY'
from __future__ import print_function

import csv
import math
import os
import time

import rosgraph
import rospy

from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty


COMMAND_TOPIC = "/tracer/mpc_reference"
DEBUG_TOPIC = "/tracer/lowlevel/swing_apex_debug"
MODEL_NAME = "a1_gazebo"

COMMAND = float(
    os.environ["TRACER_COMMAND"]
)

EXPECTED_DELTA = float(
    os.environ["TRACER_EXPECTED_DELTA"]
)

OUTPUT_CSV = os.environ[
    "TRACER_OUTPUT_CSV"
]

NEUTRAL = 0.045
PUBLISH_PERIOD = 0.02
PRELOAD_DURATION = 0.5
WARMUP_DURATION = 3.0
COLLECT_DURATION = 8.0

LEGS = (
    "FL",
    "FR",
    "RL",
    "RR",
)


def quaternion_to_euler(q):
    sinr_cosp = 2.0 * (
        q.w * q.x + q.y * q.z
    )

    cosr_cosp = 1.0 - 2.0 * (
        q.x * q.x + q.y * q.y
    )

    roll = math.atan2(
        sinr_cosp,
        cosr_cosp,
    )

    sinp = 2.0 * (
        q.w * q.y - q.z * q.x
    )

    if abs(sinp) >= 1.0:
        pitch = math.copysign(
            math.pi / 2.0,
            sinp,
        )
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (
        q.w * q.z + q.x * q.y
    )

    cosy_cosp = 1.0 - 2.0 * (
        q.y * q.y + q.z * q.z
    )

    yaw = math.atan2(
        siny_cosp,
        cosy_cosp,
    )

    return roll, pitch, yaw


rospy.init_node(
    "tracer_ll1_realized_authority_collector",
    anonymous=True,
    disable_signals=True,
)

master = rosgraph.Master(
    rospy.get_name()
)

publishers, _, _ = master.getSystemState()

existing_publishers = []

for topic, nodes in publishers:
    if topic == COMMAND_TOPIC:
        existing_publishers.extend(nodes)

if existing_publishers:
    raise RuntimeError(
        "existing command publishers: {}".format(
            existing_publishers
        )
    )

latest_model = [None]
collect_enabled = [False]
collection_wall_start = [0.0]

rows = []

packet_count = [0]
saw_swing = [False]
saw_stance = [False]

max_shape = [0.0]
max_bump = [0.0]
max_active_error = [0.0]
max_delta_error = [0.0]
max_relation_error = [0.0]
max_stance_bump = [0.0]


def model_callback(message):
    latest_model[0] = message


def current_base_state():
    models = latest_model[0]

    if (
        models is None
        or MODEL_NAME not in models.name
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    index = models.name.index(
        MODEL_NAME
    )

    pose = models.pose[index]
    twist = models.twist[index]

    roll, pitch, yaw = quaternion_to_euler(
        pose.orientation
    )

    return (
        pose.position.z,
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
        twist.linear.z,
    )


def debug_callback(message):
    if not collect_enabled[0]:
        return

    data = list(message.data)

    if len(data) != 28:
        return

    if abs(data[1] - COMMAND) > 1e-9:
        return

    packet_count[0] += 1

    max_active_error[0] = max(
        max_active_error[0],
        abs(data[0] - 1.0),
    )

    max_delta_error[0] = max(
        max_delta_error[0],
        abs(data[3] - EXPECTED_DELTA),
    )

    (
        base_z,
        roll_deg,
        pitch_deg,
        yaw_deg,
        base_vz,
    ) = current_base_state()

    wall_elapsed = (
        time.time()
        - collection_wall_start[0]
    )

    sim_time = rospy.Time.now().to_sec()

    for leg_index, leg_name in enumerate(
        LEGS
    ):
        offset = 4 + 6 * leg_index

        phase = data[offset + 0]
        swing = int(
            round(data[offset + 1])
        )
        shape = data[offset + 2]
        bump = data[offset + 3]
        target_z = data[offset + 4]
        actual_z = data[offset + 5]

        expected_bump = (
            EXPECTED_DELTA * shape
            if swing == 1
            else 0.0
        )

        max_relation_error[0] = max(
            max_relation_error[0],
            abs(bump - expected_bump),
        )

        if swing == 1:
            saw_swing[0] = True

            max_shape[0] = max(
                max_shape[0],
                shape,
            )

            max_bump[0] = max(
                max_bump[0],
                bump,
            )
        else:
            saw_stance[0] = True

            max_stance_bump[0] = max(
                max_stance_bump[0],
                abs(bump),
            )

        rows.append(
            (
                packet_count[0],
                sim_time,
                wall_elapsed,
                COMMAND,
                EXPECTED_DELTA,
                leg_name,
                leg_index,
                phase,
                swing,
                shape,
                bump,
                target_z,
                actual_z,
                actual_z - target_z,
                base_z,
                roll_deg,
                pitch_deg,
                yaw_deg,
                base_vz,
            )
        )


rospy.Subscriber(
    "/gazebo/model_states",
    ModelStates,
    model_callback,
    queue_size=1,
)

rospy.Subscriber(
    DEBUG_TOPIC,
    Float64MultiArray,
    debug_callback,
    queue_size=100,
)

publisher = rospy.Publisher(
    COMMAND_TOPIC,
    Float64MultiArray,
    queue_size=1,
)

rospy.wait_for_service(
    "/gazebo/unpause_physics",
    timeout=5.0,
)

rospy.wait_for_service(
    "/gazebo/pause_physics",
    timeout=5.0,
)

unpause = rospy.ServiceProxy(
    "/gazebo/unpause_physics",
    Empty,
)

pause = rospy.ServiceProxy(
    "/gazebo/pause_physics",
    Empty,
)

deadline = time.time() + 5.0

while (
    publisher.get_num_connections() < 1
    and time.time() < deadline
):
    time.sleep(0.05)

if publisher.get_num_connections() != 1:
    raise RuntimeError(
        "expected one command subscriber"
    )

sequence = 0


def publish_command():
    global sequence

    message = Float64MultiArray()

    message.data = [
        float(sequence),
        0.0,
        0.0,
        0.32,
        COMMAND,
        1.0,
    ]

    publisher.publish(message)
    sequence += 1


print(
    "[INFO] preload command {:.3f}".format(
        COMMAND
    )
)

preload_start = time.time()

while (
    time.time() - preload_start
    < PRELOAD_DURATION
):
    publish_command()
    time.sleep(PUBLISH_PERIOD)

unpause()

diverged = False

try:
    print(
        "[INFO] warmup {:.1f}s".format(
            WARMUP_DURATION
        )
    )

    warmup_start = time.time()

    while (
        time.time() - warmup_start
        < WARMUP_DURATION
    ):
        publish_command()

        models = latest_model[0]

        if (
            models is not None
            and MODEL_NAME in models.name
        ):
            index = models.name.index(
                MODEL_NAME
            )

            pose = models.pose[index]
            twist = models.twist[index]

            angular_norm = math.sqrt(
                twist.angular.x ** 2
                + twist.angular.y ** 2
                + twist.angular.z ** 2
            )

            if (
                pose.position.z < 0.10
                or abs(twist.linear.z) > 10.0
                or angular_norm > 20.0
            ):
                diverged = True
                break

        time.sleep(PUBLISH_PERIOD)

    if diverged:
        raise RuntimeError(
            "rollout diverged during warmup"
        )

    print(
        "[INFO] collect {:.1f}s".format(
            COLLECT_DURATION
        )
    )

    collection_wall_start[0] = time.time()
    collect_enabled[0] = True

    next_report = 1.0

    while (
        time.time() - collection_wall_start[0]
        < COLLECT_DURATION
    ):
        publish_command()

        elapsed = (
            time.time()
            - collection_wall_start[0]
        )

        if elapsed >= next_report:
            (
                base_z,
                roll_deg,
                pitch_deg,
                yaw_deg,
                base_vz,
            ) = current_base_state()

            print(
                "t={:.1f}s "
                "z={:+.4f} "
                "rpy=({:+.2f},{:+.2f},{:+.2f})deg "
                "vz={:+.4f}".format(
                    elapsed,
                    base_z,
                    roll_deg,
                    pitch_deg,
                    yaw_deg,
                    base_vz,
                )
            )

            next_report += 1.0

        time.sleep(PUBLISH_PERIOD)

finally:
    collect_enabled[0] = False
    pause()
    publisher.unregister()

    print("[OK] physics paused")

with open(OUTPUT_CSV, "wb") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(
        (
            "packet_index",
            "sim_time",
            "wall_elapsed",
            "command",
            "expected_delta",
            "leg",
            "leg_index",
            "phase",
            "swing",
            "shape",
            "bump",
            "target_z",
            "actual_z",
            "tracking_error",
            "base_z",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "base_vz",
        )
    )

    writer.writerows(rows)

print()
print("===== TRIAL METRICS =====")
print("output csv          = {}".format(OUTPUT_CSV))
print("debug packets       = {}".format(packet_count[0]))
print("rows                = {}".format(len(rows)))
print("saw swing           = {}".format(saw_swing[0]))
print("saw stance          = {}".format(saw_stance[0]))
print("max shape           = {:.9f}".format(max_shape[0]))
print("max bump            = {:.9f}".format(max_bump[0]))
print(
    "max active error    = {:.12f}".format(
        max_active_error[0]
    )
)
print(
    "max delta error     = {:.12f}".format(
        max_delta_error[0]
    )
)
print(
    "max relation error  = {:.12f}".format(
        max_relation_error[0]
    )
)
print(
    "max stance bump     = {:.12f}".format(
        max_stance_bump[0]
    )
)

if packet_count[0] < 70:
    raise RuntimeError(
        "insufficient debug coverage"
    )

if not saw_swing[0]:
    raise RuntimeError(
        "no swing samples"
    )

if not saw_stance[0]:
    raise RuntimeError(
        "no stance samples"
    )

if max_shape[0] < 0.90:
    raise RuntimeError(
        "swing apex was not sampled"
    )

if max_active_error[0] > 1e-9:
    raise RuntimeError(
        "active flag mismatch"
    )

if max_delta_error[0] > 1e-9:
    raise RuntimeError(
        "delta mismatch"
    )

if max_relation_error[0] > 1e-9:
    raise RuntimeError(
        "bump relation mismatch"
    )

if max_stance_bump[0] > 1e-9:
    raise RuntimeError(
        "stance bump was nonzero"
    )

if abs(max_bump[0] - EXPECTED_DELTA) > 1e-6:
    raise RuntimeError(
        "expected max bump {:+.6f}, "
        "observed {:+.6f}".format(
            EXPECTED_DELTA,
            max_bump[0],
        )
    )

print(
    "[OK] command {:.3f} collection passed".format(
        COMMAND
    )
)
PY
EOS

  docker cp \
    "${LL1_CONTAINER}:${remote_csv}" \
    "$host_csv"

  echo "[OK] copied: $host_csv"

  stop_ll1
}


cleanup_host() {
  set +e
  stop_ll1 >/dev/null 2>&1 || true
}

trap cleanup_host EXIT

stop_ll1

run_trial \
  "neutral_045" \
  "0.045" \
  "0.000"

run_trial \
  "high_060" \
  "0.060" \
  "0.015"


echo
echo "============================================================"
echo "MATCHED-PHASE ANALYSIS"
echo "============================================================"

HOST_OUT="$HOST_OUT" \
python3 - <<'PY'
import csv
import math
import os
import statistics
from collections import defaultdict


OUT_DIR = os.environ["HOST_OUT"]

NEUTRAL_PATH = os.path.join(
    OUT_DIR,
    "neutral_045.csv",
)

HIGH_PATH = os.path.join(
    OUT_DIR,
    "high_060.csv",
)

BIN_COUNT = 16
MIN_CELL_SAMPLES = 2

LEGS = (
    "FL",
    "FR",
    "RL",
    "RR",
)


def load(path):
    rows = []

    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            parsed = {
                "leg": row["leg"],
                "phase": float(row["phase"]),
                "swing": int(row["swing"]),
                "shape": float(row["shape"]),
                "target_z": float(row["target_z"]),
                "actual_z": float(row["actual_z"]),
                "tracking_error": float(
                    row["tracking_error"]
                ),
                "base_z": float(row["base_z"]),
                "pitch_deg": float(row["pitch_deg"]),
            }

            if (
                parsed["swing"] == 1
                and parsed["base_z"] > 0.20
                and math.isfinite(parsed["pitch_deg"])
                and abs(parsed["pitch_deg"]) <= 20.0
            ):
                rows.append(parsed)

    return rows


def build_cells(rows):
    cells = defaultdict(list)

    for row in rows:
        phase = min(
            0.999999,
            max(0.0, row["phase"]),
        )

        bin_index = min(
            BIN_COUNT - 1,
            int(phase * BIN_COUNT),
        )

        cells[
            (
                row["leg"],
                bin_index,
            )
        ].append(row)

    result = {}

    for key, samples in cells.items():
        if len(samples) < MIN_CELL_SAMPLES:
            continue

        result[key] = {
            "count": len(samples),
            "phase": statistics.mean(
                sample["phase"]
                for sample in samples
            ),
            "shape": statistics.mean(
                sample["shape"]
                for sample in samples
            ),
            "target_z": statistics.mean(
                sample["target_z"]
                for sample in samples
            ),
            "actual_z": statistics.mean(
                sample["actual_z"]
                for sample in samples
            ),
            "tracking_error": statistics.mean(
                sample["tracking_error"]
                for sample in samples
            ),
        }

    return result


neutral_rows = load(NEUTRAL_PATH)
high_rows = load(HIGH_PATH)

neutral_cells = build_cells(neutral_rows)
high_cells = build_cells(high_rows)

matched = []

for key in sorted(
    set(neutral_cells)
    & set(high_cells)
):
    leg, bin_index = key

    phase_center = (
        bin_index + 0.5
    ) / BIN_COUNT

    if not 0.375 <= phase_center <= 0.625:
        continue

    neutral = neutral_cells[key]
    high = high_cells[key]

    target_delta = (
        high["target_z"]
        - neutral["target_z"]
    )

    actual_delta = (
        high["actual_z"]
        - neutral["actual_z"]
    )

    tracking_error_change = (
        high["tracking_error"]
        - neutral["tracking_error"]
    )

    matched.append(
        {
            "leg": leg,
            "bin": bin_index,
            "phase": (
                neutral["phase"]
                + high["phase"]
            ) / 2.0,
            "shape": (
                neutral["shape"]
                + high["shape"]
            ) / 2.0,
            "target_delta": target_delta,
            "actual_delta": actual_delta,
            "tracking_error_change":
                tracking_error_change,
        }
    )

if len(matched) < 8:
    raise RuntimeError(
        "insufficient matched apex cells: {}".format(
            len(matched)
        )
    )

summary_lines = []

summary_lines.append(
    "matched apex cells = {}".format(
        len(matched)
    )
)

print(
    "matched apex cells = {}".format(
        len(matched)
    )
)

all_target_deltas = []
all_actual_deltas = []
all_error_changes = []

for leg in LEGS:
    leg_cells = [
        cell
        for cell in matched
        if cell["leg"] == leg
    ]

    if not leg_cells:
        print("{}: no matched apex cells".format(leg))
        continue

    target_delta = statistics.median(
        cell["target_delta"]
        for cell in leg_cells
    )

    actual_delta = statistics.median(
        cell["actual_delta"]
        for cell in leg_cells
    )

    error_change = statistics.median(
        cell["tracking_error_change"]
        for cell in leg_cells
    )

    all_target_deltas.extend(
        cell["target_delta"]
        for cell in leg_cells
    )

    all_actual_deltas.extend(
        cell["actual_delta"]
        for cell in leg_cells
    )

    all_error_changes.extend(
        cell["tracking_error_change"]
        for cell in leg_cells
    )

    line = (
        "{}: target Δ={:+.2f} mm, "
        "actual Δ={:+.2f} mm, "
        "tracking-error change={:+.2f} mm"
    ).format(
        leg,
        target_delta * 1000.0,
        actual_delta * 1000.0,
        error_change * 1000.0,
    )

    print(line)
    summary_lines.append(line)

median_target = statistics.median(
    all_target_deltas
)

median_actual = statistics.median(
    all_actual_deltas
)

median_error_change = statistics.median(
    all_error_changes
)

positive_actual_fraction = sum(
    delta > 0.0
    for delta in all_actual_deltas
) / float(len(all_actual_deltas))

tracking_gain = (
    median_actual / median_target
    if abs(median_target) > 1e-9
    else float("nan")
)

print()
print(
    "combined median target Δ       = {:+.3f} mm".format(
        median_target * 1000.0
    )
)

print(
    "combined median actual Δ       = {:+.3f} mm".format(
        median_actual * 1000.0
    )
)

print(
    "combined tracking-error change = {:+.3f} mm".format(
        median_error_change * 1000.0
    )
)

print(
    "positive actual cells          = {:.1%}".format(
        positive_actual_fraction
    )
)

print(
    "median realized/target ratio   = {:.3f}".format(
        tracking_gain
    )
)

summary_lines.extend(
    (
        "",
        "combined median target delta mm = "
        "{:+.6f}".format(
            median_target * 1000.0
        ),
        "combined median actual delta mm = "
        "{:+.6f}".format(
            median_actual * 1000.0
        ),
        "combined tracking-error change mm = "
        "{:+.6f}".format(
            median_error_change * 1000.0
        ),
        "positive actual cells = "
        "{:.6f}".format(
            positive_actual_fraction
        ),
        "median realized/target ratio = "
        "{:.6f}".format(
            tracking_gain
        ),
    )
)

print()

if (
    median_actual > 0.0
    and positive_actual_fraction >= 0.75
):
    conclusion = (
        "[EVIDENCE] high-clearance command produced "
        "a positive realized foot-height shift across "
        "most matched apex cells"
    )
else:
    conclusion = (
        "[INCONCLUSIVE] target authority is confirmed, "
        "but realized actual-z separation is not yet "
        "consistent across matched apex cells"
    )

print(conclusion)
summary_lines.append("")
summary_lines.append(conclusion)

summary_path = os.path.join(
    OUT_DIR,
    "matched_phase_summary.txt",
)

with open(summary_path, "w") as output:
    output.write(
        "\n".join(summary_lines)
        + "\n"
    )

print()
print("summary: {}".format(summary_path))
PY


stop_ll1
trap - EXIT

echo
echo "===== FINAL SAFE STATE ====="

docker exec -i \
  "$LL1_CONTAINER" \
  bash -s <<'EOS'
source /opt/ros/melodic/setup.bash

echo "controller:"
pgrep -ax gazebo_a1_ctrl || true

echo
echo "parameters:"
rosparam get /tracer_enable_swing_apex_residual
rosparam get /tracer_clearance_cmd_neutral
rosparam get /tracer_swing_apex_delta_min
rosparam get /tracer_swing_apex_delta_max

echo
echo "joint command authority:"
rostopic info \
  /a1_gazebo/FL_hip_controller/command \
  || true

rosservice call \
  /gazebo/pause_physics \
  >/dev/null

echo
echo "[OK] controller stopped"
echo "[OK] feature restored to false"
echo "[OK] physics paused"
EOS

echo
echo "===== OUTPUT FILES ====="

find \
  "$HOST_OUT" \
  -maxdepth 1 \
  -type f \
  -printf "%f\n" \
  | sort

echo
echo "[OK] realized-authority A/B sweep completed"
echo "results: $HOST_OUT"
