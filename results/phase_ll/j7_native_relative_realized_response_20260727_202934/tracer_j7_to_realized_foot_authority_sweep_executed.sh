#!/usr/bin/env bash
set -Eeo pipefail

ROOT="$HOME/Tracer/TRACER"
AUDIT_ROOT="$HOME/Tracer/TRACER-lowcontroller-audit"

LL1_CONTAINER="a1_cpp_ctrl_ll1_docker"
GAZEBO_CONTAINER="a1_unitree_gazebo_docker"

ROS2_SENDER="$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_ros2_mpc_ref_udp_sender.py"
ROS1_RECEIVER="/root/A1_ctrl_ws/src/a1_cpp/scripts/tracer_udp_to_ros1_mpc_ref.py"

J7_SOURCE="$ROOT/scripts/runtime/tracer_phase_j7_guarded_meta_action_ref_gate_node_v0.py"
J7_EXPECTED_SHA="2b2692af30e164db53d3c125323ff27d42849f7b6a9fc28485f4742b4580811a"
J7_RUNTIME_COPY=""
J7_WRAPPER="/tmp/tracer_j7_to_ll1_active_source.py"

SENDER_PID=""
PUBLISHER_PID=""
CLEANUP_DONE=false

RUN_TAG="tracer_j7_to_realized_foot_authority_$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="/tmp/$RUN_TAG"

mkdir -p "$RESULT_DIR"

echo "result dir: $RESULT_DIR"


source_ros2() {
  set +u

  unset COLCON_CURRENT_PREFIX
  unset AMENT_PREFIX_PATH
  unset CMAKE_PREFIX_PATH

  source /opt/ros/humble/setup.bash

  set -u
}


stop_j7_source() {
  set +e

  if [[ -n "$PUBLISHER_PID" ]] &&
     kill -0 "$PUBLISHER_PID" 2>/dev/null
  then
    kill -INT "$PUBLISHER_PID" \
      >/dev/null 2>&1 \
      || true

    for _ in $(seq 1 40); do
      if ! kill -0 "$PUBLISHER_PID" 2>/dev/null; then
        break
      fi

      sleep 0.1
    done

    if kill -0 "$PUBLISHER_PID" 2>/dev/null; then
      kill -9 "$PUBLISHER_PID" \
        >/dev/null 2>&1 \
        || true
    fi

    wait "$PUBLISHER_PID" \
      >/dev/null 2>&1 \
      || true
  fi

  PUBLISHER_PID=""

  pkill -9 -f \
    "[t]racer_j7_to_ll1_active_source.py" \
    >/dev/null 2>&1 \
    || true

  set -e
}

stop_controller() {
  docker exec \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      set +e

      source /opt/ros/melodic/setup.bash

      rosservice call \
        /gazebo/pause_physics \
        >/dev/null 2>&1 \
        || true

      launcher_pattern="/usr/bin/python /opt/ros/melodic/bin/roslaunch a1_cpp a1_ctrl.launch type:=gazebo solver_type:=mpc"

      mapfile -t launchers < <(
        pgrep -f -x "$launcher_pattern" \
          || true
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

      pkill -INT -x gazebo_a1_ctrl \
        >/dev/null 2>&1 \
        || true

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
    '
}


stop_bridge() {
  set +e

  if [[ -n "$SENDER_PID" ]] &&
     kill -0 "$SENDER_PID" 2>/dev/null
  then
    kill -INT "$SENDER_PID" \
      >/dev/null 2>&1 \
      || true

    for _ in $(seq 1 30); do
      if ! kill -0 "$SENDER_PID" 2>/dev/null; then
        break
      fi

      sleep 0.1
    done

    kill -9 "$SENDER_PID" \
      >/dev/null 2>&1 \
      || true
  fi

  SENDER_PID=""

  pkill -9 -f \
    "[t]racer_ros2_mpc_ref_udp_sender.py" \
    >/dev/null 2>&1 \
    || true

  docker exec \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      receiver_pattern="python /root/A1_ctrl_ws/src/a1_cpp/scripts/tracer_udp_to_ros1_mpc_ref.py"

      mapfile -t receiver_pids < <(
        pgrep -f -x "$receiver_pattern" \
          || true
      )

      for pid in "${receiver_pids[@]}"; do
        kill -9 "$pid" \
          >/dev/null 2>&1 \
          || true
      done
    ' \
    >/dev/null 2>&1 \
    || true

  set -e
}


cleanup() {
  if [[ "$CLEANUP_DONE" == true ]]; then
    return
  fi

  CLEANUP_DONE=true
  set +e

  echo
  echo "===== CLEANUP ====="

  stop_j7_source
  stop_controller
  stop_bridge

  docker exec \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      source /opt/ros/melodic/setup.bash

      rosservice call \
        /gazebo/pause_physics \
        >/dev/null 2>&1 \
        || true
    ' \
    >/dev/null 2>&1 \
    || true

  echo "[OK] J7 active source stopped"
  echo "[OK] LL1 controller stopped"
  echo "[OK] ROS2 sender stopped"
  echo "[OK] ROS1 receiver stopped"
  echo "[OK] feature restored to false"
  echo "[OK] physics paused"
}

trap cleanup EXIT


echo "===== PREFLIGHT ====="

echo "runtime root=$(
  realpath -e "$ROOT"
)"

echo "audit root=$(
  realpath -e "$AUDIT_ROOT"
)"

echo "runtime branch=$(
  git -C "$ROOT" branch --show-current
)"

echo "audit branch=$(
  git -C "$AUDIT_ROOT" branch --show-current
)"

if [[ "$(
  git -C "$AUDIT_ROOT" branch --show-current
)" != "TRACER-lowcontroller" ]]
then
  echo "[ERROR] wrong audit branch"
  exit 1
fi

if [[ -n "$(
  git -C "$AUDIT_ROOT" status --porcelain
)" ]]
then
  echo "[ERROR] audit worktree is not clean"
  git -C "$AUDIT_ROOT" status --short
  exit 2
fi

if [[ ! -f "$ROS2_SENDER" ]]; then
  echo "[ERROR] missing ROS2 sender: $ROS2_SENDER"
  exit 3
fi

stop_j7_source
stop_controller
stop_bridge

docker exec \
  "$LL1_CONTAINER" \
  bash --noprofile --norc -lc '
    set -eo pipefail

    source /opt/ros/melodic/setup.bash

    rosservice call \
      /gazebo/pause_physics \
      >/dev/null

    if pgrep -x gazebo_a1_ctrl >/dev/null; then
      echo "[ERROR] controller remains running"
      exit 1
    fi

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
  '


echo
echo "===== INSTALL J7 ACTIVE SOURCE ====="

test -f "$J7_SOURCE" || {
  echo "[ERROR] missing J7 source: $J7_SOURCE"
  exit 3
}

J7_ACTUAL_SHA="$(
  sha256sum "$J7_SOURCE" \
    | awk '{print $1}'
)"

echo "J7 source=$J7_SOURCE"
echo "J7 expected sha=$J7_EXPECTED_SHA"
echo "J7 actual sha=$J7_ACTUAL_SHA"

if [[ "$J7_ACTUAL_SHA" != "$J7_EXPECTED_SHA" ]]; then
  echo "[ERROR] J7 source changed since the completed audit"
  exit 4
fi

J7_RUNTIME_COPY="$RESULT_DIR/tracer_phase_j7_guarded_meta_action_ref_gate_node_v0.py"

cp \
  "$J7_SOURCE" \
  "$J7_RUNTIME_COPY"

chmod +x \
  "$J7_RUNTIME_COPY"

sha256sum \
  "$J7_RUNTIME_COPY" \
  > "$RESULT_DIR/j7_source.sha256"

source_ros2

if ros2 node list 2>/dev/null \
  | grep -qx \
      "/tracer_phase_j7_guarded_meta_action_ref_gate_node_v0"
then
  echo "[ERROR] another J7 node is already running"
  exit 5
fi

existing_publishers="$(
  ros2 topic info \
    /tracer/mpc_reference \
    2>/dev/null \
    | awk '
        /Publisher count:/ {
          print $3
        }
      ' \
    || true
)"

if [[ -n "$existing_publishers" &&
      "$existing_publishers" != "0" ]]
then
  echo "[ERROR] /tracer/mpc_reference already has a publisher"

  ros2 topic info \
    /tracer/mpc_reference \
    --verbose \
    || true

  exit 6
fi

cat > "$J7_WRAPPER" <<'PY'
#!/usr/bin/env python3

import importlib.util
import os

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def array_message(values):
    message = Float64MultiArray()
    message.data = [float(value) for value in values]
    return message


class FixedJ7Inputs(Node):
    def __init__(self):
        super().__init__("tracer_j7_to_ll1_fixed_inputs")

        self.projected_clearance = float(
            os.environ["TRACER_E2E_PROJECTED_CLEARANCE"]
        )

        self.sequence = float(
            os.environ["TRACER_E2E_COUNTER_BASE"]
        )

        self.empirical_pub = self.create_publisher(
            Float64MultiArray,
            "/tracer/empirical_mpc_reference",
            10,
        )

        self.projected_pub = self.create_publisher(
            Float64MultiArray,
            "/tracer/meta_action_projected_ref_shadow",
            10,
        )

        self.theta_pub = self.create_publisher(
            Float64MultiArray,
            "/tracer/meta_action_theta_shadow",
            10,
        )

        self.context_pub = self.create_publisher(
            String,
            "/tracer/terrain_context_label",
            10,
        )

        self.odom_pub = self.create_publisher(
            Float64MultiArray,
            "/tracer/robot_odom_flat",
            10,
        )

        # Inputs run faster than J7 so every decision sees a fresh,
        # sequence-aligned empirical/projected pair.
        self.timer = self.create_timer(
            0.025,
            self.publish_inputs,
        )

    def publish_inputs(self):
        sequence = self.sequence
        self.sequence += 1.0

        empirical = [
            sequence,
            0.0,
            0.0,
            0.320,
            0.045,
            1.0,
        ]

        projected = [
            sequence,
            0.0,
            0.0,
            0.320,
            self.projected_clearance,
            1.0,
        ]

        # theta[0] = action ID
        # theta[8] = hold override
        theta = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        context = String()
        context.data = "flat"

        self.empirical_pub.publish(
            array_message(empirical)
        )

        self.projected_pub.publish(
            array_message(projected)
        )

        self.theta_pub.publish(
            array_message(theta)
        )

        self.context_pub.publish(context)

        self.odom_pub.publish(
            array_message([0.0, 0.0])
        )


def load_j7_module(path):
    specification = importlib.util.spec_from_file_location(
        "tracer_j7_runtime_copy",
        path,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "could not load J7 source: {}".format(path)
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(module)

    return module


def main():
    os.environ[
        "TRACER_PHASE_J7_OUTPUT_REF_TOPIC"
    ] = "/tracer/mpc_reference"

    os.environ[
        "TRACER_PHASE_J7_REQUIRE_SEQ_MATCH"
    ] = "1"

    os.environ[
        "TRACER_PHASE_J7_ACTIVE_FAILSAFE_HOLD"
    ] = "1"

    os.environ[
        "TRACER_PHASE_J7_MAX_INPUT_AGE_S"
    ] = "1.0"

    os.environ[
        "TRACER_PHASE_J7_MAX_ABS_DELTA_VX"
    ] = "0.030"

    os.environ[
        "TRACER_PHASE_J7_MAX_ABS_DELTA_YAW"
    ] = "0.020"

    os.environ[
        "TRACER_PHASE_J7_MAX_ABS_DELTA_BODY_H"
    ] = "0.006"

    os.environ[
        "TRACER_PHASE_J7_MAX_ABS_DELTA_CLEARANCE"
    ] = "0.006"

    os.environ[
        "TRACER_PHASE_J7_ALLOWED_CONTEXTS"
    ] = "flat"

    os.environ[
        "TRACER_PHASE_J7_PUB_HZ"
    ] = "20.0"

    os.environ[
        "TRACER_PHASE_J7_LOG_DIR"
    ] = os.environ["TRACER_E2E_J7_LOG_DIR"]

    module = load_j7_module(
        os.environ["TRACER_E2E_J7_SOURCE_COPY"]
    )

    rclpy.init()

    input_node = FixedJ7Inputs()
    j7_node = module.J7GuardedMetaActionRefGate()

    executor = SingleThreadedExecutor()
    executor.add_node(input_node)
    executor.add_node(j7_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(input_node)
        executor.remove_node(j7_node)

        input_node.destroy_node()
        j7_node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
PY

chmod +x \
  "$J7_WRAPPER"

echo "[OK] audited J7 copied into result directory"
echo "[OK] J7 active-source wrapper installed"

echo
echo "===== START BRIDGE ====="

# Reuse the bridge launcher that already passed the independent
# ROS2 -> UDP -> ROS1 identity test.
TRACER_A1_CONTAINER="$LL1_CONTAINER" \
  "$ROOT/scripts/runtime/tracer_ensure_mpc_ref_bridge.sh"

receiver_count=0
receiver_socket_count=0
receiver_ready=false

# Do not stop waiting merely because the asynchronous child is
# absent during an early poll.
for _ in $(seq 1 60); do
  receiver_count="$(
    docker exec \
      "$LL1_CONTAINER" \
      bash --noprofile --norc -lc '
        pgrep -af \
          "[t]racer_udp_to_ros1_mpc_ref.py" \
          | grep -Ec \
              "^[0-9]+[[:space:]]+python(3)?[[:space:]]+/root/A1_ctrl_ws/src/a1_cpp/scripts/tracer_udp_to_ros1_mpc_ref.py$" \
          || true
      ' \
      | tail -1
  )"

  receiver_socket_count="$(
    ss -lunH 2>/dev/null \
      | awk '
          $5 ~ /:50110$/ ||
          $4 ~ /:50110$/ {
            count += 1
          }

          END {
            print count + 0
          }
        '
  )"

  if [[ "$receiver_count" -eq 1 &&
        "$receiver_socket_count" -eq 1 ]]
  then
    receiver_ready=true
    break
  fi

  sleep 0.25
done

echo "ROS1 receiver count=$receiver_count"
echo "UDP 50110 socket count=$receiver_socket_count"

if [[ "$receiver_ready" != true ]]; then
  echo "[ERROR] ROS1 receiver did not become ready"

  echo
  echo "===== ALL RECEIVER-RELATED PROCESSES ====="

  docker exec \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      needle="tracer_udp_to_ros1_""mpc_ref.py"

      ps -eo pid=,ppid=,stat=,args= \
        | awk -v needle="$needle" '"'"'
            index($0, needle) {
              print
            }
          '"'"'
    '

  echo
  echo "===== HOST UDP 50110 ====="

  ss -lunap 2>/dev/null \
    | grep ":50110" \
    || true

  echo
  echo "===== CONTAINER UDP 50110 ====="

  docker exec \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      ss -lunap 2>/dev/null \
        | grep ":50110" \
        || true
    '

  echo
  echo "===== RECEIVER LOG ====="

  docker exec \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      cat \
        /tmp/tracer_udp_to_ros1_mpc_ref.log \
        2>/dev/null \
        || true
    '

  exit 5
fi

echo "[OK] ROS1 receiver process and UDP socket are ready"

source_ros2

/usr/bin/python3 \
  "$ROS2_SENDER" \
  > /tmp/tracer_ll1_e2e_ros2_sender.log \
  2>&1 &

SENDER_PID=$!

sleep 1

if ! kill -0 "$SENDER_PID" 2>/dev/null; then
  echo "[ERROR] ROS2 sender failed to start"
  cat /tmp/tracer_ll1_e2e_ros2_sender.log
  exit 6
fi

echo "ROS2 sender pid=$SENDER_PID"

ros2 topic info \
  /tracer/mpc_reference \
  --verbose \
  || true


reset_standing_pose() {
  echo
  echo "===== CANONICAL STANDING RESET ====="

  docker exec \
    "$GAZEBO_CONTAINER" \
    bash --noprofile --norc -lc '
      set -Eeo pipefail

      source /opt/ros/melodic/setup.bash
      source /root/unitree_ws/devel/setup.bash

      cleanup_reset() {
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

      trap cleanup_reset EXIT

      pkill -9 -x unitree_servo \
        >/dev/null 2>&1 \
        || true

      pkill -9 -x unitree_move_kinetic \
        >/dev/null 2>&1 \
        || true

      rosservice call \
        /gazebo/pause_physics \
        >/dev/null

      python - <<'"'"'PY'"'"'
from __future__ import print_function

import time

import rospy

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


rospy.init_node(
    "tracer_ll1_e2e_reset",
    anonymous=True,
    disable_signals=True,
)

rospy.wait_for_service(
    "/gazebo/set_model_state",
    timeout=5.0,
)

set_state = rospy.ServiceProxy(
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

successes = 0

for attempt in range(5):
    response = set_state(state)

    print(
        "set_model_state {}/5: success={}".format(
            attempt + 1,
            response.success,
        )
    )

    if response.success:
        successes += 1

    time.sleep(0.1)

if successes != 5:
    raise RuntimeError(
        "only {}/5 resets succeeded".format(
            successes
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
        > /tmp/tracer_ll1_e2e_servo.log \
        2>&1 &

      servo_pid=$!

      sleep 3

      kill -INT "$servo_pid" \
        >/dev/null 2>&1 \
        || true

      wait "$servo_pid" \
        >/dev/null 2>&1 \
        || true

      rosrun \
        unitree_controller \
        unitree_move_kinetic \
        > /tmp/tracer_ll1_e2e_kinetic.log \
        2>&1 &

      kinetic_pid=$!

      sleep 3

      rosservice call \
        /gazebo/pause_physics \
        >/dev/null

      kill -INT "$kinetic_pid" \
        >/dev/null 2>&1 \
        || true

      wait "$kinetic_pid" \
        >/dev/null 2>&1 \
        || true

      trap - EXIT

      echo "[OK] standing pose restored"
      echo "[OK] physics paused"
    '
}


start_controller() {
  local trial="$1"
  local runtime_log="/tmp/tracer_ll1_e2e_${trial}_controller.log"

  docker exec \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      set -eo pipefail

      source /opt/ros/melodic/setup.bash
      source /root/unitree_ws/devel/setup.bash
      source /root/A1_ctrl_ws/devel/setup.bash

      rosservice call \
        /gazebo/pause_physics \
        >/dev/null

      if pgrep -x gazebo_a1_ctrl >/dev/null; then
        echo "[ERROR] controller already running"
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
    '

  docker exec -d \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc "
      source /opt/ros/melodic/setup.bash
      source /root/unitree_ws/devel/setup.bash
      source /root/A1_ctrl_ws/devel/setup.bash

      exec roslaunch \
        a1_cpp \
        a1_ctrl.launch \
        type:=gazebo \
        solver_type:=mpc \
        > $runtime_log \
        2>&1
    "

  local controller_count=0

  for _ in $(seq 1 60); do
    controller_count="$(
      docker exec \
        "$LL1_CONTAINER" \
        bash --noprofile --norc -lc '
          pgrep -cx gazebo_a1_ctrl \
            || true
        ' \
        | tail -1
    )"

    if [[ "$controller_count" -eq 1 ]]; then
      break
    fi

    sleep 0.25
  done

  echo "controller count=$controller_count"

  if [[ "$controller_count" -ne 1 ]]; then
    echo "[ERROR] controller failed to start"

    docker exec \
      "$LL1_CONTAINER" \
      tail -120 "$runtime_log" \
      || true

    exit 7
  fi
}


start_j7_source() {
  local projected_clearance="$1"
  local counter_base="$2"
  local process_log="$3"
  local j7_log_dir="$4"

  stop_j7_source
  source_ros2

  mkdir -p \
    "$(dirname "$process_log")" \
    "$j7_log_dir"

  TRACER_E2E_PROJECTED_CLEARANCE="$projected_clearance" \
  TRACER_E2E_COUNTER_BASE="$counter_base" \
  TRACER_E2E_J7_LOG_DIR="$j7_log_dir" \
  TRACER_E2E_J7_SOURCE_COPY="$J7_RUNTIME_COPY" \
    /usr/bin/python3 \
      "$J7_WRAPPER" \
      > "$process_log" \
      2>&1 &

  PUBLISHER_PID=$!

  source_ready=false
  publisher_count=0
  j7_node_count=0

  for _ in $(seq 1 60); do
    if ! kill -0 "$PUBLISHER_PID" 2>/dev/null; then
      echo "[ERROR] J7 active source exited during startup"
      cat "$process_log" || true
      exit 8
    fi

    publisher_count="$(
      ros2 topic info \
        /tracer/mpc_reference \
        2>/dev/null \
        | awk '
            /Publisher count:/ {
              print $3
            }
          ' \
        || true
    )"

    j7_node_count="$(
      ros2 node list \
        2>/dev/null \
        | grep -cx \
            "/tracer_phase_j7_guarded_meta_action_ref_gate_node_v0" \
        || true
    )"

    if [[ "$publisher_count" == "1" &&
          "$j7_node_count" == "1" ]]
    then
      source_ready=true
      break
    fi

    sleep 0.1
  done

  echo "ROS2 /tracer/mpc_reference publishers=$publisher_count"
  echo "J7 node count=$j7_node_count"

  if [[ "$source_ready" != true ]]; then
    echo "[ERROR] J7 active source did not become ready"
    cat "$process_log" || true

    ros2 topic info \
      /tracer/mpc_reference \
      --verbose \
      || true

    exit 9
  fi

  echo "[OK] J7 is the sole ROS2 MPC-reference publisher"

  # Allow subscriptions and sequence-aligned inputs to stabilize.
  sleep 1.0
}


validate_j7_decision() {
  local csv_path="$1"
  local expected_accept="$2"
  local expected_reason="$3"
  local expected_source="$4"
  local projected_clearance="$5"
  local expected_output_clearance="$6"
  local expected_j7_delta="$7"

  test -f "$csv_path" || {
    echo "[ERROR] missing J7 decision CSV: $csv_path"
    exit 10
  }

  /usr/bin/python3 - \
    "$csv_path" \
    "$expected_accept" \
    "$expected_reason" \
    "$expected_source" \
    "$projected_clearance" \
    "$expected_output_clearance" \
    "$expected_j7_delta" \
    <<'PY'
import csv
import math
import sys


(
    csv_path,
    expected_accept,
    expected_reason,
    expected_source,
    projected_clearance,
    expected_output_clearance,
    expected_j7_delta,
) = sys.argv[1:]

projected_clearance = float(
    projected_clearance
)

expected_output_clearance = float(
    expected_output_clearance
)

expected_j7_delta = float(
    expected_j7_delta
)

with open(csv_path, newline="") as stream:
    rows = list(csv.DictReader(stream))

if len(rows) < 10:
    raise RuntimeError(
        "insufficient J7 decision coverage: {}".format(
            len(rows)
        )
    )

stable_rows = rows[-10:]

for index, row in enumerate(stable_rows):
    checks = {
        "context": (
            row["context"] == "flat"
        ),
        "empirical_clearance": math.isclose(
            float(row["emp_clearance"]),
            0.045,
            abs_tol=1e-9,
        ),
        "projected_clearance": math.isclose(
            float(row["proj_clearance"]),
            projected_clearance,
            abs_tol=1e-9,
        ),
        "output_clearance": math.isclose(
            float(row["out_clearance"]),
            expected_output_clearance,
            abs_tol=1e-9,
        ),
        "delta_clearance": math.isclose(
            float(row["delta_clearance"]),
            expected_j7_delta,
            abs_tol=1e-9,
        ),
        "accept": (
            row["j7_accept"] == expected_accept
        ),
        "reason": (
            row["j7_reason"] == expected_reason
        ),
        "source": (
            row["j7_source"] == expected_source
        ),
        "sequence_match": (
            int(row["emp_seq"])
            == int(row["proj_seq"])
            == int(row["out_seq"])
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "J7 stable row {} failed {}: {}".format(
                index,
                failed,
                row,
            )
        )

print(
    "[OK] stable J7 decision rows={}".format(
        len(stable_rows)
    )
)

print(
    "[OK] j7_accept={}".format(
        expected_accept
    )
)

print(
    "[OK] j7_reason={}".format(
        expected_reason
    )
)

print(
    "[OK] j7_source={}".format(
        expected_source
    )
)

print(
    "[OK] projected clearance={:.3f}".format(
        projected_clearance
    )
)

print(
    "[OK] output clearance={:.3f}".format(
        expected_output_clearance
    )
)

print(
    "[OK] J7 candidate delta={:+.3f}".format(
        expected_j7_delta
    )
)
PY
}

run_trial() {
  local trial="$1"
  local projected_clearance="$2"
  local clearance="$3"
  local expected_delta="$4"
  local expected_j7_accept="$5"
  local expected_j7_reason="$6"
  local expected_j7_source="$7"
  local expected_j7_delta="$8"
  local counter_base="$9"

  local trial_dir="$RESULT_DIR/$trial"
  local j7_log_dir="$trial_dir/j7_log"
  local j7_csv="$j7_log_dir/guarded_meta_action_ref_gate_j7_v0.csv"
  local realized_csv="$trial_dir/${trial}_realized.csv"
  local container_realized_csv="/tmp/tracer_j7_${trial}_realized.csv"

  echo
  echo "============================================================"
  echo "TRIAL: $trial"
  echo "projected clearance=$projected_clearance"
  echo "expected J7 output clearance=$clearance"
  echo "expected LL1 delta=$expected_delta"
  echo "expected J7 reason=$expected_j7_reason"
  echo "============================================================"

  stop_j7_source
  stop_controller

  reset_standing_pose
  start_controller "$trial"

  start_j7_source \
    "$projected_clearance" \
    "$counter_base" \
    "$trial_dir/j7_active_source.log" \
    "$j7_log_dir"

  docker exec \
    "$LL1_CONTAINER" \
    rm -f \
    "$container_realized_csv"

  docker exec \
    -e TRACER_TRIAL="$trial" \
    -e TRACER_CLEARANCE="$clearance" \
    -e TRACER_EXPECTED_DELTA="$expected_delta" \
    -e TRACER_REALIZED_CSV="$container_realized_csv" \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      set -eo pipefail

      source /opt/ros/melodic/setup.bash
      source /root/unitree_ws/devel/setup.bash
      source /root/A1_ctrl_ws/devel/setup.bash

      python - <<'"'"'PY'"'"'
from __future__ import print_function

import csv
import sys
import math
import os
import time

import rospy

from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty


TRIAL = os.environ["TRACER_TRIAL"]

CLEARANCE = float(
    os.environ["TRACER_CLEARANCE"]
)

EXPECTED_DELTA = float(
    os.environ["TRACER_EXPECTED_DELTA"]
)

CSV_PATH = os.environ["TRACER_REALIZED_CSV"]
LEG_NAMES = ["FL", "FR", "RL", "RR"]

MODEL_NAME = "a1_gazebo"

latest_ref = [None]
latest_ref_time = [None]
latest_debug = [None]
latest_model = [None]


def on_ref(message):
    latest_ref[0] = list(message.data)
    latest_ref_time[0] = time.time()


def on_debug(message):
    latest_debug[0] = list(message.data)


def on_model(message):
    latest_model[0] = message


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
    "tracer_ll1_e2e_authority_observer",
    anonymous=True,
    disable_signals=True,
)

rospy.Subscriber(
    "/tracer/mpc_reference",
    Float64MultiArray,
    on_ref,
    queue_size=100,
)

rospy.Subscriber(
    "/tracer/lowlevel/swing_apex_debug",
    Float64MultiArray,
    on_debug,
    queue_size=100,
)

rospy.Subscriber(
    "/gazebo/model_states",
    ModelStates,
    on_model,
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

reference_deadline = time.time() + 6.0
reference_preloaded = False

while time.time() < reference_deadline:
    reference = latest_ref[0]

    reference_ok = (
        reference is not None
        and len(reference) == 6
        and abs(reference[1]) <= 1e-12
        and abs(reference[2]) <= 1e-12
        and abs(reference[3] - 0.32) <= 1e-12
        and abs(reference[4] - CLEARANCE) <= 1e-12
        and abs(reference[5] - 1.0) <= 1e-12
    )

    if reference_ok:
        reference_preloaded = True
        break

    time.sleep(0.02)

if not reference_preloaded:
    print(
        "[DEBUG] latest ROS1 reference={}".format(
            latest_ref[0]
        )
    )

    raise RuntimeError(
        "ROS2-to-UDP-to-ROS1 reference preload "
        "did not converge"
    )

print(
    "[OK] preloaded ROS1 reference {}".format(
        latest_ref[0]
    )
)

# The LL1 debug topic is generated from the Gazebo/controller
# update loop. It cannot converge while Gazebo physics is paused.
unpause()

print(
    "[OK] physics unpaused for LL1 debug convergence"
)

debug_deadline = time.time() + 4.0
debug_converged = False

while time.time() < debug_deadline:
    debug = latest_debug[0]

    debug_ok = (
        debug is not None
        and len(debug) == 28
        and abs(debug[0] - 1.0) <= 1e-12
        and abs(debug[1] - CLEARANCE) <= 1e-12
        and abs(debug[2] - 0.045) <= 1e-12
        and abs(debug[3] - EXPECTED_DELTA) <= 1e-12
    )

    if debug_ok:
        debug_converged = True
        break

    time.sleep(0.02)

if not debug_converged:
    pause()

    print(
        "[DEBUG] latest ROS1 reference={}".format(
            latest_ref[0]
        )
    )

    print(
        "[DEBUG] latest LL1 debug={}".format(
            latest_debug[0]
        )
    )

    raise RuntimeError(
        "LL1 debug did not converge after "
        "physics unpause"
    )

print(
    "[OK] preloaded LL1 header "
    "active={:.1f} raw={:.3f} neutral={:.3f} "
    "delta={:+.3f}".format(
        latest_debug[0][0],
        latest_debug[0][1],
        latest_debug[0][2],
        latest_debug[0][3],
    )
)

start = time.time()
duration = 8.0
next_report = 1.0

debug_packets = 0
reference_packets = 0

saw_swing = False
saw_stance = False

max_shape = 0.0
max_bump = 0.0
max_abs_active_error = 0.0
max_abs_raw_error = 0.0
max_abs_delta_error = 0.0
max_abs_relation_error = 0.0
max_abs_stance_bump = 0.0

last_debug_signature = None
last_ref_counter = None

realized_rows = []
packet_index = 0

bounded = True

try:
    while time.time() - start <= duration:
        elapsed = time.time() - start

        reference = latest_ref[0]

        if (
            reference is not None
            and len(reference) == 6
        ):
            counter = reference[0]

            if counter != last_ref_counter:
                last_ref_counter = counter
                reference_packets += 1

            reference_error = max(
                abs(reference[1] - 0.0),
                abs(reference[2] - 0.0),
                abs(reference[3] - 0.32),
                abs(reference[4] - CLEARANCE),
                abs(reference[5] - 1.0),
            )

            if reference_error > 1e-12:
                raise RuntimeError(
                    "ROS1 bridge identity changed during rollout"
                )

        debug = latest_debug[0]

        if (
            debug is not None
            and len(debug) == 28
        ):
            signature = tuple(debug)

            if signature != last_debug_signature:
                last_debug_signature = signature
                debug_packets += 1
                packet_index += 1

                nan = float("nan")
                sim_time = rospy.get_time()
                wall_elapsed = time.time() - start

                base_z = nan
                roll_deg = nan
                pitch_deg = nan
                yaw_deg = nan
                base_vz = nan

                realized_models = latest_model[0]

                if (
                    realized_models is not None
                    and MODEL_NAME in realized_models.name
                ):
                    realized_model_index = (
                        realized_models.name.index(
                            MODEL_NAME
                        )
                    )

                    realized_pose = (
                        realized_models.pose[
                            realized_model_index
                        ]
                    )

                    realized_twist = (
                        realized_models.twist[
                            realized_model_index
                        ]
                    )

                    realized_roll, realized_pitch, realized_yaw = (
                        quaternion_to_euler(
                            realized_pose.orientation
                        )
                    )

                    base_z = realized_pose.position.z
                    roll_deg = math.degrees(realized_roll)
                    pitch_deg = math.degrees(realized_pitch)
                    yaw_deg = math.degrees(realized_yaw)
                    base_vz = realized_twist.linear.z

                for realized_leg_index in range(4):
                    realized_offset = (
                        4 + 6 * realized_leg_index
                    )

                    realized_phase = (
                        debug[realized_offset + 0]
                    )

                    realized_swing = int(
                        round(
                            debug[realized_offset + 1]
                        )
                    )

                    realized_shape = (
                        debug[realized_offset + 2]
                    )

                    realized_bump = (
                        debug[realized_offset + 3]
                    )

                    realized_target_z = (
                        debug[realized_offset + 4]
                    )

                    realized_actual_z = (
                        debug[realized_offset + 5]
                    )

                    realized_rows.append([
                        packet_index,
                        sim_time,
                        wall_elapsed,
                        CLEARANCE,
                        EXPECTED_DELTA,
                        LEG_NAMES[realized_leg_index],
                        realized_leg_index,
                        realized_phase,
                        realized_swing,
                        realized_shape,
                        realized_bump,
                        realized_target_z,
                        realized_actual_z,
                        realized_actual_z
                        - realized_target_z,
                        base_z,
                        roll_deg,
                        pitch_deg,
                        yaw_deg,
                        base_vz,
                    ])

            max_abs_active_error = max(
                max_abs_active_error,
                abs(debug[0] - 1.0),
            )

            max_abs_raw_error = max(
                max_abs_raw_error,
                abs(debug[1] - CLEARANCE),
            )

            max_abs_delta_error = max(
                max_abs_delta_error,
                abs(debug[3] - EXPECTED_DELTA),
            )

            for leg_index in range(4):
                offset = 4 + 6 * leg_index

                swing = int(
                    round(debug[offset + 1])
                )

                shape = debug[offset + 2]
                bump = debug[offset + 3]

                expected_bump = (
                    EXPECTED_DELTA * shape
                    if swing == 1
                    else 0.0
                )

                max_abs_relation_error = max(
                    max_abs_relation_error,
                    abs(bump - expected_bump),
                )

                if swing == 1:
                    saw_swing = True

                    max_shape = max(
                        max_shape,
                        shape,
                    )

                    max_bump = max(
                        max_bump,
                        bump,
                    )
                else:
                    saw_stance = True

                    max_abs_stance_bump = max(
                        max_abs_stance_bump,
                        abs(bump),
                    )

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

            roll, pitch, yaw = quaternion_to_euler(
                pose.orientation
            )

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
                bounded = False
                break

            if elapsed >= next_report:
                print(
                    "t={:.1f}s "
                    "z={:+.4f} "
                    "rpy=({:+.2f},{:+.2f},{:+.2f})deg "
                    "vz={:+.4f}".format(
                        elapsed,
                        pose.position.z,
                        math.degrees(roll),
                        math.degrees(pitch),
                        math.degrees(yaw),
                        twist.linear.z,
                    )
                )

                next_report += 1.0

        time.sleep(0.01)

finally:
    pause()
    print("[OK] physics paused")

if sys.version_info[0] < 3:
    csv_stream = open(CSV_PATH, "wb")
else:
    csv_stream = open(CSV_PATH, "w", newline="")

try:
    csv_writer = csv.writer(csv_stream)

    csv_writer.writerow([
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
    ])

    csv_writer.writerows(realized_rows)
finally:
    csv_stream.close()

print(
    "[OK] realized rows written = {}".format(
        len(realized_rows)
    )
)

print(
    "[OK] realized CSV path = {}".format(
        CSV_PATH
    )
)

if len(realized_rows) < 240:
    raise RuntimeError(
        "insufficient realized-foot rows: {}".format(
            len(realized_rows)
        )
    )

print()
print("===== {} METRICS =====".format(TRIAL))
print("ROS1 reference packets  = {}".format(reference_packets))
print("unique debug packets    = {}".format(debug_packets))
print("saw swing               = {}".format(saw_swing))
print("saw stance              = {}".format(saw_stance))
print("max normalized shape    = {:.9f}".format(max_shape))
print("max applied bump        = {:.9f}".format(max_bump))
print(
    "max active error        = {:.12f}".format(
        max_abs_active_error
    )
)
print(
    "max raw-command error   = {:.12f}".format(
        max_abs_raw_error
    )
)
print(
    "max delta error         = {:.12f}".format(
        max_abs_delta_error
    )
)
print(
    "max relation error      = {:.12f}".format(
        max_abs_relation_error
    )
)
print(
    "max stance bump         = {:.12f}".format(
        max_abs_stance_bump
    )
)
print("bounded                 = {}".format(bounded))

if reference_packets < 50:
    raise RuntimeError(
        "insufficient ROS1 bridge coverage"
    )

if debug_packets < 40:
    raise RuntimeError(
        "insufficient LL1 debug coverage"
    )

if not saw_swing:
    raise RuntimeError(
        "no swing observed"
    )

if not saw_stance:
    raise RuntimeError(
        "no stance observed"
    )

if max_shape < 0.90:
    raise RuntimeError(
        "swing apex not sampled"
    )

if max_abs_active_error > 1e-12:
    raise RuntimeError(
        "active flag mismatch"
    )

if max_abs_raw_error > 1e-12:
    raise RuntimeError(
        "clearance changed across bridge"
    )

if max_abs_delta_error > 1e-12:
    raise RuntimeError(
        "bounded delta mismatch"
    )

if max_abs_relation_error > 1e-12:
    raise RuntimeError(
        "bump relation mismatch"
    )

if max_abs_stance_bump > 1e-12:
    raise RuntimeError(
        "stance bump was nonzero"
    )

if abs(max_bump - EXPECTED_DELTA) > 1e-6:
    raise RuntimeError(
        "expected apex bump {:+.6f}, "
        "observed {:+.6f}".format(
            EXPECTED_DELTA,
            max_bump,
        )
    )

if not bounded:
    raise RuntimeError(
        "robot state became unbounded"
    )

print()
print(
    "[OK] J7 output clearance {:.3f} reached LL1 as "
    "{:+.3f} m swing-apex residual".format(
        CLEARANCE,
        EXPECTED_DELTA,
    )
)
PY
    '

  docker cp \
    "$LL1_CONTAINER:$container_realized_csv" \
    "$realized_csv"

  test -s "$realized_csv" || {
    echo "[ERROR] missing realized CSV: $realized_csv"
    exit 11
  }

  docker exec \
    "$LL1_CONTAINER" \
    rm -f \
    "$container_realized_csv"

  echo "[OK] realized CSV copied to $realized_csv"

  stop_j7_source

  validate_j7_decision \
    "$j7_csv" \
    "$expected_j7_accept" \
    "$expected_j7_reason" \
    "$expected_j7_source" \
    "$projected_clearance" \
    "$clearance" \
    "$expected_j7_delta"

  stop_controller
}


run_trial \
  "accepted_050" \
  "0.050" \
  "0.050" \
  "0.005" \
  "1" \
  "accepted_guarded" \
  "projected" \
  "0.005" \
  "5100000"

run_trial \
  "rejected_060" \
  "0.060" \
  "0.045" \
  "0.000" \
  "0" \
  "delta_clearance_too_large" \
  "empirical" \
  "0.015" \
  "5200000"


echo

echo "===== MATCHED REALIZED-FOOT ANALYSIS ====="

ACCEPTED_REALIZED_CSV="$RESULT_DIR/accepted_050/accepted_050_realized.csv"
REJECTED_REALIZED_CSV="$RESULT_DIR/rejected_060/rejected_060_realized.csv"

MATCHED_REALIZED_CSV="$RESULT_DIR/j7_matched_realized_cells.csv"
MATCHED_REALIZED_SUMMARY="$RESULT_DIR/j7_matched_realized_summary.txt"

test -s "$ACCEPTED_REALIZED_CSV" || {
  echo "[ERROR] missing accepted realized CSV"
  exit 12
}

test -s "$REJECTED_REALIZED_CSV" || {
  echo "[ERROR] missing rejected realized CSV"
  exit 13
}

/usr/bin/python3 - \
  "$ACCEPTED_REALIZED_CSV" \
  "$REJECTED_REALIZED_CSV" \
  "$MATCHED_REALIZED_CSV" \
  "$MATCHED_REALIZED_SUMMARY" \
  <<'PY_ANALYZE_REALIZED'
import csv
import math
import statistics
import sys
from collections import defaultdict


(
    accepted_path,
    rejected_path,
    matched_path,
    summary_path,
) = sys.argv[1:]

LEG_NAMES = ["FL", "FR", "RL", "RR"]

PHASE_BIN_COUNT = 16
APEX_PHASE_MIN = 0.375
APEX_PHASE_MAX = 0.625
MIN_SAMPLES_PER_CELL = 2


def finite(value):
    return math.isfinite(float(value))


def load_rows(path):
    rows = []

    with open(path, newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {
                "packet_index": int(raw["packet_index"]),
                "command": float(raw["command"]),
                "expected_delta": float(
                    raw["expected_delta"]
                ),
                "leg": raw["leg"],
                "leg_index": int(raw["leg_index"]),
                "phase": float(raw["phase"]),
                "swing": int(raw["swing"]),
                "shape": float(raw["shape"]),
                "bump": float(raw["bump"]),
                "target_z": float(raw["target_z"]),
                "actual_z": float(raw["actual_z"]),
                "tracking_error": float(
                    raw["tracking_error"]
                ),
            }

            values = [
                row["phase"],
                row["shape"],
                row["bump"],
                row["target_z"],
                row["actual_z"],
                row["tracking_error"],
            ]

            if not all(finite(value) for value in values):
                continue

            if row["swing"] != 1:
                continue

            if not (
                APEX_PHASE_MIN
                <= row["phase"]
                <= APEX_PHASE_MAX
            ):
                continue

            phase_bin = int(
                math.floor(
                    row["phase"] * PHASE_BIN_COUNT
                )
            )

            phase_bin = max(
                0,
                min(
                    PHASE_BIN_COUNT - 1,
                    phase_bin,
                ),
            )

            row["phase_bin"] = phase_bin
            rows.append(row)

    return rows


def aggregate(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["leg"],
            row["phase_bin"],
        )

        groups[key].append(row)

    aggregated = {}

    for key, samples in groups.items():
        if len(samples) < MIN_SAMPLES_PER_CELL:
            continue

        aggregated[key] = {
            "leg": key[0],
            "phase_bin": key[1],
            "samples": len(samples),
            "phase": statistics.mean(
                sample["phase"]
                for sample in samples
            ),
            "shape": statistics.mean(
                sample["shape"]
                for sample in samples
            ),
            "bump": statistics.mean(
                sample["bump"]
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

    return aggregated


accepted_rows = load_rows(accepted_path)
rejected_rows = load_rows(rejected_path)

accepted_cells = aggregate(accepted_rows)
rejected_cells = aggregate(rejected_rows)

matched_keys = sorted(
    set(accepted_cells)
    & set(rejected_cells)
)

matched = []

for key in matched_keys:
    accepted = accepted_cells[key]
    rejected = rejected_cells[key]

    bump_delta = (
        accepted["bump"]
        - rejected["bump"]
    )

    target_delta = (
        accepted["target_z"]
        - rejected["target_z"]
    )

    actual_delta = (
        accepted["actual_z"]
        - rejected["actual_z"]
    )

    tracking_error_change = (
        accepted["tracking_error"]
        - rejected["tracking_error"]
    )

    matched.append({
        "leg": key[0],
        "phase_bin": key[1],
        "accepted_samples": accepted["samples"],
        "rejected_samples": rejected["samples"],
        "mean_phase": (
            accepted["phase"]
            + rejected["phase"]
        ) / 2.0,
        "mean_shape": (
            accepted["shape"]
            + rejected["shape"]
        ) / 2.0,
        "accepted_bump": accepted["bump"],
        "rejected_bump": rejected["bump"],
        "bump_delta": bump_delta,
        "accepted_target_z": accepted["target_z"],
        "rejected_target_z": rejected["target_z"],
        "target_delta": target_delta,
        "accepted_actual_z": accepted["actual_z"],
        "rejected_actual_z": rejected["actual_z"],
        "actual_delta": actual_delta,
        "tracking_error_change":
            tracking_error_change,
    })


if len(matched) < 8:
    raise RuntimeError(
        "insufficient matched apex cells: {}".format(
            len(matched)
        )
    )


matched_legs = sorted({
    cell["leg"]
    for cell in matched
})

if matched_legs != LEG_NAMES:
    raise RuntimeError(
        "matched-leg coverage incomplete: {}".format(
            matched_legs
        )
    )


fieldnames = [
    "leg",
    "phase_bin",
    "accepted_samples",
    "rejected_samples",
    "mean_phase",
    "mean_shape",
    "accepted_bump",
    "rejected_bump",
    "bump_delta",
    "accepted_target_z",
    "rejected_target_z",
    "target_delta",
    "accepted_actual_z",
    "rejected_actual_z",
    "actual_delta",
    "tracking_error_change",
]

with open(matched_path, "w", newline="") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for cell in matched:
        writer.writerow(cell)


bump_deltas = [
    cell["bump_delta"]
    for cell in matched
]

target_deltas = [
    cell["target_delta"]
    for cell in matched
]

actual_deltas = [
    cell["actual_delta"]
    for cell in matched
]

tracking_changes = [
    cell["tracking_error_change"]
    for cell in matched
]

median_bump_delta = statistics.median(
    bump_deltas
)

median_target_delta = statistics.median(
    target_deltas
)

median_actual_delta = statistics.median(
    actual_deltas
)

median_tracking_change = statistics.median(
    tracking_changes
)

positive_target_fraction = (
    sum(
        value > 0.0
        for value in target_deltas
    )
    / float(len(target_deltas))
)

positive_actual_fraction = (
    sum(
        value > 0.0
        for value in actual_deltas
    )
    / float(len(actual_deltas))
)

if abs(median_target_delta) > 1e-12:
    realized_target_ratio = (
        median_actual_delta
        / median_target_delta
    )
else:
    realized_target_ratio = float("nan")


summary_lines = [
    "matched apex cells = {}".format(
        len(matched)
    ),
    "matched legs = {}".format(
        ",".join(matched_legs)
    ),
    "accepted apex rows = {}".format(
        len(accepted_rows)
    ),
    "rejected apex rows = {}".format(
        len(rejected_rows)
    ),
    "combined median bump delta mm = {:+.6f}".format(
        1000.0 * median_bump_delta
    ),
    "combined median target delta mm = {:+.6f}".format(
        1000.0 * median_target_delta
    ),
    "combined median actual delta mm = {:+.6f}".format(
        1000.0 * median_actual_delta
    ),
    "combined median tracking-error change mm = {:+.6f}".format(
        1000.0 * median_tracking_change
    ),
    "positive target cells fraction = {:.6f}".format(
        positive_target_fraction
    ),
    "positive actual cells fraction = {:.6f}".format(
        positive_actual_fraction
    ),
    "median realized/target ratio = {:.6f}".format(
        realized_target_ratio
    ),
]

for leg in LEG_NAMES:
    leg_cells = [
        cell
        for cell in matched
        if cell["leg"] == leg
    ]

    summary_lines.append(
        "{}: cells={} median_target_delta_mm={:+.6f} "
        "median_actual_delta_mm={:+.6f}".format(
            leg,
            len(leg_cells),
            1000.0 * statistics.median(
                cell["target_delta"]
                for cell in leg_cells
            ),
            1000.0 * statistics.median(
                cell["actual_delta"]
                for cell in leg_cells
            ),
        )
    )


failures = []

if median_bump_delta <= 0.0030:
    failures.append(
        "median bump delta did not show the "
        "expected guarded residual"
    )

if median_target_delta <= 0.0010:
    failures.append(
        "median target-z shift was not positive enough"
    )

if positive_target_fraction < 0.75:
    failures.append(
        "positive target-cell coverage below 75%"
    )

if median_actual_delta <= 0.0005:
    failures.append(
        "median actual-z shift was not positive enough"
    )

if positive_actual_fraction < 0.65:
    failures.append(
        "positive actual-cell coverage below 65%"
    )


if failures:
    summary_lines.append("")
    summary_lines.append("[FAIL]")

    for failure in failures:
        summary_lines.append("- " + failure)
else:
    summary_lines.append("")
    summary_lines.append(
        "[EVIDENCE] same-run J7 accepted routing "
        "produced a positive realized swing-foot "
        "height shift over the empirical fallback "
        "across matched apex cells"
    )


summary_text = "\n".join(summary_lines) + "\n"

with open(summary_path, "w") as stream:
    stream.write(summary_text)

print(summary_text, end="")

if failures:
    raise RuntimeError(
        "; ".join(failures)
    )
PY_ANALYZE_REALIZED

echo "[OK] matched realized-foot analysis passed"
echo "matched CSV: $MATCHED_REALIZED_CSV"
echo "summary: $MATCHED_REALIZED_SUMMARY"

echo
echo "===== BRIDGE LOGS ====="

echo "--- ROS2 sender ---"
tail -30 \
  /tmp/tracer_ll1_e2e_ros2_sender.log \
  || true

echo
echo "--- ROS1 receiver ---"

docker exec \
  "$LL1_CONTAINER" \
  tail -30 \
    /tmp/tracer_ll1_e2e_ros1_receiver.log \
  || true


cleanup
trap - EXIT


echo
echo "===== FINAL SAFE STATE ====="

source_ros2

echo "ROS2 publisher/sender:"
pgrep -af \
  '[t]racer_j7_to_ll1_active_source.py|[t]racer_ros2_mpc_ref_udp_sender.py' \
  || true

echo
echo "UDP 50110:"
ss -lunap 2>/dev/null \
  | grep ':50110' \
  || true

echo
echo "LL1 container:"

docker exec \
  "$LL1_CONTAINER" \
  bash --noprofile --norc -lc '
    source /opt/ros/melodic/setup.bash

    echo "controller:"
    pgrep -ax gazebo_a1_ctrl || true

    echo
    echo "receiver:"
    pgrep -af \
      "[t]racer_udp_to_ros1_mpc_ref.py" \
      || true

    echo
    echo "joint command authority:"
    rostopic info \
      /a1_gazebo/FL_hip_controller/command \
      || true

    echo
    echo "parameters:"
    rosparam get /tracer_enable_swing_apex_residual
    rosparam get /tracer_clearance_cmd_neutral
    rosparam get /tracer_swing_apex_delta_min
    rosparam get /tracer_swing_apex_delta_max

    rosservice call \
      /gazebo/pause_physics \
      >/dev/null

    echo
    echo "[OK] physics paused"
  '

echo
echo "[OK] complete same-run J7-to-realized-foot authority sweep passed"
echo "result dir: $RESULT_DIR"
