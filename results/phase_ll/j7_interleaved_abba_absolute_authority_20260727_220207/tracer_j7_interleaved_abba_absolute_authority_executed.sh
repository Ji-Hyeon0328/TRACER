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

RUN_TAG="tracer_j7_interleaved_abba_absolute_authority_$(date +%Y%m%d_%H%M%S)"
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
import time

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

        self.schedule_start_epoch = float(
            os.environ["TRACER_ABBA_START_EPOCH"]
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

        schedule_offset = (
            time.time() - self.schedule_start_epoch
        )

        # A1 [0, 5): rejected projected 0.060 -> empirical 0.045
        # transition [5, 6): switch to accepted
        # B1 [6, 11): accepted projected 0.050
        # B2 [11, 16): accepted projected 0.050
        # transition [16, 17): switch to rejected
        # A2 [17, 22): rejected projected 0.060 -> empirical 0.045
        if (
            schedule_offset < 5.0
            or schedule_offset >= 16.0
        ):
            projected_clearance = 0.060
        else:
            projected_clearance = 0.050

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
            projected_clearance,
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
  local schedule_start_epoch="$1"
  local counter_base="$2"
  local process_log="$3"
  local j7_log_dir="$4"

  stop_j7_source
  source_ros2

  mkdir -p \
    "$(dirname "$process_log")" \
    "$j7_log_dir"

  TRACER_ABBA_START_EPOCH="$schedule_start_epoch" \
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
      echo "[ERROR] J7 ABBA source exited during startup"
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
    echo "[ERROR] J7 ABBA source did not become ready"
    cat "$process_log" || true

    ros2 topic info \
      /tracer/mpc_reference \
      --verbose \
      || true

    exit 9
  fi

  echo "[OK] J7 is the sole ROS2 MPC-reference publisher"
  sleep 1.0
}


validate_j7_abba_decisions() {
  local csv_path="$1"

  test -s "$csv_path" || {
    echo "[ERROR] missing J7 ABBA decision CSV: $csv_path"
    exit 10
  }

  /usr/bin/python3 - \
    "$csv_path" \
    <<'PY_VALIDATE_ABBA'
import csv
import math
import sys


csv_path = sys.argv[1]

with open(csv_path, newline="") as stream:
    rows = list(csv.DictReader(stream))

if len(rows) < 180:
    raise RuntimeError(
        "insufficient J7 ABBA decision rows: {}".format(
            len(rows)
        )
    )

accepted = []
rejected = []
invalid = []

for index, row in enumerate(rows):
    context_ok = row["context"] == "flat"

    empirical_ok = math.isclose(
        float(row["emp_clearance"]),
        0.045,
        abs_tol=1e-9,
    )

    projected = float(row["proj_clearance"])
    output = float(row["out_clearance"])
    candidate_delta = float(row["delta_clearance"])

    sequence_ok = (
        int(row["emp_seq"])
        == int(row["proj_seq"])
        == int(row["out_seq"])
    )

    if not (
        context_ok
        and empirical_ok
        and sequence_ok
    ):
        invalid.append((index, row))
        continue

    if math.isclose(
        projected,
        0.050,
        abs_tol=1e-9,
    ):
        checks = [
            math.isclose(output, 0.050, abs_tol=1e-9),
            math.isclose(
                candidate_delta,
                0.005,
                abs_tol=1e-9,
            ),
            row["j7_accept"] == "1",
            row["j7_reason"] == "accepted_guarded",
            row["j7_source"] == "projected",
        ]

        if all(checks):
            accepted.append(row)
        else:
            invalid.append((index, row))

    elif math.isclose(
        projected,
        0.060,
        abs_tol=1e-9,
    ):
        checks = [
            math.isclose(output, 0.045, abs_tol=1e-9),
            math.isclose(
                candidate_delta,
                0.015,
                abs_tol=1e-9,
            ),
            row["j7_accept"] == "0",
            (
                row["j7_reason"]
                == "delta_clearance_too_large"
            ),
            row["j7_source"] == "empirical",
        ]

        if all(checks):
            rejected.append(row)
        else:
            invalid.append((index, row))

    else:
        invalid.append((index, row))

if invalid:
    raise RuntimeError(
        "invalid J7 ABBA rows={} first={}".format(
            len(invalid),
            invalid[0],
        )
    )

if len(accepted) < 80:
    raise RuntimeError(
        "insufficient accepted J7 rows: {}".format(
            len(accepted)
        )
    )

if len(rejected) < 80:
    raise RuntimeError(
        "insufficient rejected J7 rows: {}".format(
            len(rejected)
        )
    )

print("[OK] total J7 rows={}".format(len(rows)))
print("[OK] accepted projected rows={}".format(len(accepted)))
print("[OK] rejected fallback rows={}".format(len(rejected)))
print("[OK] all J7 rows preserved sequence identity")
PY_VALIDATE_ABBA
}


run_interleaved_abba() {
  local trial="interleaved_abba"
  local trial_dir="$RESULT_DIR/$trial"
  local j7_log_dir="$trial_dir/j7_log"
  local j7_csv="$j7_log_dir/guarded_meta_action_ref_gate_j7_v0.csv"

  local realized_csv="$trial_dir/interleaved_abba_realized.csv"
  local container_realized_csv="/tmp/tracer_j7_interleaved_abba_realized.csv"

  local matched_csv="$RESULT_DIR/j7_interleaved_abba_matched_cells.csv"
  local summary_txt="$RESULT_DIR/j7_interleaved_abba_summary.txt"

  mkdir -p \
    "$trial_dir" \
    "$j7_log_dir"

  echo
  echo "============================================================"
  echo "INTERLEAVED ABBA REALIZED-AUTHORITY TEST"
  echo "A1: rejected 0.060 -> empirical 0.045"
  echo "B1: accepted 0.050"
  echo "B2: accepted 0.050"
  echo "A2: rejected 0.060 -> empirical 0.045"
  echo "single reset / single controller / single J7 process"
  echo "============================================================"

  stop_j7_source
  stop_controller

  reset_standing_pose
  start_controller "$trial"

  local schedule_start_epoch

  schedule_start_epoch="$(
    /usr/bin/python3 - <<'PY_START_EPOCH'
import time
print("{:.6f}".format(time.time() + 5.0))
PY_START_EPOCH
  )"

  echo "ABBA schedule start epoch=$schedule_start_epoch"

  start_j7_source \
    "$schedule_start_epoch" \
    "5300000" \
    "$trial_dir/j7_active_source.log" \
    "$j7_log_dir"

  docker exec \
    "$LL1_CONTAINER" \
    rm -f \
    "$container_realized_csv"

  docker exec \
    -e TRACER_ABBA_START_EPOCH="$schedule_start_epoch" \
    -e TRACER_ABBA_CSV="$container_realized_csv" \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      set -eo pipefail

      source /opt/ros/melodic/setup.bash
      source /root/unitree_ws/devel/setup.bash
      source /root/A1_ctrl_ws/devel/setup.bash

      python - <<'"'"'PY_ABBA_OBSERVER'"'"'
from __future__ import print_function

import csv
import math
import os
import sys
import time

import rospy

from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty


START_EPOCH = float(
    os.environ["TRACER_ABBA_START_EPOCH"]
)

CSV_PATH = os.environ["TRACER_ABBA_CSV"]

MODEL_NAME = "a1_gazebo"
LEG_NAMES = ["FL", "FR", "RL", "RR"]

WINDOWS = [
    {
        "name": "A1",
        "condition": "A",
        "start": 0.0,
        "end": 5.0,
        "clearance": 0.045,
        "delta": 0.000,
    },
    {
        "name": "B1",
        "condition": "B",
        "start": 6.0,
        "end": 11.0,
        "clearance": 0.050,
        "delta": 0.005,
    },
    {
        "name": "B2",
        "condition": "B",
        "start": 11.0,
        "end": 16.0,
        "clearance": 0.050,
        "delta": 0.005,
    },
    {
        "name": "A2",
        "condition": "A",
        "start": 17.0,
        "end": 22.0,
        "clearance": 0.045,
        "delta": 0.000,
    },
]

END_OFFSET = 22.0

latest_ref = [None]
latest_debug = [None]
latest_model = [None]


def on_ref(message):
    latest_ref[0] = list(message.data)


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


def window_for_offset(offset):
    for window in WINDOWS:
        if (
            window["start"]
            <= offset
            < window["end"]
        ):
            return window

    return None


def new_metrics():
    return {
        "reference_packets": 0,
        "debug_packets": 0,
        "header_mismatches": 0,
        "saw_swing": False,
        "saw_stance": False,
        "max_shape": 0.0,
        "max_bump": 0.0,
        "max_abs_active_error": 0.0,
        "max_abs_raw_error": 0.0,
        "max_abs_delta_error": 0.0,
        "max_abs_relation_error": 0.0,
        "max_abs_stance_bump": 0.0,
    }


metrics = {
    window["name"]: new_metrics()
    for window in WINDOWS
}

last_ref_counter = {
    window["name"]: None
    for window in WINDOWS
}

last_debug_signature = None
packet_index = 0
realized_rows = []
bounded = True


rospy.init_node(
    "tracer_j7_interleaved_abba_observer",
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
        and abs(reference[4] - 0.045) <= 1e-12
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
        "baseline ROS1 reference did not preload"
    )

print(
    "[OK] baseline ROS1 reference preloaded {}".format(
        latest_ref[0]
    )
)


unpause()
print("[OK] physics unpaused once for continuous ABBA run")


debug_deadline = time.time() + 4.0
debug_converged = False

while time.time() < debug_deadline:
    debug = latest_debug[0]

    debug_ok = (
        debug is not None
        and len(debug) == 28
        and abs(debug[0] - 1.0) <= 1e-12
        and abs(debug[1] - 0.045) <= 1e-12
        and abs(debug[2] - 0.045) <= 1e-12
        and abs(debug[3] - 0.000) <= 1e-12
    )

    if debug_ok:
        debug_converged = True
        break

    time.sleep(0.02)

if not debug_converged:
    pause()

    print(
        "[DEBUG] latest reference={}".format(
            latest_ref[0]
        )
    )

    print(
        "[DEBUG] latest debug={}".format(
            latest_debug[0]
        )
    )

    raise RuntimeError(
        "baseline LL1 debug did not converge"
    )

print(
    "[OK] baseline LL1 header active={:.1f} "
    "raw={:.3f} neutral={:.3f} delta={:+.3f}".format(
        latest_debug[0][0],
        latest_debug[0][1],
        latest_debug[0][2],
        latest_debug[0][3],
    )
)


if time.time() >= START_EPOCH:
    pause()

    raise RuntimeError(
        "ABBA observer missed schedule start epoch"
    )

print(
    "[OK] waiting {:.3f}s for A1 start".format(
        START_EPOCH - time.time()
    )
)

while time.time() < START_EPOCH:
    time.sleep(0.01)


next_report_offset = 1.0

try:
    while True:
        now = time.time()
        offset = now - START_EPOCH

        if offset >= END_OFFSET:
            break

        window = window_for_offset(offset)
        reference = latest_ref[0]
        debug = latest_debug[0]

        reference_matches = False

        if (
            window is not None
            and reference is not None
            and len(reference) == 6
        ):
            reference_matches = (
                abs(reference[1]) <= 1e-12
                and abs(reference[2]) <= 1e-12
                and abs(reference[3] - 0.32) <= 1e-12
                and abs(
                    reference[4] - window["clearance"]
                ) <= 1e-12
                and abs(reference[5] - 1.0) <= 1e-12
            )

            if reference_matches:
                counter = reference[0]

                if (
                    counter
                    != last_ref_counter[window["name"]]
                ):
                    last_ref_counter[window["name"]] = counter

                    metrics[
                        window["name"]
                    ]["reference_packets"] += 1

        if (
            debug is not None
            and len(debug) == 28
        ):
            signature = tuple(debug)

            if signature != last_debug_signature:
                last_debug_signature = signature

                if window is not None:
                    window_metrics = metrics[
                        window["name"]
                    ]

                    debug_header_matches = (
                        abs(debug[0] - 1.0) <= 1e-12
                        and abs(
                            debug[1]
                            - window["clearance"]
                        ) <= 1e-12
                        and abs(
                            debug[2] - 0.045
                        ) <= 1e-12
                        and abs(
                            debug[3] - window["delta"]
                        ) <= 1e-12
                    )

                    if not (
                        reference_matches
                        and debug_header_matches
                    ):
                        window_metrics[
                            "header_mismatches"
                        ] += 1
                    else:
                        packet_index += 1

                        window_metrics[
                            "debug_packets"
                        ] += 1

                        window_metrics[
                            "max_abs_active_error"
                        ] = max(
                            window_metrics[
                                "max_abs_active_error"
                            ],
                            abs(debug[0] - 1.0),
                        )

                        window_metrics[
                            "max_abs_raw_error"
                        ] = max(
                            window_metrics[
                                "max_abs_raw_error"
                            ],
                            abs(
                                debug[1]
                                - window["clearance"]
                            ),
                        )

                        window_metrics[
                            "max_abs_delta_error"
                        ] = max(
                            window_metrics[
                                "max_abs_delta_error"
                            ],
                            abs(
                                debug[3]
                                - window["delta"]
                            ),
                        )

                        nan = float("nan")

                        base_z = nan
                        roll_deg = nan
                        pitch_deg = nan
                        yaw_deg = nan
                        base_vz = nan

                        models = latest_model[0]

                        if (
                            models is not None
                            and MODEL_NAME in models.name
                        ):
                            model_index = models.name.index(
                                MODEL_NAME
                            )

                            pose = models.pose[model_index]
                            twist = models.twist[model_index]

                            roll, pitch, yaw = (
                                quaternion_to_euler(
                                    pose.orientation
                                )
                            )

                            base_z = pose.position.z
                            roll_deg = math.degrees(roll)
                            pitch_deg = math.degrees(pitch)
                            yaw_deg = math.degrees(yaw)
                            base_vz = twist.linear.z

                        for leg_index in range(4):
                            leg_offset = 4 + 6 * leg_index

                            phase = debug[
                                leg_offset + 0
                            ]

                            swing = int(
                                round(
                                    debug[
                                        leg_offset + 1
                                    ]
                                )
                            )

                            shape = debug[
                                leg_offset + 2
                            ]

                            bump = debug[
                                leg_offset + 3
                            ]

                            target_z = debug[
                                leg_offset + 4
                            ]

                            actual_z = debug[
                                leg_offset + 5
                            ]

                            expected_bump = (
                                window["delta"] * shape
                                if swing == 1
                                else 0.0
                            )

                            window_metrics[
                                "max_abs_relation_error"
                            ] = max(
                                window_metrics[
                                    "max_abs_relation_error"
                                ],
                                abs(
                                    bump
                                    - expected_bump
                                ),
                            )

                            if swing == 1:
                                window_metrics[
                                    "saw_swing"
                                ] = True

                                window_metrics[
                                    "max_shape"
                                ] = max(
                                    window_metrics[
                                        "max_shape"
                                    ],
                                    shape,
                                )

                                window_metrics[
                                    "max_bump"
                                ] = max(
                                    window_metrics[
                                        "max_bump"
                                    ],
                                    bump,
                                )
                            else:
                                window_metrics[
                                    "saw_stance"
                                ] = True

                                window_metrics[
                                    "max_abs_stance_bump"
                                ] = max(
                                    window_metrics[
                                        "max_abs_stance_bump"
                                    ],
                                    abs(bump),
                                )

                            realized_rows.append([
                                packet_index,
                                rospy.get_time(),
                                offset,
                                window["name"],
                                window["condition"],
                                window["clearance"],
                                window["delta"],
                                LEG_NAMES[leg_index],
                                leg_index,
                                phase,
                                swing,
                                shape,
                                bump,
                                target_z,
                                actual_z,
                                actual_z - target_z,
                                target_z - bump,
                                actual_z
                                - (target_z - bump),
                                base_z,
                                roll_deg,
                                pitch_deg,
                                yaw_deg,
                                base_vz,
                            ])

        models = latest_model[0]

        if (
            models is not None
            and MODEL_NAME in models.name
        ):
            model_index = models.name.index(
                MODEL_NAME
            )

            pose = models.pose[model_index]
            twist = models.twist[model_index]

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

            if offset >= next_report_offset:
                print(
                    "t={:.1f}s window={} "
                    "ref_clr={} debug_raw={} "
                    "base_z={:+.4f} vz={:+.4f}".format(
                        offset,
                        (
                            window["name"]
                            if window is not None
                            else "transition"
                        ),
                        (
                            "{:.3f}".format(reference[4])
                            if (
                                reference is not None
                                and len(reference) == 6
                            )
                            else "none"
                        ),
                        (
                            "{:.3f}".format(debug[1])
                            if (
                                debug is not None
                                and len(debug) == 28
                            )
                            else "none"
                        ),
                        pose.position.z,
                        twist.linear.z,
                    )
                )

                next_report_offset += 1.0

        time.sleep(0.01)

finally:
    pause()
    print("[OK] physics paused after one continuous ABBA run")


if sys.version_info[0] < 3:
    csv_stream = open(CSV_PATH, "wb")
else:
    csv_stream = open(CSV_PATH, "w", newline="")

try:
    writer = csv.writer(csv_stream)

    writer.writerow([
        "packet_index",
        "sim_time",
        "schedule_offset",
        "window",
        "condition",
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
        "native_target_z",
        "realized_relative_z",
        "base_z",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "base_vz",
    ])

    writer.writerows(realized_rows)

finally:
    csv_stream.close()


print()
print("===== INTERLEAVED ABBA WINDOW METRICS =====")

for window in WINDOWS:
    name = window["name"]
    value = metrics[name]

    print(
        "{}: refs={} debug={} rows={} mismatches={} "
        "shape={:.6f} bump={:.6f} "
        "relation_err={:.3e}".format(
            name,
            value["reference_packets"],
            value["debug_packets"],
            4 * value["debug_packets"],
            value["header_mismatches"],
            value["max_shape"],
            value["max_bump"],
            value["max_abs_relation_error"],
        )
    )

    if value["reference_packets"] < 60:
        raise RuntimeError(
            "{} insufficient reference coverage".format(
                name
            )
        )

    if value["debug_packets"] < 40:
        raise RuntimeError(
            "{} insufficient debug coverage".format(
                name
            )
        )

    if value["header_mismatches"] > 2:
        raise RuntimeError(
            "{} excessive header mismatches: {}".format(
                name,
                value["header_mismatches"],
            )
        )

    if not value["saw_swing"]:
        raise RuntimeError(
            "{} no swing observed".format(name)
        )

    if not value["saw_stance"]:
        raise RuntimeError(
            "{} no stance observed".format(name)
        )

    if value["max_shape"] < 0.90:
        raise RuntimeError(
            "{} swing apex not sampled".format(name)
        )

    if value["max_abs_active_error"] > 1e-12:
        raise RuntimeError(
            "{} active mismatch".format(name)
        )

    if value["max_abs_raw_error"] > 1e-12:
        raise RuntimeError(
            "{} raw-command mismatch".format(name)
        )

    if value["max_abs_delta_error"] > 1e-12:
        raise RuntimeError(
            "{} delta mismatch".format(name)
        )

    if value["max_abs_relation_error"] > 1e-12:
        raise RuntimeError(
            "{} bump relation mismatch".format(name)
        )

    if value["max_abs_stance_bump"] > 1e-12:
        raise RuntimeError(
            "{} stance bump nonzero".format(name)
        )

    if (
        abs(
            value["max_bump"]
            - window["delta"]
        )
        > 1e-6
    ):
        raise RuntimeError(
            "{} apex bump mismatch expected={} "
            "observed={}".format(
                name,
                window["delta"],
                value["max_bump"],
            )
        )

if not bounded:
    raise RuntimeError(
        "robot state became unbounded during ABBA"
    )

if len(realized_rows) < 640:
    raise RuntimeError(
        "insufficient total realized rows: {}".format(
            len(realized_rows)
        )
    )

print("[OK] total realized rows={}".format(len(realized_rows)))
print("[OK] ABBA CSV path={}".format(CSV_PATH))
PY_ABBA_OBSERVER
    '

  docker cp \
    "$LL1_CONTAINER:$container_realized_csv" \
    "$realized_csv"

  test -s "$realized_csv" || {
    echo "[ERROR] missing ABBA realized CSV: $realized_csv"
    exit 11
  }

  docker exec \
    "$LL1_CONTAINER" \
    rm -f \
    "$container_realized_csv"

  echo "[OK] ABBA realized CSV copied to $realized_csv"

  stop_j7_source

  validate_j7_abba_decisions \
    "$j7_csv"

  echo
  echo "===== INTERLEAVED ABBA MATCHED ANALYSIS ====="

  /usr/bin/python3 - \
    "$realized_csv" \
    "$matched_csv" \
    "$summary_txt" \
    <<'PY_ANALYZE_ABBA'
import csv
import math
import statistics
import sys
from collections import defaultdict


input_path, matched_path, summary_path = sys.argv[1:]

WINDOW_NAMES = ["A1", "B1", "B2", "A2"]
LEG_NAMES = ["FL", "FR", "RL", "RR"]

PHASE_BIN_COUNT = 16
APEX_PHASE_MIN = 0.375
APEX_PHASE_MAX = 0.625
MIN_SAMPLES_PER_WINDOW_CELL = 2


def mean(values):
    return statistics.mean(values)


rows = []

with open(input_path, newline="") as stream:
    for raw in csv.DictReader(stream):
        row = {
            "window": raw["window"],
            "leg": raw["leg"],
            "phase": float(raw["phase"]),
            "swing": int(raw["swing"]),
            "shape": float(raw["shape"]),
            "bump": float(raw["bump"]),
            "target_z": float(raw["target_z"]),
            "actual_z": float(raw["actual_z"]),
            "tracking_error": float(
                raw["tracking_error"]
            ),
            "native_target_z": float(
                raw["native_target_z"]
            ),
            "realized_relative_z": float(
                raw["realized_relative_z"]
            ),
        }

        numeric = [
            row["phase"],
            row["shape"],
            row["bump"],
            row["target_z"],
            row["actual_z"],
            row["tracking_error"],
            row["native_target_z"],
            row["realized_relative_z"],
        ]

        if not all(math.isfinite(value) for value in numeric):
            continue

        if row["window"] not in WINDOW_NAMES:
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


groups = defaultdict(list)

for row in rows:
    groups[
        (
            row["window"],
            row["leg"],
            row["phase_bin"],
        )
    ].append(row)


aggregated = {}

for key, samples in groups.items():
    if len(samples) < MIN_SAMPLES_PER_WINDOW_CELL:
        continue

    aggregated[key] = {
        "samples": len(samples),
        "phase": mean(
            row["phase"]
            for row in samples
        ),
        "shape": mean(
            row["shape"]
            for row in samples
        ),
        "bump": mean(
            row["bump"]
            for row in samples
        ),
        "target_z": mean(
            row["target_z"]
            for row in samples
        ),
        "actual_z": mean(
            row["actual_z"]
            for row in samples
        ),
        "tracking_error": mean(
            row["tracking_error"]
            for row in samples
        ),
        "native_target_z": mean(
            row["native_target_z"]
            for row in samples
        ),
        "realized_relative_z": mean(
            row["realized_relative_z"]
            for row in samples
        ),
    }


candidate_keys = sorted({
    (leg, phase_bin)
    for _, leg, phase_bin in aggregated
})

matched = []

for leg, phase_bin in candidate_keys:
    window_cells = {
        window: aggregated.get(
            (window, leg, phase_bin)
        )
        for window in WINDOW_NAMES
    }

    if any(
        value is None
        for value in window_cells.values()
    ):
        continue

    a1 = window_cells["A1"]
    b1 = window_cells["B1"]
    b2 = window_cells["B2"]
    a2 = window_cells["A2"]

    def abba_effect(field):
        baseline = (
            a1[field] + a2[field]
        ) / 2.0

        accepted = (
            b1[field] + b2[field]
        ) / 2.0

        return accepted - baseline

    native_effect = abba_effect(
        "native_target_z"
    )

    bump_effect = abba_effect("bump")
    target_effect = abba_effect("target_z")
    actual_effect = abba_effect("actual_z")

    tracking_effect = abba_effect(
        "tracking_error"
    )

    relative_effect = abba_effect(
        "realized_relative_z"
    )

    target_decomposition_error = (
        target_effect
        - native_effect
        - bump_effect
    )

    actual_decomposition_error = (
        actual_effect
        - native_effect
        - relative_effect
    )

    matched.append({
        "leg": leg,
        "phase_bin": phase_bin,
        "a1_samples": a1["samples"],
        "b1_samples": b1["samples"],
        "b2_samples": b2["samples"],
        "a2_samples": a2["samples"],
        "mean_phase": mean([
            a1["phase"],
            b1["phase"],
            b2["phase"],
            a2["phase"],
        ]),
        "mean_shape": mean([
            a1["shape"],
            b1["shape"],
            b2["shape"],
            a2["shape"],
        ]),
        "a_endpoint_native_drift":
            a2["native_target_z"]
            - a1["native_target_z"],
        "b_half_native_drift":
            b2["native_target_z"]
            - b1["native_target_z"],
        "native_effect": native_effect,
        "bump_effect": bump_effect,
        "target_effect": target_effect,
        "tracking_effect": tracking_effect,
        "relative_effect": relative_effect,
        "actual_effect": actual_effect,
        "target_decomposition_error":
            target_decomposition_error,
        "actual_decomposition_error":
            actual_decomposition_error,
    })


if len(matched) < 8:
    raise RuntimeError(
        "insufficient four-window matched cells: {}".format(
            len(matched)
        )
    )

matched_legs = sorted({
    row["leg"]
    for row in matched
})

if matched_legs != LEG_NAMES:
    raise RuntimeError(
        "incomplete matched leg coverage: {}".format(
            matched_legs
        )
    )


fieldnames = [
    "leg",
    "phase_bin",
    "a1_samples",
    "b1_samples",
    "b2_samples",
    "a2_samples",
    "mean_phase",
    "mean_shape",
    "a_endpoint_native_drift",
    "b_half_native_drift",
    "native_effect",
    "bump_effect",
    "target_effect",
    "tracking_effect",
    "relative_effect",
    "actual_effect",
    "target_decomposition_error",
    "actual_decomposition_error",
]

with open(matched_path, "w", newline="") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(matched)


def values(field):
    return [
        row[field]
        for row in matched
    ]


native_effects = values("native_effect")
bump_effects = values("bump_effect")
target_effects = values("target_effect")
tracking_effects = values("tracking_effect")
relative_effects = values("relative_effect")
actual_effects = values("actual_effect")

a_endpoint_drifts = values(
    "a_endpoint_native_drift"
)

b_half_drifts = values(
    "b_half_native_drift"
)


median_native_effect = statistics.median(
    native_effects
)

median_bump_effect = statistics.median(
    bump_effects
)

median_target_effect = statistics.median(
    target_effects
)

median_tracking_effect = statistics.median(
    tracking_effects
)

median_relative_effect = statistics.median(
    relative_effects
)

median_actual_effect = statistics.median(
    actual_effects
)

median_a_endpoint_drift = statistics.median(
    a_endpoint_drifts
)

median_b_half_drift = statistics.median(
    b_half_drifts
)


positive_bump_fraction = (
    sum(value > 0.0 for value in bump_effects)
    / float(len(matched))
)

positive_target_fraction = (
    sum(value > 0.0 for value in target_effects)
    / float(len(matched))
)

positive_actual_fraction = (
    sum(value > 0.0 for value in actual_effects)
    / float(len(matched))
)

positive_relative_fraction = (
    sum(value > 0.0 for value in relative_effects)
    / float(len(matched))
)


if abs(median_target_effect) > 1e-12:
    realized_target_ratio = (
        median_actual_effect
        / median_target_effect
    )
else:
    realized_target_ratio = float("nan")


max_target_decomposition_error = max(
    abs(row["target_decomposition_error"])
    for row in matched
)

max_actual_decomposition_error = max(
    abs(row["actual_decomposition_error"])
    for row in matched
)


summary_lines = [
    "four-window matched apex cells = {}".format(
        len(matched)
    ),
    "matched legs = {}".format(
        ",".join(matched_legs)
    ),
    "apex rows total = {}".format(len(rows)),
    "",
    "combined median A-endpoint native drift mm = {:+.6f}".format(
        1000.0 * median_a_endpoint_drift
    ),
    "combined median B-half native drift mm = {:+.6f}".format(
        1000.0 * median_b_half_drift
    ),
    "combined median ABBA native effect mm = {:+.6f}".format(
        1000.0 * median_native_effect
    ),
    "combined median ABBA bump effect mm = {:+.6f}".format(
        1000.0 * median_bump_effect
    ),
    "combined median ABBA final-target effect mm = {:+.6f}".format(
        1000.0 * median_target_effect
    ),
    "combined median ABBA tracking-error effect mm = {:+.6f}".format(
        1000.0 * median_tracking_effect
    ),
    "combined median ABBA native-relative effect mm = {:+.6f}".format(
        1000.0 * median_relative_effect
    ),
    "combined median ABBA absolute-actual effect mm = {:+.6f}".format(
        1000.0 * median_actual_effect
    ),
    "",
    "positive bump cells fraction = {:.6f}".format(
        positive_bump_fraction
    ),
    "positive final-target cells fraction = {:.6f}".format(
        positive_target_fraction
    ),
    "positive native-relative cells fraction = {:.6f}".format(
        positive_relative_fraction
    ),
    "positive absolute-actual cells fraction = {:.6f}".format(
        positive_actual_fraction
    ),
    "median realized/target ratio = {:.6f}".format(
        realized_target_ratio
    ),
    "",
    "max target decomposition error = {:.12e}".format(
        max_target_decomposition_error
    ),
    "max actual decomposition error = {:.12e}".format(
        max_actual_decomposition_error
    ),
]


for leg in LEG_NAMES:
    leg_rows = [
        row
        for row in matched
        if row["leg"] == leg
    ]

    summary_lines.append(
        "{}: cells={} native_mm={:+.6f} "
        "bump_mm={:+.6f} target_mm={:+.6f} "
        "actual_mm={:+.6f}".format(
            leg,
            len(leg_rows),
            1000.0 * statistics.median(
                row["native_effect"]
                for row in leg_rows
            ),
            1000.0 * statistics.median(
                row["bump_effect"]
                for row in leg_rows
            ),
            1000.0 * statistics.median(
                row["target_effect"]
                for row in leg_rows
            ),
            1000.0 * statistics.median(
                row["actual_effect"]
                for row in leg_rows
            ),
        )
    )


failures = []

if max_target_decomposition_error > 1e-12:
    failures.append(
        "target decomposition identity failed"
    )

if max_actual_decomposition_error > 1e-12:
    failures.append(
        "actual decomposition identity failed"
    )

if median_bump_effect <= 0.0030:
    failures.append(
        "ABBA bump effect did not show guarded residual"
    )

if positive_bump_fraction < 0.95:
    failures.append(
        "positive bump-cell coverage below 95%"
    )

if median_target_effect <= 0.0010:
    failures.append(
        "ABBA final-target effect was not positive enough"
    )

if positive_target_fraction < 0.75:
    failures.append(
        "positive final-target coverage below 75%"
    )

if median_actual_effect <= 0.0005:
    failures.append(
        "ABBA absolute-actual effect was not positive enough"
    )

if positive_actual_fraction < 0.65:
    failures.append(
        "positive absolute-actual coverage below 65%"
    )


summary_lines.append("")

if failures:
    summary_lines.append("[FAIL]")

    for failure in failures:
        summary_lines.append("- " + failure)
else:
    summary_lines.append(
        "[EVIDENCE] one-reset continuous J7 ABBA routing "
        "produced a positive absolute swing-foot response "
        "over the bracketing empirical fallback"
    )

summary_text = "\n".join(summary_lines) + "\n"

with open(summary_path, "w") as stream:
    stream.write(summary_text)

print(summary_text, end="")

if failures:
    raise RuntimeError(
        "; ".join(failures)
    )
PY_ANALYZE_ABBA

  echo "[OK] interleaved ABBA matched analysis passed"
  echo "matched CSV: $matched_csv"
  echo "summary: $summary_txt"

  stop_controller
}



run_interleaved_abba


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
echo "[OK] complete one-reset interleaved J7 ABBA absolute-authority sweep passed"
echo "result dir: $RESULT_DIR"
