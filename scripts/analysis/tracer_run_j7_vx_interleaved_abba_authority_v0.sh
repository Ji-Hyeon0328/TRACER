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

RUN_TAG="tracer_j7_vx_interleaved_abba_authority_$(date +%Y%m%d_%H%M%S)"
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

        # VX ABBA schedule.
        #
        # Before schedule and A windows:
        #   projected vx=0.125 gives delta=+0.035,
        #   J7 rejects and outputs empirical vx=0.090.
        #
        # B windows:
        #   projected vx=0.115 gives delta=+0.025,
        #   J7 accepts and outputs vx=0.115.
        #
        # A1: [0.0, 2.5)
        # transition: [2.5, 3.0)
        # B1: [3.0, 5.5)
        # B2: [5.5, 8.0)
        # transition: [8.0, 8.5)
        # A2: [8.5, 11.0)
        if (
            schedule_offset < 2.5
            or schedule_offset >= 8.0
        ):
            projected_vx = 0.125
        else:
            projected_vx = 0.115

        empirical = [
            sequence,
            0.090,
            0.0,
            0.320,
            0.045,
            1.0,
        ]

        projected = [
            sequence,
            projected_vx,
            0.0,
            0.320,
            0.045,
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
    echo "[ERROR] missing J7 CSV: $csv_path"
    exit 10
  }

  /usr/bin/python3 - \
    "$csv_path" \
    <<'PY_VALIDATE_VX_J7'
import csv
import math
import sys


path = sys.argv[1]


with open(
    path,
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    rows = list(csv.DictReader(stream))


if len(rows) < 200:
    raise RuntimeError(
        "insufficient J7 rows: {}".format(len(rows))
    )


accepted = []
rejected = []
invalid = []


for index, row in enumerate(rows):
    try:
        emp_vx = float(row["emp_vx"])
        proj_vx = float(row["proj_vx"])
        out_vx = float(row["out_vx"])
        delta_vx = float(row["delta_vx"])

        common_ok = (
            row["context"] == "flat"
            and int(row["emp_seq"])
                == int(row["proj_seq"])
                == int(row["out_seq"])
            and math.isclose(
                emp_vx,
                0.090,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(row["emp_body_h"]),
                0.320,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(row["out_body_h"]),
                0.320,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(row["emp_clearance"]),
                0.045,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(row["out_clearance"]),
                0.045,
                abs_tol=1e-9,
            )
        )

        if not common_ok:
            invalid.append((index, row))
            continue

        if math.isclose(
            proj_vx,
            0.115,
            abs_tol=1e-9,
        ):
            checks = [
                math.isclose(
                    out_vx,
                    0.115,
                    abs_tol=1e-9,
                ),
                math.isclose(
                    delta_vx,
                    0.025,
                    abs_tol=1e-9,
                ),
                row["j7_accept"] == "1",
                row["j7_reason"]
                    == "accepted_guarded",
                row["j7_source"] == "projected",
            ]

            if all(checks):
                accepted.append(row)
            else:
                invalid.append((index, row))

        elif math.isclose(
            proj_vx,
            0.125,
            abs_tol=1e-9,
        ):
            checks = [
                math.isclose(
                    out_vx,
                    0.090,
                    abs_tol=1e-9,
                ),
                math.isclose(
                    delta_vx,
                    0.035,
                    abs_tol=1e-9,
                ),
                row["j7_accept"] == "0",
                row["j7_reason"]
                    == "delta_vx_too_large",
                row["j7_source"] == "empirical",
            ]

            if all(checks):
                rejected.append(row)
            else:
                invalid.append((index, row))

        else:
            invalid.append((index, row))

    except Exception as error:
        invalid.append(
            (
                index,
                {
                    "error": repr(error),
                    "row": row,
                },
            )
        )


if invalid:
    raise RuntimeError(
        "invalid J7 rows={} first={}".format(
            len(invalid),
            invalid[0],
        )
    )


if len(accepted) < 60:
    raise RuntimeError(
        "insufficient accepted rows: {}".format(
            len(accepted)
        )
    )


if len(rejected) < 60:
    raise RuntimeError(
        "insufficient rejected rows: {}".format(
            len(rejected)
        )
    )


print("[OK] total J7 rows={}".format(len(rows)))
print(
    "[OK] accepted vx=0.115 rows={}".format(
        len(accepted)
    )
)
print(
    "[OK] rejected vx=0.125 fallback rows={}".format(
        len(rejected)
    )
)
print("[OK] J7 sequence identity preserved")
print("[PASS] J7 VX ABBA routing")
PY_VALIDATE_VX_J7
}


run_interleaved_abba() {
  local trial="vx_interleaved_abba"
  local trial_dir="$RESULT_DIR/$trial"
  local j7_log_dir="$trial_dir/j7_log"

  local j7_csv
  j7_csv="$j7_log_dir/guarded_meta_action_ref_gate_j7_v0.csv"

  local realized_csv
  realized_csv="$trial_dir/vx_abba_realized.csv"

  local container_realized_csv
  container_realized_csv="/tmp/tracer_j7_vx_abba_realized.csv"

  local window_csv
  window_csv="$RESULT_DIR/j7_vx_abba_window_summary.csv"

  local summary_txt
  summary_txt="$RESULT_DIR/j7_vx_abba_summary.txt"

  mkdir -p \
    "$trial_dir" \
    "$j7_log_dir"

  echo
  echo "============================================================"
  echo "J7 VX INTERLEAVED ABBA AUTHORITY TEST"
  echo "A1: projected 0.125 rejected -> empirical 0.090"
  echo "B1: projected 0.115 accepted"
  echo "B2: projected 0.115 accepted"
  echo "A2: projected 0.125 rejected -> empirical 0.090"
  echo "body_h=0.320, clearance=0.045, yaw=0"
  echo "flat guard: base_x < 1.8"
  echo "single reset / controller / J7 process"
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
    "6400000" \
    "$trial_dir/j7_active_source.log" \
    "$j7_log_dir"

  docker exec \
    "$LL1_CONTAINER" \
    rm -f "$container_realized_csv"

  docker exec \
    -e TRACER_ABBA_START_EPOCH="$schedule_start_epoch" \
    -e TRACER_ABBA_CSV="$container_realized_csv" \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      set -eo pipefail

      source /opt/ros/melodic/setup.bash
      source /root/unitree_ws/devel/setup.bash
      source /root/A1_ctrl_ws/devel/setup.bash

      python - <<'"'"'PY_VX_OBSERVER'"'"'
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
FLAT_X_MAX = 1.8

WINDOWS = [
    {
        "name": "A1",
        "condition": "A",
        "start": 0.0,
        "end": 2.5,
        "vx": 0.090,
    },
    {
        "name": "B1",
        "condition": "B",
        "start": 3.0,
        "end": 5.5,
        "vx": 0.115,
    },
    {
        "name": "B2",
        "condition": "B",
        "start": 5.5,
        "end": 8.0,
        "vx": 0.115,
    },
    {
        "name": "A2",
        "condition": "A",
        "start": 8.5,
        "end": 11.0,
        "vx": 0.090,
    },
]

END_OFFSET = 11.0

latest_ref = [None]
latest_model = [None]


def on_ref(message):
    latest_ref[0] = list(message.data)


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


def finite(value):
    try:
        value = float(value)

        if (
            math.isnan(value)
            or math.isinf(value)
        ):
            return float("nan")

        return value

    except Exception:
        return float("nan")


rospy.init_node(
    "tracer_j7_vx_abba_observer",
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


reference_deadline = time.time() + 8.0
reference_ready = False


while time.time() < reference_deadline:
    reference = latest_ref[0]

    if (
        reference is not None
        and len(reference) == 6
        and abs(reference[1] - 0.090) <= 1e-12
        and abs(reference[2]) <= 1e-12
        and abs(reference[3] - 0.320) <= 1e-12
        and abs(reference[4] - 0.045) <= 1e-12
        and abs(reference[5] - 1.0) <= 1e-12
    ):
        reference_ready = True
        break

    time.sleep(0.02)


if not reference_ready:
    raise RuntimeError(
        "baseline VX reference did not preload: {}".format(
            latest_ref[0]
        )
    )


unpause()

print(
    "[OK] physics unpaused for model-state preload"
)


model_deadline = time.time() + 8.0
model_ready = False


while time.time() < model_deadline:
    models = latest_model[0]

    if (
        models is not None
        and MODEL_NAME in models.name
    ):
        model_ready = True
        break

    time.sleep(0.02)


if not model_ready:
    pause()

    raise RuntimeError(
        "Gazebo model state did not preload"
    )


print("[OK] Gazebo model state preloaded")


if time.time() >= START_EPOCH:
    pause()

    raise RuntimeError(
        "observer missed ABBA start epoch"
    )


while time.time() < START_EPOCH:
    models = latest_model[0]

    if (
        models is not None
        and MODEL_NAME in models.name
    ):
        model_index = models.name.index(
            MODEL_NAME
        )

        if (
            models.pose[model_index].position.x
            >= FLAT_X_MAX
        ):
            pause()

            raise RuntimeError(
                "robot left flat region before A1"
            )

    time.sleep(0.01)


rows = []
sample_counts = {
    window["name"]: 0
    for window in WINDOWS
}

reference_mismatches = {
    window["name"]: 0
    for window in WINDOWS
}

rate = rospy.Rate(50)


try:
    while not rospy.is_shutdown():
        wall_now = time.time()
        offset = wall_now - START_EPOCH

        if offset >= END_OFFSET:
            break

        models = latest_model[0]

        if (
            models is None
            or MODEL_NAME not in models.name
        ):
            rate.sleep()
            continue

        model_index = models.name.index(
            MODEL_NAME
        )

        pose = models.pose[model_index]
        twist = models.twist[model_index]

        if pose.position.x >= FLAT_X_MAX:
            raise RuntimeError(
                "robot left flat authority region: "
                "x={:.6f}".format(
                    pose.position.x
                )
            )

        window = window_for_offset(offset)

        if window is None:
            rate.sleep()
            continue

        reference = latest_ref[0]

        reference_matches = (
            reference is not None
            and len(reference) == 6
            and abs(
                reference[1]
                - window["vx"]
            ) <= 1e-12
            and abs(reference[2]) <= 1e-12
            and abs(reference[3] - 0.320) <= 1e-12
            and abs(reference[4] - 0.045) <= 1e-12
            and abs(reference[5] - 1.0) <= 1e-12
        )

        if not reference_matches:
            reference_mismatches[
                window["name"]
            ] += 1

            rate.sleep()
            continue

        roll, pitch, yaw = quaternion_to_euler(
            pose.orientation
        )

        vx_world = twist.linear.x
        vy_world = twist.linear.y

        vx_body = (
            math.cos(yaw) * vx_world
            + math.sin(yaw) * vy_world
        )

        vy_body = (
            -math.sin(yaw) * vx_world
            + math.cos(yaw) * vy_world
        )

        row = {
            "wall_time": wall_now,
            "sim_time": rospy.Time.now().to_sec(),
            "offset_s": offset,
            "window": window["name"],
            "condition": window["condition"],
            "ref_sequence": reference[0],
            "ref_vx": reference[1],
            "base_x": pose.position.x,
            "base_y": pose.position.y,
            "base_z": pose.position.z,
            "vx_world": vx_world,
            "vy_world": vy_world,
            "vx_body": vx_body,
            "vy_body": vy_body,
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
        }

        numeric_values = [
            value
            for key, value in row.items()
            if key not in (
                "window",
                "condition",
            )
        ]

        if all(
            not math.isnan(finite(value))
            for value in numeric_values
        ):
            rows.append(row)

            sample_counts[
                window["name"]
            ] += 1

        rate.sleep()

finally:
    pause()


fieldnames = [
    "wall_time",
    "sim_time",
    "offset_s",
    "window",
    "condition",
    "ref_sequence",
    "ref_vx",
    "base_x",
    "base_y",
    "base_z",
    "vx_world",
    "vy_world",
    "vx_body",
    "vy_body",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
]


if sys.version_info[0] < 3:
    stream = open(CSV_PATH, "wb")
else:
    stream = open(
        CSV_PATH,
        "w",
        newline="",
    )


with stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


print("[OK] observer rows={}".format(len(rows)))

for window in WINDOWS:
    name = window["name"]

    print(
        "[OK] {} samples={} ref_mismatches={}".format(
            name,
            sample_counts[name],
            reference_mismatches[name],
        )
    )

    if sample_counts[name] < 20:
        raise RuntimeError(
            "{} has insufficient samples: {}".format(
                name,
                sample_counts[name],
            )
        )

    if reference_mismatches[name] != 0:
        raise RuntimeError(
            "{} reference mismatches={}".format(
                name,
                reference_mismatches[name],
            )
        )


print("[PASS] VX ABBA observer")
PY_VX_OBSERVER
    '

  docker cp \
    "$LL1_CONTAINER:$container_realized_csv" \
    "$realized_csv"

  test -s "$realized_csv" || {
    echo "[ERROR] missing realized VX CSV"
    exit 11
  }

  validate_j7_abba_decisions "$j7_csv"

  /usr/bin/python3 - \
    "$realized_csv" \
    "$window_csv" \
    "$summary_txt" \
    <<'PY_ANALYZE_VX'
import csv
import math
import statistics
import sys


input_path, output_path, summary_path = (
    sys.argv[1:]
)


WINDOW_ORDER = [
    "A1",
    "B1",
    "B2",
    "A2",
]


SETTLED_RANGES = {
    "A1": (1.0, 2.4),
    "B1": (4.0, 5.4),
    "B2": (6.5, 7.9),
    "A2": (9.5, 10.9),
}


EXPECTED_VX = {
    "A1": 0.090,
    "B1": 0.115,
    "B2": 0.115,
    "A2": 0.090,
}


def finite(value):
    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


rows = []


with open(
    input_path,
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    for raw in csv.DictReader(stream):
        window = raw["window"]

        if window not in WINDOW_ORDER:
            continue

        row = {
            "window": window,
            "offset_s": finite(raw["offset_s"]),
            "ref_vx": finite(raw["ref_vx"]),
            "base_x": finite(raw["base_x"]),
            "base_y": finite(raw["base_y"]),
            "base_z": finite(raw["base_z"]),
            "vx_world": finite(raw["vx_world"]),
            "vy_world": finite(raw["vy_world"]),
            "vx_body": finite(raw["vx_body"]),
            "vy_body": finite(raw["vy_body"]),
            "roll_deg": finite(raw["roll_deg"]),
            "pitch_deg": finite(raw["pitch_deg"]),
            "yaw_deg": finite(raw["yaw_deg"]),
        }

        if any(
            row[key] is None
            for key in row
            if key != "window"
        ):
            continue

        if not math.isclose(
            row["ref_vx"],
            EXPECTED_VX[window],
            abs_tol=1e-9,
        ):
            continue

        start, end = SETTLED_RANGES[window]

        if not (
            start <= row["offset_s"] < end
        ):
            continue

        rows.append(row)


by_window = {
    name: []
    for name in WINDOW_ORDER
}


for row in rows:
    by_window[row["window"]].append(row)


summaries = []


for name in WINDOW_ORDER:
    window_rows = by_window[name]

    if len(window_rows) < 10:
        raise RuntimeError(
            "{} insufficient settled rows: {}".format(
                name,
                len(window_rows),
            )
        )

    vx_body = [
        row["vx_body"]
        for row in window_rows
    ]

    vx_world = [
        row["vx_world"]
        for row in window_rows
    ]

    vy_body = [
        abs(row["vy_body"])
        for row in window_rows
    ]

    roll = [
        abs(row["roll_deg"])
        for row in window_rows
    ]

    pitch = [
        abs(row["pitch_deg"])
        for row in window_rows
    ]

    ordered = sorted(
        window_rows,
        key=lambda row: row["offset_s"],
    )

    duration = (
        ordered[-1]["offset_s"]
        - ordered[0]["offset_s"]
    )

    dx = (
        ordered[-1]["base_x"]
        - ordered[0]["base_x"]
    )

    summary = {
        "window": name,
        "condition": (
            "A"
            if name.startswith("A")
            else "B"
        ),
        "expected_vx": EXPECTED_VX[name],
        "sample_count": len(window_rows),
        "median_vx_body": statistics.median(
            vx_body
        ),
        "mean_vx_body": statistics.mean(
            vx_body
        ),
        "std_vx_body": (
            statistics.stdev(vx_body)
            if len(vx_body) > 1
            else 0.0
        ),
        "median_vx_world": statistics.median(
            vx_world
        ),
        "median_abs_vy_body": statistics.median(
            vy_body
        ),
        "window_dx": dx,
        "window_duration": duration,
        "window_dx_rate": (
            dx / duration
            if duration > 1e-9
            else float("nan")
        ),
        "median_abs_roll_deg": statistics.median(
            roll
        ),
        "max_abs_roll_deg": max(roll),
        "median_abs_pitch_deg": statistics.median(
            pitch
        ),
        "max_abs_pitch_deg": max(pitch),
        "min_base_z": min(
            row["base_z"]
            for row in window_rows
        ),
        "max_base_x": max(
            row["base_x"]
            for row in window_rows
        ),
    }

    summaries.append(summary)


by_name = {
    row["window"]: row
    for row in summaries
}


a1 = by_name["A1"]["median_vx_body"]
b1 = by_name["B1"]["median_vx_body"]
b2 = by_name["B2"]["median_vx_body"]
a2 = by_name["A2"]["median_vx_body"]


centers = {
    "A1": 1.7,
    "B1": 4.7,
    "B2": 7.2,
    "A2": 10.2,
}


def interpolated_a(time_value):
    alpha = (
        (
            time_value
            - centers["A1"]
        )
        /
        (
            centers["A2"]
            - centers["A1"]
        )
    )

    return a1 + alpha * (a2 - a1)


b1_effect = (
    b1
    - interpolated_a(
        centers["B1"]
    )
)

b2_effect = (
    b2
    - interpolated_a(
        centers["B2"]
    )
)

combined_effect = statistics.mean(
    [b1_effect, b2_effect]
)

simple_effect = (
    statistics.mean([b1, b2])
    - statistics.mean([a1, a2])
)

command_effect = 0.025

realized_target_ratio = (
    combined_effect / command_effect
)


fieldnames = [
    "window",
    "condition",
    "expected_vx",
    "sample_count",
    "median_vx_body",
    "mean_vx_body",
    "std_vx_body",
    "median_vx_world",
    "median_abs_vy_body",
    "window_dx",
    "window_duration",
    "window_dx_rate",
    "median_abs_roll_deg",
    "max_abs_roll_deg",
    "median_abs_pitch_deg",
    "max_abs_pitch_deg",
    "min_base_z",
    "max_base_x",
]


with open(
    output_path,
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(summaries)


all_roll = [
    abs(row["roll_deg"])
    for row in rows
]

all_pitch = [
    abs(row["pitch_deg"])
    for row in rows
]

all_z = [
    row["base_z"]
    for row in rows
]

all_x = [
    row["base_x"]
    for row in rows
]


lines = [
    "TRACER J7 VX interleaved ABBA authority",
    "=========================================",
    "",
    "A command vx = 0.090 m/s",
    "B command vx = 0.115 m/s",
    "command effect m/s = {:+.9f}".format(
        command_effect
    ),
    "",
]


for row in summaries:
    lines.append(
        (
            "{window}: n={sample_count} "
            "median_vx_body={median_vx_body:.9f} "
            "mean_vx_body={mean_vx_body:.9f} "
            "std_vx_body={std_vx_body:.9f} "
            "dx_rate={window_dx_rate:.9f} "
            "median_abs_vy_body={median_abs_vy_body:.9f} "
            "max_abs_roll_deg={max_abs_roll_deg:.6f} "
            "max_abs_pitch_deg={max_abs_pitch_deg:.6f}"
        ).format(**row)
    )


lines.extend([
    "",
    "A endpoint velocity drift m/s = {:+.9f}".format(
        a2 - a1
    ),
    "B-half velocity drift m/s = {:+.9f}".format(
        b2 - b1
    ),
    "simple ABBA body-vx effect m/s = {:+.9f}".format(
        simple_effect
    ),
    "B1 drift-corrected effect m/s = {:+.9f}".format(
        b1_effect
    ),
    "B2 drift-corrected effect m/s = {:+.9f}".format(
        b2_effect
    ),
    "combined drift-corrected effect m/s = {:+.9f}".format(
        combined_effect
    ),
    "realized/target ratio = {:.6f}".format(
        realized_target_ratio
    ),
    "",
    "maximum settled x m = {:.6f}".format(
        max(all_x)
    ),
    "minimum settled base-z m = {:.6f}".format(
        min(all_z)
    ),
    "maximum settled |roll| deg = {:.6f}".format(
        max(all_roll)
    ),
    "maximum settled |pitch| deg = {:.6f}".format(
        max(all_pitch)
    ),
])


failures = []


if b1_effect <= 0.005:
    failures.append(
        "B1 did not produce more than 0.005 m/s "
        "positive realized body-vx response"
    )


if b2_effect <= 0.005:
    failures.append(
        "B2 did not produce more than 0.005 m/s "
        "positive realized body-vx response"
    )


if combined_effect <= 0.008:
    failures.append(
        "combined realized VX response did not exceed "
        "0.008 m/s"
    )


if max(all_x) >= 1.8:
    failures.append(
        "flat-region x guard was exceeded"
    )


if min(all_z) <= 0.20:
    failures.append(
        "gross base-height collapse observed"
    )


if max(all_roll) >= 20.0:
    failures.append(
        "gross roll excursion observed"
    )


if max(all_pitch) >= 20.0:
    failures.append(
        "gross pitch excursion observed"
    )


if failures:
    lines.extend([
        "",
        "[FAIL]",
    ])

    lines.extend(
        "- " + failure
        for failure in failures
    )
else:
    lines.extend([
        "",
        (
            "[EVIDENCE] J7-selected VX command reached "
            "LL1 and produced a repeatable positive "
            "realized body-frame velocity response."
        ),
        (
            "[LIMIT] This establishes VX command "
            "authority only, not speed optimality, "
            "terrain benefit, stability benefit, "
            "energy benefit, or policy performance."
        ),
    ])


text = "\n".join(lines) + "\n"


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as stream:
    stream.write(text)


print(text)


if failures:
    raise RuntimeError(
        "; ".join(failures)
    )
PY_ANALYZE_VX

  echo
  echo "===== VX RESULT FILES ====="

  ls -lh \
    "$realized_csv" \
    "$window_csv" \
    "$summary_txt" \
    "$j7_csv"

  echo
  echo "===== VX SUMMARY ====="

  cat "$summary_txt"

  grep -q \
    '^\[EVIDENCE\]' \
    "$summary_txt"

  echo "[PASS] VX ABBA authority evidence"
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
echo "[OK] complete one-reset interleaved J7 VX ABBA authority test passed"
echo "result dir: $RESULT_DIR"
