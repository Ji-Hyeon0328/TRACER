#!/usr/bin/env bash
set -Eeo pipefail

ROOT="$HOME/Tracer/TRACER"
AUDIT_ROOT="$HOME/Tracer/TRACER-lowcontroller-audit"

LL1_CONTAINER="a1_cpp_ctrl_ll1_docker"
GAZEBO_CONTAINER="a1_unitree_gazebo_docker"

ROS2_SENDER="$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_ros2_mpc_ref_udp_sender.py"
ROS1_RECEIVER="/root/A1_ctrl_ws/src/a1_cpp/scripts/tracer_udp_to_ros1_mpc_ref.py"

SENDER_PID=""
PUBLISHER_PID=""
CLEANUP_DONE=false

RUN_TAG="tracer_ll1_e2e_authority_$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="/tmp/$RUN_TAG"

mkdir -p "$RESULT_DIR"

echo "result dir: $RESULT_DIR"


source_ros2() {
  set +u
  source /opt/ros/humble/setup.bash
  set -u
}


stop_fixed_publisher() {
  set +e

  if [[ -n "$PUBLISHER_PID" ]] &&
     kill -0 "$PUBLISHER_PID" 2>/dev/null
  then
    kill -INT "$PUBLISHER_PID" \
      >/dev/null 2>&1 \
      || true

    for _ in $(seq 1 30); do
      if ! kill -0 "$PUBLISHER_PID" 2>/dev/null; then
        break
      fi

      sleep 0.1
    done

    kill -9 "$PUBLISHER_PID" \
      >/dev/null 2>&1 \
      || true
  fi

  PUBLISHER_PID=""

  pkill -9 -f \
    "/tmp/tracer_ll1_e2e_fixed_ref_pub.py" \
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

  stop_fixed_publisher
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

  echo "[OK] fixed publisher stopped"
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

stop_fixed_publisher
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
echo "===== INSTALL ROS2 FIXED PUBLISHER ====="

cat > /tmp/tracer_ll1_e2e_fixed_ref_pub.py <<'PY'
#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class FixedReferencePublisher(Node):
    def __init__(self):
        super().__init__(
            "tracer_ll1_e2e_fixed_ref_pub"
        )

        self.clearance = float(
            os.environ["TRACER_E2E_CLEARANCE"]
        )

        self.counter = float(
            os.environ.get(
                "TRACER_E2E_COUNTER_BASE",
                "1000000",
            )
        )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            "/tracer/mpc_reference",
            10,
        )

        self.timer = self.create_timer(
            0.05,
            self.publish_reference,
        )

        self.get_logger().info(
            "publishing fixed clearance={:.3f}".format(
                self.clearance
            )
        )

    def publish_reference(self):
        message = Float64MultiArray()

        message.data = [
            self.counter,
            0.0,
            0.0,
            0.32,
            self.clearance,
            1.0,
        ]

        self.publisher.publish(message)
        self.counter += 1.0


def main():
    rclpy.init()
    node = FixedReferencePublisher()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
PY

chmod +x \
  /tmp/tracer_ll1_e2e_fixed_ref_pub.py


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


start_fixed_publisher() {
  local clearance="$1"
  local counter_base="$2"
  local log="$3"

  stop_fixed_publisher
  source_ros2

  TRACER_E2E_CLEARANCE="$clearance" \
  TRACER_E2E_COUNTER_BASE="$counter_base" \
    /usr/bin/python3 \
      /tmp/tracer_ll1_e2e_fixed_ref_pub.py \
      > "$log" \
      2>&1 &

  PUBLISHER_PID=$!

  sleep 1

  if ! kill -0 "$PUBLISHER_PID" 2>/dev/null; then
    echo "[ERROR] fixed publisher failed"
    cat "$log"
    exit 8
  fi

  local publisher_count

  publisher_count="$(
    ros2 topic info \
      /tracer/mpc_reference \
      2>/dev/null \
      | awk '
          /Publisher count:/ {
            print $3
          }
        '
  )"

  echo "ROS2 publisher count=$publisher_count"

  if [[ "$publisher_count" != "1" ]]; then
    echo "[ERROR] expected exactly one ROS2 publisher"
    ros2 topic info \
      /tracer/mpc_reference \
      --verbose \
      || true
    exit 9
  fi
}


run_trial() {
  local trial="$1"
  local clearance="$2"
  local expected_delta="$3"
  local counter_base="$4"

  echo
  echo "============================================================"
  echo "TRIAL: $trial"
  echo "clearance=$clearance"
  echo "expected_delta=$expected_delta"
  echo "============================================================"

  stop_fixed_publisher
  stop_controller

  reset_standing_pose
  start_controller "$trial"

  start_fixed_publisher \
    "$clearance" \
    "$counter_base" \
    "$RESULT_DIR/${trial}_ros2_publisher.log"

  docker exec \
    -e TRACER_TRIAL="$trial" \
    -e TRACER_CLEARANCE="$clearance" \
    -e TRACER_EXPECTED_DELTA="$expected_delta" \
    "$LL1_CONTAINER" \
    bash --noprofile --norc -lc '
      set -eo pipefail

      source /opt/ros/melodic/setup.bash
      source /root/unitree_ws/devel/setup.bash
      source /root/A1_ctrl_ws/devel/setup.bash

      python - <<'"'"'PY'"'"'
from __future__ import print_function

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
duration = 5.0
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
    "[OK] ROS2 clearance {:.3f} reached LL1 as "
    "{:+.3f} m swing-apex residual".format(
        CLEARANCE,
        EXPECTED_DELTA,
    )
)
PY
    '

  stop_fixed_publisher
  stop_controller
}


run_trial \
  "neutral_045" \
  "0.045" \
  "0.000" \
  "2000000"

run_trial \
  "high_060" \
  "0.060" \
  "0.015" \
  "3000000"


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
  '[t]racer_ll1_e2e_fixed_ref_pub.py|[t]racer_ros2_mpc_ref_udp_sender.py' \
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
echo "[OK] complete ROS2-to-LL1 authority sweep passed"
echo "result dir: $RESULT_DIR"
