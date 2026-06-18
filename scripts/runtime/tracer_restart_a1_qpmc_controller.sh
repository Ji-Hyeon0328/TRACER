#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

echo "[TRACER] restart A1-QP-MPC controller with TRACER patch"
echo "[TRACER] controller container: $CTRL_CONTAINER"

echo
echo "[TRACER] checking source patch..."
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
set +e
grep -R "/tracer/mpc_reference\|tracer_swing_clearance\|tracer_mpc_reference_callback" \
  /root/A1_ctrl_ws/src/A1_Ctrl/src/a1_cpp \
  -n | head -80
'

echo
echo "[TRACER] rebuilding a1_cpp..."
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
cd /root/A1_ctrl_ws
catkin build a1_cpp
source /root/A1_ctrl_ws/devel/setup.bash
ls -l /root/A1_ctrl_ws/devel/lib/a1_cpp/gazebo_a1_ctrl
'

echo
echo "[TRACER] stopping old gazebo_a1_ctrl / a1_ctrl roslaunch safely..."
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
python - <<PY
import os
import signal
import time

targets = []

for pid_s in os.listdir("/proc"):
    if not pid_s.isdigit():
        continue

    pid = int(pid_s)

    try:
        with open("/proc/%d/comm" % pid, "r") as f:
            comm = f.read().strip()
    except Exception:
        continue

    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            cmd = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
    except Exception:
        cmd = ""

    # Exact binary name: safe, does not match this shell.
    if comm == "gazebo_a1_ctrl":
        targets.append((pid, comm, cmd))
        continue

    # roslaunch process is usually python with /opt/ros/.../roslaunch in cmdline.
    # Require both real roslaunch executable path and a1_ctrl.launch.
    if "/opt/ros/melodic/bin/roslaunch" in cmd and "a1_ctrl.launch" in cmd:
        targets.append((pid, comm, cmd))
        continue

print("[TRACER:container] targets:")
for pid, comm, cmd in targets:
    print("  pid=%d comm=%s cmd=%s" % (pid, comm, cmd[:180]))

for pid, comm, cmd in targets:
    try:
        os.kill(pid, signal.SIGINT)
        print("[TRACER:container] SIGINT pid=%d comm=%s" % (pid, comm))
    except Exception as e:
        print("[TRACER:container] SIGINT failed pid=%d: %s" % (pid, e))

time.sleep(2.0)

for pid, comm, cmd in targets:
    if os.path.exists("/proc/%d" % pid):
        try:
            os.kill(pid, signal.SIGKILL)
            print("[TRACER:container] SIGKILL pid=%d comm=%s" % (pid, comm))
        except Exception as e:
            print("[TRACER:container] SIGKILL failed pid=%d: %s" % (pid, e))
PY

echo "[TRACER:container] remaining controller processes:"
ps -eo pid,comm,args | awk '\''$2=="gazebo_a1_ctrl" || ($0 ~ /\/opt\/ros\/melodic\/bin\/roslaunch/ && $0 ~ /a1_ctrl.launch/) {print}'\'' || true
'

echo
echo "[TRACER] starting a1_ctrl.launch..."
sudo docker exec -d "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
roslaunch a1_cpp a1_ctrl.launch type:=gazebo solver_type:=mpc > /tmp/tracer_a1_ctrl.launch.log 2>&1
'

echo
echo "[TRACER] waiting controller startup..."
sleep 7

echo
echo "[TRACER] controller process check:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
ps -eo pid,comm,args | awk '\''$2=="gazebo_a1_ctrl" || ($0 ~ /\/opt\/ros\/melodic\/bin\/roslaunch/ && $0 ~ /a1_ctrl.launch/) {print}'\'' || true
'

echo
echo "[TRACER] ROS node check:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rosnode list | grep -E "gazebo_a1_ctrl|a1" || true
'

echo
echo "[TRACER] launch log tail:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
tail -n 160 /tmp/tracer_a1_ctrl.launch.log 2>/dev/null || true
'

echo
echo "[TRACER] topic subscriber check:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic info /tracer/mpc_reference || true
'
