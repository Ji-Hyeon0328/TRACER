#!/usr/bin/env bash
set -euo pipefail

OUT_CSV=""
DURATION="45"
HZ="50"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-csv)
      OUT_CSV="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --hz)
      HZ="$2"
      shift 2
      ;;
    *)
      echo "[TRACER][ERROR] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$OUT_CSV" ]; then
  OUT_CSV="logs/phase_f/f1_ros1_true_metrics_$(date +%Y%m%d_%H%M%S).csv"
fi

mkdir -p "$(dirname "$OUT_CSV")"

TMP_PY="/tmp/tracer_phase_f1_ros1_true_metrics_logger_v0.py"
TMP_CSV="/tmp/tracer_phase_f1_ros1_true_metrics_v0.csv"

echo "[TRACER] Phase-F1 ROS1 true metric logger v0"
echo "  OUT_CSV=${OUT_CSV}"
echo "  DURATION=${DURATION}"
echo "  HZ=${HZ}"

echo "[TRACER] precheck ROS1 topics inside gazebo container"
docker exec a1_unitree_gazebo_docker bash -lc '
  source /opt/ros/melodic/setup.bash
  for t in /gazebo/model_states /a1_gazebo/joint_states /trunk_imu; do
    echo "[TRACER] checking message stream: $t"
    if ! timeout 5s rostopic echo -n 1 "$t" >/dev/null; then
      echo "[TRACER][ERROR] no ROS1 messages from topic: $t" >&2
      exit 3
    fi
  done
  echo "[TRACER] required ROS1 message streams are available"
'

docker exec -i a1_unitree_gazebo_docker bash -lc "cat > ${TMP_PY}" <<'PY'
from __future__ import print_function
import argparse
import csv
import math
import time

import rospy
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import WrenchStamped
from gazebo_msgs.msg import ModelStates

state = {
    "joint": None,
    "imu": None,
    "model_pose": None,
    "model_twist": None,
    "contacts": {
        "FL": None,
        "FR": None,
        "RL": None,
        "RR": None,
    },
}

def norm3(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def cb_joint(msg):
    state["joint"] = msg

def cb_imu(msg):
    state["imu"] = msg

def make_contact_cb(name):
    def cb(msg):
        state["contacts"][name] = msg
    return cb

def cb_model(msg):
    try:
        idx = msg.name.index("a1_gazebo")
    except ValueError:
        return
    state["model_pose"] = msg.pose[idx]
    state["model_twist"] = msg.twist[idx]

def safe(v):
    try:
        if v is None:
            return ""
        return float(v)
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--hz", type=float, default=50.0)
    args = ap.parse_args()

    rospy.init_node("tracer_phase_f1_ros1_true_metrics_logger_v0", anonymous=True)

    rospy.Subscriber("/a1_gazebo/joint_states", JointState, cb_joint, queue_size=1)
    rospy.Subscriber("/trunk_imu", Imu, cb_imu, queue_size=1)
    rospy.Subscriber("/gazebo/model_states", ModelStates, cb_model, queue_size=1)

    rospy.Subscriber("/visual/FL_foot_contact/the_force", WrenchStamped, make_contact_cb("FL"), queue_size=1)
    rospy.Subscriber("/visual/FR_foot_contact/the_force", WrenchStamped, make_contact_cb("FR"), queue_size=1)
    rospy.Subscriber("/visual/RL_foot_contact/the_force", WrenchStamped, make_contact_cb("RL"), queue_size=1)
    rospy.Subscriber("/visual/RR_foot_contact/the_force", WrenchStamped, make_contact_cb("RR"), queue_size=1)

    fields = [
        "t_wall", "t_ros",
        "base_x", "base_y", "base_z",
        "base_qx", "base_qy", "base_qz", "base_qw",
        "base_vx", "base_vy", "base_vz",
        "base_wx", "base_wy", "base_wz",
        "imu_qx", "imu_qy", "imu_qz", "imu_qw",
        "imu_wx", "imu_wy", "imu_wz",
        "imu_ax", "imu_ay", "imu_az",
        "joint_abs_power_sum",
        "joint_abs_effort_sum",
        "joint_sq_effort_sum",
        "joint_abs_velocity_sum",
        "joint_count",
        "contact_FL_fz", "contact_FR_fz", "contact_RL_fz", "contact_RR_fz",
        "contact_FL_norm", "contact_FR_norm", "contact_RL_norm", "contact_RR_norm",
        "contact_count_fz_gt_1",
        "contact_force_z_sum",
        "contact_force_norm_sum",
    ]

    f = open(args.out, "wb")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    start_wall = time.time()
    sleep_dt = 1.0 / max(args.hz, 1e-6)

    rows = 0
    while not rospy.is_shutdown():
        if args.duration > 0 and (time.time() - start_wall) >= args.duration:
            break

        row = dict((k, "") for k in fields)
        row["t_wall"] = time.time()
        row["t_ros"] = rospy.Time.now().to_sec()

        pose = state.get("model_pose")
        twist = state.get("model_twist")
        if pose is not None:
            row["base_x"] = safe(pose.position.x)
            row["base_y"] = safe(pose.position.y)
            row["base_z"] = safe(pose.position.z)
            row["base_qx"] = safe(pose.orientation.x)
            row["base_qy"] = safe(pose.orientation.y)
            row["base_qz"] = safe(pose.orientation.z)
            row["base_qw"] = safe(pose.orientation.w)

        if twist is not None:
            row["base_vx"] = safe(twist.linear.x)
            row["base_vy"] = safe(twist.linear.y)
            row["base_vz"] = safe(twist.linear.z)
            row["base_wx"] = safe(twist.angular.x)
            row["base_wy"] = safe(twist.angular.y)
            row["base_wz"] = safe(twist.angular.z)

        imu = state.get("imu")
        if imu is not None:
            row["imu_qx"] = safe(imu.orientation.x)
            row["imu_qy"] = safe(imu.orientation.y)
            row["imu_qz"] = safe(imu.orientation.z)
            row["imu_qw"] = safe(imu.orientation.w)
            row["imu_wx"] = safe(imu.angular_velocity.x)
            row["imu_wy"] = safe(imu.angular_velocity.y)
            row["imu_wz"] = safe(imu.angular_velocity.z)
            row["imu_ax"] = safe(imu.linear_acceleration.x)
            row["imu_ay"] = safe(imu.linear_acceleration.y)
            row["imu_az"] = safe(imu.linear_acceleration.z)

        js = state.get("joint")
        if js is not None:
            n = min(len(js.velocity), len(js.effort))
            abs_power = 0.0
            abs_effort = 0.0
            sq_effort = 0.0
            abs_vel = 0.0
            for i in range(n):
                v = js.velocity[i]
                e = js.effort[i]
                abs_power += abs(e * v)
                abs_effort += abs(e)
                sq_effort += e * e
                abs_vel += abs(v)
            row["joint_abs_power_sum"] = abs_power
            row["joint_abs_effort_sum"] = abs_effort
            row["joint_sq_effort_sum"] = sq_effort
            row["joint_abs_velocity_sum"] = abs_vel
            row["joint_count"] = n

        contact_count = 0
        force_z_sum = 0.0
        force_norm_sum = 0.0
        for name in ["FL", "FR", "RL", "RR"]:
            msg = state["contacts"].get(name)
            if msg is None:
                continue
            fx = msg.wrench.force.x
            fy = msg.wrench.force.y
            fz = msg.wrench.force.z
            fn = norm3(fx, fy, fz)
            row["contact_%s_fz" % name] = fz
            row["contact_%s_norm" % name] = fn
            if abs(fz) > 1.0:
                contact_count += 1
            force_z_sum += abs(fz)
            force_norm_sum += fn

        row["contact_count_fz_gt_1"] = contact_count
        row["contact_force_z_sum"] = force_z_sum
        row["contact_force_norm_sum"] = force_norm_sum

        writer.writerow(row)
        rows += 1
        time.sleep(sleep_dt)

    f.close()
    print("[TRACER] wrote %s rows to %s" % (rows, args.out))

if __name__ == "__main__":
    main()
PY

docker exec a1_unitree_gazebo_docker bash -lc "
  source /opt/ros/melodic/setup.bash
  source /root/unitree_ws/devel/setup.bash 2>/dev/null || true
  python ${TMP_PY} --out ${TMP_CSV} --duration ${DURATION} --hz ${HZ}
"

docker cp "a1_unitree_gazebo_docker:${TMP_CSV}" "$OUT_CSV"

echo "[TRACER] copied to ${OUT_CSV}"

python3 - "$OUT_CSV" <<'PY'
import csv
import sys
from statistics import mean

p = sys.argv[1]
rows = list(csv.DictReader(open(p)))

def vals(name):
    out = []
    for r in rows:
        try:
            if r.get(name, "") != "":
                out.append(float(r[name]))
        except Exception:
            pass
    return out

print("[TRACER] rows:", len(rows))
for k in [
    "base_vx",
    "base_vy",
    "joint_abs_power_sum",
    "joint_abs_effort_sum",
    "imu_wx",
    "imu_wy",
    "imu_wz",
    "contact_force_z_sum",
    "contact_force_norm_sum",
]:
    v = vals(k)
    if not v:
        print(f"[TRACER] {k}: no samples")
    else:
        print(f"[TRACER] {k}: mean={mean(v):.6f}, min={min(v):.6f}, max={max(v):.6f}")
PY
