#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
OUT_DIR="${TRACER_ROLLOUT_OUT_DIR:-$ROOT/data/rollout_dataset_v0}"
DURATION="${TRACER_ROLLOUT_DURATION_SEC:-30.0}"
SAMPLE_HZ="${TRACER_ROLLOUT_SAMPLE_HZ:-5.0}"
REPEATS="${TRACER_ROLLOUT_REPEATS:-1}"
TERRAINS="${TRACER_ROLLOUT_TERRAINS:-flat_normal rough_mid slope_5deg}"
POLICY_ID="${TRACER_POLICY_ID:-theta_mlp_udp_v0}"

cd "$ROOT"

set +u
source /opt/ros/humble/setup.bash
set -u

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries"

gazebo_physics() {
  local action="$1"
  local service="/gazebo/${action}_physics"

  docker exec a1_unitree_gazebo_docker bash -lc "
    source /opt/ros/melodic/setup.bash
    source /root/unitree_ws/devel/setup.bash
    timeout 5 rosservice call ${service} '{}' >/tmp/tracer_${action}_physics.log 2>&1
  " || {
    echo "[TRACER][WARN] failed to ${action} Gazebo physics via ${service}"
    docker exec a1_unitree_gazebo_docker bash -lc "cat /tmp/tracer_${action}_physics.log 2>/dev/null || true" || true
  }
}


echo "[TRACER] rollout dataset runner"
echo "[TRACER] out_dir:   $OUT_DIR"
echo "[TRACER] terrains:  $TERRAINS"
echo "[TRACER] repeats:   $REPEATS"
echo "[TRACER] duration:  $DURATION"
echo "[TRACER] policy_id: $POLICY_ID"

echo
echo "[TRACER] checking theta UDP server on 50310"
if ! ss -lunp | grep -q ':50310'; then
  echo "[TRACER][WARN] UDP 50310 does not appear to be listening."
  echo "[TRACER][WARN] Start scripts/runtime/tracer_theta_policy_mlp_udp_server_v0.py first."
fi
ss -lunp | grep ':50310' || true

for repeat in $(seq 1 "$REPEATS"); do
  for terrain in $TERRAINS; do
    ts="$(date +%Y%m%d_%H%M%S)"
    ep_id="${terrain}_${POLICY_ID}_r${repeat}_${ts}"

    echo
    echo "========== ROLLOUT $ep_id =========="

    pkill -f tracer_online_ram_udp_client_node.py || true
    pkill -f tracer_ram_gate_monitor_node.py || true
    pkill -f tracer_fusion_policy_mpc_ref_node.py || true
    pkill -f tracer_record_rollout_episode_v0.py || true
    sleep 0.5

    scripts/runtime/tracer_start_online_ram_client_v0.sh
    scripts/runtime/tracer_start_ram_gate_monitor_v0.sh "$terrain"

    TRACER_OBJECTIVE_SELECTOR_VERBOSE="${TRACER_OBJECTIVE_SELECTOR_VERBOSE:-0}" \
    TRACER_META_GAIT_POLICY_KIND="${TRACER_META_GAIT_POLICY_KIND:-udp}" \
    TRACER_META_GAIT_POLICY_UDP_HOST="${TRACER_META_GAIT_POLICY_UDP_HOST:-127.0.0.1}" \
    TRACER_META_GAIT_POLICY_UDP_PORT="${TRACER_META_GAIT_POLICY_UDP_PORT:-50310}" \
    TRACER_META_GAIT_POLICY_UDP_TIMEOUT_SEC="${TRACER_META_GAIT_POLICY_UDP_TIMEOUT_SEC:-0.20}" \
    TRACER_GMS_GATE_FRESHNESS_SEC="${TRACER_GMS_GATE_FRESHNESS_SEC:-2.0}" \
    scripts/runtime/tracer_start_fusion_policy_overlay_from_qwerty.sh "$terrain" "$DURATION"

    echo "[TRACER] unpause Gazebo physics for rollout"
    gazebo_physics unpause
    sleep 0.5

    REC_TIMEOUT_SEC="$(/usr/bin/python3 -c "d=float('$DURATION'); print(max(10.0, d + 8.0))")"

    set +e
    TRACER_TERRAIN="$terrain" \
    TRACER_POLICY_ID="$POLICY_ID" \
    TRACER_EPISODE_ID="$ep_id" \
    TRACER_ROLLOUT_DURATION_SEC="$DURATION" \
    TRACER_ROLLOUT_SAMPLE_HZ="$SAMPLE_HZ" \
    TRACER_ROLLOUT_OUT_DIR="$OUT_DIR" \
    timeout --kill-after=2s "${REC_TIMEOUT_SEC}s" \
    /usr/bin/python3 scripts/runtime/tracer_record_rollout_episode_v0.py
    rec_status=$?
    set -e

    echo "[TRACER] pause Gazebo physics after rollout"
    gazebo_physics pause

    if [ "$rec_status" -ne 0 ]; then
      echo "[TRACER][ERROR] rollout recorder failed with status=$rec_status"
      exit "$rec_status"
    fi

    echo "[TRACER] episode summary:"
    cat "$OUT_DIR/summaries/${ep_id}.json" || true
  done
done

echo
echo "[TRACER] rollout dataset collection finished"
echo "[TRACER] summaries:"
ls -lh "$OUT_DIR/summaries" | tail -n 20
