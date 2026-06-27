#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
DURATION="${1:-8.0}"

cd "$ROOT"

set +u
source /opt/ros/humble/setup.bash
set -u

parse_yaml_data_py='
from pathlib import Path
import sys

path = Path(sys.argv[1])
vals = []
in_data = False

for line in path.read_text().splitlines():
    stripped = line.strip()

    if stripped == "data:":
        in_data = True
        continue

    if not in_data:
        continue

    if stripped == "---":
        if vals:
            break
        continue

    if stripped.startswith("- "):
        token = stripped[2:].strip()
        try:
            vals.append(float(token))
        except ValueError:
            if vals:
                break
        continue

    if vals:
        break

print(" ".join(str(v) for v in vals))
'

run_case() {
  local terrain="$1"
  local exp_vx="$2"
  local exp_h="$3"
  local exp_clr="$4"
  local exp_beta_v="$5"
  local exp_beta_s="$6"
  local exp_beta_e="$7"
  local exp_semantic="$8"

  local log="/tmp/tracer_fusion_policy_mpc_ref_${terrain}.log"
  local mpc_out="/tmp/tracer_clean_routing_${terrain}_mpc.yaml"
  local beta_out="/tmp/tracer_clean_routing_${terrain}_beta.yaml"

  echo
  echo "========== CLEAN ROUTING SMOKE: ${terrain} =========="

  pkill -f tracer_fusion_policy_mpc_ref_node.py || true
  sleep 0.5

  TRACER_OBJECTIVE_SELECTOR_VERBOSE=0 \
  TRACER_GMS_GATE_FRESHNESS_SEC=2.0 \
  scripts/runtime/tracer_start_fusion_policy_overlay_from_qwerty.sh "$terrain" "$DURATION"

  sleep 1.0

  timeout 3.0 ros2 topic echo /tracer/mpc_reference --once > "$mpc_out"
  timeout 3.0 ros2 topic echo /tracer/objective_weights --once > "$beta_out"

  local mpc_vals
  local beta_vals
  mpc_vals="$(python3 -c "$parse_yaml_data_py" "$mpc_out")"
  beta_vals="$(python3 -c "$parse_yaml_data_py" "$beta_out")"

  echo "[TRACER][SMOKE] mpc_reference: $mpc_vals"
  echo "[TRACER][SMOKE] objective_weights: $beta_vals"

  python3 - "$terrain" "$exp_vx" "$exp_h" "$exp_clr" "$exp_beta_v" "$exp_beta_s" "$exp_beta_e" "$mpc_vals" "$beta_vals" <<'PY'
import sys

terrain = sys.argv[1]
exp = {
    "vx": float(sys.argv[2]),
    "h": float(sys.argv[3]),
    "clr": float(sys.argv[4]),
    "beta_v": float(sys.argv[5]),
    "beta_s": float(sys.argv[6]),
    "beta_e": float(sys.argv[7]),
}
mpc = [float(x) for x in sys.argv[8].split()] if sys.argv[8] else []
beta = [float(x) for x in sys.argv[9].split()] if sys.argv[9] else []

errors = []
tol = 1e-6

if len(mpc) < 6:
    errors.append(f"mpc_reference too short: {mpc}")
else:
    checks = [
        ("vx", mpc[1], exp["vx"]),
        ("yaw", mpc[2], 0.0),
        ("body_height", mpc[3], exp["h"]),
        ("clearance", mpc[4], exp["clr"]),
        ("enable", mpc[5], 1.0),
    ]
    for name, got, want in checks:
        if abs(got - want) > tol:
            errors.append(f"{name}: expected {want}, got {got}")

if len(beta) < 3:
    errors.append(f"objective_weights too short: {beta}")
else:
    checks = [
        ("beta_v", beta[0], exp["beta_v"]),
        ("beta_s", beta[1], exp["beta_s"]),
        ("beta_e", beta[2], exp["beta_e"]),
    ]
    for name, got, want in checks:
        if abs(got - want) > tol:
            errors.append(f"{name}: expected {want}, got {got}")

if errors:
    print(f"[TRACER][SMOKE][FAIL] {terrain}")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print(f"[TRACER][SMOKE][PASS] {terrain}")
PY

  echo "[TRACER][SMOKE] key log lines:"
  grep -E "policy_json|terrain=.*semantic|publishing /tracer/mpc_reference" "$log" | tail -n 8 || true

  if ! grep -q "semantic=${exp_semantic}" "$log"; then
    echo "[TRACER][SMOKE][FAIL] expected semantic=${exp_semantic} in $log"
    exit 1
  fi
}

run_case flat_normal 0.28 0.295 0.03 0.55 0.25 0.20 validated_locomotion
run_case rough_mid 0.055 0.305 0.05 0.30 0.50 0.20 cautious_probe
run_case slope_5deg 0.045 0.315 0.06 0.25 0.55 0.20 high_clearance_slow_probe

echo
echo "[TRACER][SMOKE][PASS] clean objective selector routing for flat/rough/slope"
