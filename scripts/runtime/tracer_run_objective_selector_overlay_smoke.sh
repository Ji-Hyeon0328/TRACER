#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-flat_normal}"
DURATION="${2:-8.0}"

LOG="/tmp/tracer_fusion_policy_mpc_ref_${TERRAIN}.log"
MPC_OUT="/tmp/tracer_objective_selector_smoke_mpc_ref.yaml"
BETA_OUT="/tmp/tracer_objective_selector_smoke_objective_weights.yaml"

cd "$ROOT"

set +u
source /opt/ros/humble/setup.bash
set -u

echo "[TRACER][SMOKE] objective selector overlay smoke"
echo "[TRACER][SMOKE] terrain=$TERRAIN duration=$DURATION"

pkill -f tracer_fusion_policy_mpc_ref_node.py || true
pkill -f tracer_ram_gate_monitor_node.py || true
pkill -f tracer_online_ram_udp_client.py || true
sleep 0.5

TRACER_OBJECTIVE_SELECTOR_VERBOSE="${TRACER_OBJECTIVE_SELECTOR_VERBOSE:-0}" \
TRACER_GMS_GATE_FRESHNESS_SEC="${TRACER_GMS_GATE_FRESHNESS_SEC:-5.0}" \
scripts/runtime/tracer_start_fusion_policy_overlay_from_qwerty.sh "$TERRAIN" "$DURATION"

sleep 1.0

echo "[TRACER][SMOKE] publish fake risky RAM gate"
python3 scripts/runtime/tracer_publish_fake_ram_gate_advice.py \
  --level unstable \
  --action would_conservative_probe \
  --would-override 1.0 \
  --control-risk 0.80 \
  --fallen-prob 0.50 \
  --recovery-prob 0.70 \
  --duration 2.0 \
  --hz 5.0

sleep 0.5

echo "[TRACER][SMOKE] read /tracer/mpc_reference"
timeout 5.0 ros2 topic echo /tracer/mpc_reference --once > "$MPC_OUT"

echo "[TRACER][SMOKE] read /tracer/objective_weights"
timeout 5.0 ros2 topic echo /tracer/objective_weights --once > "$BETA_OUT"

python3 - <<'PY'
from pathlib import Path
import math
import re
import sys

mpc_path = Path("/tmp/tracer_objective_selector_smoke_mpc_ref.yaml")
beta_path = Path("/tmp/tracer_objective_selector_smoke_objective_weights.yaml")

def parse_data(path: Path) -> list[float]:
    vals = []
    in_data = False

    for line in path.read_text().splitlines():
        stripped = line.strip()

        if stripped == "data:":
            in_data = True
            continue

        if not in_data:
            continue

        # ros2 topic echo appends YAML separators like "---".
        if stripped == "---":
            if vals:
                break
            continue

        # Parse YAML list entries like "- 0.28".
        if stripped.startswith("- "):
            token = stripped[2:].strip()
            try:
                vals.append(float(token))
            except ValueError:
                # Ignore non-numeric YAML/list artifacts.
                if vals:
                    break
            continue

        # Stop once the data block ended after collecting values.
        if vals:
            break

    return vals

mpc = parse_data(mpc_path)
beta = parse_data(beta_path)

print("[TRACER][SMOKE] mpc_reference =", mpc)
print("[TRACER][SMOKE] objective_weights =", beta)

errors = []

if len(mpc) < 6:
    errors.append(f"mpc_reference too short: {mpc}")
else:
    if abs(mpc[1]) > 1e-6:
        errors.append(f"expected vx=0.0 after risky gate, got {mpc[1]}")
    if abs(mpc[2]) > 1e-6:
        errors.append(f"expected yaw_rate=0.0 after risky gate, got {mpc[2]}")
    if abs(mpc[5]) > 1e-6:
        errors.append(f"expected enable=0.0 after risky gate, got {mpc[5]}")

if len(beta) < 3:
    errors.append(f"objective_weights too short: {beta}")
else:
    expected = [0.09935897435897437, 0.7506410256410256, 0.15]
    for i, (got, exp) in enumerate(zip(beta[:3], expected)):
        if abs(got - exp) > 1e-6:
            errors.append(f"expected beta[{i}]={exp}, got {got}")

if errors:
    print("[TRACER][SMOKE][FAIL]")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("[TRACER][SMOKE][PASS] risky gate blocked forward deployment")
PY

echo
echo "[TRACER][SMOKE] relevant fusion log:"
grep -E "objective_selector kind|GMS enable|publishing /tracer/mpc_reference" "$LOG" | tail -n 20 || true
