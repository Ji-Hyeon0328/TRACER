#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

POLICY_JSON="${TRACER_FUSION_POLICY_JSON:-$ROOT/configs/highlevel_policy/tracer_fusion_policy_v1.json}"
STAMP="$(date +%Y%m%d_%H%M%S)"
PREFIX="${TRACER_GATE_EVAL_PREFIX:-gate_eval_v2_${STAMP}}"

RAM_MONITOR_CSV="${TRACER_RAM_MONITOR_CSV:-$ROOT/data/sanity_results/tracer_ram_monitor_results_v2.csv}"
RAM_GATE_CSV="${TRACER_RAM_GATE_CSV:-$ROOT/data/sanity_results/tracer_ram_gate_monitor_results.csv}"
JOINED_OUT_CSV="${TRACER_JOINED_OUT_CSV:-$ROOT/data/sanity_results/tracer_fusion_ram_gate_joined_${PREFIX}.csv}"

echo "============================================================"
echo "[TRACER] gate eval v2 batch"
echo "============================================================"
echo "[TRACER] root:       $ROOT"
echo "[TRACER] policy:     $POLICY_JSON"
echo "[TRACER] prefix:     $PREFIX"
echo "[TRACER] ram csv:    $RAM_MONITOR_CSV"
echo "[TRACER] gate csv:   $RAM_GATE_CSV"
echo "[TRACER] joined csv: $JOINED_OUT_CSV"
echo "============================================================"

echo
echo "========== RAM server check =========="
if pgrep -af "tracer_ram_udp_server_v0.py" >/dev/null 2>&1; then
  echo "[TRACER] RAM UDP server already running:"
  pgrep -af "tracer_ram_udp_server_v0.py" || true
else
  echo "[TRACER] RAM UDP server not found. Starting it with current python."
  nohup python3 "$ROOT/scripts/runtime/tracer_ram_udp_server_v0.py" \
    > /tmp/tracer_ram_udp_server_v0.log 2>&1 &
  sleep 2
  echo "[TRACER] RAM server process:"
  pgrep -af "tracer_ram_udp_server_v0.py" || true
  echo "[TRACER] RAM server log tail:"
  tail -n 20 /tmp/tracer_ram_udp_server_v0.log || true
fi

run_one() {
  local terrain="$1"
  local world="$2"
  local duration="$3"
  local tag="$4"

  echo
  echo "============================================================"
  echo "[TRACER] gate eval case: $tag"
  echo "============================================================"
  echo "[TRACER] terrain:  $terrain"
  echo "[TRACER] world:    $world"
  echo "[TRACER] duration: $duration"
  echo "[TRACER] tag:      $tag"

  TRACER_ENABLE_RAM_MONITOR=1 \
  TRACER_ENABLE_RAM_GATE_MONITOR=1 \
  TRACER_RAM_REQUIRE_ODOM=1 \
  TRACER_GATE_SKIP_INITIAL_MSGS="${TRACER_GATE_SKIP_INITIAL_MSGS:-3}" \
  TRACER_FUSION_POLICY_JSON="$POLICY_JSON" \
  TRACER_RAM_RESULT_CSV="$RAM_MONITOR_CSV" \
  TRACER_RAM_GATE_RESULT_CSV="$RAM_GATE_CSV" \
  "$ROOT/scripts/runtime/tracer_run_fusion_policy_sanity_once.sh" \
    "$terrain" \
    "$world" \
    "$duration" \
    "$tag"
}

run_one \
  "flat_normal" \
  "earth" \
  "15" \
  "${PREFIX}_flat_normal"

run_one \
  "sponge_firm_downslope_5deg_forward" \
  "tracer_sponge_firm_downslope_5deg" \
  "15" \
  "${PREFIX}_sponge_firm_downslope"

run_one \
  "slippery_mid_downslope_5deg_forward_postfix" \
  "tracer_slippery_mid_downslope_5deg" \
  "12" \
  "${PREFIX}_slippery_mid_downslope_avoid"

echo
echo "============================================================"
echo "[TRACER] joining fusion + RAM + gate"
echo "============================================================"

TRACER_RAM_MONITOR_CSV="$RAM_MONITOR_CSV" \
TRACER_RAM_GATE_CSV="$RAM_GATE_CSV" \
TRACER_JOINED_OUT_CSV="$JOINED_OUT_CSV" \
TRACER_JOIN_FILTER_PREFIX="$PREFIX" \
"$ROOT/scripts/runtime/tracer_join_fusion_ram_gate_results.py"

echo
echo "============================================================"
echo "[TRACER] compact gate eval table"
echo "============================================================"

PREFIX="$PREFIX" JOINED_OUT_CSV="$JOINED_OUT_CSV" python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd

prefix = os.environ["PREFIX"]
p = Path(os.environ["JOINED_OUT_CSV"])
df = pd.read_csv(p)

df = df[df["sanity_tag"].astype(str).str.startswith(prefix)].copy()

cols = [
    "sanity_tag",
    "terrain_key_fusion",
    "fused_mode",
    "semantic_mode",
    "decision",
    "valid_locomotion_candidate",
    "stable_hold_candidate",
    "fallen_relative_z_based",
    "delta_x",
    "delta_y",
    "min_rel_z",
    "ram_level_last",
    "action_last",
    "would_override_mean",
    "unstable_frac",
    "ctrl_ema_tail5_mean_gate",
    "fallen_tail5_mean_gate",
    "sigma_tail5_mean_gate",
]

cols = [c for c in cols if c in df.columns]

if df.empty:
    print(f"[TRACER] no joined rows found for prefix={prefix}")
else:
    print(df[cols].to_string(index=False))
PY

echo
echo "[TRACER] gate eval v2 done"
echo "[TRACER] joined csv: $JOINED_OUT_CSV"
