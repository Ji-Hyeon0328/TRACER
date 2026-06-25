#!/usr/bin/env python3
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = ROOT / "configs/highlevel_policy/tracer_fusion_policy_v1.json"
dst = ROOT / "configs/highlevel_policy/tracer_fusion_policy_slippery_direct_primitive_sweep_v0.json"

d = json.loads(src.read_text())

d["name"] = "tracer_fusion_policy_slippery_direct_primitive_sweep_v0"
d["policy_version"] = "slippery_direct_primitive_sweep_v0"
d["base_policy"] = str(src)
d["note"] = (
    "Direct primitive sweep policy. This file intentionally creates synthetic terrain keys "
    "that differ only by the low-level reference command, to test primitive viability on "
    "slippery flat and slippery downslope terrains."
)

terrains = d.setdefault("terrains", {})
style_commands = d.setdefault("style_commands", {})

base_mid = terrains["slippery_mid_flat"]
base_down = terrains["slippery_downslope_5deg_forward"]

def add_variant(base, key, style, vx, h, clr, enable, mode=None, semantic=None, note=""):
    item = copy.deepcopy(base)
    item["terrain"] = key
    item["suggested_style"] = style
    item["command"] = {
        "vx": float(vx),
        "yaw_rate": 0.0,
        "body_height": float(h),
        "swing_clearance": float(clr),
        "enable": float(enable),
    }
    if mode is not None:
        item["fused_mode"] = mode
    if semantic is not None:
        item["semantic_mode"] = semantic

    item["direct_primitive_sweep"] = True
    item["sweep_note"] = note
    item["validated_by_online_sanity"] = False

    terrains[key] = item
    style_commands[style] = copy.deepcopy(item["command"])

# Slippery flat: test whether low-speed crawl variants can survive.
add_variant(
    base_mid,
    "slippery_mid_flat_crawl_02",
    "slip_crawl_02_lowclear",
    0.02, 0.318, 0.05, 1.0,
    mode="cautious_locomotion",
    semantic="direct_primitive_test",
    note="Low-speed crawl on slippery flat."
)

add_variant(
    base_mid,
    "slippery_mid_flat_crawl_04",
    "slip_crawl_04_lowclear",
    0.04, 0.318, 0.05, 1.0,
    mode="cautious_locomotion",
    semantic="direct_primitive_test",
    note="Medium-low-speed crawl on slippery flat."
)

add_variant(
    base_mid,
    "slippery_mid_flat_crawl_06",
    "slip_crawl_06_lowclear",
    0.06, 0.318, 0.05, 1.0,
    mode="cautious_locomotion",
    semantic="direct_primitive_test",
    note="Original slip crawl baseline on slippery flat."
)

add_variant(
    base_mid,
    "slippery_mid_flat_tall_crawl_04",
    "slip_tall_crawl_04_h335_c080",
    0.04, 0.335, 0.08, 1.0,
    mode="cautious_locomotion",
    semantic="direct_primitive_test",
    note="Taller body and higher clearance crawl on slippery flat."
)

add_variant(
    base_mid,
    "slippery_mid_flat_active_hold_h335_c080",
    "slip_active_hold_h335_c080",
    0.0, 0.335, 0.08, 1.0,
    mode="recovery_needed",
    semantic="direct_primitive_test",
    note="Active hold on slippery flat, controller enabled."
)

# Slippery downslope: safe_stop enable=0 failed. Test active enable=1 primitives.
add_variant(
    base_down,
    "slippery_downslope_active_hold_h305_c055",
    "slip_down_active_hold_h305_c055",
    0.0, 0.305, 0.055, 1.0,
    mode="recovery_needed",
    semantic="direct_primitive_test",
    note="Active hold on slippery downslope; replaces passive safe_stop."
)

add_variant(
    base_down,
    "slippery_downslope_active_hold_h335_c080",
    "slip_down_active_hold_h335_c080",
    0.0, 0.335, 0.08, 1.0,
    mode="recovery_needed",
    semantic="direct_primitive_test",
    note="Taller active hold on slippery downslope."
)

add_variant(
    base_down,
    "slippery_downslope_micro_backstep_02",
    "slip_down_micro_backstep_02_h335_c080",
    -0.02, 0.335, 0.08, 1.0,
    mode="recovery_needed",
    semantic="direct_primitive_test",
    note="Micro backstep on slippery downslope."
)

add_variant(
    base_down,
    "slippery_downslope_micro_backstep_04",
    "slip_down_micro_backstep_04_h335_c080",
    -0.04, 0.335, 0.08, 1.0,
    mode="recovery_needed",
    semantic="direct_primitive_test",
    note="Stronger micro backstep on slippery downslope."
)

add_variant(
    base_down,
    "slippery_downslope_micro_backstep_06",
    "slip_down_micro_backstep_06_h335_c080",
    -0.06, 0.335, 0.08, 1.0,
    mode="recovery_needed",
    semantic="direct_primitive_test",
    note="Strongest micro backstep in this sweep."
)

add_variant(
    base_down,
    "slippery_downslope_forward_crawl_02",
    "slip_down_forward_crawl_02_h335_c080",
    0.02, 0.335, 0.08, 1.0,
    mode="recovery_needed",
    semantic="direct_primitive_test",
    note="Very slow forward crawl on slippery downslope."
)

dst.write_text(json.dumps(d, indent=2))
print("[TRACER] wrote", dst)
print("[TRACER] added variants:")
for k in sorted(terrains):
    if k.startswith("slippery_mid_flat_") or k.startswith("slippery_downslope_"):
        if terrains[k].get("direct_primitive_sweep"):
            print(" ", k, terrains[k]["command"])
