#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/highlevel_policy/tracer_fusion_policy_v0.json"
OUT_DIR = ROOT / "configs/highlevel_policy/overrides"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE_COMMANDS = {
    "nominal": {"vx": 0.21, "yaw_rate": 0.0, "body_height": 0.300, "swing_clearance": 0.035, "enable": 1.0},
    "fast": {"vx": 0.28, "yaw_rate": 0.0, "body_height": 0.295, "swing_clearance": 0.030, "enable": 1.0},
    "cautious": {"vx": 0.12, "yaw_rate": 0.0, "body_height": 0.310, "swing_clearance": 0.060, "enable": 1.0},
    "slip_crawl_06_lowclear": {"vx": 0.06, "yaw_rate": 0.0, "body_height": 0.318, "swing_clearance": 0.050, "enable": 1.0},
    "sponge_tall_10_c080": {"vx": 0.10, "yaw_rate": 0.0, "body_height": 0.335, "swing_clearance": 0.080, "enable": 1.0},
    "safe_stop": {"vx": 0.00, "yaw_rate": 0.0, "body_height": 0.310, "swing_clearance": 0.050, "enable": 0.0},
    "active_hold": {"vx": 0.00, "yaw_rate": 0.0, "body_height": 0.318, "swing_clearance": 0.050, "enable": 1.0},
}

def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: tracer_make_fusion_policy_override_copy.py <terrain_key> <style> <out_name>")

    terrain, style, out_name = sys.argv[1], sys.argv[2], sys.argv[3]

    if style not in STYLE_COMMANDS:
        raise SystemExit(f"unknown style: {style}")

    data = json.loads(BASE.read_text())

    if terrain not in data["terrains"]:
        raise SystemExit(f"unknown terrain key: {terrain}")

    entry = data["terrains"][terrain]
    entry["suggested_style"] = style
    entry["command"] = STYLE_COMMANDS[style]
    entry["manual_override"] = True
    entry["manual_override_style"] = style

    if style == "safe_stop":
        entry["fused_mode"] = "recovery_needed"
    elif "slippery" in terrain:
        entry["fused_mode"] = "cautious_locomotion"
    else:
        entry["fused_mode"] = "locomotion"

    out = OUT_DIR / out_name
    if out.suffix != ".json":
        out = out.with_suffix(".json")

    out.write_text(json.dumps(data, indent=2))
    print("[TRACER] wrote override copy:", out)
    print("[TRACER] base policy unchanged:", BASE)
    print("[TRACER] terrain:", terrain)
    print("[TRACER] style:", style)
    print("[TRACER] command:", STYLE_COMMANDS[style])

if __name__ == "__main__":
    main()
