#!/usr/bin/env python3
import json
from pathlib import Path

OUT_JSON = Path("configs/phase_j/meta_action_bank_v0.json")
OUT_MD = Path("reports/phase_j/j0_meta_action_bank_summary_v0.md")

ACTIONS = [
    {
        "id": 0,
        "name": "nominal_cruise",
        "intended_contexts": ["flat", "start_flat", "goal_flat"],
        "description": "Default forward traversal action close to the current empirical policy.",
        "theta": {
            "vx_scale": 1.00,
            "vx_delta": 0.000,
            "yaw_gain_scale": 1.00,
            "body_h_delta": 0.000,
            "clearance_delta": 0.000,
            "stability_bias": 0.00,
            "energy_bias": 0.00,
            "hold_override": False
        }
    },
    {
        "id": 1,
        "name": "fast_motion",
        "intended_contexts": ["flat", "start_flat"],
        "description": "Slightly faster motion-oriented traversal for low-risk flat terrain.",
        "theta": {
            "vx_scale": 1.08,
            "vx_delta": 0.000,
            "yaw_gain_scale": 1.00,
            "body_h_delta": 0.000,
            "clearance_delta": -0.003,
            "stability_bias": -0.10,
            "energy_bias": -0.05,
            "hold_override": False
        }
    },
    {
        "id": 2,
        "name": "energy_saver",
        "intended_contexts": ["flat", "goal_flat"],
        "description": "Moderate slower gait with lower clearance to reduce unnecessary actuation.",
        "theta": {
            "vx_scale": 0.88,
            "vx_delta": 0.000,
            "yaw_gain_scale": 0.90,
            "body_h_delta": 0.000,
            "clearance_delta": -0.004,
            "stability_bias": 0.00,
            "energy_bias": 0.15,
            "hold_override": False
        }
    },
    {
        "id": 3,
        "name": "rough_high_clearance",
        "intended_contexts": ["rough"],
        "description": "Increase foot clearance and slightly reduce speed for rough terrain traversal.",
        "theta": {
            "vx_scale": 0.92,
            "vx_delta": 0.000,
            "yaw_gain_scale": 1.00,
            "body_h_delta": 0.000,
            "clearance_delta": 0.010,
            "stability_bias": 0.15,
            "energy_bias": -0.05,
            "hold_override": False
        }
    },
    {
        "id": 4,
        "name": "rough_stability",
        "intended_contexts": ["rough"],
        "description": "More conservative rough-terrain action emphasizing stability over speed.",
        "theta": {
            "vx_scale": 0.82,
            "vx_delta": 0.000,
            "yaw_gain_scale": 0.85,
            "body_h_delta": -0.005,
            "clearance_delta": 0.012,
            "stability_bias": 0.25,
            "energy_bias": -0.10,
            "hold_override": False
        }
    },
    {
        "id": 5,
        "name": "upslope_push",
        "intended_contexts": ["upslope"],
        "description": "Maintain forward progress on upslope with mild stability margin.",
        "theta": {
            "vx_scale": 0.96,
            "vx_delta": 0.000,
            "yaw_gain_scale": 0.95,
            "body_h_delta": 0.000,
            "clearance_delta": 0.004,
            "stability_bias": 0.12,
            "energy_bias": -0.05,
            "hold_override": False
        }
    },
    {
        "id": 6,
        "name": "downslope_stable",
        "intended_contexts": ["downslope"],
        "description": "Reduce speed and increase stability margin for downslope traversal.",
        "theta": {
            "vx_scale": 0.78,
            "vx_delta": 0.000,
            "yaw_gain_scale": 0.80,
            "body_h_delta": -0.006,
            "clearance_delta": 0.004,
            "stability_bias": 0.30,
            "energy_bias": -0.10,
            "hold_override": False
        }
    },
    {
        "id": 7,
        "name": "lateral_recovery_soft",
        "intended_contexts": ["rough", "downslope", "unknown"],
        "description": "Conservative action for lateral drift or mismatch; not a full recovery primitive.",
        "theta": {
            "vx_scale": 0.65,
            "vx_delta": 0.000,
            "yaw_gain_scale": 0.70,
            "body_h_delta": -0.008,
            "clearance_delta": 0.008,
            "stability_bias": 0.35,
            "energy_bias": -0.15,
            "hold_override": False
        }
    },
    {
        "id": 8,
        "name": "goal_hold",
        "intended_contexts": ["goal_flat"],
        "description": "Goal/hold behavior. This action should only be selected near the goal or by the existing stop guard.",
        "theta": {
            "vx_scale": 0.00,
            "vx_delta": 0.025,
            "yaw_gain_scale": 0.00,
            "body_h_delta": 0.000,
            "clearance_delta": 0.000,
            "stability_bias": 0.25,
            "energy_bias": 0.10,
            "hold_override": True
        }
    }
]

CONFIG = {
    "phase": "J0",
    "name": "TRACER meta-action bank v0",
    "safe_claim": "Defines candidate high-level meta-actions only; not active control and not final RL.",
    "action_space_type": "discrete_meta_action_bank",
    "base_reference_convention": {
        "input_ref": "[seq, vx, yaw_rate, body_h, clearance, enable]",
        "application": "theta modifies the empirical/D7-conditioned high-level reference before safety projection",
        "not_yet_active": True
    },
    "safety_projection": {
        "vx_min": 0.025,
        "vx_max": 0.220,
        "yaw_rate_min": -0.200,
        "yaw_rate_max": 0.200,
        "body_h_min": 0.300,
        "body_h_max": 0.340,
        "clearance_min": 0.035,
        "clearance_max": 0.065,
        "enable_values": [1.0],
        "hold_vx": 0.025,
        "notes": [
            "The action bank must never bypass the existing goal/hold guard.",
            "The action bank should first be evaluated offline or in shadow mode.",
            "The action bank is intended for contextual-bandit/offline-RL preparation before online PPO."
        ]
    },
    "state_inputs_for_future_policy": [
        "terrain_context",
        "D7_beta",
        "legacy_ram_or_I5_safe_ram_shadow",
        "x/y progress",
        "recent tracking/drift window features"
    ],
    "actions": ACTIONS
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(CONFIG, indent=2))

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-J0 Meta-Action Bank v0\n\n")
    f.write("This defines a discrete high-level meta-action bank for future contextual-bandit / offline-RL meta-plan learning.\n\n")
    f.write(f"- output config: `{OUT_JSON}`\n")
    f.write(f"- number of actions: `{len(ACTIONS)}`\n")
    f.write("- active control: `false`\n")
    f.write("- intended next step: J1 offline/surrogate action scoring\n\n")

    f.write("## Safety projection\n\n")
    sp = CONFIG["safety_projection"]
    f.write(f"- vx: `{sp['vx_min']}` to `{sp['vx_max']}`\n")
    f.write(f"- yaw_rate: `{sp['yaw_rate_min']}` to `{sp['yaw_rate_max']}`\n")
    f.write(f"- body_h: `{sp['body_h_min']}` to `{sp['body_h_max']}`\n")
    f.write(f"- clearance: `{sp['clearance_min']}` to `{sp['clearance_max']}`\n")
    f.write(f"- hold_vx: `{sp['hold_vx']}`\n\n")

    f.write("## Actions\n\n")
    f.write("| id | name | intended contexts | vx scale | body_h delta | clearance delta | stability bias | energy bias | hold |\n")
    f.write("|---:|---|---|---:|---:|---:|---:|---:|---|\n")
    for a in ACTIONS:
        th = a["theta"]
        f.write(
            f"| {a['id']} | {a['name']} | {', '.join(a['intended_contexts'])} | "
            f"{th['vx_scale']:.3f} | {th['body_h_delta']:.3f} | {th['clearance_delta']:.3f} | "
            f"{th['stability_bias']:.3f} | {th['energy_bias']:.3f} | {th['hold_override']} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("- J0 only defines candidate high-level meta-actions.\n")
    f.write("- J0 does not modify active control.\n")
    f.write("- The existing D7 Objective Selector should remain the active beta source.\n")
    f.write("- I5 safe RAM may be used as a shadow/auxiliary feature, but not as active RAM yet.\n")
    f.write("- J1 should score these actions offline or in shadow before any active rollout.\n")

print(f"[TRACER] wrote {OUT_JSON}")
print(f"[TRACER] wrote {OUT_MD}")
print(f"[TRACER] actions={len(ACTIONS)}")
