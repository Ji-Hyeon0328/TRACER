from pprint import pprint

from tracer_core.highlevel.gait_mode_selector import (
    input_from_policy_entry,
    select_gait_mode,
)
from tracer_core.highlevel.policy_input_builder import (
    build_high_level_policy_input,
    high_level_policy_input_summary,
)


def run_case(name, entry, *, gate=None, ram_level="unknown", ram_gate_action="unknown"):
    gms_in = input_from_policy_entry(
        entry,
        ram_level=ram_level,
        ram_gate_action=ram_gate_action,
        sigma_mean=gate.get("sigma_mean") if gate else None,
        control_risk=gate.get("control_risk", 0.0) if gate else 0.0,
        fallen_prob=gate.get("fallen_prob", 0.0) if gate else 0.0,
        recovery_prob=gate.get("recovery_prob", 0.0) if gate else 0.0,
    )
    gms_out = select_gait_mode(gms_in)
    inp = build_high_level_policy_input(
        policy_entry=entry,
        gms_in=gms_in,
        gms_out=gms_out,
        gate=gate,
        context=[],
        robot_state=[],
        goal=[],
    )

    print(f"\n== {name} ==")
    print("gms:", gms_out)
    print("policy_input:", inp)
    print("summary:")
    pprint(high_level_policy_input_summary(inp))


def main():
    run_case(
        "flat_fast",
        {
            "fused_mode": "locomotion",
            "semantic_mode": "validated_locomotion",
            "suggested_style": "fast",
            "beta": {"motion": 0.55, "stability": 0.25, "energy": 0.20},
            "command": {
                "vx": 0.28,
                "yaw_rate": 0.0,
                "body_height": 0.295,
                "swing_clearance": 0.03,
                "enable": 1.0,
            },
        },
        ram_level="stable",
        ram_gate_action="keep",
    )

    run_case(
        "sponge_gate_unstable",
        {
            "fused_mode": "locomotion",
            "semantic_mode": "cautious_locomotion",
            "suggested_style": "sponge_tall_10_c080",
            "beta": {"motion": 0.1853, "stability": 0.6358, "energy": 0.1789},
            "command": {
                "vx": 0.10,
                "yaw_rate": 0.0,
                "body_height": 0.335,
                "swing_clearance": 0.08,
                "enable": 1.0,
            },
        },
        gate={
            "ram_level": "unstable",
            "ram_gate_action": "would_conservative_probe",
            "control_risk": 0.2428,
            "fallen_prob": 0.7808,
            "recovery_prob": 0.10,
            "sigma_mean": 0.83,
            "rho_norm": 1.606,
        },
        ram_level="unstable",
        ram_gate_action="would_conservative_probe",
    )

    run_case(
        "slippery_no_valid",
        {
            "fused_mode": "avoid_required",
            "semantic_mode": "no_valid_high_level_velocity_primitive",
            "suggested_style": "no_deployable_high_level_fallback",
            "beta": {"motion": 0.1228, "stability": 0.7594, "energy": 0.1178},
            "command": {
                "vx": 0.0,
                "yaw_rate": 0.0,
                "body_height": 0.305,
                "swing_clearance": 0.055,
                "enable": 0.0,
            },
        },
        gate={
            "ram_level": "unstable",
            "ram_gate_action": "no_valid_keep",
            "control_risk": 0.0,
            "fallen_prob": 1.0,
            "recovery_prob": 0.0,
            "sigma_mean": 1.0,
            "rho_norm": 2.0,
        },
        ram_level="unstable",
        ram_gate_action="no_valid_keep",
    )


if __name__ == "__main__":
    main()
