from tracer_core.highlevel.gait_mode_selector import (
    apply_gait_mode_to_command,
    input_from_policy_entry,
    select_gait_mode,
)


def run_case(name, entry, **ram):
    gms_in = input_from_policy_entry(entry, **ram)
    gms_out = select_gait_mode(gms_in)
    cmd = apply_gait_mode_to_command(entry.get("command", {}), gms_out)
    print(f"\n== {name} ==")
    print("input:", gms_in)
    print("output:", gms_out)
    print("command:", cmd)


def main():
    run_case(
        "flat_valid_fast",
        {
            "fused_mode": "locomotion",
            "semantic_mode": "validated_locomotion",
            "suggested_style": "fast",
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
        "sponge_cautious",
        {
            "fused_mode": "locomotion",
            "semantic_mode": "cautious_locomotion",
            "suggested_style": "sponge_tall_10_c080",
            "command": {
                "vx": 0.10,
                "yaw_rate": 0.0,
                "body_height": 0.335,
                "swing_clearance": 0.08,
                "enable": 1.0,
            },
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
            "command": {
                "vx": 0.0,
                "yaw_rate": 0.0,
                "body_height": 0.305,
                "swing_clearance": 0.055,
                "enable": 0.0,
            },
        },
        ram_level="unstable",
        ram_gate_action="no_valid_keep",
    )


if __name__ == "__main__":
    main()
