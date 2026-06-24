from tracer_core.highlevel.decoder_mapper import (
    decode_meta_gait_to_low_level_ref,
    meta_gait_from_gms,
)
from tracer_core.highlevel.gait_mode_selector import (
    input_from_policy_entry,
    select_gait_mode,
)
from tracer_core.highlevel.meta_gait import ObjectiveWeights


def run_case(name, entry, *, ram_level="unknown", ram_gate_action="unknown"):
    beta = ObjectiveWeights.from_mapping(entry.get("beta"))
    gms_in = input_from_policy_entry(
        entry,
        ram_level=ram_level,
        ram_gate_action=ram_gate_action,
    )
    gms_out = select_gait_mode(gms_in)
    meta = meta_gait_from_gms(entry["command"], gms_out, beta=beta)
    ref = decode_meta_gait_to_low_level_ref(meta)

    print(f"\n== {name} ==")
    print("gms:", gms_out)
    print("meta:", meta)
    print("low_level_ref:", ref)
    print("mpc_array:", ref.to_mpc_reference_array(counter=0.0))


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
        "sponge_conservative",
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
        ram_level="unstable",
        ram_gate_action="no_valid_keep",
    )


if __name__ == "__main__":
    main()
