from tracer_core.highlevel.decoder_mapper import decode_meta_gait_to_low_level_ref
from tracer_core.highlevel.meta_gait import HighLevelPolicyInput, ObjectiveWeights, RamSignal
from tracer_core.highlevel.meta_gait_policy import make_meta_gait_policy


def run_case(name, inp):
    policy = make_meta_gait_policy("rule_based")
    meta = policy.predict(inp)
    ref = decode_meta_gait_to_low_level_ref(meta)

    print(f"\n== {name} ==")
    print("input:", inp)
    print("meta:", meta)
    print("low_ref:", ref)
    print("mpc_array:", ref.to_mpc_reference_array(counter=0.0))


def main():
    run_case(
        "flat_fast",
        HighLevelPolicyInput(
            beta=ObjectiveWeights(motion=0.55, stability=0.25, energy=0.20),
            ram=RamSignal(ram_level="stable", sigma=0.0),
            gait_mode="fast",
        ),
    )

    run_case(
        "sponge_conservative",
        HighLevelPolicyInput(
            beta=ObjectiveWeights(motion=0.1853, stability=0.6358, energy=0.1789),
            ram=RamSignal(
                rho=(1.606,),
                sigma=0.83,
                ram_level="unstable",
                control_risk=0.2428,
                fallen_prob=0.7808,
                recovery_prob=0.1,
            ),
            gait_mode="conservative",
        ),
    )

    run_case(
        "slippery_no_valid",
        HighLevelPolicyInput(
            beta=ObjectiveWeights(motion=0.1228, stability=0.7594, energy=0.1178),
            ram=RamSignal(
                rho=(2.0,),
                sigma=1.0,
                ram_level="unstable",
                fallen_prob=1.0,
            ),
            gait_mode="no_valid",
        ),
    )


if __name__ == "__main__":
    main()
