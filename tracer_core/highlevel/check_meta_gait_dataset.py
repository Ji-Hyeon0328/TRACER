from pathlib import Path

from tracer_core.highlevel.meta_gait_dataset import (
    FEATURE_NAMES,
    TARGET_NAMES,
    build_meta_gait_samples_from_policy_json,
)


def main():
    root = Path(__file__).resolve().parents[2]
    policy_json = root / "configs/highlevel_policy/tracer_fusion_policy_slippery_micro_brake_v3.json"

    samples = build_meta_gait_samples_from_policy_json(policy_json)

    print(f"policy_json: {policy_json}")
    print(f"num_samples: {len(samples)}")
    print(f"feature_dim: {len(FEATURE_NAMES)}")
    print(f"target_dim: {len(TARGET_NAMES)}")

    for key in [
        "sponge_firm_downslope_5deg_forward",
        "slippery_mid_downslope_5deg_forward_postfix",
    ]:
        matched = [s for s in samples if s.terrain_key == key]
        if not matched:
            print(f"\n== {key} ==")
            print("missing")
            continue

        s = matched[0]
        print(f"\n== {key} ==")
        print("input summary:", dict(s.policy_input_summary))
        print("feature vector:", list(s.feature_vector))
        print("target:", dict(zip(TARGET_NAMES, s.target_vector)))


if __name__ == "__main__":
    main()
