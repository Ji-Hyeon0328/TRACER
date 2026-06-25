from pathlib import Path

from tracer_core.highlevel.meta_gait_dataset import build_meta_gait_samples_from_policy_json
from tracer_core.highlevel.meta_gait_policy import make_meta_gait_policy


def main():
    root = Path(__file__).resolve().parents[2]
    policy_json = root / "configs/highlevel_policy/tracer_fusion_policy_slippery_micro_brake_v3.json"
    samples = build_meta_gait_samples_from_policy_json(policy_json)

    udp_policy = make_meta_gait_policy("udp")

    print("UDP meta-gait policy client check")
    print("requires server:")
    print("  scripts/runtime/tracer_meta_gait_policy_udp_server_v0.py")

    for key in [
        "sponge_firm_downslope_5deg_forward",
        "slippery_mid_downslope_5deg_forward_postfix",
    ]:
        matched = [s for s in samples if s.terrain_key == key]
        if not matched:
            print(f"\n== {key} == missing")
            continue

        s = matched[0]
        pred = udp_policy.predict(s.policy_input)

        print(f"\n== {key} ==")
        print("gait_mode:", s.policy_input.gait_mode)
        print(
            "pred:",
            {
                "vx": pred.vx,
                "body_height": pred.body_height,
                "swing_clearance": pred.swing_clearance,
                "enable": pred.enable,
                "source_reason": pred.source_reason,
            },
        )


if __name__ == "__main__":
    main()
