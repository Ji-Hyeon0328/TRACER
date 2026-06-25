from pathlib import Path

from tracer_core.highlevel.meta_gait_dataset import (
    TARGET_NAMES,
    build_meta_gait_samples_from_policy_json,
)
from tracer_core.highlevel.meta_gait_policy import make_meta_gait_policy


def main():
    root = Path(__file__).resolve().parents[2]
    model_path = root / "artifacts/meta_gait_policy_v0/model.pt"
    policy_json = root / "configs/highlevel_policy/tracer_fusion_policy_slippery_micro_brake_v3.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"missing learned model: {model_path}\\n"
            "Run scripts/training/tracer_train_meta_gait_policy_v0.py first."
        )

    samples = build_meta_gait_samples_from_policy_json(policy_json)
    learned = make_meta_gait_policy("learned", str(model_path))

    print(f"model_path: {model_path}")
    print(f"samples: {len(samples)}")

    for key in [
        "flat_normal",
        "sponge_firm_downslope_5deg_forward",
        "slippery_mid_downslope_5deg_forward_postfix",
    ]:
        matched = [s for s in samples if s.terrain_key == key]
        if not matched:
            print(f"\\n== {key} == missing")
            continue

        s = matched[0]
        pred = learned.predict(s.policy_input)

        pred_vec = [
            pred.vx,
            pred.yaw_rate,
            pred.body_height,
            pred.swing_clearance,
            pred.enable,
            pred.gait_period,
            pred.duty_factor,
            pred.step_length,
            pred.stance_width,
            pred.impedance_scale,
            pred.residual_gain_scale,
            pred.risk_scale,
        ]

        print(f"\\n== {key} ==")
        print("gait_mode:", s.policy_input.gait_mode)
        print("target:", dict(zip(TARGET_NAMES, s.target_vector)))
        print("pred:  ", dict(zip(TARGET_NAMES, pred_vec)))


if __name__ == "__main__":
    main()
