#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

from tracer_core.highlevel.objective_selector import (
    RuntimeBaselineObjectiveSelector,
    objective_selector_input_from_dict,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data/preference_datasets/tracer_objective_selector_runtime_v0_baseline_model.json"


CASES = [
    {
        "name": "flat_fast_clean",
        "terrain_key": "flat_normal",
        "world": "earth",
        "semantic_prior": "validated_locomotion",
        "style_prior": "fast",
        "ramgate_enabled": True,
        "ramgate_calibrated": True,
        "ramgate_observed": True,
        "ram_monitor_observed": True,
        "ram_ctrl_risk": 0.0,
        "ram_fallen_prob": 0.0,
        "ram_recovery_prob": 0.0,
        "gate_stable_prob": 1.0,
        "gate_caution_prob": 0.0,
        "gate_unstable_prob": 0.0,
        "gate_override_prob": 0.0,
        "gate_first_action": "keep",
        "gate_last_action": "keep",
    },
    {
        "name": "rough_cautious_clean",
        "terrain_key": "rough_mid",
        "world": "tracer_rough_mid",
        "semantic_prior": "cautious_probe",
        "style_prior": "cautious",
        "ram_ctrl_risk": 0.0,
        "ram_fallen_prob": 0.0,
        "gate_stable_prob": 1.0,
        "gate_first_action": "keep",
        "gate_last_action": "keep",
    },
    {
        "name": "slope_high_clearance_clean",
        "terrain_key": "slope_5deg",
        "world": "tracer_slope_5deg",
        "semantic_prior": "high_clearance_slow_probe",
        "style_prior": "high_clearance",
        "ram_ctrl_risk": 0.0,
        "ram_fallen_prob": 0.0,
        "gate_stable_prob": 1.0,
        "gate_first_action": "keep",
        "gate_last_action": "keep",
    },
    {
        "name": "flat_fast_risky_gate_override",
        "terrain_key": "flat_normal",
        "world": "earth",
        "semantic_prior": "validated_locomotion",
        "style_prior": "fast",
        "ram_ctrl_risk": 0.19,
        "ram_fallen_prob": 0.50,
        "ram_recovery_prob": 0.21,
        "gate_stable_prob": 0.08,
        "gate_caution_prob": 0.31,
        "gate_unstable_prob": 0.62,
        "gate_override_prob": 0.0,
        "gate_first_action": "would_conservative_probe",
        "gate_last_action": "keep",
    },
    {
        "name": "sponge_downslope_risky",
        "terrain_key": "sponge_firm_downslope_5deg_forward",
        "world": "tracer_sponge_firm_downslope_5deg",
        "semantic_prior": "validated_locomotion",
        "style_prior": "sponge_tall_10_c080",
        "ram_ctrl_risk": 0.30,
        "ram_fallen_prob": 0.99,
        "gate_unstable_prob": 1.0,
        "gate_override_prob": 1.0,
        "gate_first_action": "would_conservative_probe",
        "gate_last_action": "would_conservative_probe",
    },
    {
        "name": "slippery_unknown_risky",
        "terrain_key": "slippery_mid_downslope_5deg_forward_postfix",
        "world": "tracer_slippery_mid_downslope_5deg",
        "semantic_prior": "no_valid_high_level_velocity_primitive",
        "style_prior": "no_deployable_high_level_fallback",
        "ram_ctrl_risk": 0.32,
        "ram_fallen_prob": 0.53,
        "gate_unstable_prob": 0.0,
        "gate_override_prob": 1.0,
        "gate_first_action": "no_valid_keep",
        "gate_last_action": "no_valid_keep",
    },
]


def main() -> None:
    selector = RuntimeBaselineObjectiveSelector(MODEL)

    rows = []
    for c in CASES:
        x = objective_selector_input_from_dict(c)
        y = selector.predict(x)
        row = {
            "name": c["name"],
            "terrain_key": x.terrain_key,
            "terrain_family": x.terrain_family,
            "semantic_prior": x.semantic_prior,
            "style_prior": x.style_prior,
            "semantic_target": y.semantic_target,
            "deploy_label": y.deploy_label,
            "beta": list(y.beta),
            "safety_override": y.safety_override,
            "reason": y.reason,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    expected = {
        "flat_fast_clean": ("validated_locomotion", "deploy_forward"),
        "rough_cautious_clean": ("cautious_probe", "deploy_forward"),
        "slope_high_clearance_clean": ("high_clearance_slow_probe", "deploy_forward"),
        "flat_fast_risky_gate_override": ("no_valid_forward_recovery_needed", "do_not_deploy_forward"),
        "sponge_downslope_risky": ("no_valid_forward_recovery_needed", "do_not_deploy_forward"),
        "slippery_unknown_risky": ("no_valid_forward_recovery_needed", "do_not_deploy_forward"),
    }

    bad = []
    for r in rows:
        exp_sem, exp_dep = expected[r["name"]]
        if r["semantic_target"] != exp_sem or r["deploy_label"] != exp_dep:
            bad.append((r["name"], exp_sem, exp_dep, r["semantic_target"], r["deploy_label"]))

    if bad:
        print("[TRACER][FAIL] objective selector check failed:")
        for b in bad:
            print("  ", b)
        raise SystemExit(1)

    print("[TRACER] objective selector runtime v0 check passed")


if __name__ == "__main__":
    main()
