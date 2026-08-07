#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tracer_core.highlevel.decoder_mapper import decode_meta_gait_to_low_level_ref
from tracer_core.highlevel.theta_direct_mapper import meta_gait_from_normalized_theta


PRESETS = {
    "nominal": [0, 0, 0, 0, 0, 0, 0, 0],
    "short_step_low": [0.2, 0.2, -0.8, 0.2, 0.0, 0.0, 0.3, 0.2],
    "high_clearance": [0.4, 0.4, -0.5, 0.4, 0.4, 0.8, 0.5, 0.4],
    "longer_step": [-0.2, -0.2, 0.6, 0.0, 0.0, 0.0, -0.2, 0.0],
    "high_body_safe": [0.5, 0.5, -0.4, 0.5, 0.8, 0.8, 0.8, 0.6],
}

BETAS = {
    "balanced": {"motion": 0.34, "stability": 0.33, "energy": 0.33},
    "motion": {"motion": 0.60, "stability": 0.25, "energy": 0.15},
    "stability": {"motion": 0.20, "stability": 0.65, "energy": 0.15},
}


def main() -> int:
    rows = []

    for beta_name, beta in BETAS.items():
        for theta_name, theta in PRESETS.items():
            meta = meta_gait_from_normalized_theta(theta, beta=beta)
            ref = decode_meta_gait_to_low_level_ref(meta)

            row = {
                "beta_name": beta_name,
                "theta_name": theta_name,
                "theta_norm": theta,
                "meta": {
                    "vx": meta.vx,
                    "yaw_rate": meta.yaw_rate,
                    "body_height": meta.body_height,
                    "swing_clearance": meta.swing_clearance,
                    "gait_period": meta.gait_period,
                    "duty_factor": meta.duty_factor,
                    "step_length": meta.step_length,
                    "stance_width": meta.stance_width,
                    "impedance_scale": meta.impedance_scale,
                    "residual_gain_scale": meta.residual_gain_scale,
                    "risk_scale": meta.risk_scale,
                },
                "mpc_reference": ref.to_mpc_reference_array(counter=0.0),
                "extras": dict(meta.extras),
            }

            rows.append(row)
            print(json.dumps(row, indent=2, sort_keys=True))

    print(f"[TRACER] checked {len(rows)} theta-direct mappings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
