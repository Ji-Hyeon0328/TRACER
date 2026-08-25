#!/usr/bin/env python3

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from tracer_core.highlevel_rl.reward_v3 import (
    compute_simplified_tracer_costs_v3,
)

from tracer_core.highlevel_rl.reward_v4 import (
    compute_simplified_tracer_costs_v4,
)

from tracer_core.highlevel_rl.m5_transport import (
    _traction_cost_interval,
    _latched_traction_cost_interval,
)


def main():
    print("=" * 96)
    print(
        "ICRA27 OS-T4.7c V3/V4 REWARD CONTRACT"
    )
    print("=" * 96)

    # --------------------------------------------------------
    # A. Reward algebra must be identical for identical
    #    supplied traction scalar.
    # --------------------------------------------------------

    kwargs = dict(
        previous_goal_distance=1.20,
        goal_distance=1.15,
        decision_dt=0.20,

        roll=0.03,
        pitch=-0.04,

        roll_unsafe_rad=0.35,
        pitch_unsafe_rad=0.35,

        traction_cost=0.42,

        applied_abs_energy_j=12.0,
        energy_dt_s=0.20,
    )

    v3 = compute_simplified_tracer_costs_v3(
        **kwargs
    )

    v4 = compute_simplified_tracer_costs_v4(
        **kwargs
    )

    d3 = v3.as_dict()
    d4 = v4.as_dict()

    if d3.keys() != d4.keys():
        raise RuntimeError(
            "v3/v4 reward keys differ"
        )

    max_diff = max(
        abs(
            float(d3[k])
            - float(d4[k])
        )
        for k in d3
    )

    print(
        f"same-input reward max diff : "
        f"{max_diff:.3e}"
    )

    if max_diff > 1e-12:
        raise RuntimeError(
            "v3/v4 objective algebra differs"
        )

    # --------------------------------------------------------
    # B. Transport provenance must differ independently.
    # --------------------------------------------------------

    common_role = (
        "candidate_contact_stability_cost_v0"
    )

    latched_role = (
        "candidate_latched_contact_stability_cost_v0"
    )

    start = {
        "eval_stance_slip": {
            "traction_cost_role":
                common_role,

            "latched_traction_cost_role":
                latched_role,

            "established_contact_time_s":
                1.000,

            "established_traction_cost_time_sum_s":
                0.200,

            "latched_established_contact_time_s":
                1.000,

            "latched_established_traction_cost_time_sum_s":
                0.200,
        }
    }

    end = {
        "eval_stance_slip": {
            "traction_cost_role":
                common_role,

            "latched_traction_cost_role":
                latched_role,

            # Historical v3 interval:
            # 0.40 cost integral / 0.40 s = 1.0
            "established_contact_time_s":
                1.400,

            "established_traction_cost_time_sum_s":
                0.600,

            # v4 latched interval:
            # 0.30 cost integral / 0.50 s = 0.6
            "latched_established_contact_time_s":
                1.500,

            "latched_established_traction_cost_time_sum_s":
                0.500,
        }
    }

    i3 = _traction_cost_interval(
        start,
        end,
    )

    i4 = _latched_traction_cost_interval(
        start,
        end,
    )

    if i3 is None or i4 is None:
        raise RuntimeError(
            "transport interval unexpectedly None"
        )

    print(
        "v3 transport mean          : "
        f"{i3.mean_cost:.6f}"
    )

    print(
        "v4 transport mean          : "
        f"{i4.mean_cost:.6f}"
    )

    if not math.isclose(
        i3.mean_cost,
        1.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "unexpected v3 transport result"
        )

    if not math.isclose(
        i4.mean_cost,
        0.6,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "unexpected v4 transport result"
        )

    print()
    print(
        "INTERPRETATION:"
    )
    print(
        "  reward algebra       : identical"
    )
    print(
        "  traction provenance  : independent"
    )
    print(
        "  v3 historical path   : preserved"
    )
    print(
        "  v4 latched path       : active"
    )

    print()
    print(
        "[ICRA27] OS-T4.7c "
        "v3/v4 reward contract: PASS"
    )


if __name__ == "__main__":
    main()
