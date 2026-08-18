#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import mujoco


ROOT = Path(__file__).resolve().parents[2]
ICRA27 = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27))


import run_pympc_lowlevel_udp as m5

from tracer_core.highlevel_rl.state_protocol import (
    SCHEMA,
    encode_state_payload,
)

from tracer_core.highlevel_rl.mechanical_energy import (
    MechanicalEnergyAccumulator,
    applied_generalized_power,
    commanded_joint_power,
)


standalone = m5.standalone


ORIGINAL_STANDALONE_COMPUTE_ACTIONS = (
    standalone.standalone_compute_actions
)

ORIGINAL_STANDALONE_WRAPPER_RESET = (
    standalone.standalone_wrapper_reset
)

ORIGINAL_M5_UDP_ENV_STEP = (
    m5.udp_env_step
)


LATEST_STATE: dict | None = None

# Evaluation-only same-tick PyMPC planned contact.
#
# Captured immediately after the frozen standalone controller
# computes its same-tick contact plan, before env.step().
LATEST_PLANNED_CONTACT: dict[str, bool] | None = None

EPISODE_INDEX = -1
STATE_SENDER = None


ENERGY_ACCUMULATOR = (
    MechanicalEnergyAccumulator()
)

LATEST_ENERGY_SAMPLE: dict | None = None

ENERGY_LOG_PATH = (
    os.environ.get(
        "TRACER_M7_ENERGY_LOG",
        "",
    ).strip()
)


# ============================================================
# Evaluation-only 500 Hz per-foot slip trace.
#
# This is a file-only side channel used by OS-T4.5k6f.
# It does NOT alter reward, observation, controller state,
# safety logic, UDP state packets, or the existing accumulator.
# ============================================================

SLIP_TRACE_PATH = (
    os.environ.get(
        "TRACER_M7_SLIP_TRACE",
        "",
    ).strip()
)

SLIP_TRACE_TAG = (
    os.environ.get(
        "TRACER_M7_SLIP_TRACE_TAG",
        "",
    ).strip()
)


def _optional_trace_env_float(
    name,
):
    raw = (
        os.environ.get(
            name,
            "",
        ).strip()
    )

    if not raw:
        return None

    value = float(raw)

    if not np.isfinite(value):
        raise RuntimeError(
            f"Non-finite {name}: {value!r}"
        )

    return value


SLIP_TRACE_TMIN_S = (
    _optional_trace_env_float(
        "TRACER_M7_SLIP_TRACE_TMIN_S"
    )
)

SLIP_TRACE_TMAX_S = (
    _optional_trace_env_float(
        "TRACER_M7_SLIP_TRACE_TMAX_S"
    )
)

if SLIP_TRACE_PATH:
    if (
        SLIP_TRACE_TMIN_S is None
        or SLIP_TRACE_TMAX_S is None
    ):
        raise RuntimeError(
            "TRACER_M7_SLIP_TRACE requires both "
            "TMIN and TMAX."
        )

    if (
        SLIP_TRACE_TMAX_S
        <= SLIP_TRACE_TMIN_S
    ):
        raise RuntimeError(
            "Invalid slip-trace time window: "
            f"({SLIP_TRACE_TMIN_S}, "
            f"{SLIP_TRACE_TMAX_S}]"
        )


# ============================================================
# Evaluation-only stance-foot slip telemetry.
#
# IMPORTANT:
#   - NOT part of reward
#   - NOT part of observation
#   - NOT fed to PyMPC/controller
#   - read-only MuJoCo diagnostic
# ============================================================

FOOT_LEGS = (
    "FL",
    "FR",
    "RL",
    "RR",
)

FOOT_GEOM_IDS = None
FOOT_GEOM_NAMES = None


# Evaluation-only contact-age resolution.
#
# 0--100 ms is retained at 10 ms resolution, followed by one
# >=100 ms sustained-contact tail bin. No reward threshold is
# implied by these diagnostic bins.
CONTACT_AGE_BIN_WIDTH_S = 0.010
CONTACT_AGE_TAIL_START_S = 0.100


# Frozen from OS-T4.1b TRAIN9:
#
# 40 ms is the earliest contact age at which nominal-flat
# touchdown slip has reached its post-impact plateau while
# retaining terrain-induced rough/contact disturbance.
#
# Evaluation-only for now. This is NOT yet connected to reward.
ESTABLISHED_STANCE_GATE_S = 0.040


# Frozen from OS-T4.1:
#
#   nominal established-stance flat Q95
#       -> 0.002 m/s
#
#   marginally viable friction boundary (mu=0.152)
#   pooled established-stance Q95
#       -> 0.150 m/s
#
# These define the candidate traction-risk normalization.
# This state-tap patch only accumulates the candidate cost;
# it does NOT yet alter PPO reward or observation.
TRACTION_SLIP_DEADBAND_MPS = 0.002
TRACTION_SLIP_UNSAFE_MPS = 0.150


def _pointwise_traction_cost(
    speed_mps,
):
    speed = float(speed_mps)

    if (
        not np.isfinite(speed)
        or speed < 0.0
    ):
        raise RuntimeError(
            "Invalid traction slip speed: "
            f"{speed!r}"
        )

    scale = (
        TRACTION_SLIP_UNSAFE_MPS
        - TRACTION_SLIP_DEADBAND_MPS
    )

    if scale <= 0.0:
        raise RuntimeError(
            "Invalid traction normalization scale."
        )

    excess = max(
        0.0,
        speed
        - TRACTION_SLIP_DEADBAND_MPS,
    )

    return float(
        (excess / scale) ** 2
    )


# Diagnostic speed histogram only.
#
# Fine resolution near nominal flat established-stance slip,
# wider bins in the high-slip tail.
ESTABLISHED_SLIP_SPEED_EDGES_MPS = (
    0.0,
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0030,
    0.0040,
    0.0050,
    0.0060,
    0.0080,
    0.0100,
    0.0125,
    0.0150,
    0.0200,
    0.0300,
    0.0400,
    0.0500,
    0.0750,
    0.1000,
    0.1250,
    0.1500,
    0.1550,
    0.1600,
    0.1650,
    0.1700,
    0.1750,
    0.1800,
    0.1850,
    0.1900,
    0.1950,
    0.2000,
    0.2250,
    0.2500,
    0.2750,
    0.3000,
    0.5000,
    1.0000,
)


def _new_established_slip_histogram():
    bins = []

    edges = (
        ESTABLISHED_SLIP_SPEED_EDGES_MPS
    )

    for index in range(
        len(edges) - 1
    ):
        bins.append(
            {
                "low_mps":
                    float(edges[index]),

                "high_mps":
                    float(edges[index + 1]),

                "contact_time_s":
                    0.0,

                "contact_samples":
                    0,
            }
        )

    bins.append(
        {
            "low_mps":
                float(edges[-1]),

            "high_mps":
                None,

            "contact_time_s":
                0.0,

            "contact_samples":
                0,
        }
    )

    return bins


def _established_slip_histogram_index(
    speed_mps,
):
    speed_mps = float(speed_mps)

    if (
        not np.isfinite(speed_mps)
        or speed_mps < 0.0
    ):
        raise RuntimeError(
            "Invalid established stance slip speed: "
            f"{speed_mps!r}"
        )

    edges = (
        ESTABLISHED_SLIP_SPEED_EDGES_MPS
    )

    for index in range(
        len(edges) - 1
    ):
        if (
            speed_mps
            < edges[index + 1]
        ):
            return index

    return len(edges) - 1


def _new_contact_age_bins():
    bins = []

    bin_count = int(
        round(
            CONTACT_AGE_TAIL_START_S
            / CONTACT_AGE_BIN_WIDTH_S
        )
    )

    for index in range(bin_count):
        low_s = (
            index
            * CONTACT_AGE_BIN_WIDTH_S
        )

        high_s = (
            (index + 1)
            * CONTACT_AGE_BIN_WIDTH_S
        )

        bins.append(
            {
                "low_s":
                    float(low_s),

                "high_s":
                    float(high_s),

                "contact_time_s":
                    0.0,

                "contact_samples":
                    0,

                "speed_time_sum_m":
                    0.0,

                "speed_sq_time_sum_m2ps":
                    0.0,

                "speed_max_mps":
                    0.0,
            }
        )

    bins.append(
        {
            "low_s":
                float(
                    CONTACT_AGE_TAIL_START_S
                ),

            "high_s":
                None,

            "contact_time_s":
                0.0,

            "contact_samples":
                0,

            "speed_time_sum_m":
                0.0,

            "speed_sq_time_sum_m2ps":
                0.0,

            "speed_max_mps":
                0.0,
        }
    )

    return bins


def _contact_age_bin_index(
    age_s,
):
    age_s = float(age_s)

    if age_s < 0.0:
        raise RuntimeError(
            f"Negative contact age: {age_s}"
        )

    if (
        age_s
        >= CONTACT_AGE_TAIL_START_S
    ):
        return int(
            round(
                CONTACT_AGE_TAIL_START_S
                / CONTACT_AGE_BIN_WIDTH_S
            )
        )

    return min(
        int(
            age_s
            / CONTACT_AGE_BIN_WIDTH_S
        ),
        int(
            round(
                CONTACT_AGE_TAIL_START_S
                / CONTACT_AGE_BIN_WIDTH_S
            )
        )
        - 1,
    )


def _new_slip_accumulator():
    return {
        "speed_time_sum_m":
            0.0,

        "speed_sq_time_sum_m2ps":
            0.0,

        "contact_time_s":
            0.0,

        "speed_max_mps":
            0.0,

        "contact_samples":
            0,

        "physics_steps_with_contact":
            0,

        "per_foot_contact_samples": {
            leg: 0
            for leg in FOOT_LEGS
        },

        # Consecutive contact duration for each foot.
        # Reset to zero whenever the foot is not in contact.
        "contact_age_s": {
            leg: 0.0
            for leg in FOOT_LEGS
        },

        # ----------------------------------------------------
        # OS-T4.6b planned-stance traction SHADOW.
        #
        # Unlike contact_age_s above, this age is advanced by
        # the PyMPC planned stance state, not by binary MuJoCo
        # contact. It is evaluation-only and does NOT alter the
        # existing tracer_cost_v3 accumulator.
        # ----------------------------------------------------
        "planned_stance_age_s": {
            leg: 0.0
            for leg in FOOT_LEGS
        },

        # ----------------------------------------------------
        # OS-T4.6c candidate:
        # first physical contact starts the transient clock;
        # a micro contact loss does NOT reset it while PyMPC
        # still plans that leg to remain in stance.
        # ----------------------------------------------------
        "latched_stance_contact_started": {
            leg: False
            for leg in FOOT_LEGS
        },

        "latched_stance_contact_age_s": {
            leg: 0.0
            for leg in FOOT_LEGS
        },

        "latched_established_contact_time_s":
            0.0,

        "latched_established_contact_samples":
            0,

        "latched_established_traction_cost_time_sum_s":
            0.0,

        "planned_established_contact_time_s":
            0.0,

        "planned_established_contact_samples":
            0,

        "planned_established_traction_cost_time_sum_s":
            0.0,

        # Diagnostic sufficient statistics indexed by
        # consecutive stance-contact age.
        "contact_age_bins":
            _new_contact_age_bins(),

        "established_slip_histogram":
            _new_established_slip_histogram(),

        "established_contact_time_s":
            0.0,

        "established_contact_samples":
            0,

        # Integral of the dimensionless pointwise candidate
        # traction cost over established foot-contact time:
        #
        #     I_T = integral c_T(s) dt
        #
        # Units are seconds because c_T is dimensionless.
        "established_traction_cost_time_sum_s":
            0.0,
    }


SLIP_ACCUMULATOR = (
    _new_slip_accumulator()
)


def _contact_enabled_geom(
    model,
    geom_id,
):
    geom_id = int(geom_id)

    return bool(
        int(
            model.geom_contype[
                geom_id
            ]
        ) != 0
        or int(
            model.geom_conaffinity[
                geom_id
            ]
        ) != 0
    )


def _resolve_foot_geom_ids(
    self,
):
    """
    Resolve the four active foot collision geoms from the
    *runtime* MuJoCo model.

    Do not hard-code a robot model: the gym-quadruped package
    contains several robot XMLs.
    """

    global FOOT_GEOM_IDS
    global FOOT_GEOM_NAMES

    if FOOT_GEOM_IDS is not None:
        return dict(
            FOOT_GEOM_IDS
        )

    model = self.mjModel

    all_named_geoms = []

    for geom_id in range(
        int(model.ngeom)
    ):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        if name is not None:
            all_named_geoms.append(
                (
                    int(geom_id),
                    str(name),
                )
            )

    resolved = {}
    resolved_names = {}

    for leg in FOOT_LEGS:
        exact_names = (
            leg,
            f"{leg}_foot",
            f"{leg}_foot_geom",
            f"{leg}_foot_collision",
            f"{leg.lower()}_foot",
        )

        candidates = []

        for name in exact_names:
            geom_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )

            if (
                geom_id >= 0
                and _contact_enabled_geom(
                    model,
                    geom_id,
                )
            ):
                candidates.append(
                    (
                        int(geom_id),
                        str(name),
                    )
                )

        # Generic fallback for models that use names such as
        # FL_foot_collision or front-like variants beginning
        # with the standard leg prefix.
        if not candidates:
            leg_lower = leg.lower()

            for geom_id, name in (
                all_named_geoms
            ):
                low = name.lower()

                name_matches = (
                    low == leg_lower
                    or (
                        low.startswith(
                            leg_lower
                        )
                        and "foot" in low
                    )
                )

                if (
                    name_matches
                    and _contact_enabled_geom(
                        model,
                        geom_id,
                    )
                ):
                    candidates.append(
                        (
                            int(geom_id),
                            str(name),
                        )
                    )

        # Preserve order while removing duplicate IDs.
        unique = {}

        for geom_id, name in candidates:
            unique.setdefault(
                geom_id,
                name,
            )

        candidates = list(
            unique.items()
        )

        if not candidates:
            contact_named = [
                name
                for geom_id, name
                in all_named_geoms
                if _contact_enabled_geom(
                    model,
                    geom_id,
                )
            ]

            raise RuntimeError(
                "Could not resolve contact-enabled "
                f"foot geom for {leg}. "
                "Contact-enabled named geoms include: "
                f"{contact_named[:80]}"
            )

        # Prefer the first exact/runtime match.
        geom_id, name = candidates[0]

        resolved[leg] = int(
            geom_id
        )

        resolved_names[leg] = str(
            name
        )

    ids = list(
        resolved.values()
    )

    if len(set(ids)) != 4:
        raise RuntimeError(
            "Foot geom resolution produced "
            f"non-unique IDs: {resolved}"
        )

    FOOT_GEOM_IDS = dict(
        resolved
    )

    FOOT_GEOM_NAMES = dict(
        resolved_names
    )

    print(
        "[M7 slip tap] resolved runtime "
        f"foot geoms: {FOOT_GEOM_NAMES}"
    )

    return dict(
        FOOT_GEOM_IDS
    )


def _static_environment_geom(
    model,
    geom_id,
):
    """
    Treat world-body geoms and fixed/static bodies as
    environment geometry.

    This supports flat ground as well as static generated
    terrain without requiring terrain geom names.
    """

    body_id = int(
        model.geom_bodyid[
            int(geom_id)
        ]
    )

    if body_id == 0:
        return True

    return (
        int(
            model.body_dofnum[
                body_id
            ]
        )
        == 0
    )


def _stance_slip_speeds(
    self,
):
    """
    Return one tangential contact-point speed per contacting
    foot for the current post-step MuJoCo state.

    If a foot has multiple contacts, keep the maximum
    tangential speed for that foot during this physics step.

    Contact-point translational velocity is computed via the
    MuJoCo body-point Jacobian:

        v_contact = J_p(q) qdot

    and projected onto the contact tangent plane:

        v_t = v - (v . n) n
    """

    model = self.mjModel
    data = self.mjData

    foot_ids = (
        _resolve_foot_geom_ids(
            self
        )
    )

    foot_by_geom = {
        int(geom_id): leg
        for leg, geom_id
        in foot_ids.items()
    }

    qvel = np.asarray(
        data.qvel,
        dtype=float,
    )

    best = {}

    for contact_index in range(
        int(data.ncon)
    ):
        contact = data.contact[
            contact_index
        ]

        geom1 = int(
            contact.geom1
        )

        geom2 = int(
            contact.geom2
        )

        if geom1 in foot_by_geom:
            foot_geom = geom1
            other_geom = geom2

        elif geom2 in foot_by_geom:
            foot_geom = geom2
            other_geom = geom1

        else:
            continue

        if not _static_environment_geom(
            model,
            other_geom,
        ):
            continue

        leg = foot_by_geom[
            foot_geom
        ]

        body_id = int(
            model.geom_bodyid[
                foot_geom
            ]
        )

        point = np.asarray(
            contact.pos,
            dtype=float,
        ).reshape(3)

        jacp = np.zeros(
            (
                3,
                int(model.nv),
            ),
            dtype=float,
        )

        jacr = np.zeros(
            (
                3,
                int(model.nv),
            ),
            dtype=float,
        )

        mujoco.mj_jac(
            model,
            data,
            jacp,
            jacr,
            point,
            body_id,
        )

        velocity_world = (
            jacp @ qvel
        )

        frame = np.asarray(
            contact.frame,
            dtype=float,
        ).reshape(-1)

        if frame.size < 3:
            raise RuntimeError(
                "Unexpected MuJoCo contact frame "
                f"size: {frame.size}"
            )

        normal = frame[:3].copy()

        normal_norm = float(
            np.linalg.norm(
                normal
            )
        )

        if (
            not np.isfinite(
                normal_norm
            )
            or normal_norm <= 1e-12
        ):
            raise RuntimeError(
                "Invalid MuJoCo contact normal"
            )

        normal /= normal_norm

        tangential_velocity = (
            velocity_world
            - float(
                np.dot(
                    velocity_world,
                    normal,
                )
            )
            * normal
        )

        speed = float(
            np.linalg.norm(
                tangential_velocity
            )
        )

        if not np.isfinite(speed):
            raise RuntimeError(
                "Non-finite stance slip speed"
            )

        best[leg] = max(
            speed,
            best.get(
                leg,
                0.0,
            ),
        )

    return best


def _planned_contact_snapshot(
    self,
):
    """
    Read the same PyMPC planned-contact state used by the
    standalone M4/M5 integration.

    This is read-only controller telemetry.
    """

    planned = np.asarray(
        standalone.contact_array(
            self.wb_interface.current_contact
        ),
        dtype=bool,
    ).reshape(-1)

    if planned.size != len(
        FOOT_LEGS
    ):
        raise RuntimeError(
            "Unexpected planned-contact vector size: "
            f"{planned.size}"
        )

    return {
        leg: bool(
            planned[index]
        )
        for index, leg
        in enumerate(
            FOOT_LEGS
        )
    }


def _accumulate_planned_stance_slip_shadow(
    self,
    *,
    dt,
):
    """
    OS-T4.6b evaluation-only traction shadow.

    planned stance:
        controls transient age

    actual MuJoCo contact:
        controls whether a slip sample exists

    Therefore a micro contact loss during a still-planned
    stance does NOT create a new 40 ms grace period.
    """

    global SLIP_ACCUMULATOR

    dt = float(dt)

    if (
        not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise RuntimeError(
            "Invalid planned-stance shadow dt: "
            f"{dt!r}"
        )

    if LATEST_PLANNED_CONTACT is None:
        return

    planned = (
        LATEST_PLANNED_CONTACT
    )

    # Planned stance age advances regardless of whether the
    # physical foot momentarily loses contact.
    for leg in FOOT_LEGS:
        if bool(
            planned[leg]
        ):
            SLIP_ACCUMULATOR[
                "planned_stance_age_s"
            ][leg] += dt

        else:
            SLIP_ACCUMULATOR[
                "planned_stance_age_s"
            ][leg] = 0.0

    # Slip itself still requires physical contact.
    speeds = (
        _stance_slip_speeds(
            self
        )
    )

    for leg, speed in speeds.items():
        if not bool(
            planned[leg]
        ):
            continue

        age_s = float(
            SLIP_ACCUMULATOR[
                "planned_stance_age_s"
            ][leg]
        )

        # Preserve the existing 40 ms numerical comparator
        # exactly for this shadow experiment. OS-T4.6b changes
        # ONLY the age semantics, not the gate threshold.
        if (
            age_s
            < ESTABLISHED_STANCE_GATE_S
        ):
            continue

        traction_cost = (
            _pointwise_traction_cost(
                float(speed)
            )
        )

        SLIP_ACCUMULATOR[
            "planned_established_contact_time_s"
        ] += dt

        SLIP_ACCUMULATOR[
            "planned_established_contact_samples"
        ] += 1

        SLIP_ACCUMULATOR[
            "planned_established_traction_cost_time_sum_s"
        ] += (
            traction_cost
            * dt
        )


def _accumulate_latched_stance_slip_shadow(
    self,
    *,
    dt,
):
    """
    OS-T4.6c evaluation-only candidate.

    Gate clock starts at the FIRST actual physical contact
    within each PyMPC-planned stance episode.

    Once started, short actual-contact losses do not reset
    the clock. Only planned swing terminates the episode.

    This preserves the original 40 ms touchdown-transient
    suppression while removing the v3 micro-contact-reset
    loophole.
    """

    global SLIP_ACCUMULATOR

    dt = float(dt)

    if (
        not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise RuntimeError(
            "Invalid latched-stance shadow dt: "
            f"{dt!r}"
        )

    if LATEST_PLANNED_CONTACT is None:
        return

    planned = (
        LATEST_PLANNED_CONTACT
    )

    speeds = (
        _stance_slip_speeds(
            self
        )
    )

    # --------------------------------------------------------
    # Update episode state.
    # --------------------------------------------------------
    for leg in FOOT_LEGS:
        planned_stance = bool(
            planned[leg]
        )

        actual_contact = bool(
            leg in speeds
        )

        if not planned_stance:
            # Planned lift-off is the ONLY reset event.
            SLIP_ACCUMULATOR[
                "latched_stance_contact_started"
            ][leg] = False

            SLIP_ACCUMULATOR[
                "latched_stance_contact_age_s"
            ][leg] = 0.0

            continue

        started = bool(
            SLIP_ACCUMULATOR[
                "latched_stance_contact_started"
            ][leg]
        )

        if not started:
            if actual_contact:
                SLIP_ACCUMULATOR[
                    "latched_stance_contact_started"
                ][leg] = True

                # First physical-contact sample has age=dt,
                # matching the existing v3 convention.
                SLIP_ACCUMULATOR[
                    "latched_stance_contact_age_s"
                ][leg] = dt

            else:
                # Planned stance but no touchdown yet.
                SLIP_ACCUMULATOR[
                    "latched_stance_contact_age_s"
                ][leg] = 0.0

            continue

        # Contact was previously established inside this
        # planned stance episode. Age advances even through a
        # momentary physical-contact loss.
        SLIP_ACCUMULATOR[
            "latched_stance_contact_age_s"
        ][leg] += dt

    # --------------------------------------------------------
    # Cost still requires CURRENT physical contact.
    # --------------------------------------------------------
    for leg, speed in speeds.items():
        if not bool(
            planned[leg]
        ):
            continue

        if not bool(
            SLIP_ACCUMULATOR[
                "latched_stance_contact_started"
            ][leg]
        ):
            continue

        age_s = float(
            SLIP_ACCUMULATOR[
                "latched_stance_contact_age_s"
            ][leg]
        )

        if (
            age_s
            < ESTABLISHED_STANCE_GATE_S
        ):
            continue

        traction_cost = (
            _pointwise_traction_cost(
                float(speed)
            )
        )

        SLIP_ACCUMULATOR[
            "latched_established_contact_time_s"
        ] += dt

        SLIP_ACCUMULATOR[
            "latched_established_contact_samples"
        ] += 1

        SLIP_ACCUMULATOR[
            "latched_established_traction_cost_time_sum_s"
        ] += (
            traction_cost
            * dt
        )


def _accumulate_stance_slip(
    self,
    *,
    dt,
):
    global SLIP_ACCUMULATOR

    dt = float(dt)

    if (
        not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise RuntimeError(
            f"Invalid slip sample dt: {dt!r}"
        )

    speeds = (
        _stance_slip_speeds(
            self
        )
    )

    # Update consecutive stance-contact age first.
    #
    # First contact sample therefore has age=dt.
    for leg in FOOT_LEGS:
        if leg in speeds:
            SLIP_ACCUMULATOR[
                "contact_age_s"
            ][leg] += dt

        else:
            SLIP_ACCUMULATOR[
                "contact_age_s"
            ][leg] = 0.0

    if not speeds:
        return

    SLIP_ACCUMULATOR[
        "physics_steps_with_contact"
    ] += 1

    for leg, speed in speeds.items():
        speed = float(speed)

        age_s = float(
            SLIP_ACCUMULATOR[
                "contact_age_s"
            ][leg]
        )

        bin_index = (
            _contact_age_bin_index(
                age_s
            )
        )

        age_bin = (
            SLIP_ACCUMULATOR[
                "contact_age_bins"
            ][bin_index]
        )

        # ----------------------------------------------------
        # Existing episode-wide sufficient statistics.
        # ----------------------------------------------------
        SLIP_ACCUMULATOR[
            "speed_time_sum_m"
        ] += (
            speed * dt
        )

        SLIP_ACCUMULATOR[
            "speed_sq_time_sum_m2ps"
        ] += (
            speed
            * speed
            * dt
        )

        SLIP_ACCUMULATOR[
            "contact_time_s"
        ] += dt

        SLIP_ACCUMULATOR[
            "speed_max_mps"
        ] = max(
            float(
                SLIP_ACCUMULATOR[
                    "speed_max_mps"
                ]
            ),
            speed,
        )

        SLIP_ACCUMULATOR[
            "contact_samples"
        ] += 1

        SLIP_ACCUMULATOR[
            "per_foot_contact_samples"
        ][leg] += 1

        # ----------------------------------------------------
        # New evaluation-only contact-age profile.
        # ----------------------------------------------------
        age_bin[
            "contact_time_s"
        ] += dt

        age_bin[
            "contact_samples"
        ] += 1

        age_bin[
            "speed_time_sum_m"
        ] += (
            speed * dt
        )

        age_bin[
            "speed_sq_time_sum_m2ps"
        ] += (
            speed
            * speed
            * dt
        )

        age_bin[
            "speed_max_mps"
        ] = max(
            float(
                age_bin[
                    "speed_max_mps"
                ]
            ),
            speed,
        )

        # ----------------------------------------------------
        # Established-stance distribution diagnostic.
        #
        # Only samples after the frozen 40 ms contact-age gate
        # contribute. No reward signal is changed here.
        # ----------------------------------------------------
        if (
            age_s
            >= ESTABLISHED_STANCE_GATE_S
        ):
            hist_index = (
                _established_slip_histogram_index(
                    speed
                )
            )

            hist_bin = (
                SLIP_ACCUMULATOR[
                    "established_slip_histogram"
                ][hist_index]
            )

            hist_bin[
                "contact_time_s"
            ] += dt

            hist_bin[
                "contact_samples"
            ] += 1

            SLIP_ACCUMULATOR[
                "established_contact_time_s"
            ] += dt

            SLIP_ACCUMULATOR[
                "established_contact_samples"
            ] += 1

            traction_cost = (
                _pointwise_traction_cost(
                    speed
                )
            )

            SLIP_ACCUMULATOR[
                "established_traction_cost_time_sum_s"
            ] += (
                traction_cost
                * dt
            )


def _contact_age_bins_payload():
    result = []

    for age_bin in (
        SLIP_ACCUMULATOR[
            "contact_age_bins"
        ]
    ):
        contact_time_s = float(
            age_bin[
                "contact_time_s"
            ]
        )

        if contact_time_s > 0.0:
            mean_mps = (
                float(
                    age_bin[
                        "speed_time_sum_m"
                    ]
                )
                / contact_time_s
            )

            rms_mps = float(
                np.sqrt(
                    max(
                        0.0,
                        float(
                            age_bin[
                                "speed_sq_time_sum_m2ps"
                            ]
                        )
                        / contact_time_s,
                    )
                )
            )

        else:
            mean_mps = None
            rms_mps = None

        result.append(
            {
                "low_s":
                    float(
                        age_bin[
                            "low_s"
                        ]
                    ),

                "high_s":
                    (
                        None
                        if age_bin[
                            "high_s"
                        ] is None
                        else float(
                            age_bin[
                                "high_s"
                            ]
                        )
                    ),

                "contact_time_s":
                    contact_time_s,

                "contact_samples":
                    int(
                        age_bin[
                            "contact_samples"
                        ]
                    ),

                "speed_time_sum_m":
                    float(
                        age_bin[
                            "speed_time_sum_m"
                        ]
                    ),

                "speed_sq_time_sum_m2ps":
                    float(
                        age_bin[
                            "speed_sq_time_sum_m2ps"
                        ]
                    ),

                "speed_mean_mps":
                    mean_mps,

                "speed_rms_mps":
                    rms_mps,

                "speed_max_mps":
                    float(
                        age_bin[
                            "speed_max_mps"
                        ]
                    ),
            }
        )

    return result


def _established_slip_histogram_payload():
    result = []

    total_time_s = float(
        SLIP_ACCUMULATOR[
            "established_contact_time_s"
        ]
    )

    for item in (
        SLIP_ACCUMULATOR[
            "established_slip_histogram"
        ]
    ):
        contact_time_s = float(
            item[
                "contact_time_s"
            ]
        )

        fraction = (
            0.0
            if total_time_s <= 0.0
            else (
                contact_time_s
                / total_time_s
            )
        )

        result.append(
            {
                "low_mps":
                    float(
                        item[
                            "low_mps"
                        ]
                    ),

                "high_mps":
                    (
                        None
                        if item[
                            "high_mps"
                        ] is None
                        else float(
                            item[
                                "high_mps"
                            ]
                        )
                    ),

                "contact_time_s":
                    contact_time_s,

                "contact_samples":
                    int(
                        item[
                            "contact_samples"
                        ]
                    ),

                "time_fraction":
                    float(
                        fraction
                    ),
            }
        )

    return result


def _slip_payload():
    contact_time = float(
        SLIP_ACCUMULATOR[
            "contact_time_s"
        ]
    )

    if contact_time > 0.0:
        mean_speed = (
            float(
                SLIP_ACCUMULATOR[
                    "speed_time_sum_m"
                ]
            )
            / contact_time
        )

        rms_speed = float(
            np.sqrt(
                max(
                    0.0,
                    float(
                        SLIP_ACCUMULATOR[
                            "speed_sq_time_sum_m2ps"
                        ]
                    )
                    / contact_time,
                )
            )
        )

    else:
        mean_speed = None
        rms_speed = None

    return {
        "measurement_role":
            "evaluation_only_contact_slip_v0",

        "speed_time_sum_m":
            float(
                SLIP_ACCUMULATOR[
                    "speed_time_sum_m"
                ]
            ),

        "speed_sq_time_sum_m2ps":
            float(
                SLIP_ACCUMULATOR[
                    "speed_sq_time_sum_m2ps"
                ]
            ),

        "contact_time_s":
            contact_time,

        "speed_mean_mps":
            mean_speed,

        "speed_rms_mps":
            rms_speed,

        "speed_max_mps":
            float(
                SLIP_ACCUMULATOR[
                    "speed_max_mps"
                ]
            ),

        "contact_samples":
            int(
                SLIP_ACCUMULATOR[
                    "contact_samples"
                ]
            ),

        "physics_steps_with_contact":
            int(
                SLIP_ACCUMULATOR[
                    "physics_steps_with_contact"
                ]
            ),

        "per_foot_contact_samples":
            dict(
                SLIP_ACCUMULATOR[
                    "per_foot_contact_samples"
                ]
            ),

        "foot_geom_names":
            (
                None
                if FOOT_GEOM_NAMES is None
                else dict(
                    FOOT_GEOM_NAMES
                )
            ),

        "contact_age_profile_role":
            "evaluation_only_contact_age_profile_v0",

        "contact_age_bin_width_s":
            float(
                CONTACT_AGE_BIN_WIDTH_S
            ),

        "contact_age_tail_start_s":
            float(
                CONTACT_AGE_TAIL_START_S
            ),

        "contact_age_bins":
            _contact_age_bins_payload(),

        "established_stance_role":
            "evaluation_only_established_stance_slip_v0",

        "established_stance_gate_s":
            float(
                ESTABLISHED_STANCE_GATE_S
            ),

        "established_contact_time_s":
            float(
                SLIP_ACCUMULATOR[
                    "established_contact_time_s"
                ]
            ),

        "established_contact_samples":
            int(
                SLIP_ACCUMULATOR[
                    "established_contact_samples"
                ]
            ),

        "traction_cost_role":
            "candidate_contact_stability_cost_v0",

        "traction_slip_deadband_mps":
            float(
                TRACTION_SLIP_DEADBAND_MPS
            ),

        "traction_slip_unsafe_mps":
            float(
                TRACTION_SLIP_UNSAFE_MPS
            ),

        "established_traction_cost_time_sum_s":
            float(
                SLIP_ACCUMULATOR[
                    "established_traction_cost_time_sum_s"
                ]
            ),

        # ----------------------------------------------------
        # OS-T4.6b planned-stance traction SHADOW.
        #
        # These fields are deliberately not consumed by
        # tracer_cost_v3 / M5 transport yet.
        # ----------------------------------------------------
        "latched_stance_shadow_role":
            "evaluation_only_latched_stance_traction_shadow_v0",

        # OS-T4.7a canonical v4 transport-facing role.
        #
        # The historical shadow role above is intentionally
        # preserved for audit reproducibility.
        "latched_traction_cost_role":
            "candidate_latched_contact_stability_cost_v0",

        "latched_stance_contact_started":
            {
                leg: bool(
                    SLIP_ACCUMULATOR[
                        "latched_stance_contact_started"
                    ][leg]
                )
                for leg in FOOT_LEGS
            },

        "latched_stance_contact_age_s":
            {
                leg: float(
                    SLIP_ACCUMULATOR[
                        "latched_stance_contact_age_s"
                    ][leg]
                )
                for leg in FOOT_LEGS
            },

        "latched_established_contact_time_s":
            float(
                SLIP_ACCUMULATOR[
                    "latched_established_contact_time_s"
                ]
            ),

        "latched_established_contact_samples":
            int(
                SLIP_ACCUMULATOR[
                    "latched_established_contact_samples"
                ]
            ),

        "latched_established_traction_cost_time_sum_s":
            float(
                SLIP_ACCUMULATOR[
                    "latched_established_traction_cost_time_sum_s"
                ]
            ),

        "planned_stance_shadow_role":
            "evaluation_only_planned_stance_traction_shadow_v0",

        "planned_stance_age_s":
            {
                leg: float(
                    SLIP_ACCUMULATOR[
                        "planned_stance_age_s"
                    ][leg]
                )
                for leg in FOOT_LEGS
            },

        "planned_established_contact_time_s":
            float(
                SLIP_ACCUMULATOR[
                    "planned_established_contact_time_s"
                ]
            ),

        "planned_established_contact_samples":
            int(
                SLIP_ACCUMULATOR[
                    "planned_established_contact_samples"
                ]
            ),

        "planned_established_traction_cost_time_sum_s":
            float(
                SLIP_ACCUMULATOR[
                    "planned_established_traction_cost_time_sum_s"
                ]
            ),

        "established_slip_histogram":
            _established_slip_histogram_payload(),
    }


def _append_slip_trace_post_step(
    self,
    *,
    time_pre_s,
    time_post_s,
    dt,
):
    """
    Evaluation-only physics-step trace.

    The caller invokes this immediately after the existing
    _accumulate_stance_slip() update, therefore contact_age_s
    corresponds to the same post-step physical state.

    No traced value is fed back into control or reward.
    """

    if not SLIP_TRACE_PATH:
        return

    if (
        SLIP_TRACE_TMIN_S is None
        or SLIP_TRACE_TMAX_S is None
    ):
        raise RuntimeError(
            "Slip trace enabled without "
            "a complete time window."
        )

    time_pre_s = float(
        time_pre_s
    )

    time_post_s = float(
        time_post_s
    )

    dt = float(dt)

    if (
        not np.isfinite(time_pre_s)
        or not np.isfinite(time_post_s)
        or not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise RuntimeError(
            "Invalid physics-step timing for "
            "slip trace."
        )

    # k6f contract:
    #
    #     (TMIN, TMAX]
    #
    # Use a small tolerance on the OPEN lower boundary so
    # accumulated floating-point time at exactly TMIN does not
    # accidentally create a 101st row.
    if not (
        time_post_s
        > float(SLIP_TRACE_TMIN_S)
        + 1e-12
        and time_post_s
        <= float(SLIP_TRACE_TMAX_S)
        + 1e-12
    ):
        return

    # Read the same post-step MuJoCo contact/slip state used by
    # the existing accumulator. This function is read-only.
    speeds = (
        _stance_slip_speeds(
            self
        )
    )

    physics_step_num_post = int(
        round(
            time_post_s
            / dt
        )
    )

    feet = {}

    for leg in FOOT_LEGS:
        contact = bool(
            leg in speeds
        )

        if contact:
            speed = float(
                speeds[leg]
            )

            age_s = float(
                SLIP_ACCUMULATOR[
                    "contact_age_s"
                ][leg]
            )

            established = bool(
                age_s
                >= ESTABLISHED_STANCE_GATE_S
            )

            traction_cost = float(
                _pointwise_traction_cost(
                    speed
                )
            )

        else:
            speed = None
            age_s = 0.0
            established = False
            traction_cost = None

        feet[leg] = {
            "contact":
                contact,

            "established":
                established,

            "contact_age_s":
                float(age_s),

            "slip_speed_mps":
                (
                    None
                    if speed is None
                    else float(speed)
                ),

            "traction_cost":
                (
                    None
                    if traction_cost is None
                    else float(
                        traction_cost
                    )
                ),
        }

    planned_stance_shadow = {}

    for leg in FOOT_LEGS:
        if LATEST_PLANNED_CONTACT is None:
            planned_contact = None
            planned_age_s = None
            planned_established = None

        else:
            planned_contact = bool(
                LATEST_PLANNED_CONTACT[
                    leg
                ]
            )

            planned_age_s = float(
                SLIP_ACCUMULATOR[
                    "planned_stance_age_s"
                ][leg]
            )

            # Eligibility requires BOTH:
            #   planned established stance
            #   actual physical contact
            planned_established = bool(
                planned_contact
                and feet[leg]["contact"]
                and (
                    planned_age_s
                    >= ESTABLISHED_STANCE_GATE_S
                )
            )

        planned_stance_shadow[
            leg
        ] = {
            "planned_contact":
                planned_contact,

            "planned_stance_age_s":
                planned_age_s,

            "eligible":
                planned_established,
        }

    row = {
        "schema":
            "icra27_m7_per_foot_slip_500hz_v0",

        "tag":
            SLIP_TRACE_TAG,

        "episode_index":
            int(EPISODE_INDEX),

        "time_pre_s":
            time_pre_s,

        "time_post_s":
            time_post_s,

        "dt_s":
            dt,

        "physics_step_num_post":
            physics_step_num_post,

        "established_stance_gate_s":
            float(
                ESTABLISHED_STANCE_GATE_S
            ),

        "feet":
            feet,

        "planned_stance_shadow":
            planned_stance_shadow,

        "latched_stance_shadow":
            {
                leg: {
                    "started":
                        bool(
                            SLIP_ACCUMULATOR[
                                "latched_stance_contact_started"
                            ][leg]
                        ),

                    "age_s":
                        float(
                            SLIP_ACCUMULATOR[
                                "latched_stance_contact_age_s"
                            ][leg]
                        ),

                    "eligible":
                        bool(
                            LATEST_PLANNED_CONTACT
                            is not None
                            and bool(
                                LATEST_PLANNED_CONTACT[
                                    leg
                                ]
                            )
                            and bool(
                                feet[leg][
                                    "contact"
                                ]
                            )
                            and bool(
                                SLIP_ACCUMULATOR[
                                    "latched_stance_contact_started"
                                ][leg]
                            )
                            and (
                                float(
                                    SLIP_ACCUMULATOR[
                                        "latched_stance_contact_age_s"
                                    ][leg]
                                )
                                >=
                                ESTABLISHED_STANCE_GATE_S
                            )
                        ),
                }
                for leg in FOOT_LEGS
            },
    }

    path = Path(
        SLIP_TRACE_PATH
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                row,
                separators=(",", ":"),
            )
            + "\n"
        )


def _append_energy_log(
    row: dict,
) -> None:
    """
    Optional logging-only mechanical-energy side channel.

    This does not alter M5 telemetry, M7 observation,
    task reward, or controller behavior.
    """

    if not ENERGY_LOG_PATH:
        return

    path = Path(
        ENERGY_LOG_PATH
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                row,
                separators=(",", ":"),
            )
            + "\n"
        )


class M7StateUdpSender:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        state_hz: float,
    ) -> None:
        self.addr = (
            str(host),
            int(port),
        )

        self.period_s = (
            1.0
            / max(float(state_hz), 1e-6)
        )

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.seq = 0
        self.last_send_time = None

    def ready(self) -> bool:
        now = time.monotonic()

        if self.last_send_time is None:
            return True

        return (
            now - self.last_send_time
            >= self.period_s
        )

    def send(
        self,
        payload: dict,
    ) -> None:
        payload = dict(payload)
        payload["seq"] = self.seq

        raw = encode_state_payload(
            payload
        )

        self.sock.sendto(
            raw,
            self.addr,
        )

        self.seq += 1
        self.last_send_time = (
            time.monotonic()
        )

    def close(self) -> None:
        self.sock.close()


def _arg(
    args,
    kwargs,
    *,
    name: str,
    index: int,
):
    if name in kwargs:
        return kwargs[name]

    if len(args) <= index:
        raise RuntimeError(
            f"Missing compute_actions argument "
            f"{name!r} at index {index}"
        )

    return args[index]


def m7_compute_actions(
    self,
    *args,
    **kwargs,
):
    """
    Read-only wrapper around the already validated
    standalone M4/M5 compute-actions hook.

    The captured physical state is the controller-input
    state immediately before env.step().
    """

    global LATEST_STATE
    global LATEST_PLANNED_CONTACT

    base_pos = _arg(
        args,
        kwargs,
        name="base_pos",
        index=1,
    )

    base_lin_vel = _arg(
        args,
        kwargs,
        name="base_lin_vel",
        index=2,
    )

    base_ori = _arg(
        args,
        kwargs,
        name="base_ori_euler_xyz",
        index=3,
    )

    base_ang_vel = _arg(
        args,
        kwargs,
        name="base_ang_vel",
        index=4,
    )

    simulation_dt = _arg(
        args,
        kwargs,
        name="simulation_dt",
        index=10,
    )

    step_num = _arg(
        args,
        kwargs,
        name="step_num",
        index=13,
    )

    pos = np.asarray(
        base_pos,
        dtype=float,
    ).reshape(-1)

    vel_world = np.asarray(
        base_lin_vel,
        dtype=float,
    ).reshape(-1)

    rpy = np.asarray(
        base_ori,
        dtype=float,
    ).reshape(-1)

    ang_base = np.asarray(
        base_ang_vel,
        dtype=float,
    ).reshape(-1)

    if (
        len(pos) < 3
        or len(vel_world) < 3
        or len(rpy) < 3
        or len(ang_base) < 3
    ):
        raise RuntimeError(
            "Unexpected base-state vector size"
        )

    yaw = float(rpy[2])

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))

    vx_world = float(
        vel_world[0]
    )

    vy_world = float(
        vel_world[1]
    )

    # Same yaw-projection convention already used
    # by the validated M3 integration instrumentation.
    vx_body = (
        c * vx_world
        + s * vy_world
    )

    vy_body = (
        -s * vx_world
        + c * vy_world
    )

    dt = float(simulation_dt)

    LATEST_STATE = {
        "schema": SCHEMA,
        "episode_index":
            int(EPISODE_INDEX),

        "sample_time_s":
            float(step_num) * dt,

        "lowlevel_dt_s":
            dt,

        "sample_phase":
            "controller_input_pre_env_step",

        "base_position_world": [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
        ],

        "base_rpy": [
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        ],

        "base_linear_velocity_world": [
            float(vel_world[0]),
            float(vel_world[1]),
            float(vel_world[2]),
        ],

        "base_linear_velocity_body_yaw": [
            float(vx_body),
            float(vy_body),
            float(vel_world[2]),
        ],

        "base_angular_velocity_base": [
            float(ang_base[0]),
            float(ang_base[1]),
            float(ang_base[2]),
        ],
    }

    # Execute the exact frozen controller first.
    #
    # WBInterface.update_state_and_reference() updates the
    # existing PyMPC TerrainEstimator from this same
    # controller-input state.
    result = ORIGINAL_STANDALONE_COMPUTE_ACTIONS(
        self,
        *args,
        **kwargs,
    )

    # --------------------------------------------------------
    # OS-T4.6b same-tick planned-contact snapshot.
    #
    # The frozen standalone controller has now updated
    # WBInterface.current_contact for this physics tick.
    # env.step() has NOT happened yet.
    # --------------------------------------------------------
    LATEST_PLANNED_CONTACT = (
        _planned_contact_snapshot(
            self
        )
    )

    # --------------------------------------------------------
    # Read-only reward/diagnostic telemetry.
    #
    # Do NOT feed these scalars into the frozen 21-D M7
    # observation. They expose estimates already computed
    # internally by the frozen PyMPC controller.
    # --------------------------------------------------------
    terrain_computation = (
        self.wb_interface.terrain_computation
    )

    terrain_height = float(
        terrain_computation.terrain_height
    )

    robot_height = float(
        terrain_computation.robot_height
    )

    if not np.isfinite(terrain_height):
        raise RuntimeError(
            "Non-finite PyMPC terrain-height estimate: "
            f"{terrain_height!r}"
        )

    if not np.isfinite(robot_height):
        raise RuntimeError(
            "Non-finite PyMPC robot-height estimate: "
            f"{robot_height!r}"
        )

    if LATEST_STATE is None:
        raise RuntimeError(
            "PyMPC estimator updated before "
            "LATEST_STATE was captured"
        )

    LATEST_STATE[
        "pympc_terrain_height_estimate_world"
    ] = terrain_height

    LATEST_STATE[
        "pympc_robot_height_estimate"
    ] = robot_height

    LATEST_STATE[
        "pympc_height_estimate_phase"
    ] = "post_controller_compute_pre_env_step"

    return result


def _native_reward(
    result,
):
    try:
        return float(result[1])
    except Exception:
        return None


def m7_udp_env_step(
    self,
    action,
):
    """
    Preserve the exact frozen M5 env-step semantics,
    while adding a read-only mechanical-energy tap.

    Timing:
      1. snapshot qvel/time before mj_step
      2. compute commanded torque power
      3. execute frozen M5/MuJoCo step
      4. read qfrc_actuator generated during that step
      5. pair it with the pre-step qvel snapshot
    """

    global ENERGY_ACCUMULATOR
    global LATEST_ENERGY_SAMPLE
    global SLIP_ACCUMULATOR

    qvel_pre = np.asarray(
        self.mjData.qvel,
        dtype=float,
    ).copy()

    base_position_pre = np.asarray(
        self.mjData.qpos[:3],
        dtype=float,
    ).copy()

    time_pre = float(
        self.mjData.time
    )

    commanded_power = (
        commanded_joint_power(
            self,
            action,
        )
    )

    # Exact frozen M5/MuJoCo behavior.
    result = ORIGINAL_M5_UDP_ENV_STEP(
        self,
        action,
    )

    time_post = float(
        self.mjData.time
    )

    base_position_post = np.asarray(
        self.mjData.qpos[:3],
        dtype=float,
    ).copy()

    robot_mass_kg = float(
        np.sum(
            np.asarray(
                self.mjModel.body_mass,
                dtype=float,
            )
        )
    )

    if (
        not np.isfinite(robot_mass_kg)
        or robot_mass_kg <= 0.0
    ):
        raise RuntimeError(
            "Invalid MuJoCo robot mass: "
            f"{robot_mass_kg!r}"
        )

    dt = (
        time_post
        - time_pre
    )

    if not np.isfinite(dt) or dt <= 0.0:
        raise RuntimeError(
            "Unexpected MuJoCo step dt: "
            f"{dt!r}"
        )

    applied_power = (
        applied_generalized_power(
            self,
            qvel_snapshot=qvel_pre,
        )
    )

    ENERGY_ACCUMULATOR.add(
        dt=dt,
        commanded=commanded_power,
        applied=applied_power,
    )

    # Evaluation-only post-step stance-foot slip tap.
    # This does not feed reward, observation, safety, or
    # controller state.
    _accumulate_stance_slip(
        self,
        dt=dt,
    )

    # OS-T4.6b evaluation-only planned-stance shadow.
    #
    # This does NOT replace or modify the v3 accumulator above.
    _accumulate_planned_stance_slip_shadow(
        self,
        dt=dt,
    )

    # OS-T4.6c second shadow candidate.
    #
    # first physical touchdown starts the 40 ms clock;
    # planned swing, not a micro contact break, resets it.
    _accumulate_latched_stance_slip_shadow(
        self,
        dt=dt,
    )

    # Evaluation-only 500 Hz per-foot snapshot.
    #
    # IMPORTANT:
    # this is deliberately AFTER the existing accumulator
    # update and BEFORE any state packet rate limiting.
    _append_slip_trace_post_step(
        self,
        time_pre_s=time_pre,
        time_post_s=time_post,
        dt=dt,
    )

    LATEST_ENERGY_SAMPLE = {
        "episode_index":
            int(EPISODE_INDEX),

        "time_pre_s":
            float(time_pre),

        "time_post_s":
            float(time_post),

        "dt_s":
            float(dt),

        "robot_mass_kg":
            float(robot_mass_kg),

        "base_position_pre_world":
            [
                float(x)
                for x in base_position_pre
            ],

        "base_position_post_world":
            [
                float(x)
                for x in base_position_post
            ],

        "commanded_power_w":
            dict(commanded_power),

        "applied_power_w":
            dict(applied_power),

        "cumulative_energy":
            ENERGY_ACCUMULATOR.as_dict(),
    }

    # Keep energy as a separate local diagnostic channel.
    # Do not modify the validated M7 state protocol.
    _append_energy_log(
        LATEST_ENERGY_SAMPLE
    )

    if (
        STATE_SENDER is None
        or LATEST_STATE is None
        or not STATE_SENDER.ready()
    ):
        return result

    payload = dict(
        LATEST_STATE
    )

    payload["terminated"] = bool(
        result[2]
    )

    payload["truncated"] = bool(
        result[3]
    )

    payload["native_reward"] = (
        _native_reward(result)
    )

    if LATEST_ENERGY_SAMPLE is None:
        raise RuntimeError(
            "State packet is ready but "
            "mechanical-energy sample is missing"
        )

    payload["energy_sample_time_s"] = float(
        LATEST_ENERGY_SAMPLE[
            "time_post_s"
        ]
    )

    payload["mechanical_energy"] = dict(
        LATEST_ENERGY_SAMPLE[
            "cumulative_energy"
        ]
    )

    payload["eval_stance_slip"] = (
        _slip_payload()
    )

    STATE_SENDER.send(
        payload
    )

    return result


def m7_wrapper_reset(
    self,
    initial_feet_pos,
):
    global LATEST_STATE
    global LATEST_PLANNED_CONTACT
    global EPISODE_INDEX
    global ENERGY_ACCUMULATOR
    global LATEST_ENERGY_SAMPLE
    global SLIP_ACCUMULATOR

    result = (
        ORIGINAL_STANDALONE_WRAPPER_RESET(
            self,
            initial_feet_pos,
        )
    )

    EPISODE_INDEX += 1
    LATEST_STATE = None
    LATEST_PLANNED_CONTACT = None

    ENERGY_ACCUMULATOR = (
        MechanicalEnergyAccumulator()
    )

    LATEST_ENERGY_SAMPLE = None

    SLIP_ACCUMULATOR = (
        _new_slip_accumulator()
    )

    print(
        "[M7 state tap] reset "
        f"episode={EPISODE_INDEX}"
    )

    return result


def main():
    global STATE_SENDER

    host = os.environ.get(
        "TRACER_M7_STATE_HOST",
        "127.0.0.1",
    )

    port = int(
        os.environ.get(
            "TRACER_M7_STATE_PORT",
            "50512",
        )
    )

    state_hz = float(
        os.environ.get(
            "TRACER_M7_STATE_HZ",
            "20.0",
        )
    )

    STATE_SENDER = M7StateUdpSender(
        host=host,
        port=port,
        state_hz=state_hz,
    )

    # M5 main() installs these functions into the
    # simulator. Replace only module-level hook targets;
    # no frozen source file is changed.
    standalone.standalone_compute_actions = (
        m7_compute_actions
    )

    standalone.standalone_wrapper_reset = (
        m7_wrapper_reset
    )

    m5.udp_env_step = (
        m7_udp_env_step
    )

    print("=" * 72)
    print(
        "ICRA27 M7 READ-ONLY PYMPC STATE TAP"
    )
    print("=" * 72)
    print(
        f"state UDP       : {host}:{port}"
    )
    print(
        f"state publish Hz: {state_hz:.3f}"
    )
    print(
        "sample phase    : "
        "controller_input_pre_env_step"
    )
    print(
        "frozen M5 source: unchanged"
    )
    print("=" * 72)

    try:
        m5.main()

    finally:
        standalone.standalone_compute_actions = (
            ORIGINAL_STANDALONE_COMPUTE_ACTIONS
        )

        standalone.standalone_wrapper_reset = (
            ORIGINAL_STANDALONE_WRAPPER_RESET
        )

        m5.udp_env_step = (
            ORIGINAL_M5_UDP_ENV_STEP
        )

        if STATE_SENDER is not None:
            STATE_SENDER.close()


if __name__ == "__main__":
    main()
