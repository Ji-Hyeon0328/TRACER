#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5k6_lowfriction_action_mechanism_v0"
    / "pairwise_deltas.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5k6c_transient_phase_v0"
)

UPDATES = (
    10,
    20,
    30,
    40,
)

BETAS = (
    "stability",
    "energy",
)

HL_DT_S = 0.20

GAIT_PERIOD_S = (
    0.7142857142857143
)

PHASE_REPEAT_STEPS = 25

# Diagnostic split only.
#
# IMPORTANT:
# This does NOT alter the frozen semantic gate.
# It only asks whether the observed gate result is
# dominated by the initial policy-launch transient.
EARLY_LAST_STEP = 2


def f(
    row,
    key,
):
    return float(
        row[key]
    )


def i(
    row,
    key,
):
    return int(
        float(
            row[key]
        )
    )


def mean(
    values,
):
    if not values:
        return float("nan")

    return (
        sum(values)
        / len(values)
    )


def sign(
    value,
    tol=1e-12,
):
    if value > tol:
        return 1

    if value < -tol:
        return -1

    return 0


def fmt(
    value,
):
    if not math.isfinite(
        value
    ):
        return "nan"

    return f"{value:+.6f}"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists; "
            "refusing overwrite: "
            f"{OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    with INPUT.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as fobj:
        rows = list(
            csv.DictReader(
                fobj
            )
        )

    if not rows:
        raise RuntimeError(
            "pairwise_deltas.csv is empty"
        )

    # --------------------------------------------------------
    # Arithmetic phase-repeat contract.
    # --------------------------------------------------------

    repeat_time = (
        PHASE_REPEAT_STEPS
        * HL_DT_S
    )

    repeat_gaits = (
        repeat_time
        / GAIT_PERIOD_S
    )

    if abs(
        repeat_gaits
        - round(
            repeat_gaits
        )
    ) > 1e-9:
        raise RuntimeError(
            "Expected 25-step HL/gait "
            "phase recurrence is not exact."
        )

    print("=" * 118)
    print(
        "ICRA27 OS-T4.5k6c "
        "TRANSIENT / PHASE RECURRENCE AUDIT"
    )
    print("=" * 118)

    print(
        f"HL dt          : "
        f"{HL_DT_S:.6f} s"
    )

    print(
        f"gait period    : "
        f"{GAIT_PERIOD_S:.12f} s"
    )

    print(
        f"phase repeat   : "
        f"{PHASE_REPEAT_STEPS} HL steps "
        f"= {repeat_time:.3f} s "
        f"= {repeat_gaits:.0f} gait cycles"
    )

    print(
        f"early window   : "
        f"k=0..{EARLY_LAST_STEP} "
        "(diagnostic only)"
    )

    print(
        "gate thresholds: UNCHANGED"
    )

    print()

    decomposition_rows = []

    # ========================================================
    # 1. Episode mean decomposition:
    #
    # total dCtr
    #   = launch transient contribution
    #   + post-launch contribution
    #
    # This is descriptive only.
    # ========================================================

    print("=" * 118)
    print(
        "A. LAUNCH-TRANSIENT VS POST-LAUNCH "
        "TRACTION DECOMPOSITION"
    )
    print("=" * 118)

    for update in UPDATES:
        for beta in BETAS:
            x = [
                row
                for row in rows
                if (
                    i(
                        row,
                        "update",
                    )
                    == update
                    and row[
                        "specialized_beta"
                    ]
                    == beta
                )
            ]

            x.sort(
                key=lambda row: i(
                    row,
                    "policy_step",
                )
            )

            if not x:
                raise RuntimeError(
                    f"No rows for "
                    f"u{update:02d}/{beta}"
                )

            early = [
                row
                for row in x
                if i(
                    row,
                    "policy_step",
                )
                <= EARLY_LAST_STEP
            ]

            post = [
                row
                for row in x
                if i(
                    row,
                    "policy_step",
                )
                > EARLY_LAST_STEP
            ]

            total_values = [
                f(
                    row,
                    "delta_cost_traction",
                )
                for row in x
            ]

            early_values = [
                f(
                    row,
                    "delta_cost_traction",
                )
                for row in early
            ]

            post_values = [
                f(
                    row,
                    "delta_cost_traction",
                )
                for row in post
            ]

            total_sum = sum(
                total_values
            )

            early_sum = sum(
                early_values
            )

            post_sum = sum(
                post_values
            )

            if abs(
                (
                    early_sum
                    + post_sum
                )
                - total_sum
            ) > 1e-10:
                raise RuntimeError(
                    "Transient decomposition "
                    "does not reconstruct total."
                )

            contribution_fraction = (
                float("nan")
                if abs(total_sum) < 1e-12
                else early_sum
                / total_sum
            )

            result = {
                "update":
                    update,

                "beta":
                    beta,

                "steps_total":
                    len(x),

                "steps_early":
                    len(early),

                "steps_post":
                    len(post),

                "mean_dCtr_total":
                    mean(
                        total_values
                    ),

                "mean_dCtr_early":
                    mean(
                        early_values
                    ),

                "mean_dCtr_post":
                    mean(
                        post_values
                    ),

                "sum_dCtr_total":
                    total_sum,

                "sum_dCtr_early":
                    early_sum,

                "sum_dCtr_post":
                    post_sum,

                "early_fraction_of_signed_total":
                    contribution_fraction,

                "post_sign":
                    sign(
                        mean(
                            post_values
                        )
                    ),
            }

            decomposition_rows.append(
                result
            )

            print(
                f"u{update:02d} "
                f"{beta.upper():<9} "
                f"total={fmt(result['mean_dCtr_total'])} "
                f"early={fmt(result['mean_dCtr_early'])} "
                f"post={fmt(result['mean_dCtr_post'])} "
                f"| early sum="
                f"{fmt(result['sum_dCtr_early'])} "
                f"post sum="
                f"{fmt(result['sum_dCtr_post'])} "
                f"share="
                f"{result['early_fraction_of_signed_total']:+.2f}"
            )

    # ========================================================
    # 2. Same nominal gait-phase recurrence:
    #
    # compare k and k+25 within the SAME update/beta.
    # ========================================================

    print()
    print("=" * 118)
    print(
        "B. 25-STEP SAME-NOMINAL-GAIT-PHASE PAIRS"
    )
    print("=" * 118)

    recurrence_rows = []

    for update in UPDATES:
        for beta in BETAS:
            x = [
                row
                for row in rows
                if (
                    i(
                        row,
                        "update",
                    )
                    == update
                    and row[
                        "specialized_beta"
                    ]
                    == beta
                )
            ]

            by_step = {
                i(
                    row,
                    "policy_step",
                ):
                    row
                for row in x
            }

            pairs = []

            for k in sorted(
                by_step
            ):
                k2 = (
                    k
                    + PHASE_REPEAT_STEPS
                )

                if k2 not in by_step:
                    continue

                a = by_step[k]
                b = by_step[k2]

                da = f(
                    a,
                    "delta_cost_traction",
                )

                db = f(
                    b,
                    "delta_cost_traction",
                )

                pairs.append(
                    {
                        "update":
                            update,

                        "beta":
                            beta,

                        "k0":
                            k,

                        "k1":
                            k2,

                        "dCtr_k":
                            da,

                        "dCtr_k_plus_25":
                            db,

                        "abs_difference":
                            abs(
                                db - da
                            ),

                        "same_sign":
                            int(
                                sign(da)
                                == sign(db)
                                and sign(da)
                                != 0
                            ),

                        "dVxRef_k":
                            f(
                                a,
                                "delta_req_vx_mps",
                            ),

                        "dVxRef_k_plus_25":
                            f(
                                b,
                                "delta_req_vx_mps",
                            ),

                        "dYawRef_k":
                            f(
                                a,
                                "delta_req_yaw_rate_rps",
                            ),

                        "dYawRef_k_plus_25":
                            f(
                                b,
                                "delta_req_yaw_rate_rps",
                            ),

                        "dHRef_k":
                            f(
                                a,
                                "delta_req_body_height_m",
                            ),

                        "dHRef_k_plus_25":
                            f(
                                b,
                                "delta_req_body_height_m",
                            ),
                    }
                )

            recurrence_rows.extend(
                pairs
            )

            if not pairs:
                print(
                    f"u{update:02d} "
                    f"{beta.upper():<9} "
                    "no repeated pairs"
                )
                continue

            sign_agree = sum(
                row[
                    "same_sign"
                ]
                for row in pairs
            )

            mean_abs_diff = mean(
                [
                    row[
                        "abs_difference"
                    ]
                    for row in pairs
                ]
            )

            print(
                f"u{update:02d} "
                f"{beta.upper():<9} "
                f"pairs={len(pairs):>2} "
                f"same-sign="
                f"{sign_agree}/{len(pairs)} "
                f"mean |dCtr(k+25)-dCtr(k)|="
                f"{mean_abs_diff:.6f}"
            )

    # ========================================================
    # 3. Stability-vs-Energy contrast at identical update/step.
    #
    # Positive contrast:
    #   stability traction outcome is worse than energy.
    # ========================================================

    print()
    print("=" * 118)
    print(
        "C. STABILITY-vs-ENERGY "
        "TRACTION CONTRAST"
    )
    print("=" * 118)

    contrast_rows = []

    for update in UPDATES:
        s_rows = {
            i(
                row,
                "policy_step",
            ):
                row
            for row in rows
            if (
                i(
                    row,
                    "update",
                )
                == update
                and row[
                    "specialized_beta"
                ]
                == "stability"
            )
        }

        e_rows = {
            i(
                row,
                "policy_step",
            ):
                row
            for row in rows
            if (
                i(
                    row,
                    "update",
                )
                == update
                and row[
                    "specialized_beta"
                ]
                == "energy"
            )
        }

        for k in sorted(
            set(s_rows)
            & set(e_rows)
        ):
            s = s_rows[k]
            e = e_rows[k]

            contrast_rows.append(
                {
                    "update":
                        update,

                    "policy_step":
                        k,

                    "phase_mod_25":
                        k
                        % PHASE_REPEAT_STEPS,

                    "stability_dCtr":
                        f(
                            s,
                            "delta_cost_traction",
                        ),

                    "energy_dCtr":
                        f(
                            e,
                            "delta_cost_traction",
                        ),

                    "S_minus_E_dCtr":
                        (
                            f(
                                s,
                                "delta_cost_traction",
                            )
                            - f(
                                e,
                                "delta_cost_traction",
                            )
                        ),

                    "S_minus_E_dVxRef":
                        (
                            f(
                                s,
                                "delta_req_vx_mps",
                            )
                            - f(
                                e,
                                "delta_req_vx_mps",
                            )
                        ),

                    "S_minus_E_dYawRef":
                        (
                            f(
                                s,
                                "delta_req_yaw_rate_rps",
                            )
                            - f(
                                e,
                                "delta_req_yaw_rate_rps",
                            )
                        ),

                    "S_minus_E_dHRef":
                        (
                            f(
                                s,
                                "delta_req_body_height_m",
                            )
                            - f(
                                e,
                                "delta_req_body_height_m",
                            )
                        ),
                }
            )

    top_all = sorted(
        contrast_rows,
        key=lambda row: (
            row[
                "S_minus_E_dCtr"
            ]
        ),
        reverse=True,
    )[:12]

    print(
        "Top Stability-worse-than-Energy events:"
    )

    for row in top_all:
        print(
            f"  u{row['update']:02d} "
            f"k={row['policy_step']:>2} "
            f"phase={row['phase_mod_25']:>2} "
            f"S-E dCtr="
            f"{row['S_minus_E_dCtr']:+.6f} "
            f"(S={row['stability_dCtr']:+.6f}, "
            f"E={row['energy_dCtr']:+.6f}) "
            f"dVxRef="
            f"{row['S_minus_E_dVxRef']:+.5f} "
            f"dYawRef="
            f"{row['S_minus_E_dYawRef']:+.5f} "
            f"dHRef="
            f"{1000.0 * row['S_minus_E_dHRef']:+.2f}mm"
        )

    # Same ranking excluding initial launch transient.
    steady = [
        row
        for row in contrast_rows
        if row[
            "policy_step"
        ] > EARLY_LAST_STEP
    ]

    top_steady = sorted(
        steady,
        key=lambda row: (
            row[
                "S_minus_E_dCtr"
            ]
        ),
        reverse=True,
    )[:12]

    print()
    print(
        "Top POST-LAUNCH "
        "Stability-worse-than-Energy events:"
    )

    for row in top_steady:
        print(
            f"  u{row['update']:02d} "
            f"k={row['policy_step']:>2} "
            f"phase={row['phase_mod_25']:>2} "
            f"S-E dCtr="
            f"{row['S_minus_E_dCtr']:+.6f} "
            f"(S={row['stability_dCtr']:+.6f}, "
            f"E={row['energy_dCtr']:+.6f})"
        )

    # --------------------------------------------------------
    # Persist.
    # --------------------------------------------------------

    def write_csv(
        path,
        data,
    ):
        if not data:
            raise RuntimeError(
                f"No rows for {path}"
            )

        fields = list(
            data[0].keys()
        )

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as fobj:
            writer = csv.DictWriter(
                fobj,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                data
            )

    write_csv(
        OUT_DIR
        / "transient_decomposition.csv",
        decomposition_rows,
    )

    write_csv(
        OUT_DIR
        / "phase_recurrence_pairs.csv",
        recurrence_rows,
    )

    write_csv(
        OUT_DIR
        / "stability_energy_contrast.csv",
        contrast_rows,
    )

    print()
    print("=" * 118)
    print(
        "[ICRA27] OS-T4.5k6c "
        "transient / phase audit: COMPUTE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
