#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

RESULTS = (
    ROOT
    / "results"
    / "icra27"
)

UPDATES = (
    10,
    30,
    40,
)

BETAS = (
    "balanced",
    "stability",
    "energy",
)

LEGS = (
    "FL",
    "FR",
    "RL",
    "RR",
)


def trace_dir(update):
    if update == 30:
        return (
            RESULTS
            / "os_t4p5k6f_"
              "u30_k11_perfoot_trace_v0"
        )

    return (
        RESULTS
        / (
            "os_t4p5k6h_"
            f"u{update}_"
            "k11_perfoot_trace_v0"
        )
    )


def trace_path(
    update,
    beta,
):
    return (
        trace_dir(update)
        / (
            f"u{update}_{beta}_"
            "low27200_k11_"
            "slip500.jsonl"
        )
    )


def read_jsonl(path):
    if not path.exists():
        raise RuntimeError(
            f"Missing trace: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_no}: "
                    "invalid JSON"
                ) from exc

    if len(rows) != 100:
        raise RuntimeError(
            f"{path}: expected 100 rows, "
            f"got {len(rows)}"
        )

    return rows


def aggregate(rows):
    raw_num = 0.0
    raw_den = 0.0

    est_num = 0.0
    est_den = 0.0

    excluded_num = 0.0

    for row in rows:
        dt = float(
            row["dt_s"]
        )

        for leg in LEGS:
            foot = (
                row["feet"][leg]
            )

            if not bool(
                foot["contact"]
            ):
                continue

            cost = float(
                foot["traction_cost"]
            )

            raw_num += (
                cost * dt
            )

            raw_den += dt

            if bool(
                foot["established"]
            ):
                est_num += (
                    cost * dt
                )

                est_den += dt

            else:
                excluded_num += (
                    cost * dt
                )

    return {
        "raw":
            (
                raw_num / raw_den
                if raw_den > 0.0
                else 0.0
            ),

        "established":
            (
                est_num / est_den
                if est_den > 0.0
                else 0.0
            ),

        "raw_num":
            raw_num,

        "est_num":
            est_num,

        "excluded_num":
            excluded_num,

        "raw_den":
            raw_den,

        "est_den":
            est_den,
    }


def rr_stats(rows):
    raw_num = 0.0
    raw_den = 0.0

    est_num = 0.0
    est_den = 0.0

    excluded_num = 0.0
    excluded_den = 0.0

    excluded_tail30 = 0.0
    max_excluded_speed = 0.0

    off_samples = 0
    contact_breaks = 0
    recontacts = 0

    previous_contact = None

    for row in rows:
        dt = float(
            row["dt_s"]
        )

        foot = (
            row["feet"]["RR"]
        )

        contact = bool(
            foot["contact"]
        )

        if previous_contact is not None:
            if (
                previous_contact
                and not contact
            ):
                contact_breaks += 1

            if (
                not previous_contact
                and contact
            ):
                recontacts += 1

        previous_contact = contact

        if not contact:
            off_samples += 1
            continue

        speed = float(
            foot["slip_speed_mps"]
        )

        cost = float(
            foot["traction_cost"]
        )

        raw_num += (
            cost * dt
        )
        raw_den += dt

        if bool(
            foot["established"]
        ):
            est_num += (
                cost * dt
            )
            est_den += dt
        else:
            excluded_num += (
                cost * dt
            )
            excluded_den += dt

            max_excluded_speed = max(
                max_excluded_speed,
                speed,
            )

            if speed >= 0.30:
                excluded_tail30 += dt

    return {
        "raw":
            (
                raw_num / raw_den
                if raw_den > 0.0
                else 0.0
            ),

        "est":
            (
                est_num / est_den
                if est_den > 0.0
                else 0.0
            ),

        "excluded":
            (
                excluded_num
                / excluded_den
                if excluded_den > 0.0
                else 0.0
            ),

        "excluded_tail30_s":
            excluded_tail30,

        "max_excluded_speed":
            max_excluded_speed,

        "contact_breaks":
            contact_breaks,

        "recontacts":
            recontacts,

        "off_samples":
            off_samples,
    }


def classify(
    d_raw,
    d_est,
):
    eps = 1e-12

    if (
        d_raw < -eps
        and d_est > eps
    ):
        return (
            "GATE_SIGN_INVERSION"
        )

    if (
        d_raw > eps
        and d_est > eps
    ):
        return (
            "SAME_SIGN_S_WORSE"
        )

    if (
        d_raw < -eps
        and d_est < -eps
    ):
        return (
            "SAME_SIGN_S_BETTER"
        )

    return (
        "MIXED_OR_NEAR_ZERO"
    )


def main():
    print("=" * 124)
    print(
        "ICRA27 OS-T4.5k6h "
        "CROSS-CHECKPOINT CONTACT-AGE "
        "GATE VALIDATION"
    )
    print("=" * 124)

    print(
        "terrain : low_friction"
    )
    print(
        "seed    : 27200"
    )
    print(
        "event   : k=11, "
        "(3.200, 3.400] s"
    )
    print(
        "updates : u10, u30, u40"
    )
    print()

    results = {}

    for update in UPDATES:
        traces = {
            beta:
                read_jsonl(
                    trace_path(
                        update,
                        beta,
                    )
                )
            for beta in BETAS
        }

        agg = {
            beta:
                aggregate(
                    traces[beta]
                )
            for beta in BETAS
        }

        rr = {
            beta:
                rr_stats(
                    traces[beta]
                )
            for beta in BETAS
        }

        d_raw_se = (
            agg["stability"]["raw"]
            - agg["energy"]["raw"]
        )

        d_est_se = (
            agg["stability"][
                "established"
            ]
            - agg["energy"][
                "established"
            ]
        )

        gate_shift = (
            d_est_se
            - d_raw_se
        )

        verdict = classify(
            d_raw_se,
            d_est_se,
        )

        results[update] = {
            "d_raw_se":
                d_raw_se,

            "d_est_se":
                d_est_se,

            "gate_shift":
                gate_shift,

            "verdict":
                verdict,
        }

        print("=" * 124)
        print(f"UPDATE u{update}")
        print("=" * 124)

        for beta in BETAS:
            a = agg[beta]

            excluded_share = (
                0.0
                if a["raw_num"] <= 0.0
                else (
                    a["excluded_num"]
                    / a["raw_num"]
                )
            )

            r = rr[beta]

            print(
                f"{beta:<9} "
                f"raw={a['raw']:.6f} "
                f"est={a['established']:.6f} "
                f"excludedCostTime="
                f"{100*excluded_share:6.2f}% "
                f"| RR raw={r['raw']:.6f} "
                f"est={r['est']:.6f} "
                f"excluded={r['excluded']:.6f}"
            )

            print(
                " " * 11
                + "RR "
                + f"excluded>=.30="
                + f"{1000*r['excluded_tail30_s']:.1f}ms "
                + f"maxExcluded="
                + f"{r['max_excluded_speed']:.4f} "
                + f"break/recontact="
                + f"{r['contact_breaks']}/"
                + f"{r['recontacts']}"
            )

        print()

        print(
            "S-E "
            f"dRaw={d_raw_se:+.6f} "
            f"dEstablished={d_est_se:+.6f} "
            f"gateShift={gate_shift:+.6f}"
        )

        print(
            "VERDICT:",
            verdict,
        )

        print()

    inversion_updates = [
        update
        for update in UPDATES
        if (
            results[update][
                "verdict"
            ]
            == "GATE_SIGN_INVERSION"
        )
    ]

    print("=" * 124)
    print("CROSS-CHECKPOINT VERDICT")
    print("=" * 124)

    for update in UPDATES:
        result = (
            results[update]
        )

        print(
            f"u{update}: "
            f"{result['verdict']:<22} "
            f"dRaw={result['d_raw_se']:+.6f} "
            f"dEst={result['d_est_se']:+.6f} "
            f"shift={result['gate_shift']:+.6f}"
        )

    print()

    if len(
        inversion_updates
    ) == len(UPDATES):
        print(
            "RESULT: REPLICATED "
            "GATE-INDUCED SIGN INVERSION "
            "AT u10/u30/u40"
        )

        print(
            "INTERPRETATION: current "
            "established-contact traction "
            "semantic is structurally "
            "confounded by contact-age reset."
        )

        print(
            "NEXT: freeze v3 artifacts; "
            "design tracer_cost_v4 traction "
            "gating before any retraining."
        )

    else:
        print(
            "RESULT: finding did NOT "
            "replicate across all three "
            "checkpoints."
        )

        print(
            "NEXT: inspect the differing "
            "checkpoint(s) before changing "
            "the reward."
        )

    print()
    print(
        "[ICRA27] OS-T4.5k6h "
        "cross-checkpoint gate validation: "
        "COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
