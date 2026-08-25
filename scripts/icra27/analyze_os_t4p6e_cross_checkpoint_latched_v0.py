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
        # Latest T4.6c trace contains the latched shadow.
        return (
            RESULTS
            / "os_t4p6b_u30_k11_"
              "planned_stance_shadow_v0"
        )

    return (
        RESULTS
        / (
            f"os_t4p6e_u{update}_"
            "k11_latched_shadow_v0"
        )
    )


def load(
    update,
    beta,
):
    directory = (
        trace_dir(update)
    )

    matches = list(
        directory.glob(
            f"u{update}_{beta}_"
            "*slip500.jsonl"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"u{update}/{beta}: "
            f"expected one trace, "
            f"got {matches}"
        )

    rows = [
        json.loads(line)
        for line
        in matches[
            0
        ].read_text().splitlines()
        if line.strip()
    ]

    if len(rows) != 100:
        raise RuntimeError(
            f"u{update}/{beta}: "
            f"expected 100 rows, "
            f"got {len(rows)}"
        )

    for row in rows:
        if (
            "latched_stance_shadow"
            not in row
        ):
            raise RuntimeError(
                f"u{update}/{beta}: "
                "missing latched shadow"
            )

    return rows


def aggregate(rows):
    raw_num = 0.0
    raw_den = 0.0

    v3_num = 0.0
    v3_den = 0.0

    latched_num = 0.0
    latched_den = 0.0

    rr_raw_num = 0.0
    rr_raw_den = 0.0

    rr_v3_num = 0.0
    rr_v3_den = 0.0

    rr_latched_num = 0.0
    rr_latched_den = 0.0

    for row in rows:
        dt = float(
            row["dt_s"]
        )

        for leg in LEGS:
            foot = (
                row[
                    "feet"
                ][leg]
            )

            if not bool(
                foot["contact"]
            ):
                continue

            cost = float(
                foot[
                    "traction_cost"
                ]
            )

            raw_num += (
                cost * dt
            )
            raw_den += dt

            if leg == "RR":
                rr_raw_num += (
                    cost * dt
                )
                rr_raw_den += dt

            if bool(
                foot[
                    "established"
                ]
            ):
                v3_num += (
                    cost * dt
                )
                v3_den += dt

                if leg == "RR":
                    rr_v3_num += (
                        cost * dt
                    )
                    rr_v3_den += dt

            latched = (
                row[
                    "latched_stance_shadow"
                ][leg]
            )

            if bool(
                latched[
                    "eligible"
                ]
            ):
                latched_num += (
                    cost * dt
                )
                latched_den += dt

                if leg == "RR":
                    rr_latched_num += (
                        cost * dt
                    )
                    rr_latched_den += dt

    def safe_mean(
        num,
        den,
    ):
        return (
            num / den
            if den > 0.0
            else 0.0
        )

    return {
        "raw":
            safe_mean(
                raw_num,
                raw_den,
            ),

        "v3":
            safe_mean(
                v3_num,
                v3_den,
            ),

        "latched":
            safe_mean(
                latched_num,
                latched_den,
            ),

        "raw_dt":
            raw_den,

        "v3_dt":
            v3_den,

        "latched_dt":
            latched_den,

        "rr_raw":
            safe_mean(
                rr_raw_num,
                rr_raw_den,
            ),

        "rr_v3":
            safe_mean(
                rr_v3_num,
                rr_v3_den,
            ),

        "rr_latched":
            safe_mean(
                rr_latched_num,
                rr_latched_den,
            ),
    }


def sign_name(value):
    if value < 0.0:
        return "S_BETTER"

    if value > 0.0:
        return "S_WORSE"

    return "TIE"


def main():
    print("=" * 126)
    print(
        "ICRA27 OS-T4.6e "
        "CROSS-CHECKPOINT LATCHED-STANCE "
        "TRACTION VALIDATION"
    )
    print("=" * 126)

    all_pass = True

    summaries = {}

    for update in UPDATES:
        agg = {
            beta:
                aggregate(
                    load(
                        update,
                        beta,
                    )
                )
            for beta in BETAS
        }

        d_raw = (
            agg[
                "stability"
            ]["raw"]
            - agg[
                "energy"
            ]["raw"]
        )

        d_v3 = (
            agg[
                "stability"
            ]["v3"]
            - agg[
                "energy"
            ]["v3"]
        )

        d_latched = (
            agg[
                "stability"
            ]["latched"]
            - agg[
                "energy"
            ]["latched"]
        )

        d_rr_raw = (
            agg[
                "stability"
            ]["rr_raw"]
            - agg[
                "energy"
            ]["rr_raw"]
        )

        d_rr_v3 = (
            agg[
                "stability"
            ]["rr_v3"]
            - agg[
                "energy"
            ]["rr_v3"]
        )

        d_rr_latched = (
            agg[
                "stability"
            ]["rr_latched"]
            - agg[
                "energy"
            ]["rr_latched"]
        )

        # Historical problem:
        #
        # raw       says Stability is better
        # v3        says Stability is worse
        #
        # Candidate succeeds if it agrees with raw sign.
        update_pass = (
            d_raw < 0.0
            and d_v3 > 0.0
            and d_latched < 0.0
        )

        all_pass = (
            all_pass
            and update_pass
        )

        summaries[
            update
        ] = {
            "raw": d_raw,
            "v3": d_v3,
            "latched": d_latched,
            "pass": update_pass,
        }

        print()
        print("-" * 126)
        print(
            f"UPDATE u{update}"
        )
        print("-" * 126)

        for beta in BETAS:
            a = agg[beta]

            print(
                f"{beta:<9} "
                f"raw={a['raw']:.6f} "
                f"v3={a['v3']:.6f} "
                f"latched={a['latched']:.6f} "
                f"| eligible dt "
                f"v3={1000*a['v3_dt']:.1f} "
                f"L={1000*a['latched_dt']:.1f} ms"
            )

        print()

        print(
            "ALL-FOOT S-E: "
            f"raw={d_raw:+.6f} "
            f"v3={d_v3:+.6f} "
            f"latched={d_latched:+.6f}"
        )

        print(
            "RR-ONLY  S-E: "
            f"raw={d_rr_raw:+.6f} "
            f"v3={d_rr_v3:+.6f} "
            f"latched={d_rr_latched:+.6f}"
        )

        print(
            "signs         : "
            f"raw={sign_name(d_raw)} "
            f"v3={sign_name(d_v3)} "
            f"latched={sign_name(d_latched)}"
        )

        print(
            "verdict       :",
            "PASS"
            if update_pass
            else "FAIL",
        )

    print()
    print("=" * 126)
    print(
        "CROSS-CHECKPOINT VERDICT"
    )
    print("=" * 126)

    for update in UPDATES:
        s = summaries[
            update
        ]

        print(
            f"u{update}: "
            f"dRaw={s['raw']:+.6f} "
            f"dV3={s['v3']:+.6f} "
            f"dLatched={s['latched']:+.6f} "
            f"=> "
            f"{'PASS' if s['pass'] else 'FAIL'}"
        )

    print()

    if all_pass:
        print(
            "RESULT: LATCHED-STANCE TRACTION "
            "SEMANTICS CORRECT THE v3 SIGN "
            "INVERSION AT u10/u30/u40"
        )

        print(
            "INTERPRETATION: the v3 failure was "
            "systematically caused by binary-contact "
            "age reset, while the latched candidate "
            "preserves physical ordering."
        )

        print(
            "[ICRA27] OS-T4.6e "
            "cross-checkpoint latched validation: PASS"
        )

    else:
        print(
            "RESULT: candidate is not yet "
            "cross-checkpoint consistent."
        )

        print(
            "[ICRA27] OS-T4.6e "
            "cross-checkpoint latched validation: "
            "REQUIRES FURTHER AUDIT"
        )

    print("=" * 126)


if __name__ == "__main__":
    main()
