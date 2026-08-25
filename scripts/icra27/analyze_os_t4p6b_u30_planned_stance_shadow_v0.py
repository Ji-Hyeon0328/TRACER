#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TRACE_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p6b_u30_k11_planned_stance_shadow_v0"
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


def load(beta):
    path = (
        TRACE_DIR
        / (
            f"u30_{beta}_"
            "low27200_k11_slip500.jsonl"
        )
    )

    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    if len(rows) != 100:
        raise RuntimeError(
            f"{beta}: expected 100 rows, "
            f"got {len(rows)}"
        )

    return rows


def aggregate(rows):
    raw_num = 0.0
    raw_den = 0.0

    v3_num = 0.0
    v3_den = 0.0

    shadow_num = 0.0
    shadow_den = 0.0

    for row in rows:
        dt = float(
            row["dt_s"]
        )

        shadow = row.get(
            "planned_stance_shadow"
        )

        if shadow is None:
            raise RuntimeError(
                "Missing planned_stance_shadow"
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

            raw_num += cost * dt
            raw_den += dt

            if bool(
                foot["established"]
            ):
                v3_num += cost * dt
                v3_den += dt

            if bool(
                shadow[
                    leg
                ]["eligible"]
            ):
                shadow_num += cost * dt
                shadow_den += dt

    def mean(num, den):
        return (
            num / den
            if den > 0.0
            else 0.0
        )

    return {
        "raw":
            mean(
                raw_num,
                raw_den,
            ),

        "v3":
            mean(
                v3_num,
                v3_den,
            ),

        "shadow":
            mean(
                shadow_num,
                shadow_den,
            ),

        "raw_dt":
            raw_den,

        "v3_dt":
            v3_den,

        "shadow_dt":
            shadow_den,
    }


def main():
    traces = {
        beta:
            load(beta)
        for beta in BETAS
    }

    result = {
        beta:
            aggregate(
                traces[beta]
            )
        for beta in BETAS
    }

    print("=" * 112)
    print(
        "ICRA27 OS-T4.6b u30 "
        "PLANNED-STANCE TRACTION SHADOW"
    )
    print("=" * 112)

    for beta in BETAS:
        r = result[beta]

        print(
            f"{beta:<9} "
            f"raw={r['raw']:.6f} "
            f"v3={r['v3']:.6f} "
            f"shadow={r['shadow']:.6f} "
            f"| dt raw/v3/shadow="
            f"{1000*r['raw_dt']:.1f}/"
            f"{1000*r['v3_dt']:.1f}/"
            f"{1000*r['shadow_dt']:.1f} ms"
        )

    d_raw = (
        result["stability"]["raw"]
        - result["energy"]["raw"]
    )

    d_v3 = (
        result["stability"]["v3"]
        - result["energy"]["v3"]
    )

    d_shadow = (
        result["stability"]["shadow"]
        - result["energy"]["shadow"]
    )

    print()
    print(
        f"S-E dRaw   = {d_raw:+.6f}"
    )

    print(
        f"S-E dV3    = {d_v3:+.6f}"
    )

    print(
        f"S-E dShadow= {d_shadow:+.6f}"
    )

    print()
    print("=" * 112)
    print(
        "RR TIMELINE 3.282--3.304 s"
    )
    print("=" * 112)

    by_step = {
        beta: {
            int(
                row[
                    "physics_step_num_post"
                ]
            ): row
            for row in traces[beta]
        }
        for beta in BETAS
    }

    common = sorted(
        set(by_step["balanced"])
        & set(by_step["stability"])
        & set(by_step["energy"])
    )

    for step in common:
        t = float(
            by_step[
                "balanced"
            ][step]["time_post_s"]
        )

        if not (
            3.282
            <= t
            <= 3.304
        ):
            continue

        print(
            f"t={t:.3f} "
            f"tick={step}"
        )

        for beta in BETAS:
            row = (
                by_step[
                    beta
                ][step]
            )

            foot = (
                row[
                    "feet"
                ]["RR"]
            )

            sh = (
                row[
                    "planned_stance_shadow"
                ]["RR"]
            )

            if foot["contact"]:
                speed = (
                    f"{float(foot['slip_speed_mps']):.4f}"
                )
            else:
                speed = "OFF"

            age = (
                "None"
                if sh[
                    "planned_stance_age_s"
                ] is None
                else (
                    f"{float(sh['planned_stance_age_s']):.3f}"
                )
            )

            print(
                f"  {beta[0].upper()}: "
                f"actual={int(bool(foot['contact']))} "
                f"planned={int(bool(sh['planned_contact']))} "
                f"pAge={age} "
                f"eligible={int(bool(sh['eligible']))} "
                f"v={speed}"
            )

    print()
    print("=" * 112)

    if (
        d_raw < 0.0
        and d_v3 > 0.0
        and d_shadow < 0.0
    ):
        print(
            "RESULT: PLANNED-STANCE SHADOW "
            "REMOVES THE v3 SIGN INVERSION"
        )

        print(
            "[ICRA27] OS-T4.6b "
            "planned-stance shadow: PASS"
        )

    else:
        print(
            "RESULT: shadow does not yet "
            "recover the raw semantic ordering"
        )

        print(
            "[ICRA27] OS-T4.6b "
            "planned-stance shadow: "
            "REQUIRES FURTHER AUDIT"
        )

    print("=" * 112)


if __name__ == "__main__":
    main()
