#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

K5_ROOT = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5k5_lockstep_checkpoint_sweep_v0"
    / "train9"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5k6d_interval_slip_distribution_v0"
)

UPDATES = (
    10,
    20,
    30,
    40,
)

BETAS = (
    "balanced",
    "stability",
    "energy",
)

TARGET_TERRAIN = "low_friction"
TARGET_SEED = 27200

SETTLING_STEPS = 5

# Representative events selected BEFORE this analysis
# from frozen k6b/k6c diagnostics.
#
# u30 k0  : strongest launch S-vs-E contrast
# u20 k0  : launch-dominated checkpoint
# u40 k1  : largest u40 launch Stability spike
# k11     : repeated post-launch S-vs-E contrast
PRINT_CASES = (
    (20, 0),
    (30, 0),
    (40, 1),
    (10, 11),
    (30, 11),
    (40, 11),
)


def read_jsonl(
    path: Path,
):
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
                    f"{path}:{line_no}: invalid JSON"
                ) from exc

    return rows


def find_seed(
    rows,
):
    for row in rows:
        if (
            row.get("event") == "reset"
            and row.get("seed") is not None
        ):
            return int(
                row["seed"]
            )

    for row in rows:
        if row.get("seed") is not None:
            return int(
                row["seed"]
            )

    return None


def find_episode(
    *,
    update,
    beta,
):
    log_dir = (
        K5_ROOT
        / f"u{update:02d}"
        / "env_logs"
        / beta
        / TARGET_TERRAIN
    )

    if not log_dir.exists():
        raise FileNotFoundError(
            log_dir
        )

    matches = []

    for path in sorted(
        log_dir.glob(
            "episode_*.jsonl"
        )
    ):
        rows = read_jsonl(
            path
        )

        if find_seed(rows) == TARGET_SEED:
            matches.append(
                (
                    path,
                    rows,
                )
            )

    if len(matches) != 1:
        raise RuntimeError(
            f"u{update:02d}/{beta}: "
            f"expected one seed={TARGET_SEED} "
            f"episode, found={len(matches)}"
        )

    return matches[0]


def numeric(
    obj,
    key,
    default=None,
):
    value = obj.get(
        key,
        default,
    )

    if value is None:
        return None

    value = float(value)

    if not math.isfinite(value):
        raise RuntimeError(
            f"Non-finite {key}: {value}"
        )

    return value


def delta_nonnegative(
    current,
    previous,
    *,
    name,
    tol=1e-9,
):
    value = (
        float(current)
        - float(previous)
    )

    if value < -tol:
        raise RuntimeError(
            f"Cumulative field decreased: "
            f"{name} delta={value}"
        )

    if value < 0.0:
        value = 0.0

    return value


def slip_payload(
    row,
):
    slip = row.get(
        "eval_stance_slip"
    )

    if not isinstance(
        slip,
        dict,
    ):
        raise RuntimeError(
            "Missing eval_stance_slip at "
            f"episode_step="
            f"{row.get('episode_step')}"
        )

    return slip


def reward_ctr(
    row,
):
    rc = row.get(
        "reward_components"
    )

    if not isinstance(
        rc,
        dict,
    ):
        raise RuntimeError(
            "Missing reward_components"
        )

    if "cost_traction" not in rc:
        raise RuntimeError(
            "Missing reward cost_traction"
        )

    return float(
        rc["cost_traction"]
    )


def metadata_only(
    payload,
    cumulative_keys,
):
    excluded = set(
        cumulative_keys
    )

    excluded.update(
        (
            "speed_mean_mps",
            "speed_rms_mps",
            "speed_max_mps",
        )
    )

    return {
        key:
            value
        for key, value
        in payload.items()
        if key not in excluded
    }


def age_interval_rows(
    *,
    update,
    beta,
    policy_step,
    current,
    previous,
):
    cur = slip_payload(
        current
    ).get(
        "contact_age_bins"
    )

    prev = slip_payload(
        previous
    ).get(
        "contact_age_bins"
    )

    if not isinstance(cur, list):
        return []

    if not isinstance(prev, list):
        return []

    if len(cur) != len(prev):
        raise RuntimeError(
            "contact_age_bins length changed"
        )

    cumulative = (
        "contact_time_s",
        "contact_samples",
        "speed_time_sum_m",
        "speed_sq_time_sum_m2ps",
    )

    result = []

    for index, (
        c,
        p,
    ) in enumerate(
        zip(
            cur,
            prev,
        )
    ):
        dt = delta_nonnegative(
            numeric(
                c,
                "contact_time_s",
                0.0,
            ),
            numeric(
                p,
                "contact_time_s",
                0.0,
            ),
            name=(
                f"age[{index}].contact_time"
            ),
        )

        samples = delta_nonnegative(
            numeric(
                c,
                "contact_samples",
                0.0,
            ),
            numeric(
                p,
                "contact_samples",
                0.0,
            ),
            name=(
                f"age[{index}].samples"
            ),
        )

        speed_sum = delta_nonnegative(
            numeric(
                c,
                "speed_time_sum_m",
                0.0,
            ),
            numeric(
                p,
                "speed_time_sum_m",
                0.0,
            ),
            name=(
                f"age[{index}].speed_sum"
            ),
        )

        speed_sq_sum = delta_nonnegative(
            numeric(
                c,
                "speed_sq_time_sum_m2ps",
                0.0,
            ),
            numeric(
                p,
                "speed_sq_time_sum_m2ps",
                0.0,
            ),
            name=(
                f"age[{index}].speed_sq_sum"
            ),
        )

        mean_speed = (
            float("nan")
            if dt <= 0.0
            else speed_sum / dt
        )

        rms_speed = (
            float("nan")
            if dt <= 0.0
            else math.sqrt(
                max(
                    0.0,
                    speed_sq_sum / dt,
                )
            )
        )

        result.append(
            {
                "update":
                    update,

                "beta":
                    beta,

                "policy_step":
                    policy_step,

                "bin_index":
                    index,

                "contact_time_s":
                    dt,

                "contact_samples":
                    int(
                        round(
                            samples
                        )
                    ),

                "speed_time_sum_m":
                    speed_sum,

                "speed_sq_time_sum_m2ps":
                    speed_sq_sum,

                "interval_mean_mps":
                    mean_speed,

                "interval_rms_mps":
                    rms_speed,

                "metadata_json":
                    json.dumps(
                        metadata_only(
                            c,
                            cumulative,
                        ),
                        sort_keys=True,
                        separators=(
                            ",",
                            ":",
                        ),
                    ),
            }
        )

    return result


def histogram_interval_rows(
    *,
    update,
    beta,
    policy_step,
    current,
    previous,
):
    cur = slip_payload(
        current
    ).get(
        "established_slip_histogram"
    )

    prev = slip_payload(
        previous
    ).get(
        "established_slip_histogram"
    )

    if not isinstance(cur, list):
        return []

    if not isinstance(prev, list):
        return []

    if len(cur) != len(prev):
        raise RuntimeError(
            "established histogram "
            "length changed"
        )

    cumulative = (
        "contact_time_s",
        "contact_samples",
    )

    result = []

    for index, (
        c,
        p,
    ) in enumerate(
        zip(
            cur,
            prev,
        )
    ):
        dt = delta_nonnegative(
            numeric(
                c,
                "contact_time_s",
                0.0,
            ),
            numeric(
                p,
                "contact_time_s",
                0.0,
            ),
            name=(
                f"hist[{index}].contact_time"
            ),
        )

        samples = delta_nonnegative(
            numeric(
                c,
                "contact_samples",
                0.0,
            ),
            numeric(
                p,
                "contact_samples",
                0.0,
            ),
            name=(
                f"hist[{index}].samples"
            ),
        )

        result.append(
            {
                "update":
                    update,

                "beta":
                    beta,

                "policy_step":
                    policy_step,

                "bin_index":
                    index,

                "contact_time_s":
                    dt,

                "contact_samples":
                    int(
                        round(
                            samples
                        )
                    ),

                "metadata_json":
                    json.dumps(
                        metadata_only(
                            c,
                            cumulative,
                        ),
                        sort_keys=True,
                        separators=(
                            ",",
                            ":",
                        ),
                    ),
            }
        )

    return result


def interval_summary(
    *,
    update,
    beta,
    policy_step,
    current,
    previous,
    source_log,
):
    cur_slip = slip_payload(
        current
    )

    prev_slip = slip_payload(
        previous
    )

    established_dt = delta_nonnegative(
        numeric(
            cur_slip,
            "established_contact_time_s",
            0.0,
        ),
        numeric(
            prev_slip,
            "established_contact_time_s",
            0.0,
        ),
        name="established_contact_time_s",
    )

    established_samples = (
        delta_nonnegative(
            numeric(
                cur_slip,
                "established_contact_samples",
                0.0,
            ),
            numeric(
                prev_slip,
                "established_contact_samples",
                0.0,
            ),
            name=(
                "established_contact_samples"
            ),
        )
    )

    cost_sum_cur = numeric(
        cur_slip,
        "established_traction_cost_time_sum_s",
        None,
    )

    cost_sum_prev = numeric(
        prev_slip,
        "established_traction_cost_time_sum_s",
        None,
    )

    if (
        cost_sum_cur is not None
        and cost_sum_prev is not None
    ):
        cost_sum_delta = (
            delta_nonnegative(
                cost_sum_cur,
                cost_sum_prev,
                name=(
                    "established_traction_"
                    "cost_time_sum_s"
                ),
            )
        )
    else:
        cost_sum_delta = float(
            "nan"
        )

    reconstructed_ctr = (
        float("nan")
        if (
            established_dt <= 0.0
            or not math.isfinite(
                cost_sum_delta
            )
        )
        else (
            cost_sum_delta
            / established_dt
        )
    )

    ctr = reward_ctr(
        current
    )

    per_foot_cur = cur_slip.get(
        "per_foot_contact_samples",
        {},
    )

    per_foot_prev = prev_slip.get(
        "per_foot_contact_samples",
        {},
    )

    per_foot_delta = {}

    for leg in (
        "FL",
        "FR",
        "RL",
        "RR",
    ):
        per_foot_delta[
            leg
        ] = int(
            round(
                delta_nonnegative(
                    float(
                        per_foot_cur.get(
                            leg,
                            0,
                        )
                    ),
                    float(
                        per_foot_prev.get(
                            leg,
                            0,
                        )
                    ),
                    name=(
                        f"{leg}.contact_samples"
                    ),
                )
            )
        )

    return {
        "update":
            update,

        "beta":
            beta,

        "policy_step":
            policy_step,

        "episode_step":
            int(
                current[
                    "episode_step"
                ]
            ),

        "state_sim_time_s":
            float(
                current.get(
                    "state_sim_time_s",
                    float("nan"),
                )
            ),

        "reward_cost_traction":
            ctr,

        "established_contact_time_s":
            established_dt,

        "established_contact_samples":
            int(
                round(
                    established_samples
                )
            ),

        "traction_cost_time_sum_delta_s":
            cost_sum_delta,

        "reconstructed_cost_traction":
            reconstructed_ctr,

        "reconstruction_abs_error":
            (
                float("nan")
                if not math.isfinite(
                    reconstructed_ctr
                )
                else abs(
                    reconstructed_ctr
                    - ctr
                )
            ),

        "FL_contact_samples":
            per_foot_delta[
                "FL"
            ],

        "FR_contact_samples":
            per_foot_delta[
                "FR"
            ],

        "RL_contact_samples":
            per_foot_delta[
                "RL"
            ],

        "RR_contact_samples":
            per_foot_delta[
                "RR"
            ],

        "source_log":
            source_log,
    }


def write_csv(
    path,
    rows,
):
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def main():
    if not K5_ROOT.exists():
        raise FileNotFoundError(
            K5_ROOT
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

    summaries = []
    age_rows = []
    histogram_rows = []

    print("=" * 120)
    print(
        "ICRA27 OS-T4.5k6d "
        "INTERVAL SLIP DISTRIBUTION AUDIT"
    )
    print("=" * 120)

    print(
        "source  : k5 clean lockstep JSONL"
    )

    print(
        "terrain : low_friction"
    )

    print(
        "seed    :",
        TARGET_SEED,
    )

    print(
        "scope   : OFFLINE cumulative-difference "
        "analysis; no simulation"
    )

    print()

    by_key = {}

    for update in UPDATES:
        for beta in BETAS:
            path, rows = find_episode(
                update=update,
                beta=beta,
            )

            step_rows = [
                row
                for row in rows
                if row.get(
                    "event"
                ) == "step"
            ]

            by_episode_step = {
                int(
                    row[
                        "episode_step"
                    ]
                ):
                    row
                for row in step_rows
            }

            for current in step_rows:
                episode_step = int(
                    current[
                        "episode_step"
                    ]
                )

                if episode_step <= (
                    SETTLING_STEPS
                ):
                    continue

                policy_step = (
                    episode_step
                    - SETTLING_STEPS
                    - 1
                )

                previous_step = (
                    episode_step - 1
                )

                if previous_step not in (
                    by_episode_step
                ):
                    raise RuntimeError(
                        f"Missing previous step "
                        f"{previous_step} in {path}"
                    )

                previous = (
                    by_episode_step[
                        previous_step
                    ]
                )

                summary = interval_summary(
                    update=update,
                    beta=beta,
                    policy_step=policy_step,
                    current=current,
                    previous=previous,
                    source_log=str(
                        path.relative_to(
                            ROOT
                        )
                    ),
                )

                summaries.append(
                    summary
                )

                ages = age_interval_rows(
                    update=update,
                    beta=beta,
                    policy_step=policy_step,
                    current=current,
                    previous=previous,
                )

                hists = (
                    histogram_interval_rows(
                        update=update,
                        beta=beta,
                        policy_step=policy_step,
                        current=current,
                        previous=previous,
                    )
                )

                age_rows.extend(
                    ages
                )

                histogram_rows.extend(
                    hists
                )

                by_key[
                    (
                        update,
                        beta,
                        policy_step,
                    )
                ] = (
                    summary,
                    ages,
                    hists,
                )

    write_csv(
        OUT_DIR
        / "interval_summary.csv",
        summaries,
    )

    write_csv(
        OUT_DIR
        / "contact_age_interval.csv",
        age_rows,
    )

    write_csv(
        OUT_DIR
        / "established_histogram_interval.csv",
        histogram_rows,
    )

    # --------------------------------------------------------
    # Validate reward-v3 interval traction reconstruction
    # wherever cumulative cost sufficient statistics exist.
    # --------------------------------------------------------

    errors = [
        row[
            "reconstruction_abs_error"
        ]
        for row in summaries
        if math.isfinite(
            row[
                "reconstruction_abs_error"
            ]
        )
    ]

    print(
        "reconstructed intervals:",
        len(errors),
    )

    if errors:
        print(
            "max |Ctr_reconstructed - Ctr_reward|:",
            f"{max(errors):.12e}",
        )

    print()

    # --------------------------------------------------------
    # Representative launch/post-launch windows.
    # --------------------------------------------------------

    for (
        update,
        policy_step,
    ) in PRINT_CASES:

        print("=" * 120)
        print(
            f"u{update:02d} "
            f"policy k={policy_step}"
        )
        print("=" * 120)

        for beta in BETAS:
            key = (
                update,
                beta,
                policy_step,
            )

            if key not in by_key:
                raise RuntimeError(
                    f"Missing {key}"
                )

            (
                summary,
                ages,
                hists,
            ) = by_key[key]

            print()
            print(
                f"{beta.upper():<9} "
                f"t="
                f"{summary['state_sim_time_s']:.3f} "
                f"Ctr="
                f"{summary['reward_cost_traction']:.6f} "
                f"est_dt="
                f"{summary['established_contact_time_s']:.4f} "
                f"est_n="
                f"{summary['established_contact_samples']} "
                f"feet="
                f"{summary['FL_contact_samples']}/"
                f"{summary['FR_contact_samples']}/"
                f"{summary['RL_contact_samples']}/"
                f"{summary['RR_contact_samples']}"
            )

            if math.isfinite(
                summary[
                    "reconstructed_cost_traction"
                ]
            ):
                print(
                    "  reconstructed Ctr="
                    f"{summary['reconstructed_cost_traction']:.6f} "
                    "err="
                    f"{summary['reconstruction_abs_error']:.3e}"
                )

            active_age = [
                row
                for row in ages
                if (
                    row[
                        "contact_time_s"
                    ]
                    > 0.0
                )
            ]

            active_age.sort(
                key=lambda row: (
                    (
                        -1.0
                        if not math.isfinite(
                            row[
                                "interval_rms_mps"
                            ]
                        )
                        else row[
                            "interval_rms_mps"
                        ]
                    )
                ),
                reverse=True,
            )

            print(
                "  highest-RMS contact-age bins:"
            )

            for row in active_age[:4]:
                print(
                    "   ",
                    f"age_bin={row['bin_index']:>2} "
                    f"dt={row['contact_time_s']:.4f} "
                    f"n={row['contact_samples']:>3} "
                    f"mean="
                    f"{row['interval_mean_mps']:.4f} "
                    f"rms="
                    f"{row['interval_rms_mps']:.4f} "
                    f"meta="
                    f"{row['metadata_json']}"
                )

            active_hist = [
                row
                for row in hists
                if (
                    row[
                        "contact_samples"
                    ]
                    > 0
                )
            ]

            # Preserve histogram index ordering.
            # Metadata is printed so we do not assume
            # how the source names/ranges speed bins.
            active_hist.sort(
                key=lambda row: (
                    row[
                        "bin_index"
                    ]
                )
            )

            print(
                "  established-slip histogram "
                "active bins:"
            )

            for row in active_hist:
                print(
                    "   ",
                    f"hist_bin={row['bin_index']:>2} "
                    f"dt={row['contact_time_s']:.4f} "
                    f"n={row['contact_samples']:>3} "
                    f"meta="
                    f"{row['metadata_json']}"
                )

        print()

    print("=" * 120)
    print(
        "[ICRA27] OS-T4.5k6d "
        "interval slip distribution: COMPUTE PASS"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
