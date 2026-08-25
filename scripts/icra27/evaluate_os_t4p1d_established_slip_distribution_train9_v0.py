#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import socket
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


PHASE2B_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_phase2b_beta_train_archive_v0.py"
)

DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "icra27"
    / "phase1a_beta_conditioned_v2_seed27027"
    / "phase1_beta_identifiable_selected_u20.pt"
)

DEFAULT_OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p1d_established_slip_distribution_v0"
    / "u0020_train9"
)


ESTABLISHED_GATE_S = 0.040


GROUPS = (
    {
        "name": "flat",
        "terrain": "flat",
        "seeds": (27100, 27101, 27102),
    },
    {
        "name": "low_friction",
        "terrain": "low_friction",
        "seeds": (27200, 27201, 27202),
    },
    {
        "name": "rough",
        "terrain": "rough_perlin",
        "seeds": (13, 7, 15),
    },
)


# All values are existing histogram edges.
# These are diagnostic candidates, NOT frozen reward parameters.
DEADBAND_CANDIDATES_MPS = (
    0.0010,
    0.0015,
    0.0020,
    0.0030,
    0.0040,
    0.0050,
    0.0060,
    0.0080,
    0.0100,
)


QUANTILES = (
    0.90,
    0.95,
    0.975,
    0.99,
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {path}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def udp_bindable(port):
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    try:
        sock.bind(
            ("127.0.0.1", int(port))
        )
        return True

    except OSError:
        return False

    finally:
        sock.close()


def choose_base_port():
    offsets = (
        0, 1, 2,
        10, 11, 12,
        20, 21, 22,
    )

    for base in range(
        62000,
        64501,
        100,
    ):
        ports = [
            base + offset
            for offset in offsets
        ]

        if all(
            udp_bindable(port)
            for port in ports
        ):
            return base

    raise RuntimeError(
        "No free TRAIN9 UDP layout found."
    )


def median(values):
    return float(
        statistics.median(
            float(x)
            for x in values
        )
    )


def extract_established_histogram(env):
    info = getattr(
        env.unwrapped,
        "last_info",
        None,
    )

    if info is None:
        raise RuntimeError(
            "Missing env.unwrapped.last_info"
        )

    slip = info.get(
        "eval_stance_slip"
    )

    if slip is None:
        raise RuntimeError(
            "Missing eval_stance_slip"
        )

    if (
        slip.get(
            "established_stance_role"
        )
        !=
        "evaluation_only_established_stance_slip_v0"
    ):
        raise RuntimeError(
            "Unexpected established-stance role."
        )

    gate_s = float(
        slip[
            "established_stance_gate_s"
        ]
    )

    if abs(
        gate_s
        - ESTABLISHED_GATE_S
    ) > 1e-12:
        raise RuntimeError(
            "Unexpected established-stance gate: "
            f"{gate_s}"
        )

    hist = slip.get(
        "established_slip_histogram"
    )

    if not hist:
        raise RuntimeError(
            "Missing established-slip histogram."
        )

    total_time_s = float(
        slip[
            "established_contact_time_s"
        ]
    )

    total_samples = int(
        slip[
            "established_contact_samples"
        ]
    )

    if (
        total_time_s <= 0.0
        or total_samples <= 0
    ):
        raise RuntimeError(
            "No established-stance contact data."
        )

    return {
        "gate_s":
            gate_s,

        "total_time_s":
            total_time_s,

        "total_samples":
            total_samples,

        "histogram": [
            dict(item)
            for item in hist
        ],
    }


def same_edge(a, b):
    if (
        a is None
        or b is None
    ):
        return a is None and b is None

    return abs(
        float(a) - float(b)
    ) <= 1e-12


def pool_histograms(episodes):
    if not episodes:
        raise RuntimeError(
            "Cannot pool empty episode set."
        )

    reference = episodes[
        0
    ][
        "established"
    ][
        "histogram"
    ]

    pooled = []

    for ref_bin in reference:
        pooled.append(
            {
                "low_mps":
                    float(
                        ref_bin[
                            "low_mps"
                        ]
                    ),

                "high_mps":
                    (
                        None
                        if ref_bin[
                            "high_mps"
                        ] is None
                        else float(
                            ref_bin[
                                "high_mps"
                            ]
                        )
                    ),

                "contact_time_s":
                    0.0,

                "contact_samples":
                    0,
            }
        )

    for episode in episodes:
        hist = episode[
            "established"
        ][
            "histogram"
        ]

        if len(hist) != len(pooled):
            raise RuntimeError(
                "Histogram length mismatch."
            )

        for index, item in enumerate(hist):
            dst = pooled[index]

            if not same_edge(
                item["low_mps"],
                dst["low_mps"],
            ):
                raise RuntimeError(
                    "Histogram low-edge mismatch."
                )

            if not same_edge(
                item["high_mps"],
                dst["high_mps"],
            ):
                raise RuntimeError(
                    "Histogram high-edge mismatch."
                )

            dst[
                "contact_time_s"
            ] += float(
                item[
                    "contact_time_s"
                ]
            )

            dst[
                "contact_samples"
            ] += int(
                item[
                    "contact_samples"
                ]
            )

    total_time = sum(
        item["contact_time_s"]
        for item in pooled
    )

    if total_time <= 0.0:
        raise RuntimeError(
            "Pooled histogram has zero time."
        )

    for item in pooled:
        item[
            "time_fraction"
        ] = float(
            item[
                "contact_time_s"
            ]
            / total_time
        )

    return pooled


def exceedance_fraction(
    histogram,
    threshold_mps,
):
    threshold_mps = float(
        threshold_mps
    )

    total = sum(
        float(
            item[
                "contact_time_s"
            ]
        )
        for item in histogram
    )

    if total <= 0.0:
        raise RuntimeError(
            "Zero histogram duration."
        )

    above = 0.0

    for item in histogram:
        low = float(
            item[
                "low_mps"
            ]
        )

        high = item[
            "high_mps"
        ]

        # Candidates must coincide with existing bin edges,
        # so no partial-bin interpolation is required.
        if low + 1e-12 >= threshold_mps:
            above += float(
                item[
                    "contact_time_s"
                ]
            )

        elif (
            high is not None
            and float(high) - 1e-12
            > threshold_mps
        ):
            raise RuntimeError(
                "Threshold falls inside histogram bin: "
                f"{threshold_mps} in "
                f"[{low}, {high})"
            )

    return float(
        above / total
    )


def conservative_quantile(
    histogram,
    q,
):
    q = float(q)

    total = sum(
        float(
            item[
                "contact_time_s"
            ]
        )
        for item in histogram
    )

    target = q * total
    cumulative = 0.0

    for item in histogram:
        cumulative += float(
            item[
                "contact_time_s"
            ]
        )

        if cumulative + 1e-15 >= target:
            high = item[
                "high_mps"
            ]

            return {
                "quantile":
                    q,

                "bin_low_mps":
                    float(
                        item[
                            "low_mps"
                        ]
                    ),

                "bin_high_mps":
                    (
                        None
                        if high is None
                        else float(high)
                    ),

                # Conservative because the precise within-bin
                # sample distribution is intentionally unknown.
                "conservative_upper_mps":
                    (
                        float(
                            item[
                                "low_mps"
                            ]
                        )
                        if high is None
                        else float(high)
                    ),
            }

    raise RuntimeError(
        f"Could not evaluate quantile q={q}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    args = ap.parse_args()

    checkpoint = (
        args.checkpoint.resolve()
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    out_dir = (
        args.out_dir.resolve()
    )

    if out_dir.exists():
        raise RuntimeError(
            "Output directory already exists; "
            "use a fresh --out-dir: "
            f"{out_dir}"
        )

    out_dir.mkdir(
        parents=True
    )

    phase2b = load_module(
        PHASE2B_PATH,
        "os_t4p1d_phase2b",
    )

    beta_bank = {
        str(name): tuple(
            float(x)
            for x in beta
        )
        for name, beta
        in phase2b.BETA_BANK
    }

    beta = beta_bank[
        "balanced"
    ]

    policy = phase2b.base.load_policy(
        name="beta_conditioned",
        path=checkpoint,
        expected_reward_mode=(
            "tracer_cost_v2"
        ),
    )

    phase2b.validate_checkpoint_beta_bank(
        policy["payload"]
    )

    base_port = (
        choose_base_port()
    )

    print("=" * 100)
    print(
        "ICRA27 OS-T4.1d ESTABLISHED-STANCE "
        "SLIP DISTRIBUTION TRAIN9"
    )
    print("=" * 100)

    print(
        "checkpoint :",
        checkpoint,
    )

    print(
        "beta       :",
        beta,
    )

    print(
        "gate       :",
        ESTABLISHED_GATE_S,
    )

    print(
        "base port  :",
        base_port,
    )

    print(
        "output     :",
        out_dir,
    )

    print()

    episodes = []

    episodes_path = (
        out_dir
        / "episodes.jsonl"
    )

    for group_index, group in enumerate(
        GROUPS
    ):
        command_port = (
            base_port
            + 10 * group_index
        )

        env = phase2b.make_env(
            terrain=group["terrain"],
            beta=beta,
            command_port=command_port,
            log_dir=(
                out_dir
                / "env_logs"
                / group["name"]
            ),
        )

        print(
            f"[{group['name']}] "
            f"terrain={group['terrain']} "
            f"seeds={list(group['seeds'])}"
        )

        try:
            for seed in group[
                "seeds"
            ]:
                row = (
                    phase2b.base.run_episode(
                        env,
                        policy_name=(
                            "beta_conditioned"
                        ),
                        checkpoint=(
                            policy["path"]
                        ),
                        trained_reward_mode=(
                            policy[
                                "trained_reward_mode"
                            ]
                        ),
                        model=(
                            policy["model"]
                        ),
                        group=(
                            "os_t4p1d_"
                            + group["name"]
                        ),
                        terrain=(
                            group["terrain"]
                        ),
                        seed=seed,
                    )
                )

                established = (
                    extract_established_histogram(
                        env
                    )
                )

                episode = {
                    "group":
                        group["name"],

                    "terrain":
                        group["terrain"],

                    "seed":
                        int(seed),

                    "success":
                        bool(
                            row.get(
                                "success",
                                False,
                            )
                        ),

                    "attitude_cost":
                        float(
                            row[
                                "mean_cost_stability"
                            ]
                        ),

                    "established":
                        established,
                }

                episodes.append(
                    episode
                )

                # Incremental snapshot.
                with episodes_path.open(
                    "w"
                ) as f:
                    for item in episodes:
                        f.write(
                            json.dumps(
                                item,
                                sort_keys=True,
                            )
                            + "\n"
                        )

                print(
                    f"  seed={seed:<6} "
                    f"success={episode['success']} "
                    f"contact_time="
                    f"{established['total_time_s']:.3f}s "
                    f"samples="
                    f"{established['total_samples']}"
                )

        finally:
            env.close()

        print()

    if len(episodes) != 9:
        raise RuntimeError(
            f"Expected 9 episodes, got "
            f"{len(episodes)}"
        )

    if not all(
        item["success"]
        for item in episodes
    ):
        raise RuntimeError(
            "At least one episode failed."
        )

    summary = {
        "schema":
            "icra27_os_t4p1d_established_slip_distribution_train9_v0",

        "scope":
            "TRAIN-only balanced-beta diagnostic",

        "checkpoint":
            str(checkpoint),

        "beta":
            list(beta),

        "established_stance_gate_s":
            ESTABLISHED_GATE_S,

        "deadband_candidates_mps":
            list(
                DEADBAND_CANDIDATES_MPS
            ),

        "quantiles":
            list(
                QUANTILES
            ),

        "terrains":
            {},

        "candidate_exceedance":
            {},
    }

    pooled_by_group = {}

    for group in GROUPS:
        name = group[
            "name"
        ]

        subset = [
            item
            for item in episodes
            if item["group"] == name
        ]

        pooled = pool_histograms(
            subset
        )

        pooled_by_group[
            name
        ] = pooled

        summary[
            "terrains"
        ][name] = {
            "seeds":
                list(
                    group["seeds"]
                ),

            "pooled_histogram":
                pooled,

            "pooled_quantiles":
                {
                    f"{q:.3f}":
                        conservative_quantile(
                            pooled,
                            q,
                        )
                    for q in QUANTILES
                },
        }

    candidate_rows = []

    for threshold in (
        DEADBAND_CANDIDATES_MPS
    ):
        key = f"{threshold:.4f}"

        summary[
            "candidate_exceedance"
        ][key] = {}

        for group in GROUPS:
            name = group[
                "name"
            ]

            subset = [
                item
                for item in episodes
                if item["group"] == name
            ]

            pooled_exceed = (
                exceedance_fraction(
                    pooled_by_group[name],
                    threshold,
                )
            )

            episode_exceeds = [
                exceedance_fraction(
                    item[
                        "established"
                    ][
                        "histogram"
                    ],
                    threshold,
                )
                for item in subset
            ]

            item = {
                "pooled_time_fraction_above":
                    pooled_exceed,

                "median_episode_fraction_above":
                    median(
                        episode_exceeds
                    ),

                "max_episode_fraction_above":
                    max(
                        float(x)
                        for x in episode_exceeds
                    ),

                "episode_fractions_above":
                    [
                        float(x)
                        for x in episode_exceeds
                    ],
            }

            summary[
                "candidate_exceedance"
            ][key][name] = item

            candidate_rows.append(
                {
                    "threshold_mps":
                        threshold,

                    "terrain":
                        name,

                    "pooled_fraction_above":
                        pooled_exceed,

                    "median_episode_fraction_above":
                        item[
                            "median_episode_fraction_above"
                        ],

                    "max_episode_fraction_above":
                        item[
                            "max_episode_fraction_above"
                        ],
                }
            )

    summary_path = (
        out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    csv_path = (
        out_dir
        / "deadband_candidates.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "threshold_mps",
                "terrain",
                "pooled_fraction_above",
                "median_episode_fraction_above",
                "max_episode_fraction_above",
            ),
        )

        writer.writeheader()
        writer.writerows(
            candidate_rows
        )

    print("=" * 100)
    print(
        "OS-T4.1d DEAD-BAND CANDIDATE SUMMARY"
    )
    print("=" * 100)

    print()
    print("POOLED QUANTILES")
    print("-" * 100)

    for group in GROUPS:
        name = group["name"]

        print(
            f"{name}:"
        )

        for q in QUANTILES:
            item = summary[
                "terrains"
            ][name][
                "pooled_quantiles"
            ][
                f"{q:.3f}"
            ]

            print(
                f"  q={q:5.3f} "
                f"bin=["
                f"{item['bin_low_mps']:.4f}, "
                f"{item['bin_high_mps']}"
                f") "
                f"upper="
                f"{item['conservative_upper_mps']:.4f}"
            )

    print()
    print(
        f"{'threshold':>10} "
        f"{'flat pooled':>12} "
        f"{'flat max':>10} "
        f"{'low pooled':>12} "
        f"{'rough pooled':>13}"
    )

    print("-" * 65)

    for threshold in (
        DEADBAND_CANDIDATES_MPS
    ):
        key = f"{threshold:.4f}"

        flat = summary[
            "candidate_exceedance"
        ][key]["flat"]

        low = summary[
            "candidate_exceedance"
        ][key]["low_friction"]

        rough = summary[
            "candidate_exceedance"
        ][key]["rough"]

        print(
            f"{threshold:10.4f} "
            f"{flat['pooled_time_fraction_above']:12.4f} "
            f"{flat['max_episode_fraction_above']:10.4f} "
            f"{low['pooled_time_fraction_above']:12.4f} "
            f"{rough['pooled_time_fraction_above']:13.4f}"
        )

    print()
    print(
        "episodes:",
        episodes_path,
    )

    print(
        "summary :",
        summary_path,
    )

    print(
        "csv     :",
        csv_path,
    )


if __name__ == "__main__":
    main()
