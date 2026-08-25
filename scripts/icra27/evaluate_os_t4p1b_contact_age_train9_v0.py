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
    / "os_t4p1b_contact_age_train9_v0"
    / "u0020_train9"
)


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


GATE_CANDIDATES_S = (
    0.020,
    0.030,
    0.040,
    0.050,
    0.060,
    0.080,
    0.100,
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
        61000,
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
    values = [
        float(x)
        for x in values
    ]

    return float(
        statistics.median(values)
    )


def extract_profile(env):
    info = getattr(
        env.unwrapped,
        "last_info",
        None,
    )

    if info is None:
        raise RuntimeError(
            "Missing final env info."
        )

    slip = info.get(
        "eval_stance_slip"
    )

    if slip is None:
        raise RuntimeError(
            "Missing eval_stance_slip."
        )

    if (
        slip.get(
            "contact_age_profile_role"
        )
        !=
        "evaluation_only_contact_age_profile_v0"
    ):
        raise RuntimeError(
            "Unexpected contact-age profile role."
        )

    bins = slip.get(
        "contact_age_bins"
    )

    if not bins:
        raise RuntimeError(
            "Empty contact-age bins."
        )

    return {
        "overall_slip_mean_mps":
            float(
                slip["speed_mean_mps"]
            ),

        "overall_slip_rms_mps":
            float(
                slip["speed_rms_mps"]
            ),

        "overall_slip_max_mps":
            float(
                slip["speed_max_mps"]
            ),

        "bins":
            bins,
    }


def aggregate_after_gate(
    bins,
    gate_s,
):
    speed_time_sum = 0.0
    speed_sq_time_sum = 0.0
    contact_time = 0.0
    contact_samples = 0
    speed_max = 0.0

    for b in bins:
        low_s = float(
            b["low_s"]
        )

        if low_s + 1e-12 < gate_s:
            continue

        speed_time_sum += float(
            b["speed_time_sum_m"]
        )

        speed_sq_time_sum += float(
            b["speed_sq_time_sum_m2ps"]
        )

        contact_time += float(
            b["contact_time_s"]
        )

        contact_samples += int(
            b["contact_samples"]
        )

        speed_max = max(
            speed_max,
            float(
                b["speed_max_mps"]
            ),
        )

    if contact_time <= 0.0:
        return {
            "mean_mps": None,
            "rms_mps": None,
            "max_mps": None,
            "contact_time_s": 0.0,
            "contact_samples": 0,
        }

    mean_mps = (
        speed_time_sum
        / contact_time
    )

    rms_mps = (
        speed_sq_time_sum
        / contact_time
    ) ** 0.5

    return {
        "mean_mps":
            float(mean_mps),

        "rms_mps":
            float(rms_mps),

        "max_mps":
            float(speed_max),

        "contact_time_s":
            float(contact_time),

        "contact_samples":
            int(contact_samples),
    }


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
        "os_t4p1b_phase2b",
    )

    beta_bank = {
        name: tuple(
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
        "ICRA27 OS-T4.1b CONTACT-AGE "
        "STANCE-SLIP TRAIN9"
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
        "base port  :",
        base_port,
    )

    print(
        "gate candidates:",
        GATE_CANDIDATES_S,
    )

    print(
        "output     :",
        out_dir,
    )

    print()

    episodes = []

    episode_jsonl = (
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
                            "os_t4p1b_"
                            + group["name"]
                        ),
                        terrain=(
                            group["terrain"]
                        ),
                        seed=seed,
                    )
                )

                profile = (
                    extract_profile(
                        env
                    )
                )

                gates = {
                    f"{gate:.3f}":
                        aggregate_after_gate(
                            profile["bins"],
                            gate,
                        )
                    for gate
                    in GATE_CANDIDATES_S
                }

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

                    "roll_rms_rad":
                        float(
                            row[
                                "roll_rms_rad"
                            ]
                        ),

                    "pitch_rms_rad":
                        float(
                            row[
                                "pitch_rms_rad"
                            ]
                        ),

                    **profile,

                    "post_gate":
                        gates,
                }

                episodes.append(
                    episode
                )

                with episode_jsonl.open(
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
                    f"success="
                    f"{episode['success']} "
                    f"rawRMS="
                    f"{episode['overall_slip_rms_mps']:.6f}"
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
            "At least one TRAIN9 episode failed."
        )

    # --------------------------------------------------------
    # Gate summary.
    # --------------------------------------------------------
    gate_rows = []

    summary = {
        "schema":
            "icra27_os_t4p1b_contact_age_train9_v0",

        "checkpoint":
            str(checkpoint),

        "beta":
            list(beta),

        "all_success":
            True,

        "gate_candidates_s":
            list(
                GATE_CANDIDATES_S
            ),

        "terrain_gate_medians":
            {},
    }

    for group in GROUPS:
        name = group[
            "name"
        ]

        subset = [
            item
            for item in episodes
            if item["group"] == name
        ]

        summary[
            "terrain_gate_medians"
        ][name] = {}

        for gate in GATE_CANDIDATES_S:
            key = f"{gate:.3f}"

            rms_values = [
                item[
                    "post_gate"
                ][key][
                    "rms_mps"
                ]
                for item in subset
            ]

            mean_values = [
                item[
                    "post_gate"
                ][key][
                    "mean_mps"
                ]
                for item in subset
            ]

            time_values = [
                item[
                    "post_gate"
                ][key][
                    "contact_time_s"
                ]
                for item in subset
            ]

            if any(
                x is None
                for x in rms_values
            ):
                raise RuntimeError(
                    f"Missing RMS for "
                    f"{name} gate={gate}"
                )

            item = {
                "median_rms_mps":
                    median(
                        rms_values
                    ),

                "median_mean_mps":
                    median(
                        mean_values
                    ),

                "median_contact_time_s":
                    median(
                        time_values
                    ),
            }

            summary[
                "terrain_gate_medians"
            ][name][key] = item

            gate_rows.append(
                {
                    "terrain":
                        name,

                    "gate_s":
                        gate,

                    **item,
                }
            )

    # --------------------------------------------------------
    # Ratios relative to flat.
    # --------------------------------------------------------
    summary[
        "terrain_ratios_vs_flat"
    ] = {}

    for gate in GATE_CANDIDATES_S:
        key = f"{gate:.3f}"

        flat = summary[
            "terrain_gate_medians"
        ][
            "flat"
        ][key][
            "median_rms_mps"
        ]

        low = summary[
            "terrain_gate_medians"
        ][
            "low_friction"
        ][key][
            "median_rms_mps"
        ]

        rough = summary[
            "terrain_gate_medians"
        ][
            "rough"
        ][key][
            "median_rms_mps"
        ]

        summary[
            "terrain_ratios_vs_flat"
        ][key] = {
            "low_over_flat":
                float(
                    low / flat
                ),

            "rough_over_flat":
                float(
                    rough / flat
                ),
        }

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
        / "gate_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "terrain",
                "gate_s",
                "median_rms_mps",
                "median_mean_mps",
                "median_contact_time_s",
            ),
        )

        writer.writeheader()
        writer.writerows(
            gate_rows
        )

    print("=" * 100)
    print(
        "OS-T4.1b ESTABLISHED-STANCE "
        "GATE SUMMARY"
    )
    print("=" * 100)

    print(
        f"{'gate':>7} "
        f"{'flat RMS':>12} "
        f"{'low RMS':>12} "
        f"{'rough RMS':>12} "
        f"{'low/flat':>10} "
        f"{'rough/flat':>12}"
    )

    print("-" * 75)

    for gate in GATE_CANDIDATES_S:
        key = f"{gate:.3f}"

        flat = summary[
            "terrain_gate_medians"
        ]["flat"][key][
            "median_rms_mps"
        ]

        low = summary[
            "terrain_gate_medians"
        ]["low_friction"][key][
            "median_rms_mps"
        ]

        rough = summary[
            "terrain_gate_medians"
        ]["rough"][key][
            "median_rms_mps"
        ]

        ratio = summary[
            "terrain_ratios_vs_flat"
        ][key]

        print(
            f"{gate:7.3f} "
            f"{flat:12.6f} "
            f"{low:12.6f} "
            f"{rough:12.6f} "
            f"{ratio['low_over_flat']:10.3f} "
            f"{ratio['rough_over_flat']:12.3f}"
        )

    print()
    print(
        "episodes :",
        episode_jsonl,
    )

    print(
        "summary  :",
        summary_path,
    )

    print(
        "gate csv :",
        csv_path,
    )


if __name__ == "__main__":
    main()
