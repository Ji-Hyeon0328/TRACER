#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


SOURCE_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5i_same_state_beta_sensitivity_v0"
)

STATE_BANK_PATH = (
    SOURCE_DIR
    / "common_state_bank.npz"
)

HELPER_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "diagnose_os_t4p5i_same_state_beta_sensitivity_v0.py"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5i2_checkpoint_vs_beta_drift_v0"
)


TERRAINS = (
    "flat",
    "low_friction",
    "rough_perlin",
)

BETAS = (
    "balanced",
    "motion",
    "stability",
    "energy",
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load module: {path}"
        )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def stats(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return {
        "mean":
            float(np.mean(x)),

        "median":
            float(np.median(x)),

        "q25":
            float(np.quantile(x, 0.25)),

        "q75":
            float(np.quantile(x, 0.75)),

        "max":
            float(np.max(x)),
    }


def row_cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    num = np.sum(
        a * b,
        axis=1,
    )

    den = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
    )

    result = np.full(
        num.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = den > 1e-12

    result[valid] = (
        num[valid]
        / den[valid]
    )

    return result


def main():
    if not STATE_BANK_PATH.exists():
        raise RuntimeError(
            f"Missing state bank: {STATE_BANK_PATH}"
        )

    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists; "
            f"refusing overwrite: {OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    global helper
    global trainer
    global grad_helper
    global ppo

    from tracer_core.highlevel_rl import (
        ppo as ppo_module
    )

    ppo = ppo_module

    helper = load_module(
        HELPER_PATH,
        "os_t4p5i2_helper",
    )

    # The helper's own __main__ block is not executed when
    # imported as a module, so initialize its dependencies
    # explicitly here.
    trainer = load_module(
        helper.TRAINER_PATH,
        "os_t4p5i2_trainer",
    )

    grad_helper = load_module(
        helper.GRAD_HELPER_PATH,
        "os_t4p5i2_grad_helper",
    )

    grad_helper.ppo = ppo

    helper.trainer = trainer
    helper.grad_helper = grad_helper
    helper.ppo = ppo

    beta_bank = {
        name:
            tuple(
                float(x)
                for x in beta
            )
        for name, beta
        in trainer.BETA_BANK
    }

    checkpoint_names = (
        "base_u30",
        "base_u40",
    )

    models = {}

    for name in checkpoint_names:
        model, _cfg, _payload = (
            grad_helper.load_checkpoint(
                grad_helper.CHECKPOINTS[
                    name
                ]
            )
        )

        models[name] = model

    loaded = np.load(
        STATE_BANK_PATH
    )

    state_bank = {
        terrain:
            np.asarray(
                loaded[terrain],
                dtype=np.float32,
            )
        for terrain
        in TERRAINS
    }

    actions = {}

    for checkpoint_name in checkpoint_names:
        actions[
            checkpoint_name
        ] = {}

        for terrain in TERRAINS:
            actions[
                checkpoint_name
            ][
                terrain
            ] = {}

            for beta_name in BETAS:
                actions[
                    checkpoint_name
                ][
                    terrain
                ][
                    beta_name
                ] = (
                    helper.deterministic_mean_actions(
                        models[
                            checkpoint_name
                        ],

                        state_bank[
                            terrain
                        ],

                        beta_bank[
                            beta_name
                        ],
                    )
                )

    manifest = {
        "schema":
            "icra27_os_t4p5i2_checkpoint_vs_beta_drift_v0",

        "scope":
            "TRAIN-only same-state offline checkpoint drift diagnostic",

        "state_bank":
            str(
                STATE_BANK_PATH
            ),

        "comparison":
            "base_u30 versus base_u40",

        "terrains": {},
    }

    print("=" * 126)
    print(
        "OS-T4.5i2 CHECKPOINT DRIFT "
        "VERSUS BETA MODULATION"
    )
    print("=" * 126)

    for terrain in TERRAINS:
        print()
        print(terrain)

        terrain_out = {}

        a30_bal = actions[
            "base_u30"
        ][terrain]["balanced"]

        a40_bal = actions[
            "base_u40"
        ][terrain]["balanced"]

        balanced_drift = np.linalg.norm(
            a40_bal - a30_bal,
            axis=1,
        )

        terrain_out[
            "balanced_checkpoint_drift"
        ] = stats(
            balanced_drift
        )

        print(
            "  balanced ckpt drift "
            f"median="
            f"{np.median(balanced_drift):.6f}"
        )

        specialized = {}

        for beta_name in (
            "motion",
            "stability",
            "energy",
        ):
            a30 = actions[
                "base_u30"
            ][terrain][beta_name]

            a40 = actions[
                "base_u40"
            ][terrain][beta_name]

            delta30 = (
                a30
                - a30_bal
            )

            delta40 = (
                a40
                - a40_bal
            )

            beta30 = np.linalg.norm(
                delta30,
                axis=1,
            )

            beta40 = np.linalg.norm(
                delta40,
                axis=1,
            )

            checkpoint_drift = (
                np.linalg.norm(
                    a40 - a30,
                    axis=1,
                )
            )

            beta_reference = (
                0.5
                * (
                    beta30
                    + beta40
                )
            )

            ratio = (
                checkpoint_drift
                / (
                    beta_reference
                    + 1e-12
                )
            )

            cos = row_cosine(
                delta30,
                delta40,
            )

            finite_cos = cos[
                np.isfinite(cos)
            ]

            specialized[
                beta_name
            ] = {
                "checkpoint_drift":
                    stats(
                        checkpoint_drift
                    ),

                "beta_modulation_u30":
                    stats(
                        beta30
                    ),

                "beta_modulation_u40":
                    stats(
                        beta40
                    ),

                "checkpoint_to_beta_ratio":
                    stats(
                        ratio
                    ),

                "beta_direction_cosine":
                    stats(
                        finite_cos
                    ),
            }

            print(
                f"  {beta_name:<10} "
                f"ckpt_med="
                f"{np.median(checkpoint_drift):.6f} "
                f"beta30_med="
                f"{np.median(beta30):.6f} "
                f"beta40_med="
                f"{np.median(beta40):.6f} "
                f"ratio_med="
                f"{np.median(ratio):.3f} "
                f"dir_cos_med="
                f"{np.median(finite_cos):+.4f}"
            )

        terrain_out[
            "specialized"
        ] = specialized

        manifest[
            "terrains"
        ][
            terrain
        ] = terrain_out

    out = (
        OUT_DIR
        / "checkpoint_vs_beta_drift_manifest.json"
    )

    out.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "manifest:",
        out,
    )

    print(
        "[ICRA27] OS-T4.5i2 checkpoint-vs-beta drift: PASS"
    )


if __name__ == "__main__":
    main()
