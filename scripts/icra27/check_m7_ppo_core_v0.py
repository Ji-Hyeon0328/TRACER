#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.ppo import (
    PPOActorCritic,
    PPOConfig,
    PPORolloutBuffer,
    load_ppo_checkpoint,
    ppo_update,
    save_ppo_checkpoint,
)


def finite_dict(d):
    for key, value in d.items():
        if not np.isfinite(
            float(value)
        ):
            raise AssertionError(
                f"{key} is non-finite: "
                f"{value}"
            )


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    cfg = PPOConfig(
        update_epochs=2,
        minibatch_size=8,
    )

    model = PPOActorCritic(
        obs_dim=21,
        act_dim=4,
        initial_std=0.15,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
    )

    zero_obs = np.zeros(
        21,
        dtype=np.float32,
    )

    (
        deterministic_action,
        _,
        _,
    ) = model.act(
        zero_obs,
        deterministic=True,
    )

    if not np.allclose(
        deterministic_action,
        0.0,
        atol=1e-7,
    ):
        raise AssertionError(
            "initial deterministic policy "
            "must map to normalized zero"
        )

    buffer = PPORolloutBuffer()

    obs = zero_obs.copy()

    for i in range(16):
        action, logp, value = (
            model.act(obs)
        )

        next_obs = (
            obs
            + 0.01
            * np.random.randn(21)
            .astype(np.float32)
        )

        reward = (
            0.1
            + 0.02
            * float(action[0])
        )

        next_value = model.value(
            next_obs
        )

        continuation = (
            0.0
            if i == 15
            else 1.0
        )

        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            value=value,
            log_prob=logp,
            next_value=next_value,
            continuation=continuation,
        )

        obs = next_obs

    batch = buffer.as_batch(
        cfg=cfg
    )

    metrics = ppo_update(
        model=model,
        optimizer=optimizer,
        batch=batch,
        cfg=cfg,
    )

    finite_dict(metrics)

    out_dir = (
        ROOT
        / "results"
        / "icra27"
        / "m7_ppo_core_smoke"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ckpt = (
        out_dir
        / "ppo_core_smoke.pt"
    )

    save_ppo_checkpoint(
        ckpt,
        model=model,
        cfg=cfg,
        extra={
            "purpose":
                "m7_ppo_core_smoke",
        },
    )

    (
        loaded,
        loaded_cfg,
        _,
    ) = load_ppo_checkpoint(
        ckpt
    )

    original_action, _, _ = (
        model.act(
            zero_obs,
            deterministic=True,
        )
    )

    loaded_action, _, _ = (
        loaded.act(
            zero_obs,
            deterministic=True,
        )
    )

    reload_diff = float(
        np.max(
            np.abs(
                original_action
                - loaded_action
            )
        )
    )

    if reload_diff > 1e-7:
        raise AssertionError(
            "checkpoint reload mismatch: "
            f"{reload_diff}"
        )

    if (
        loaded_cfg.gamma
        != cfg.gamma
    ):
        raise AssertionError(
            "PPO config reload mismatch"
        )

    print(
        "# ICRA27 M7 PPO CORE CHECK"
    )
    print()

    print(
        "initial zero-mean policy : PASS"
    )

    print(
        "continuous 4D sampling   : PASS"
    )

    print(
        "GAE rollout buffer       : PASS"
    )

    print(
        "PPO clipped update       : PASS"
    )

    print(
        "checkpoint save/reload   : PASS"
    )

    print(
        "reload max action diff   :",
        reload_diff,
    )

    print()
    print(
        "metrics:"
    )

    for key, value in metrics.items():
        print(
            f"  {key:16s}: "
            f"{value:+.6f}"
        )

    print()
    print(
        "[ICRA27] M7 PPO core: PASS"
    )


if __name__ == "__main__":
    main()
