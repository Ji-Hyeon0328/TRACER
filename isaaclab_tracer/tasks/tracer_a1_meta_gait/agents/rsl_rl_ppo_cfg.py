from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TracerA1MetaGaitPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 7
    num_steps_per_env = 192
    max_iterations = 100
    save_interval = 25
    experiment_name = "tracer_a1_meta_gait_v0"
    run_name = "vx_only_rsl_rl_smoke"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.20,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
