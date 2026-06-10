import torch

from isaaclab_tracer.utils.paths import tracer_config_path

from isaaclab_tracer.envs.tracer_task_spec import TracerTaskSpec
from isaaclab_tracer.envs.tracer_mock_env import TracerMockEnv
from isaaclab_tracer.envs.tracer_isaac_env_cfg import TracerIsaacEnvCfg


def main():
    spec = TracerTaskSpec.from_json(tracer_config_path("configs/isaac_lab/tracer_task_v0.json"))

    env = TracerMockEnv(
        spec=spec,
        num_envs=8,
        device="cpu",
    )

    cfg = TracerIsaacEnvCfg()

    print("cfg name:", cfg.name)
    print("obs dim:", env.observation_dim)
    print("action dim:", env.action_dim)

    obs = env.reset()
    print("reset obs:", tuple(obs.shape))

    for i in range(10):
        action = torch.randn(env.num_envs, env.action_dim) * 0.1
        out = env.step(action)

        print("step", i)
        print("  obs:", tuple(out.obs.shape))
        print("  reward mean:", out.reward.mean().item())
        print("  done:", int(out.done.sum().item()))
        print("  goal dist mean:", out.extras["goal_distance"].mean().item())
        print("  rho/sigma mean:", out.extras["rho"].mean().item(), out.extras["sigma"].mean().item())
        print("  beta mean:", [
            out.extras["beta_v"].mean().item(),
            out.extras["beta_s"].mean().item(),
            out.extras["beta_e"].mean().item(),
        ])

    assert obs.shape[-1] == env.observation_dim
    assert out.obs.shape[-1] == env.observation_dim
    assert out.reward.shape[0] == env.num_envs
    assert out.done.shape[0] == env.num_envs

    print("M27 mock env smoke test passed.")


if __name__ == "__main__":
    main()
