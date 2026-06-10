import torch

from isaaclab_tracer.utils.paths import tracer_config_path

from isaaclab_tracer.envs.tracer_task_spec import TracerTaskSpec
from isaaclab_tracer.envs.tracer_highlevel_loop import TracerHighLevelLoop
from isaaclab_tracer.envs.tracer_step_adapter import TracerStepAdapter
from isaaclab_tracer.mdp.tracer_policy_obs import policy_obs_dim_v0


def make_fake_obs(num_envs: int, device: str, step_idx: int):
    base_xy = torch.zeros(num_envs, 2, device=device)
    base_xy[:, 0] = 0.02 * step_idx

    base_yaw = torch.zeros(num_envs, 1, device=device)

    base_lin_vel_body = torch.zeros(num_envs, 3, device=device)
    base_lin_vel_body[:, 0:1] = 0.05

    base_ang_vel_body = torch.zeros(num_envs, 3, device=device)

    base_yaw_rate = torch.zeros(num_envs, 1, device=device)

    projected_gravity = torch.zeros(num_envs, 3, device=device)
    projected_gravity[:, 2] = -1.0

    height_scan = torch.randn(num_envs, 10, device=device) * 0.01

    base_height = torch.ones(num_envs, 1, device=device) * 0.30
    roll = torch.zeros(num_envs, 1, device=device)
    pitch = torch.zeros(num_envs, 1, device=device)

    joint_pos = torch.zeros(num_envs, 12, device=device)
    joint_vel = torch.zeros(num_envs, 12, device=device)
    previous_action = torch.zeros(num_envs, 12, device=device)

    return {
        "base_xy": base_xy,
        "base_yaw": base_yaw,
        "base_lin_vel_body": base_lin_vel_body,
        "base_ang_vel_body": base_ang_vel_body,
        "base_yaw_rate": base_yaw_rate,
        "projected_gravity": projected_gravity,
        "height_scan": height_scan,
        "base_height": base_height,
        "roll": roll,
        "pitch": pitch,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "previous_action": previous_action,
    }


def main():
    spec = TracerTaskSpec.from_json(tracer_config_path("configs/isaac_lab/tracer_task_v0.json"))

    num_envs = 8
    device = "cpu"

    loop = TracerHighLevelLoop(
        spec=spec,
        num_envs=num_envs,
        device=device,
    )

    goal_xy = torch.zeros(num_envs, 2, device=device)
    goal_xy[:, 0] = 1.0
    goal_xy[:, 1] = 0.5
    loop.set_goal_xy(goal_xy)

    default_joint_pos = torch.zeros(num_envs, 12, device=device)

    adapter = TracerStepAdapter(
        highlevel_loop=loop,
        default_joint_pos=default_joint_pos,
        success_tolerance=0.30,
        residual_scale=0.25,
    )

    expected_obs_dim = policy_obs_dim_v0(num_dofs=12, action_dim=12, theta_dim=6)
    print("expected policy obs dim:", expected_obs_dim)

    for i in range(5):
        obs = make_fake_obs(num_envs, device, i)
        action = torch.randn(num_envs, 12, device=device) * 0.1

        out = adapter.step(obs, action)

        print("step", i)
        print("  policy_obs:", tuple(out.policy_obs.shape))
        print("  joint_pos_target:", tuple(out.joint_pos_target.shape))
        print("  reward:", tuple(out.reward.shape), out.reward.mean().item())
        print("  done:", tuple(out.done.shape), out.done.sum().item())
        print("  tracer_cmd:", tuple(out.tracer.tracer_cmd.shape))
        print("  beta mean:", out.tracer.beta.mean(dim=0).detach().cpu().numpy())
        print("  reward keys:", sorted(out.reward_info.keys()))

    assert out.policy_obs.shape[-1] == expected_obs_dim
    print("M26 step adapter smoke test passed.")


if __name__ == "__main__":
    main()
