import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import traceback
import os
import sys


def main():
    print("TRACER A1 adapter DirectRLEnv check", flush=True)
    print("exe:", sys.executable, flush=True)
    print("python:", sys.version.replace("\\n", " "), flush=True)
    print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""), flush=True)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""), flush=True)
    print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""), flush=True)
    print("", flush=True)

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_scale", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=80)
    parser.add_argument("--hold_default_pose", action="store_true")
    parser.add_argument("--use_external_lowlevel", action="store_true")
    parser.add_argument("--external_lowlevel_timeout_s", type=float, default=0.20)
    parser.add_argument("--external_lowlevel_env_index", type=int, default=0)

    parser.add_argument("--use_nominal_gait", action="store_true")
    parser.add_argument("--gait_cmd_x", type=float, default=0.0)
    parser.add_argument("--gait_cmd_y", type=float, default=0.0)
    parser.add_argument("--gait_clearance2", type=float, default=0.20)
    parser.add_argument("--gait_warmup_steps", type=int, default=40)
    parser.add_argument("--gait_counter_speed", type=float, default=0.5)
    parser.add_argument("--gait_pattern", type=str, default="trot", choices=["trot", "crawl"])

    parser.add_argument("--use_grf_torque", action="store_true")
    parser.add_argument("--grf_mass", type=float, default=12.5)
    parser.add_argument("--grf_scale", type=float, default=1.0)
    parser.add_argument("--grf_sign", type=float, default=1.0)
    parser.add_argument("--grf_joint_damping", type=float, default=0.25)
    parser.add_argument("--grf_tau_limit", type=float, default=33.0)
    parser.add_argument("--grf_height_target", type=float, default=0.34)
    parser.add_argument("--grf_height_kp", type=float, default=35.0)
    parser.add_argument("--grf_height_kd", type=float, default=5.0)
    parser.add_argument("--grf_fz_min", type=float, default=5.0)
    parser.add_argument("--grf_fz_max", type=float, default=120.0)
    parser.add_argument("--grf_roll_kp", type=float, default=8.0)
    parser.add_argument("--grf_roll_kd", type=float, default=1.0)
    parser.add_argument("--grf_pitch_kp", type=float, default=8.0)
    parser.add_argument("--grf_pitch_kd", type=float, default=1.0)
    parser.add_argument("--grf_wrench_damping", type=float, default=1.0e-3)
    parser.add_argument("--grf_vx_kp", type=float, default=8.0)
    parser.add_argument("--grf_vy_kp", type=float, default=4.0)
    parser.add_argument("--grf_y_pos_kp", type=float, default=2.0)
    parser.add_argument("--grf_fx_max", type=float, default=20.0)
    parser.add_argument("--grf_fy_max", type=float, default=15.0)
    parser.add_argument("--grf_yaw_kp", type=float, default=2.0)
    parser.add_argument("--grf_yaw_kd", type=float, default=0.5)
    parser.add_argument("--grf_xy_wrench_damping", type=float, default=1.0e-3)

    parser.add_argument(
        "--action_mode",
        type=str,
        default="zero",
        choices=[
            "zero",
            "constant",
            "random",
            "sinusoid",
            "single_leg_wave",
            "single_leg_swing",
            "fl_static_pose",
        ],
    )
    parser.add_argument("--action_value", type=float, default=0.5)
    parser.add_argument("--sin_period", type=float, default=20.0)
    parser.add_argument("--fl_hip", type=float, default=0.0)
    parser.add_argument("--fl_thigh", type=float, default=0.0)
    parser.add_argument("--fl_calf", type=float, default=0.0)

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    print("[M41] launching Isaac app...", flush=True)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print("[M41] Isaac app launched.", flush=True)

    try:
        print("[M41] before import torch", flush=True)
        import torch
        print("[M41] after import torch", flush=True)
        print("[M41] before import env factory", flush=True)
        try:
            print("[M41] before import lowlevel_udp_client", flush=True)
            from isaaclab_tracer.utils.lowlevel_udp_client import LowLevelUdpClient
            print("[M41] after import lowlevel_udp_client", flush=True)

            print("[M41] before import tracer_a1_adapter_env", flush=True)
            from isaaclab_tracer.envs.tracer_a1_adapter_env import (
                make_tracer_a1_adapter_env_class,
            )
            print("[M41] after import tracer_a1_adapter_env", flush=True)
        except BaseException as exc:
            print("[M41][ERROR] env factory import failed:", repr(exc), flush=True)
            traceback.print_exc()
            raise

        print("[M41] before make_tracer_a1_adapter_env_class()", flush=True)
        try:
            TracerA1AdapterEnv, TracerA1AdapterEnvCfg = make_tracer_a1_adapter_env_class()
            print("[M41] after make_tracer_a1_adapter_env_class()", flush=True)
        except BaseException as exc:
            print("[M41][ERROR] make_tracer_a1_adapter_env_class failed:", repr(exc), flush=True)
            traceback.print_exc()
            raise
        print("[M41] imported env class factory.", flush=True)
        cfg = TracerA1AdapterEnvCfg()

        cfg.residual_scale = args_cli.residual_scale
        cfg.hold_default_pose = args_cli.hold_default_pose
        cfg.use_external_lowlevel = args_cli.use_external_lowlevel
        cfg.external_lowlevel_timeout_s = args_cli.external_lowlevel_timeout_s
        cfg.external_lowlevel_env_index = args_cli.external_lowlevel_env_index
        if args_cli.use_external_lowlevel:
            cfg.scene.num_envs = 1

        cfg.use_nominal_gait = args_cli.use_nominal_gait
        cfg.gait_cmd_x = args_cli.gait_cmd_x
        cfg.gait_cmd_y = args_cli.gait_cmd_y
        cfg.gait_clearance2 = args_cli.gait_clearance2
        cfg.gait_warmup_steps = args_cli.gait_warmup_steps
        cfg.gait_counter_speed = args_cli.gait_counter_speed
        cfg.gait_pattern = args_cli.gait_pattern

        cfg.use_grf_torque = args_cli.use_grf_torque
        cfg.grf_mass = args_cli.grf_mass
        cfg.grf_scale = args_cli.grf_scale
        cfg.grf_sign = args_cli.grf_sign
        cfg.grf_joint_damping = args_cli.grf_joint_damping
        cfg.grf_tau_limit = args_cli.grf_tau_limit
        cfg.grf_height_target = args_cli.grf_height_target
        cfg.grf_height_kp = args_cli.grf_height_kp
        cfg.grf_height_kd = args_cli.grf_height_kd
        cfg.grf_fz_min = args_cli.grf_fz_min
        cfg.grf_fz_max = args_cli.grf_fz_max
        cfg.grf_roll_kp = args_cli.grf_roll_kp
        cfg.grf_roll_kd = args_cli.grf_roll_kd
        cfg.grf_pitch_kp = args_cli.grf_pitch_kp
        cfg.grf_pitch_kd = args_cli.grf_pitch_kd
        cfg.grf_wrench_damping = args_cli.grf_wrench_damping
        cfg.grf_vx_kp = args_cli.grf_vx_kp
        cfg.grf_vy_kp = args_cli.grf_vy_kp
        cfg.grf_y_pos_kp = args_cli.grf_y_pos_kp
        cfg.grf_fx_max = args_cli.grf_fx_max
        cfg.grf_fy_max = args_cli.grf_fy_max
        cfg.grf_yaw_kp = args_cli.grf_yaw_kp
        cfg.grf_yaw_kd = args_cli.grf_yaw_kd
        cfg.grf_xy_wrench_damping = args_cli.grf_xy_wrench_damping

        print("cfg robot prim path:", cfg.robot.prim_path, flush=True)
        print("cfg num_envs:", cfg.scene.num_envs, flush=True)
        print("cfg obs dim:", cfg.observation_space, flush=True)
        print("cfg action dim:", cfg.action_space, flush=True)
        print("residual_scale:", cfg.residual_scale, flush=True)
        print("num_steps:", args_cli.num_steps, flush=True)
        print("use_nominal_gait:", args_cli.use_nominal_gait, flush=True)
        print("gait_cmd_x:", args_cli.gait_cmd_x, flush=True)
        print("gait_cmd_y:", args_cli.gait_cmd_y, flush=True)
        print("gait_clearance2:", args_cli.gait_clearance2, flush=True)
        print("gait_warmup_steps:", args_cli.gait_warmup_steps, flush=True)
        print("gait_counter_speed:", args_cli.gait_counter_speed, flush=True)
        print("gait_pattern:", args_cli.gait_pattern, flush=True)
        print("use_grf_torque:", args_cli.use_grf_torque, flush=True)
        print("grf_mass:", args_cli.grf_mass, flush=True)
        print("grf_scale:", args_cli.grf_scale, flush=True)
        print("grf_sign:", args_cli.grf_sign, flush=True)
        print("grf_joint_damping:", args_cli.grf_joint_damping, flush=True)
        print("grf_tau_limit:", args_cli.grf_tau_limit, flush=True)
        print("grf_height_target:", args_cli.grf_height_target, flush=True)
        print("grf_height_kp:", args_cli.grf_height_kp, flush=True)
        print("grf_height_kd:", args_cli.grf_height_kd, flush=True)
        print("grf_fz_min:", args_cli.grf_fz_min, flush=True)
        print("grf_fz_max:", args_cli.grf_fz_max, flush=True)
        print("grf_roll_kp:", args_cli.grf_roll_kp, flush=True)
        print("grf_roll_kd:", args_cli.grf_roll_kd, flush=True)
        print("grf_pitch_kp:", args_cli.grf_pitch_kp, flush=True)
        print("grf_pitch_kd:", args_cli.grf_pitch_kd, flush=True)
        print("grf_wrench_damping:", args_cli.grf_wrench_damping, flush=True)
        print("grf_vx_kp:", args_cli.grf_vx_kp, flush=True)
        print("grf_vy_kp:", args_cli.grf_vy_kp, flush=True)
        print("grf_y_pos_kp:", args_cli.grf_y_pos_kp, flush=True)
        print("grf_fx_max:", args_cli.grf_fx_max, flush=True)
        print("grf_fy_max:", args_cli.grf_fy_max, flush=True)
        print("grf_yaw_kp:", args_cli.grf_yaw_kp, flush=True)
        print("grf_yaw_kd:", args_cli.grf_yaw_kd, flush=True)
        print("grf_xy_wrench_damping:", args_cli.grf_xy_wrench_damping, flush=True)
        print("hold_default_pose:", cfg.hold_default_pose, flush=True)
        print("action_mode:", args_cli.action_mode, flush=True)
        print("action_value:", args_cli.action_value, flush=True)
        print("sin_period:", args_cli.sin_period, flush=True)
        print("fl_hip:", args_cli.fl_hip, flush=True)
        print("fl_thigh:", args_cli.fl_thigh, flush=True)
        print("fl_calf:", args_cli.fl_calf, flush=True)
        print("", flush=True)

        env = TracerA1AdapterEnv(cfg)
        print("env created", flush=True)
        print("env.num_envs:", env.num_envs, flush=True)
        print("env.device:", env.device, flush=True)
        print("joint names:", env.robot.data.joint_names, flush=True)
        print("body names:", env.robot.data.body_names, flush=True)
        print("", flush=True)

        obs, extras = env.reset()
        print("reset obs policy:", tuple(obs["policy"].shape), flush=True)
        print("reset root height mean:", env.robot.data.root_pos_w[:, 2].mean().item(), flush=True)
        print("reset default joint pos mean:", env.robot.data.default_joint_pos.mean().item(), flush=True)
        print("reset joint pos mean:", env.robot.data.joint_pos.mean().item(), flush=True)
        print("reset default joint pos[0]:", env.robot.data.default_joint_pos[0].detach().cpu().numpy(), flush=True)
        print("reset joint pos[0]:", env.robot.data.joint_pos[0].detach().cpu().numpy(), flush=True)
        print("", flush=True)

        min_h = 999.0
        max_h = -999.0
        terminated_total = 0
        truncated_total = 0
        root_xy0 = env.robot.data.root_pos_w[:, :2].detach().clone()
        gait_xy0 = None
        gait_yaw0 = None

        for i in range(args_cli.num_steps):
            if args_cli.action_mode == "zero":
                actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)

            elif args_cli.action_mode == "constant":
                actions = torch.full(
                    (env.num_envs, cfg.action_space),
                    float(args_cli.action_value),
                    device=env.device,
                )

            elif args_cli.action_mode == "random":
                actions = 2.0 * torch.rand(env.num_envs, cfg.action_space, device=env.device) - 1.0
                actions = float(args_cli.action_value) * actions

            elif args_cli.action_mode == "sinusoid":
                phase = torch.tensor(
                    2.0 * 3.141592653589793 * (float(i) / float(args_cli.sin_period)),
                    device=env.device,
                )
                s0 = torch.sin(phase)
                s1 = torch.sin(phase + 3.141592653589793)
                amp = float(args_cli.action_value)
                actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
                actions[:, 4] = amp * s0
                actions[:, 5] = amp * s1
                actions[:, 6] = amp * s1
                actions[:, 7] = amp * s0
                actions[:, 8] = -0.5 * amp * s0
                actions[:, 9] = -0.5 * amp * s1
                actions[:, 10] = -0.5 * amp * s1
                actions[:, 11] = -0.5 * amp * s0

            elif args_cli.action_mode == "single_leg_wave":
                phase = torch.tensor(
                    2.0 * 3.141592653589793 * (float(i) / float(args_cli.sin_period)),
                    device=env.device,
                )
                s0 = torch.sin(phase)
                amp = float(args_cli.action_value)
                actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
                actions[:, 4] = amp * s0
                actions[:, 8] = -amp * s0

            elif args_cli.action_mode == "single_leg_swing":
                phase01 = (float(i) % float(args_cli.sin_period)) / float(args_cli.sin_period)
                phase = torch.tensor(phase01, device=env.device)
                lift = torch.sin(torch.pi * phase).clamp(min=0.0)
                amp = float(args_cli.action_value)
                actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
                actions[:, 4] = amp * lift
                actions[:, 8] = -1.2 * amp * lift

            elif args_cli.action_mode == "fl_static_pose":
                actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
                actions[:, 0] = float(args_cli.fl_hip)
                actions[:, 4] = float(args_cli.fl_thigh)
                actions[:, 8] = float(args_cli.fl_calf)

            else:
                raise RuntimeError(f"Unknown action_mode: {args_cli.action_mode}")

            obs, reward, terminated, truncated, extras = env.step(actions)

            if i == args_cli.gait_warmup_steps:
                gait_xy0 = env.robot.data.root_pos_w[:, :2].detach().clone()
                try:
                    gait_yaw0 = env._make_tracer_obs_dict()["yaw"].detach().clone()
                except Exception:
                    gait_yaw0 = None

            h = env.robot.data.root_pos_w[:, 2].mean().item()
            min_h = min(min_h, h)
            max_h = max(max_h, h)
            terminated_total += int(terminated.sum().item())
            truncated_total += int(truncated.sum().item())

            if (
                i < 12
                or i % 10 == 0
                or i == args_cli.num_steps - 1
            ):
                target = env._joint_pos_target
                default = env.robot.data.default_joint_pos
                print(f"step {i}", flush=True)
                print("  obs policy:", tuple(obs["policy"].shape), flush=True)
                print("  reward mean:", reward.mean().item(), flush=True)
                print("  terminated:", int(terminated.sum().item()), flush=True)
                print("  truncated:", int(truncated.sum().item()), flush=True)

                root_xy = env.robot.data.root_pos_w[:, :2].detach()
                root_dxy = root_xy - root_xy0
                print("  root height mean:", h, flush=True)
                print("  root dx mean:", root_dxy[:, 0].mean().item(), flush=True)
                print("  root dy mean:", root_dxy[:, 1].mean().item(), flush=True)

                if gait_xy0 is not None:
                    gait_dxy = root_xy - gait_xy0
                    print("  gait dx mean:", gait_dxy[:, 0].mean().item(), flush=True)
                    print("  gait dy mean:", gait_dxy[:, 1].mean().item(), flush=True)

                    if gait_yaw0 is not None:
                        try:
                            yaw_now = env._make_tracer_obs_dict()["yaw"].detach()
                            gait_yaw = yaw_now - gait_yaw0
                            print("  gait yaw mean:", gait_yaw.mean().item(), flush=True)
                        except Exception:
                            pass

                print("  target mean:", target.mean().item(), flush=True)
                print("  target-default abs mean:", (target - default).abs().mean().item(), flush=True)

        print("", flush=True)
        print("summary", flush=True)
        print("  residual_scale:", args_cli.residual_scale, flush=True)
        print("  min_h:", min_h, flush=True)
        print("  max_h:", max_h, flush=True)

        final_dxy = env.robot.data.root_pos_w[:, :2].detach() - root_xy0
        print("  final_h:", env.robot.data.root_pos_w[:, 2].mean().item(), flush=True)
        print("  final_dx:", final_dxy[:, 0].mean().item(), flush=True)
        print("  final_dy:", final_dxy[:, 1].mean().item(), flush=True)

        if gait_xy0 is not None:
            final_gait_dxy = env.robot.data.root_pos_w[:, :2].detach() - gait_xy0
            print("  final_gait_dx:", final_gait_dxy[:, 0].mean().item(), flush=True)
            print("  final_gait_dy:", final_gait_dxy[:, 1].mean().item(), flush=True)

        print("  terminated_total:", terminated_total, flush=True)
        print("  truncated_total:", truncated_total, flush=True)
        print("", flush=True)
        print("M41 A1 adapter residual-scale check completed.", flush=True)

        env.close()

    finally:
        print("[M41] closing Isaac app", flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
