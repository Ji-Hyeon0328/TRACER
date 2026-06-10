import argparse
import os
import sys


def main():
    print("TRACER DirectRLEnv import check")
    print("exe:", sys.executable)
    print("python:", sys.version.replace("\n", " "))
    print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))
    print("")

    print("Launching Isaac Sim runtime...")

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    print("")
    print("Isaac Sim runtime launched.")
    print("Loading TRACER DirectRLEnv skeleton...")

    from isaaclab_tracer.envs.tracer_direct_env_runtime import (
        describe_tracer_direct_env_skeleton,
    )

    info = describe_tracer_direct_env_skeleton()

    print("")
    print("DirectRLEnv:", info["DirectRLEnv"])
    print("DirectRLEnvCfg:", info["DirectRLEnvCfg"])
    print("TracerDirectEnv:", info["TracerDirectEnv"])
    print("issubclass(TracerDirectEnv, DirectRLEnv):", info["is_subclass"])

    print("")
    print("Required method status:")
    for name, ok in info["method_status"].items():
        print("  %s: %s" % (name, ok))

    assert info["is_subclass"]
    assert all(info["method_status"].values())

    print("")
    print("M29 DirectRLEnv skeleton import passed.")

    simulation_app.close()


if __name__ == "__main__":
    main()
