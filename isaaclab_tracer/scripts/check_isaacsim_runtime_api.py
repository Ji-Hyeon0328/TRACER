import argparse
import importlib
import sys
import os


def try_import(name):
    try:
        mod = importlib.import_module(name)
        print("[OK] %s: %s" % (name, mod))
        return mod
    except Exception as e:
        print("[NO] %s: %s" % (name, e))
        return None


def main():
    print("Before launching Isaac Sim runtime:")
    print("exe:", sys.executable)
    print("python:", sys.version.replace("\n", " "))
    print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
    print("")

    print("Pre-launch imports:")
    for name in ["isaaclab", "isaacsim", "pxr", "omni", "omni.physics", "carb"]:
        try_import(name)

    print("")
    print("Launching Isaac Sim runtime through Isaac Lab AppLauncher...")

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    print("")
    print("After launching Isaac Sim runtime:")
    for name in [
        "pxr",
        "omni",
        "omni.physics",
        "isaacsim",
        "isaacsim.core",
        "isaaclab.envs",
        "isaaclab.sim",
        "isaaclab.assets",
        "isaaclab.scene",
        "isaaclab.terrains",
        "isaaclab.managers",
    ]:
        try_import(name)

    print("")
    print("Common Isaac Lab classes:")
    checks = [
        ("isaaclab.envs", "DirectRLEnv"),
        ("isaaclab.envs", "DirectRLEnvCfg"),
        ("isaaclab.envs", "ManagerBasedRLEnv"),
        ("isaaclab.envs", "ManagerBasedRLEnvCfg"),
        ("isaaclab.scene", "InteractiveSceneCfg"),
        ("isaaclab.sim", "SimulationCfg"),
        ("isaaclab.assets", "ArticulationCfg"),
    ]

    for module_name, attr in checks:
        mod = try_import(module_name)
        if mod is None:
            continue
        print("  %s.%s: %s" % (module_name, attr, hasattr(mod, attr)))

    print("")
    print("Closing simulation app.")
    simulation_app.close()


if __name__ == "__main__":
    main()
