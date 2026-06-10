import argparse
import importlib
import os
import sys


CANDIDATE_NAMES = [
    "UNITREE_A1_CFG",
    "UNITREE_GO1_CFG",
    "UNITREE_GO2_CFG",
    "ANYMAL_C_CFG",
    "ANYMAL_D_CFG",
    "ANYMAL_B_CFG",
    "SPOT_CFG",
]


def try_import(name):
    try:
        mod = importlib.import_module(name)
        print("[OK] import", name, "->", mod)
        return mod
    except Exception as e:
        print("[NO] import", name, "->", e)
        return None


def describe_cfg(name, cfg):
    print("")
    print("=== Candidate:", name, "===")
    print("type:", type(cfg))

    for attr in ["prim_path", "spawn", "init_state", "actuators", "soft_joint_pos_limit_factor"]:
        try:
            value = getattr(cfg, attr)
            print(attr + ":", value)
        except Exception as e:
            print(attr + ": <unavailable>", e)

    try:
        print("actuator keys:", list(cfg.actuators.keys()))
    except Exception as e:
        print("actuator keys: <unavailable>", e)

    try:
        print("init joint pos keys:", list(cfg.init_state.joint_pos.keys())[:20])
    except Exception as e:
        print("init joint pos keys: <unavailable>", e)


def main():
    print("TRACER robot asset probe")
    print("exe:", sys.executable)
    print("python:", sys.version.replace("\n", " "))
    print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
    print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
    print("")

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        print("")
        print("Runtime launched. Probing asset modules...")
        print("")

        modules = [
            "isaaclab_assets",
            "isaaclab_assets.robots",
            "isaaclab_assets.robots.unitree",
            "isaaclab_assets.robots.anymal",
            "isaaclab_assets.robots.boston_dynamics",
        ]

        imported = []
        for m in modules:
            mod = try_import(m)
            if mod is not None:
                imported.append(mod)

        print("")
        print("Searching candidate robot cfgs...")

        found = {}

        for mod in imported:
            print("")
            print("--- module:", mod.__name__, "---")

            names = dir(mod)
            matching = [n for n in names if "CFG" in n and any(k in n.upper() for k in ["A1", "GO1", "GO2", "ANYMAL", "SPOT"])]
            print("matching CFG names:", matching)

            for name in CANDIDATE_NAMES:
                if hasattr(mod, name):
                    found[name] = getattr(mod, name)

        if not found:
            print("")
            print("[WARN] No known quadruped cfg candidate found.")
            print("This does not necessarily mean no asset exists.")
            print("We may need to inspect isaaclab_assets package structure manually.")
        else:
            print("")
            print("Found candidates:", sorted(found.keys()))
            for name, cfg in found.items():
                describe_cfg(name, cfg)

        print("")
        print("M31 robot asset probe completed.")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
