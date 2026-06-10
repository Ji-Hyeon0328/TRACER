from __future__ import annotations

import importlib


MODULES = [
    "isaaclab",
    "isaaclab.app",
    "isaaclab.envs",
    "isaaclab.envs.direct_rl_env",
    "isaaclab.envs.manager_based_rl_env",
    "isaaclab.managers",
    "isaaclab.scene",
    "isaaclab.sim",
    "isaaclab.assets",
    "isaaclab.terrains",
    "isaaclab.utils",
    "isaaclab.utils.configclass",
]


def try_import(name: str):
    try:
        mod = importlib.import_module(name)
        print(f"[OK] {name}: {mod}")
        return mod
    except Exception as e:
        print(f"[NO] {name}: {e}")
        return None


def main():
    print("Checking Isaac Lab API availability...")
    print("")

    for name in MODULES:
        try_import(name)

    print("")
    print("Checking common classes...")

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
        print(f"  {module_name}.{attr}:", hasattr(mod, attr))


if __name__ == "__main__":
    main()
