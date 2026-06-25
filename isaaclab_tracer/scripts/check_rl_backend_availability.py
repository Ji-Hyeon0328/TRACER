#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run(cmd):
    print()
    print("[cmd]", " ".join(cmd))
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        print(out[:12000])
    except subprocess.CalledProcessError as e:
        print(e.output[:12000])


def main():
    print("===== Python RL module availability =====")
    modules = [
        "gymnasium",
        "rsl_rl",
        "skrl",
        "rl_games",
        "stable_baselines3",
        "isaaclab_rl",
        "isaaclab_tasks",
    ]
    for m in modules:
        print(f"{m:24s}: {has_module(m)}")

    print()
    print("===== Important paths =====")
    candidates = [
        Path.home() / "IsaacLab",
        Path.home() / "IsaacLab" / "scripts",
        Path.home() / "IsaacLab" / "scripts" / "reinforcement_learning",
        Path.home() / "IsaacLab" / "source",
    ]
    for p in candidates:
        print(f"{p}: exists={p.exists()}")

    isaaclab = Path.home() / "IsaacLab"
    if isaaclab.exists():
        print()
        print("===== RL train scripts under ~/IsaacLab =====")
        run(["bash", "-lc", "find ~/IsaacLab -maxdepth 5 -type f \\( -name 'train.py' -o -name '*train*.py' \\) | grep -Ei 'rsl|skrl|rl_games|sb3|reinforcement' | sort | head -80"])

        print()
        print("===== rsl_rl references =====")
        run(["bash", "-lc", "grep -R \"rsl_rl\" -n ~/IsaacLab/scripts ~/IsaacLab/source 2>/dev/null | head -80"])

        print()
        print("===== skrl references =====")
        run(["bash", "-lc", "grep -R \"skrl\" -n ~/IsaacLab/scripts ~/IsaacLab/source 2>/dev/null | head -80"])

        print()
        print("===== gymnasium register examples =====")
        run(["bash", "-lc", "grep -R \"gymnasium.register\\|register(\" -n ~/IsaacLab/source 2>/dev/null | head -120"])

    print()
    print("===== Recommendation hint =====")
    if has_module("rsl_rl") and has_module("isaaclab_rl"):
        print("Preferred next path: Isaac Lab rsl_rl workflow.")
    elif has_module("skrl") and has_module("isaaclab_rl"):
        print("Preferred next path: Isaac Lab skrl workflow.")
    elif has_module("rl_games") and has_module("isaaclab_rl"):
        print("Preferred next path: Isaac Lab rl_games workflow.")
    else:
        print("No obvious Isaac Lab RL backend found yet. Need inspect IsaacLab install or install backend.")


if __name__ == "__main__":
    main()
