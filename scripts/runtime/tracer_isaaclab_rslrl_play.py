from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    play_py = Path.home() / "IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py"
    play_dir = play_py.parent

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if str(play_dir) not in sys.path:
        sys.path.insert(0, str(play_dir))

    # Register TRACER gym tasks before Isaac Lab play.py resolves --task.
    import isaaclab_tracer.tasks  # noqa: F401

    sys.argv[0] = str(play_py)
    runpy.run_path(str(play_py), run_name="__main__")


if __name__ == "__main__":
    main()
