from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    train_py = Path.home() / "IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"
    train_dir = train_py.parent

    # 1) Make TRACER importable.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # 2) Make Isaac Lab rsl_rl local imports work, e.g. `import cli_args`.
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))

    # 3) Register TRACER gym tasks before Isaac Lab train.py resolves --task.
    # This import must remain lightweight: no isaaclab_tasks / pxr import here.
    import isaaclab_tracer.tasks  # noqa: F401

    # Preserve CLI args, but make it look as if train.py was invoked directly.
    sys.argv[0] = str(train_py)
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
