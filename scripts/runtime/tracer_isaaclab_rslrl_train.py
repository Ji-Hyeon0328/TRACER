from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    # Ensure the TRACER package is importable and its gym registrations are loaded.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import isaaclab_tracer.tasks  # noqa: F401

    train_py = Path.home() / "IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"

    # Preserve CLI args, but make it look as if the Isaac Lab train.py was invoked directly.
    sys.argv[0] = str(train_py)
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
