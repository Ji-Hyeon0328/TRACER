#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = Path.home() / "IsaacLab"
RSL_RL_TRAIN = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Isaac Lab's rsl_rl/train.py imports its sibling cli_args.py as `import cli_args`.
# When we execute it through runpy, we must manually add that directory.
if str(RSL_RL_TRAIN.parent) not in sys.path:
    sys.path.insert(0, str(RSL_RL_TRAIN.parent))

# Register custom TRACER task before Isaac Lab train.py parses the task registry.
import isaaclab_tracer.tasks.tracer_a1_meta_gait  # noqa: F401,E402


if not RSL_RL_TRAIN.exists():
    raise FileNotFoundError(f"Could not find Isaac Lab rsl_rl train script: {RSL_RL_TRAIN}")

sys.argv[0] = str(RSL_RL_TRAIN)
runpy.run_path(str(RSL_RL_TRAIN), run_name="__main__")
