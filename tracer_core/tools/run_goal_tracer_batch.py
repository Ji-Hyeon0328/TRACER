#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import subprocess


def main():
    if len(sys.argv) < 2:
        print("Usage: python tracer_core/tools/run_goal_tracer_batch.py <config1.json> [config2.json ...]")
        sys.exit(1)

    project_root = os.environ.get("TRACER_ROOT", "/root/TRACER")
    runner = os.path.join(project_root, "tracer_core/tools/run_goal_tracer_experiment.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

    for cfg in sys.argv[1:]:
        print("")
        print("=" * 80)
        print("Running config:", cfg)
        print("=" * 80)
        ret = subprocess.call(["python", runner, cfg], env=env)

        if ret != 0:
            print("Experiment failed. Stopping batch. Return code:", ret)
            sys.exit(ret)

        print("")
        print("Please confirm robot is stable before the next run.")
        print("Press Enter to continue, or Ctrl-C to stop.")
        try:
            raw_input()
        except KeyboardInterrupt:
            print("Batch interrupted.")
            sys.exit(1)


if __name__ == "__main__":
    main()
