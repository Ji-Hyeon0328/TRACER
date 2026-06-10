#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import json
import subprocess


def bool_to_ros(v):
    return "true" if bool(v) else "false"


def main():
    if len(sys.argv) < 2:
        print("Usage: python tracer_core/tools/run_goal_tracer_experiment.py <config.json>")
        sys.exit(1)

    cfg_path = sys.argv[1]

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    project_root = os.environ.get("TRACER_ROOT", "/root/TRACER")
    node_path = os.path.join(project_root, "tracer_core/nodes/tracer_goal_tracer_bridge.py")

    if not os.path.exists(node_path):
        print("Node not found:", node_path)
        sys.exit(1)

    cmd = ["python", node_path]

    for k in sorted(cfg.keys()):
        v = cfg[k]

        if isinstance(v, bool):
            v_str = bool_to_ros(v)
        else:
            v_str = str(v)

        cmd.append("_%s:=%s" % (k, v_str))

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

    print("Config:", cfg_path)
    print("Command:")
    print(" ".join(cmd))
    print("")

    ret = subprocess.call(cmd, env=env)

    print("")
    print("Experiment finished with return code:", ret)

    # Optional quick summary.
    log_dir = cfg.get("log_dir", "/root/TRACER/logs")
    tag = cfg.get("tag", "")

    if tag:
        print("")
        print("Latest matching log:")
        shell_cmd = (
            "latest=$(ls -t {log_dir}/tracer_goal_tracer_{tag}_*.csv 2>/dev/null | head -1); "
            "echo $latest; "
            "if [ -n \"$latest\" ]; then "
            "python {root}/tracer_core/tools/analyze_goal_tracer_csv.py $latest; "
            "fi"
        ).format(log_dir=log_dir, tag=tag, root=project_root)

        subprocess.call(shell_cmd, shell=True, env=env)


if __name__ == "__main__":
    main()
