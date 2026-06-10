#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import subprocess


def try_import(name):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "unknown")
        print("%s: available, version=%s" % (name, ver))
        return True
    except Exception as e:
        print("%s: not available (%s)" % (name, str(e)))
        return False


def main():
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("")
    print("Environment:")
    for k in [
        "ISAACSIM_PATH",
        "ISAACLAB_PATH",
        "PYTHONPATH",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV"
    ]:
        print("%s=%s" % (k, os.environ.get(k, "")))

    print("")
    print("Imports:")
    try_import("torch")
    try_import("omni")
    try_import("isaaclab")
    try_import("gymnasium")
    try_import("numpy")

    print("")
    print("Possible Isaac/IsaacLab directories:")
    candidates = [
        os.path.expanduser("~/IsaacLab"),
        os.path.expanduser("~/isaaclab"),
        os.path.expanduser("~/isaac_sim"),
        os.path.expanduser("~/isaac-sim"),
        "/workspace/isaaclab",
        "/workspace/IsaacLab",
        "/root/IsaacLab",
        "/root/isaaclab"
    ]

    for c in candidates:
        if os.path.exists(c):
            print("exists:", c)


if __name__ == "__main__":
    main()
