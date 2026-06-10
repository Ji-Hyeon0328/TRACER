#!/usr/bin/env bash
set -e

if [ -z "$CONDA_PREFIX" ]; then
  echo "[ERROR] CONDA_PREFIX is empty. Please activate env_isaaclab first."
  exit 1
fi

export TRACER_ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"

export PYTHONPATH="$TRACER_ROOT:$PYTHONPATH"

cd "$ISAACLAB_ROOT"

LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" ./isaaclab.sh -p "$@"
