#!/usr/bin/env bash
set -e

if [ -z "$CONDA_PREFIX" ]; then
  echo "[ERROR] CONDA_PREFIX is empty. Please activate env_isaaclab first."
  exit 1
fi

export TRACER_ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"

export PYTHONPATH="$TRACER_ROOT:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$ISAACLAB_ROOT"

./isaaclab.sh -p "$@"
