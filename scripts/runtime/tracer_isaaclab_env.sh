#!/usr/bin/env bash
# Source this file before running TRACER Isaac Lab scripts.
#
# Why:
# - Isaac Lab / isaaclab_tasks may import matplotlib extensions requiring CXXABI_1.3.15.
# - Ubuntu system libstdc++ may be older than conda's libstdc++.
# - Prefer conda libstdc++ via LD_LIBRARY_PATH.
#
# Usage:
#   source scripts/runtime/tracer_isaaclab_env.sh

set -e

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

# Prefer conda runtime libraries.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Avoid forcing LD_PRELOAD globally unless debugging a specific CXXABI issue.
unset LD_PRELOAD

# Keep TRACER repo importable.
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[TRACER] env_isaaclab activated"
echo "[TRACER] CONDA_PREFIX=$CONDA_PREFIX"
echo "[TRACER] python=$(which python)"
echo "[TRACER] LD_LIBRARY_PATH begins with: $CONDA_PREFIX/lib"
