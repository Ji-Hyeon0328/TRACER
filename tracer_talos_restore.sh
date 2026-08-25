#!/usr/bin/env bash
set -euo pipefail

BASE="/data/jihyeony/TRACER/repos"
EXISTING="$BASE/TRACER"
DEST="$BASE/TRACER-ICRA27"
REMOTE="https://github.com/Ji-Hyeon0328/TRACER.git"
BRANCH="icra27-lowlevel-pympc"

mkdir -p "$BASE"

echo "Existing repo is intentionally left untouched:"
echo "  $EXISTING"

if [ -e "$DEST" ]; then
  echo "REFUSING TO OVERWRITE: $DEST"
  exit 1
fi

git clone --recurse-submodules --branch "$BRANCH" "$REMOTE" "$DEST"

cd "$DEST"
git submodule update --init --recursive

echo
echo "===== TALOS CHECKOUT ====="
git status --short --branch
git log -1 --oneline --decorate
git submodule status --recursive

echo
echo "Created:"
echo "  $DEST"
