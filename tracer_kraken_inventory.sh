#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/share/nas/Yoo/Tracer/TRACER-ICRA27}"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/migration_snapshot_$STAMP"
mkdir -p "$OUT"

{
  echo "===== DATE ====="
  date -Is
  echo
  echo "===== HOST ====="
  hostname
  uname -a
  echo
  echo "===== REPO ====="
  pwd
  git branch --show-current
  git rev-parse HEAD
  git log -1 --oneline --decorate
  echo
  echo "===== REMOTES ====="
  git remote -v
  echo
  echo "===== STATUS ====="
  git status --short --branch
  echo
  echo "===== WORKTREES ====="
  git worktree list --porcelain
  echo
  echo "===== SUBMODULES ====="
  git submodule status --recursive || true
} > "$OUT/repo_state.txt"

git status --porcelain=v1 -uall > "$OUT/git_status_porcelain.txt"

{
  echo "size_bytes,path"
  find . -type f -printf '%s,%p\n' \
    | sort -t, -k1,1nr \
    | sed -n '1,500p'
} > "$OUT/largest_files_top500.csv"

{
  echo "===== files >= 50 MiB ====="
  find . -type f -size +50M -printf '%s %p\n' | sort -nr || true
} > "$OUT/large_files_over_50MiB.txt"

{
  echo "===== RESULTS TREE ====="
  find results/icra27 -maxdepth 2 -type f \
    -printf '%s %p\n' 2>/dev/null | sort -k2 || true
} > "$OUT/icra27_results_inventory.txt"

{
  echo "===== ICRA27 SCRIPTS ====="
  find scripts/icra27 -maxdepth 1 -type f -printf '%p\n' 2>/dev/null | sort || true
  echo
  echo "===== CONFIGS ====="
  find configs -type f -printf '%p\n' 2>/dev/null | sort || true
} > "$OUT/code_inventory.txt"

{
  echo "===== IGNORED FILES ====="
  git status --ignored --short | sed -n '1,2000p'
} > "$OUT/ignored_files.txt"

if command -v conda >/dev/null 2>&1; then
  conda env export --no-builds > "$OUT/conda_environment.yml" 2>/dev/null || true
  conda list --explicit > "$OUT/conda_explicit.txt" 2>/dev/null || true
fi

python -m pip freeze > "$OUT/pip_freeze.txt" 2>/dev/null || true
python --version > "$OUT/python_version.txt" 2>&1 || true

# Hash compact experiment evidence that is usually suitable for Git.
find results/icra27 \
  -type f \
  \( -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.md' \) \
  -print0 2>/dev/null \
  | sort -z \
  | xargs -0 -r sha256sum \
  > "$OUT/result_metadata_sha256.txt"

echo
echo "Migration snapshot written to:"
echo "  $OUT"
echo
echo "Please inspect:"
echo "  $OUT/repo_state.txt"
echo "  $OUT/git_status_porcelain.txt"
echo "  $OUT/large_files_over_50MiB.txt"
echo "  $OUT/icra27_results_inventory.txt"
echo
echo "No Git commit or push was performed."
