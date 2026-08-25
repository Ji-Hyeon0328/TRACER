# TRACER ICRA27 — Kraken → Talos Migration Plan
Date: 2026-08-24

## Known repository locations

Kraken primary ICRA27 worktree:
- /mnt/share/nas/Yoo/Tracer/TRACER-ICRA27
- branch: icra27-lowlevel-pympc
- remote: Ji-Hyeon0328/TRACER

Talos repository area:
- /data/jihyeony/TRACER/repos/
- Keep the existing /data/jihyeony/TRACER/repos/TRACER untouched.
- Recommended destination: /data/jihyeony/TRACER/repos/TRACER-ICRA27

## Migration principle

Preserve three layers separately:

1. Git reproducibility layer
   - source code
   - scripts/icra27
   - configs
   - docs
   - ROS2 packages
   - submodule pointers
   - small result CSV/JSON/manifests used to support conclusions

2. Large experiment artifacts
   - checkpoints (*.pt, *.pth)
   - raw rollouts
   - large npz/npy/bag files
   - videos/images
   Use Git LFS or rsync/archive; do not blindly put these into ordinary Git.

3. Environment/provenance layer
   - git HEAD / branch / remote
   - git status
   - submodule revisions
   - conda environment export
   - pip freeze
   - result file hashes and sizes

## Current Objective Selector checkpoint to preserve

Recent milestones include:
- OS-T5.5e1 / e1l / f4 selector studies
- OS-T6.2 raw-patch balanced Q-surface LOCO
- OS-T6.3+ high-resolution/context investigations
- OS-T6.9b/c augmented-context physical/regret surrogate LOCO
- Next planned diagnostic: confidence/fallback selector audit

Keep the scripts that generated each milestone together with:
- result manifest JSON
- summary CSV
- candidate/fold prediction CSV
- any exact configuration files used by the run

## Git strategy

Recommended migration checkpoint branch/tag:

- branch: icra27-lowlevel-pympc
- checkpoint commit message:
  chore(icra27): freeze kraken selector migration checkpoint
- tag:
  tracer-icra27-kraken-freeze-20260824

Do not create the commit until the inventory has been reviewed for secrets,
huge binaries, unrelated logs, and ignored/generated files.
