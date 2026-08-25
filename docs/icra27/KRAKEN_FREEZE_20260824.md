# TRACER ICRA27 — Kraken Freeze

Freeze date: 2026-08-24

## Purpose

This checkpoint preserves the reproducible research state before
migration from Kraken to Talos.

## Source machine

Primary worktree:

    /mnt/share/nas/Yoo/Tracer/TRACER-ICRA27

Branch:

    icra27-lowlevel-pympc

## Research lineage preserved

Objective Selector / reward-weight selector research:

- OS T4.x
  - reward/contact/traction validation
  - beta-conditioned policy diagnostics
  - physical response validation

- OS T5.x
  - physical beta-response atlas
  - Pareto/preference construction
  - context descriptors
  - height-patch representations
  - selector LOCO studies
  - regret-surface studies

- OS T6.x
  - T6.1 physical quality surface
  - T6.2 raw-patch Q-surface LOCO
  - T6.3 high-resolution terrain context
  - T6.4 physical-response/Q-transfer analysis
  - T6.5 determinism audit
  - T6.6 semantic/geometry identifiability
  - T6.7 nested local Q surface
  - T6.8 pre-policy probe response
  - T6.9 augmented causal context
  - T6.10 eta semantic-transfer analyses
  - T6.11 factorized regret surface
  - T6.12 final factorized selector and untouched evaluation
  - T6.13 beta/physical tradeoff visualization

Phase 2:

- expressible-preference atlas
- mission oracle
- inverse-beta analyses
- selector self-assessment
- beta training archive
- context-mission IRL posterior
- preference IRL posterior
- selector supervision
- observable-context compatibility
- shared-latent/shared-support analyses
- terrain compromise analysis
- terrain-objective vulnerability analysis

## Migration policy

Tracked in Git:

- source code
- analysis/evaluation/training scripts
- configuration and manifests
- compact CSV/JSON/NPZ research evidence
- compact final selector checkpoints

Transferred separately:

- raw environment JSONL logs
- bulk intermediate training checkpoints
- large datasets
- machine-generated build/install/log directories

The Talos checkout must preserve the existing Talos TRACER repository.
The migration target is a separate checkout:

    /data/jihyeony/TRACER/repos/TRACER-ICRA27
