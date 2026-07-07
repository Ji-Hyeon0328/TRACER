# Phase-B Common Action Bank Selector Report v1

This report compares a single uniform fixed command against terrain-specific high-level selection.

Important: this is an offline/counterfactual selector analysis over directly sampled fixed-command rollouts, not yet an online mixed-terrain switching experiment.

## Dataset

- Validity report: `reports/phase_b_common_action_bank_pilot_v1_validity_episode_labels_v1c.json`
- Worlds: `6`
- Actions: `6`
- World-action groups: `36`
- Total rollout episodes: `360`

## Uniform command candidates

| uniform action | n | reach | deploy_success | hard_invalid | warning | final | drift | score_v1c |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fast solid trot | 60 | 0.733 | 0.583 | 0.250 | 0.233 | 0.620 | 0.426 | 2.640 |
| Medium trot | 60 | 0.383 | 0.283 | 0.200 | 0.217 | 0.585 | 0.380 | 2.190 |
| Sponge safe probe | 60 | 0.133 | 0.033 | 0.250 | 0.400 | 0.556 | 0.303 | 1.463 |
| Sponge late-hold | 60 | 0.133 | 0.017 | 0.183 | 0.883 | 0.684 | 0.358 | 1.036 |
| Sponge reach-biased | 60 | 0.150 | 0.000 | 0.200 | 0.900 | 0.705 | 0.328 | 0.963 |
| Sponge high-clear | 60 | 0.117 | 0.000 | 0.183 | 0.933 | 0.728 | 0.337 | 0.882 |

## Selector comparison

| mode | n | reach | deploy_success | hard_invalid | warning | final | drift | score_v1c |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Best single uniform command | 60 | 0.733 | 0.583 | 0.250 | 0.233 | 0.620 | 0.426 | 2.640 |
| Terrain-specific reach-only | 60 | 0.767 | 0.583 | 0.267 | 0.283 | 0.577 | 0.378 | 2.696 |
| Terrain-specific task objective | 60 | 0.767 | 0.583 | 0.267 | 0.283 | 0.573 | 0.377 | 2.706 |
| Terrain-specific risk filter | 60 | 0.667 | 0.550 | 0.183 | 0.200 | 0.587 | 0.395 | 2.809 |
| Terrain-specific task + risk | 60 | 0.700 | 0.550 | 0.200 | 0.250 | 0.540 | 0.353 | 2.875 |
| Terrain-specific oracle | 60 | 0.700 | 0.550 | 0.200 | 0.250 | 0.540 | 0.353 | 2.875 |

## Per-world selected actions

| mode | Flat | Stairs | Rough low | Rough mid | Slippery | Sponge |
|---|---|---|---|---|---|---|
| Best single uniform command | Fast solid trot | Fast solid trot | Fast solid trot | Fast solid trot | Fast solid trot | Fast solid trot |
| Terrain-specific reach-only | Fast solid trot | Fast solid trot | Fast solid trot | Fast solid trot | Sponge high-clear | Sponge reach-biased |
| Terrain-specific task objective | Fast solid trot | Fast solid trot | Fast solid trot | Fast solid trot | Sponge safe probe | Sponge reach-biased |
| Terrain-specific risk filter | Fast solid trot | Medium trot | Medium trot | Fast solid trot | Sponge late-hold | Fast solid trot |
| Terrain-specific task + risk | Fast solid trot | Medium trot | Medium trot | Fast solid trot | Sponge reach-biased | Sponge reach-biased |
| Terrain-specific oracle | Fast solid trot | Medium trot | Medium trot | Fast solid trot | Sponge reach-biased | Sponge reach-biased |
