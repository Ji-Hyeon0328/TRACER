# TRACER Phase-I3 I2 vs Legacy RAM Comparison v0

This compares Phase-I2 learned RAM shadow outputs against the legacy/proxy RAM stream by terrain context.

- input csv: `logs/phase_i/i2_ram_shadow_active_20260721_160236/ram_shadow_i2_v0.csv`
- output csv: `datasets/phase_i/i3_i2_active_20260721_160236_vs_legacy_ram_by_context_v0.csv`
- rows: `3036`

## Overall / known-context summary

| group | rows | legacy rows | I2 slip | I2 rough | I2 sigma | legacy slip | legacy rough | legacy sigma | L1 vs legacy | sat slip | sat rough | sat sigma | obs/imputed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ALL | 3036 | 2280 | 0.181896 | 0.607689 | 0.371379 | 0.092961 | 0.164583 | 0.189079 | 0.963292 | 0.925560 | 0.880435 | 0.582016 | 19.413043/3.586957 |
| KNOWN_CONTEXT_ONLY | 2307 | 2280 | 0.239374 | 0.483720 | 0.488733 | 0.092961 | 0.164583 | 0.189079 | 0.963292 | 0.902037 | 0.842653 | 0.449935 | 22.998700/0.001300 |

## By context

| context | rows | legacy rows | I2 slip | I2 rough | I2 sigma | legacy slip | legacy rough | legacy sigma | L1 vs legacy | sat slip | sat rough | sat sigma | y mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| downslope | 513 | 513 | 0.968937 | 1.000000 | 0.995948 | 0.250000 | 0.150000 | 0.300000 | 2.266082 | 0.920078 | 1.000000 | 0.972710 | -0.122705 |
| flat | 457 | 430 | 0.000000 | 0.143260 | 0.425857 | 0.000000 | 0.050000 | 0.050000 | 0.542992 | 1.000000 | 0.975930 | 0.164114 | 0.011131 |
| goal_flat | 500 | 500 | 0.069671 | 0.332619 | 0.100236 | 0.000000 | 0.050000 | 0.050000 | 0.503801 | 0.916000 | 0.324000 | 0.928000 | -0.277415 |
| rough | 422 | 422 | 0.048188 | 0.879535 | 0.643167 | 0.100000 | 0.400000 | 0.350000 | 0.962244 | 0.661137 | 0.966825 | 0.000000 | -0.068087 |
| unknown | 729 | 0 | 0.000000 | 1.000000 | 0.000000 | NA | NA | NA | NA | 1.000000 | 1.000000 | 1.000000 | NA |
| upslope | 415 | 415 | 0.000000 | 0.000000 | 0.242010 | 0.100000 | 0.200000 | 0.200000 | 0.343015 | 1.000000 | 1.000000 | 0.000000 | -0.019234 |

## Safe interpretation

- I2 is a learned RAM shadow trained on heuristic teacher seeds, not a replacement for legacy RAM yet.
- Large L1 from legacy RAM is expected if I2 is capturing different stress/mismatch targets.
- High saturation rates indicate that the ridge+clamp model is too sharp for active deployment.
- The next step should either damp/calibrate I2 outputs or use them only as auxiliary shadow features for H2/H3 diagnostics.
