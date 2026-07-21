# TRACER Phase-F21 Mean vs Robust Calibrator Comparison v0

This compares the earlier mean-target grouped calibrator against the robust-target calibrator. Conditions absent from the older F14 baseline are marked as `NA` rather than treated as zero.

- output csv: `datasets/phase_f/f21_mean_vs_robust_calibrator_comparison_v0.csv`
- conditions: `0`
- comparable conditions: `0`
- new conditions without F14 baseline: `0`
- conditions improved by robust target: `0` / `0`
- conditions worsened by robust target: `0` / `0`

## Leave-one-tag-out comparison

| base_tag | mean beta | robust beta | target shift L1 | F14 LOTO L1 | F20 LOTO L1 | improvement | F17 max rollout delta |
|---|---|---|---:|---:|---:|---:|---:|

## Safe interpretation

Rows with `NA` in the F14 columns are new anchor conditions and should not be interpreted as degradation or improvement against the older baseline.
