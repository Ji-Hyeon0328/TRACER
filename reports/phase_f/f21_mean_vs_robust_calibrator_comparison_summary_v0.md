# TRACER Phase-F21 Mean vs Robust Calibrator Comparison v0

This compares the earlier mean-target grouped calibrator against the robust-target calibrator. Conditions absent from the older F14 baseline are marked as `NA` rather than treated as zero.

- output csv: `datasets/phase_f/f21_mean_vs_robust_calibrator_comparison_v0.csv`
- conditions: `8`
- comparable conditions: `7`
- new conditions without F14 baseline: `1`
- conditions improved by robust target: `7` / `7`
- conditions worsened by robust target: `0` / `7`

## Leave-one-tag-out comparison

| base_tag | mean beta | robust beta | target shift L1 | F14 LOTO L1 | F20 LOTO L1 | improvement | F17 max rollout delta |
|---|---|---|---:|---:|---:|---:|---:|
| m060 | (0.5427, 0.2420, 0.2153) | (0.4489, 0.2333, 0.3178) | 0.204993 | 0.306800 | 0.142900 | 0.163900 | NA |
| p030 | (0.5340, 0.2660, 0.2000) | (0.4623, 0.3365, 0.2012) | 0.143304 | 0.246400 | 0.110900 | 0.135500 | NA |
| p060 | (0.0255, 0.5348, 0.4397) | (0.0265, 0.5573, 0.4162) | 0.047050 | 0.515000 | 0.388600 | 0.126400 | NA |
| clean | (0.3934, 0.2129, 0.3937) | (0.3655, 0.2691, 0.3655) | 0.112371 | 0.169300 | 0.084900 | 0.084400 | NA |
| m030 | (0.3931, 0.2043, 0.4026) | (0.3313, 0.1809, 0.4878) | 0.170372 | 0.075500 | 0.036100 | 0.039400 | NA |
| m015 | (0.4707, 0.0589, 0.4704) | (0.4287, 0.2704, 0.3009) | 0.423054 | 0.182200 | 0.150900 | 0.031300 | NA |
| p015 | (0.4407, 0.2429, 0.3164) | (0.3672, 0.2869, 0.3460) | 0.147177 | 0.050300 | 0.032900 | 0.017400 | NA |
| p045 | (0.7087, 0.2560, 0.0353) | (0.5711, 0.3831, 0.0458) | 0.275049 | NA | 0.229900 | NA | NA |

## Safe interpretation

Rows with `NA` in the F14 columns are new anchor conditions and should not be interpreted as degradation or improvement against the older baseline.
