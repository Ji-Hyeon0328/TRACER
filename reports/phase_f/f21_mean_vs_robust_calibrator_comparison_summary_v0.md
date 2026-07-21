# TRACER Phase-F21 Mean vs Robust Calibrator Comparison v0

This compares the earlier mean-target grouped calibrator against the robust-target calibrator. Conditions absent from the older F14 baseline are marked as `NA` rather than treated as zero.

- output csv: `datasets/phase_f/f21_mean_vs_robust_calibrator_comparison_v0.csv`
- conditions: `8`
- comparable conditions: `7`
- new conditions without F14 baseline: `1`
- conditions improved by robust target: `6` / `7`
- conditions worsened by robust target: `1` / `7`

## Leave-one-tag-out comparison

| base_tag | mean beta | robust beta | target shift L1 | F14 LOTO L1 | F20 LOTO L1 | improvement | F17 max rollout delta |
|---|---|---|---:|---:|---:|---:|---:|
| m015 | (0.4707, 0.0589, 0.4704) | (0.4287, 0.2704, 0.3009) | 0.423054 | 0.554850 | 0.202642 | 0.352208 | NA |
| m060 | (0.5427, 0.2420, 0.2153) | (0.4489, 0.2333, 0.3178) | 0.204993 | 0.625265 | 0.275416 | 0.349849 | NA |
| p030 | (0.5340, 0.2660, 0.2000) | (0.4623, 0.3365, 0.2012) | 0.143304 | 0.626425 | 0.282248 | 0.344177 | NA |
| p060 | (0.0255, 0.5348, 0.4397) | (0.0265, 0.5573, 0.4162) | 0.047050 | 1.027067 | 0.781274 | 0.245793 | NA |
| clean | (0.3934, 0.2129, 0.3937) | (0.3655, 0.2691, 0.3655) | 0.112371 | 0.356959 | 0.168541 | 0.188418 | NA |
| p015 | (0.4407, 0.2429, 0.3164) | (0.3672, 0.2869, 0.3460) | 0.147177 | 0.116401 | 0.097460 | 0.018941 | NA |
| m030 | (0.3931, 0.2043, 0.4026) | (0.3313, 0.1809, 0.4878) | 0.170372 | 0.286960 | 0.330251 | -0.043291 | NA |
| p045 | (0.7087, 0.2560, 0.0353) | (0.5711, 0.3831, 0.0458) | 0.275049 | NA | 0.586065 | NA | NA |

## Safe interpretation

Rows with `NA` in the F14 columns are new anchor conditions and should not be interpreted as degradation or improvement against the older baseline.
