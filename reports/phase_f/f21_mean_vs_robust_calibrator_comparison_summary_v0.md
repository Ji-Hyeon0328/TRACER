# TRACER Phase-F21 Mean vs Robust Calibrator Comparison v0

This compares the Phase-F14 mean-target runtime-safe beta calibrator against the Phase-F20 robust-median-target calibrator.

- F14 model: `models/phase_f/f14_grouped_runtime_safe_beta_calibrator_ridge_v0.json`
- F20 model: `models/phase_f/f20_robust_runtime_safe_beta_calibrator_ridge_v0.json`
- F18 robust target table: `datasets/phase_f/f18_robust_grouped_true_metric_beta_targets_v0.csv`
- F17 rollout outlier audit: `datasets/phase_f/f17_rollout_outlier_variance_audit_v0.csv`
- output csv: `datasets/phase_f/f21_mean_vs_robust_calibrator_comparison_v0.csv`

## Leave-one-tag-out comparison

| base_tag | mean beta | robust beta | target shift L1 | F14 LOTO L1 | F20 LOTO L1 | improvement | F17 max rollout delta |
|---|---|---|---:|---:|---:|---:|---:|
| m015 | (0.4707, 0.0589, 0.4704) | (0.4287, 0.2704, 0.3009) | 0.423054 | 0.554850 | 0.293169 | 0.261680 | 0.304370 |
| m060 | (0.5427, 0.2420, 0.2153) | (0.4489, 0.2333, 0.3178) | 0.204993 | 0.625265 | 0.557075 | 0.068190 | 0.259011 |
| p060 | (0.0255, 0.5348, 0.4397) | (0.0265, 0.5573, 0.4162) | 0.047050 | 1.027067 | 0.960739 | 0.066328 | 0.721040 |
| clean | (0.3934, 0.2129, 0.3937) | (0.3655, 0.2691, 0.3655) | 0.112371 | 0.356959 | 0.304147 | 0.052813 | 0.155065 |
| p030 | (0.5340, 0.2660, 0.2000) | (0.4623, 0.3365, 0.2012) | 0.143304 | 0.626425 | 0.621789 | 0.004636 | 0.286104 |
| p015 | (0.4407, 0.2429, 0.3164) | (0.3672, 0.2869, 0.3460) | 0.147177 | 0.116401 | 0.131179 | -0.014778 | 0.203246 |
| m030 | (0.3931, 0.2043, 0.4026) | (0.3313, 0.1809, 0.4878) | 0.170372 | 0.286960 | 0.482283 | -0.195323 | 0.047588 |

## Summary

- conditions improved by robust target: `5` / `7`
- conditions worsened by robust target: `2` / `7`

- strongest improvement: `m015` with LOTO L1 improvement `0.261680`
- strongest degradation: `m030` with LOTO L1 change `-0.195323`

## Safe interpretation

F20 is still diagnostic, not deployable. If robust targets improve outlier-sensitive conditions but high-error conditions remain, the bottleneck is likely runtime feature expressiveness rather than only target noise. The next step should be feature/context design, not simply a larger linear ridge model.
