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
| m015 | (0.5308, 0.0342, 0.4351) | (0.5301, 0.2194, 0.2505) | 0.370508 | 0.554850 | 0.293169 | 0.261680 | 0.304370 |
| m060 | (0.6776, 0.2774, 0.0450) | (0.6204, 0.1594, 0.2202) | 0.350431 | 0.625265 | 0.557075 | 0.068190 | 0.259011 |
| p060 | (0.0268, 0.5622, 0.4110) | (0.0287, 0.6033, 0.3680) | 0.086091 | 1.027067 | 0.960739 | 0.066328 | 0.721040 |
| clean | (0.3975, 0.2048, 0.3978) | (0.3807, 0.2387, 0.3807) | 0.067795 | 0.356959 | 0.304147 | 0.052813 | 0.155065 |
| p030 | (0.6601, 0.3081, 0.0318) | (0.6232, 0.3430, 0.0338) | 0.073835 | 0.626425 | 0.621789 | 0.004636 | 0.286104 |
| p015 | (0.4765, 0.2502, 0.2733) | (0.3885, 0.2666, 0.3449) | 0.175938 | 0.116401 | 0.131179 | -0.014778 | 0.203246 |
| m030 | (0.3976, 0.1967, 0.4057) | (0.3610, 0.1145, 0.5245) | 0.237566 | 0.286960 | 0.482283 | -0.195323 | 0.047588 |

## Summary

- conditions improved by robust target: `5` / `7`
- conditions worsened by robust target: `2` / `7`

- strongest improvement: `m015` with LOTO L1 improvement `0.261680`
- strongest degradation: `m030` with LOTO L1 change `-0.195323`

## Safe interpretation

F20 is still diagnostic, not deployable. If robust targets improve outlier-sensitive conditions but high-error conditions remain, the bottleneck is likely runtime feature expressiveness rather than only target noise. The next step should be feature/context design, not simply a larger linear ridge model.
