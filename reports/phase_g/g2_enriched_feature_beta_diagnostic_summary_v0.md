# TRACER Phase-G2 Enriched Feature Beta Diagnostic v0

This evaluates enriched rollout-level RAM/context feature suites for robust true-metric beta prediction using leave-one-base-condition-out validation.

- input: `datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv`
- output csv: `datasets/phase_g/g2_enriched_feature_beta_diagnostic_v0.csv`
- rollout rows: `22`
- alpha: `1.0`

## Suite summary

| rank | suite | n_features | mean LOTO L1 | max LOTO L1 | worst condition | best condition |
|---:|---|---:|---:|---:|---|---|
| 1 | true_metric_upper_bound_g2_leaky | 30 | 0.312125 | 0.890352 | p060 | clean |
| 2 | tracking_energy_proxy_g2_diagnostic | 10 | 0.366466 | 1.065904 | p060 | clean |
| 3 | ram_ref_context_g2 | 50 | 0.387122 | 0.727846 | p060 | p030 |
| 4 | online_safe_enriched_g2 | 187 | 0.433056 | 0.913592 | p060 | m060 |
| 5 | all_nonleak_enriched_g2 | 187 | 0.433056 | 0.913592 | p060 | m060 |
| 6 | d7_beta_only_g2 | 108 | 0.440729 | 0.830377 | p060 | m060 |

## Per-condition LOTO for top suites

| suite | heldout | target beta | pred beta | LOTO L1 |
|---|---|---|---|---:|
| true_metric_upper_bound_g2_leaky | clean | (0.3655, 0.2691, 0.3655) | (0.3643, 0.2376, 0.3981) | 0.098746 |
| true_metric_upper_bound_g2_leaky | m015 | (0.4287, 0.2704, 0.3009) | (0.5718, 0.1948, 0.2333) | 0.475442 |
| true_metric_upper_bound_g2_leaky | m030 | (0.3313, 0.1809, 0.4878) | (0.3821, 0.2226, 0.3953) | 0.210181 |
| true_metric_upper_bound_g2_leaky | m060 | (0.4489, 0.2333, 0.3178) | (0.3964, 0.2755, 0.3281) | 0.148947 |
| true_metric_upper_bound_g2_leaky | p015 | (0.3672, 0.2869, 0.3460) | (0.3313, 0.3439, 0.3248) | 0.169058 |
| true_metric_upper_bound_g2_leaky | p030 | (0.4623, 0.3365, 0.2012) | (0.3534, 0.3504, 0.2962) | 0.219758 |
| true_metric_upper_bound_g2_leaky | p045 | (0.5711, 0.3831, 0.0458) | (0.4289, 0.3865, 0.1846) | 0.284515 |
| true_metric_upper_bound_g2_leaky | p060 | (0.0265, 0.5573, 0.4162) | (0.2683, 0.2003, 0.5315) | 0.890352 |
| tracking_energy_proxy_g2_diagnostic | clean | (0.3655, 0.2691, 0.3655) | (0.4021, 0.2603, 0.3376) | 0.075323 |
| tracking_energy_proxy_g2_diagnostic | m015 | (0.4287, 0.2704, 0.3009) | (0.2840, 0.4947, 0.2213) | 0.490017 |
| tracking_energy_proxy_g2_diagnostic | m030 | (0.3313, 0.1809, 0.4878) | (0.3882, 0.2556, 0.3562) | 0.263267 |
| tracking_energy_proxy_g2_diagnostic | m060 | (0.4489, 0.2333, 0.3178) | (0.3649, 0.2544, 0.3808) | 0.196727 |
| tracking_energy_proxy_g2_diagnostic | p015 | (0.3672, 0.2869, 0.3460) | (0.3531, 0.3407, 0.3063) | 0.116919 |
| tracking_energy_proxy_g2_diagnostic | p030 | (0.4623, 0.3365, 0.2012) | (0.3455, 0.3330, 0.3215) | 0.263600 |
| tracking_energy_proxy_g2_diagnostic | p045 | (0.5711, 0.3831, 0.0458) | (0.3816, 0.3426, 0.2758) | 0.459975 |
| tracking_energy_proxy_g2_diagnostic | p060 | (0.0265, 0.5573, 0.4162) | (0.5595, 0.2375, 0.2030) | 1.065904 |
| ram_ref_context_g2 | clean | (0.3655, 0.2691, 0.3655) | (0.2898, 0.3579, 0.3523) | 0.248872 |
| ram_ref_context_g2 | m015 | (0.4287, 0.2704, 0.3009) | (0.3526, 0.2878, 0.3596) | 0.444246 |
| ram_ref_context_g2 | m030 | (0.3313, 0.1809, 0.4878) | (0.3793, 0.2753, 0.3453) | 0.289686 |
| ram_ref_context_g2 | m060 | (0.4489, 0.2333, 0.3178) | (0.3224, 0.2781, 0.3995) | 0.253051 |
| ram_ref_context_g2 | p015 | (0.3672, 0.2869, 0.3460) | (0.3225, 0.4369, 0.2407) | 0.425691 |
| ram_ref_context_g2 | p030 | (0.4623, 0.3365, 0.2012) | (0.3927, 0.3251, 0.2823) | 0.221238 |
| ram_ref_context_g2 | p045 | (0.5711, 0.3831, 0.0458) | (0.4397, 0.2713, 0.2890) | 0.486346 |
| ram_ref_context_g2 | p060 | (0.0265, 0.5573, 0.4162) | (0.3707, 0.2788, 0.3504) | 0.727846 |
| online_safe_enriched_g2 | clean | (0.3655, 0.2691, 0.3655) | (0.5012, 0.3325, 0.1664) | 0.398191 |
| online_safe_enriched_g2 | m015 | (0.4287, 0.2704, 0.3009) | (0.5691, 0.1859, 0.2449) | 0.511497 |
| online_safe_enriched_g2 | m030 | (0.3313, 0.1809, 0.4878) | (0.4039, 0.2569, 0.3392) | 0.323841 |
| online_safe_enriched_g2 | m060 | (0.4489, 0.2333, 0.3178) | (0.4146, 0.1910, 0.3944) | 0.231771 |
| online_safe_enriched_g2 | p015 | (0.3672, 0.2869, 0.3460) | (0.3130, 0.3624, 0.3246) | 0.321191 |
| online_safe_enriched_g2 | p030 | (0.4623, 0.3365, 0.2012) | (0.3028, 0.3157, 0.3815) | 0.381680 |
| online_safe_enriched_g2 | p045 | (0.5711, 0.3831, 0.0458) | (0.4032, 0.3597, 0.2371) | 0.382688 |
| online_safe_enriched_g2 | p060 | (0.0265, 0.5573, 0.4162) | (0.4833, 0.3026, 0.2140) | 0.913592 |

## Suite meanings

- `online_safe_enriched_g2`: excludes direct rollout true metrics and target labels.
- `tracking_energy_proxy_g2_diagnostic`: uses features such as velocity tracking ratio and power/contact per velocity. These are diagnostic now, but can become online temporal-window features later.
- `true_metric_upper_bound_g2_leaky`: intentionally uses rollout true metrics. This is an upper bound only and must not be deployed.

## Safe interpretation

If online-safe or tracking-proxy suites improve over F20/F22b, Phase-G features are useful. If only the leaky true-metric upper bound improves, the current runtime logs still lack deployable explanatory features.
