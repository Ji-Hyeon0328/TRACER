# TRACER Phase-G2 Enriched Feature Beta Diagnostic v0

This evaluates enriched rollout-level RAM/context feature suites for robust true-metric beta prediction using leave-one-base-condition-out validation.

- input: `datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv`
- output csv: `datasets/phase_g/g2_enriched_feature_beta_diagnostic_v0.csv`
- rollout rows: `21`
- alpha: `1.0`

## Suite summary

| rank | suite | n_features | mean LOTO L1 | max LOTO L1 | worst condition | best condition |
|---:|---|---:|---:|---:|---|---|
| 1 | true_metric_upper_bound_g2_leaky | 30 | 0.428244 | 1.012411 | p060 | clean |
| 2 | tracking_energy_proxy_g2_diagnostic | 10 | 0.522679 | 1.245095 | p060 | p015 |
| 3 | ram_ref_context_g2 | 50 | 0.565804 | 0.918083 | p060 | clean |
| 4 | online_safe_enriched_g2 | 187 | 0.608253 | 1.043200 | p060 | p015 |
| 5 | all_nonleak_enriched_g2 | 187 | 0.608253 | 1.043200 | p060 | p015 |
| 6 | d7_beta_only_g2 | 108 | 0.639909 | 1.022533 | p060 | p015 |

## Per-condition LOTO for top suites

| suite | heldout | target beta | pred beta | LOTO L1 |
|---|---|---|---|---:|
| true_metric_upper_bound_g2_leaky | clean | (0.3807, 0.2387, 0.3807) | (0.4246, 0.1888, 0.3866) | 0.178446 |
| true_metric_upper_bound_g2_leaky | m015 | (0.5301, 0.2194, 0.2505) | (0.5262, 0.2581, 0.2158) | 0.374935 |
| true_metric_upper_bound_g2_leaky | m030 | (0.3610, 0.1145, 0.5245) | (0.4802, 0.1672, 0.3526) | 0.370469 |
| true_metric_upper_bound_g2_leaky | m060 | (0.6204, 0.1594, 0.2202) | (0.4562, 0.2523, 0.2915) | 0.341635 |
| true_metric_upper_bound_g2_leaky | p015 | (0.3885, 0.2666, 0.3449) | (0.3972, 0.3324, 0.2704) | 0.223186 |
| true_metric_upper_bound_g2_leaky | p030 | (0.6232, 0.3430, 0.0338) | (0.3947, 0.3313, 0.2739) | 0.496628 |
| true_metric_upper_bound_g2_leaky | p060 | (0.0287, 0.6033, 0.3680) | (0.3147, 0.1917, 0.4936) | 1.012411 |
| tracking_energy_proxy_g2_diagnostic | clean | (0.3807, 0.2387, 0.3807) | (0.4876, 0.2149, 0.2975) | 0.213902 |
| tracking_energy_proxy_g2_diagnostic | m015 | (0.5301, 0.2194, 0.2505) | (0.2829, 0.4354, 0.2817) | 0.517309 |
| tracking_energy_proxy_g2_diagnostic | m030 | (0.3610, 0.1145, 0.5245) | (0.4976, 0.2080, 0.2944) | 0.460011 |
| tracking_energy_proxy_g2_diagnostic | m060 | (0.6204, 0.1594, 0.2202) | (0.4032, 0.2157, 0.3812) | 0.452448 |
| tracking_energy_proxy_g2_diagnostic | p015 | (0.3885, 0.2666, 0.3449) | (0.4212, 0.3237, 0.2552) | 0.179403 |
| tracking_energy_proxy_g2_diagnostic | p030 | (0.6232, 0.3430, 0.0338) | (0.3647, 0.3062, 0.3291) | 0.590586 |
| tracking_energy_proxy_g2_diagnostic | p060 | (0.0287, 0.6033, 0.3680) | (0.6513, 0.1803, 0.1684) | 1.245095 |
| ram_ref_context_g2 | clean | (0.3807, 0.2387, 0.3807) | (0.2943, 0.3240, 0.3817) | 0.382179 |
| ram_ref_context_g2 | m015 | (0.5301, 0.2194, 0.2505) | (0.4177, 0.2470, 0.3353) | 0.651520 |
| ram_ref_context_g2 | m030 | (0.3610, 0.1145, 0.5245) | (0.4788, 0.2396, 0.2816) | 0.485651 |
| ram_ref_context_g2 | m060 | (0.6204, 0.1594, 0.2202) | (0.3950, 0.2578, 0.3472) | 0.450779 |
| ram_ref_context_g2 | p015 | (0.3885, 0.2666, 0.3449) | (0.3475, 0.4326, 0.2199) | 0.508293 |
| ram_ref_context_g2 | p030 | (0.6232, 0.3430, 0.0338) | (0.4242, 0.2689, 0.3069) | 0.564122 |
| ram_ref_context_g2 | p060 | (0.0287, 0.6033, 0.3680) | (0.4531, 0.2669, 0.2800) | 0.918083 |
| online_safe_enriched_g2 | clean | (0.3807, 0.2387, 0.3807) | (0.5574, 0.2805, 0.1621) | 0.579790 |
| online_safe_enriched_g2 | m015 | (0.5301, 0.2194, 0.2505) | (0.5642, 0.2206, 0.2152) | 0.432613 |
| online_safe_enriched_g2 | m030 | (0.3610, 0.1145, 0.5245) | (0.6055, 0.2039, 0.1906) | 0.718555 |
| online_safe_enriched_g2 | m060 | (0.6204, 0.1594, 0.2202) | (0.4594, 0.1289, 0.4116) | 0.427659 |
| online_safe_enriched_g2 | p015 | (0.3885, 0.2666, 0.3449) | (0.3603, 0.3379, 0.3019) | 0.304413 |
| online_safe_enriched_g2 | p030 | (0.6232, 0.3430, 0.0338) | (0.3080, 0.2851, 0.4069) | 0.751538 |
| online_safe_enriched_g2 | p060 | (0.0287, 0.6033, 0.3680) | (0.5503, 0.2788, 0.1709) | 1.043200 |

## Suite meanings

- `online_safe_enriched_g2`: excludes direct rollout true metrics and target labels.
- `tracking_energy_proxy_g2_diagnostic`: uses features such as velocity tracking ratio and power/contact per velocity. These are diagnostic now, but can become online temporal-window features later.
- `true_metric_upper_bound_g2_leaky`: intentionally uses rollout true metrics. This is an upper bound only and must not be deployed.

## Safe interpretation

If online-safe or tracking-proxy suites improve over F20/F22b, Phase-G features are useful. If only the leaky true-metric upper bound improves, the current runtime logs still lack deployable explanatory features.
