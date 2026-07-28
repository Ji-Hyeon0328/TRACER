# TRACER LL1 Clearance Energy A/B/C Balanced Analysis

## Collection validity

- runs: 9
- balanced conditions: A/B/C = 3/3/3
- all runs rc=0: 9/9
- energy window ends at first `goal_flat` entry; post-goal hold is excluded.
- energy values are commanded mechanical-work/effort proxies, not battery electrical energy.

## Per-run outcome and energy

| run | block | condition | goal sim time | goal |y| | max |y| | rough |Δy| | work |τq̇| | positive work | τ² integral | work/m |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | A (neutral) | 44.700 | 0.238 | 0.502 | 0.002 | 4637.4 | 3544.0 | 15885.5 | 579.7 |
| 2 | 1 | B (rough_only_high) | 45.420 | 0.585 | 0.598 | 0.011 | 4591.9 | 3465.6 | 16187.5 | 574.0 |
| 3 | 1 | C (fixed_high) | 44.760 | 0.376 | 0.376 | 0.280 | 5198.4 | 3858.0 | 16129.4 | 649.7 |
| 4 | 2 | C (fixed_high) | 44.720 | 0.473 | 0.497 | 0.145 | 4935.0 | 3745.9 | 15787.5 | 616.8 |
| 5 | 2 | A (neutral) | 44.860 | 0.060 | 0.102 | 0.066 | 4525.7 | 3441.1 | 15888.0 | 565.8 |
| 6 | 2 | B (rough_only_high) | 44.580 | 0.538 | 0.538 | 0.201 | 4501.1 | 3460.2 | 16030.2 | 562.5 |
| 7 | 3 | B (rough_only_high) | 44.640 | 0.271 | 0.406 | 0.151 | 4706.5 | 3583.1 | 15722.7 | 588.2 |
| 8 | 3 | C (fixed_high) | 45.240 | 0.499 | 0.499 | 0.058 | 4897.1 | 3572.9 | 16005.7 | 612.0 |
| 9 | 3 | A (neutral) | 44.880 | 0.139 | 0.142 | 0.073 | 4463.4 | 3396.7 | 15705.5 | 558.0 |

## Condition summary

| condition | goal time | goal |y| | max |y| | rough |Δy| | work |τq̇| | positive work | τ² integral | work/m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (neutral) | 44.813 ± 0.099 | 0.146 ± 0.089 | 0.249 ± 0.220 | 0.047 ± 0.039 | 4542.2 ± 88.2 | 3460.6 ± 75.6 | 15826.3 ± 104.6 | 567.8 ± 11.0 |
| B (rough_only_high) | 44.880 ± 0.469 | 0.465 ± 0.169 | 0.514 ± 0.098 | 0.121 ± 0.098 | 4599.8 ± 102.9 | 3503.0 ± 69.4 | 15980.1 ± 236.4 | 574.9 ± 12.9 |
| C (fixed_high) | 44.907 ± 0.289 | 0.449 ± 0.065 | 0.457 ± 0.070 | 0.161 ± 0.112 | 5010.2 ± 164.1 | 3725.6 ± 143.6 | 15974.2 ± 173.1 | 626.2 ± 20.5 |

## Segment work summary

| condition | context | work |τq̇| mean±std | positive work mean±std | mean power |
|---|---|---:|---:|---:|
| A (neutral) | flat | 1020.8 ± 93.7 | 815.3 ± 52.4 | 95.18 ± 8.57 |
| A (neutral) | upslope | 1027.9 ± 135.3 | 808.9 ± 94.0 | 97.66 ± 13.74 |
| A (neutral) | rough | 1159.6 ± 67.3 | 852.5 ± 77.5 | 105.44 ± 4.14 |
| A (neutral) | downslope | 1333.2 ± 136.1 | 983.3 ± 98.5 | 106.11 ± 9.15 |
| B (rough_only_high) | flat | 1061.2 ± 31.3 | 831.4 ± 50.8 | 98.24 ± 3.27 |
| B (rough_only_high) | upslope | 985.3 ± 71.0 | 797.3 ± 73.4 | 90.45 ± 8.80 |
| B (rough_only_high) | rough | 1263.6 ± 44.0 | 934.3 ± 31.5 | 115.30 ± 3.64 |
| B (rough_only_high) | downslope | 1288.9 ± 36.6 | 939.3 ± 45.3 | 105.69 ± 1.29 |
| C (fixed_high) | flat | 1144.6 ± 56.1 | 886.1 ± 11.8 | 104.13 ± 6.06 |
| C (fixed_high) | upslope | 1065.0 ± 99.8 | 862.6 ± 68.6 | 101.65 ± 10.79 |
| C (fixed_high) | rough | 1332.9 ± 78.5 | 909.9 ± 95.1 | 120.71 ± 9.51 |
| C (fixed_high) | downslope | 1466.9 ± 45.0 | 1066.8 ± 69.1 | 118.75 ± 5.59 |

## Paired block effects

Effects are left minus right. For time, lateral deviation, work, and effort, negative is preferable.

| contrast | metric | mean effect | std | relative effect | signs (+/-) |
|---|---|---:|---:|---:|---:|
| B_minus_A | goal_time_sim_s | 0.0667 | 0.5659 | 0.15% | 1/2 |
| B_minus_A | goal_entry_abs_y | 0.3190 | 0.1746 | 345.69% | 3/0 |
| B_minus_A | max_abs_y_pre_goal | 0.2653 | 0.1698 | 210.63% | 3/0 |
| B_minus_A | rough_abs_dy | 0.0741 | 0.0634 | 234.36% | 3/0 |
| B_minus_A | goal_work_abs_J_proxy | 57.6661 | 160.9385 | 1.31% | 1/2 |
| B_minus_A | goal_work_positive_J_proxy | 42.3401 | 133.9090 | 1.28% | 2/1 |
| B_minus_A | goal_tau_sq_integral | 153.7867 | 142.7668 | 0.97% | 3/0 |
| B_minus_A | goal_dq_sq_integral | 178.2101 | 103.4161 | 1.24% | 3/0 |
| B_minus_A | work_abs_per_m_J_proxy | 7.0714 | 20.0892 | 1.28% | 1/2 |
| B_minus_A | rough_work_abs_J_proxy | 103.9838 | 49.8036 | 9.11% | 3/0 |
| B_minus_A | nonrough_work_abs_J_proxy | -46.5062 | 132.6451 | -1.32% | 1/2 |
| C_minus_A | goal_time_sim_s | 0.0933 | 0.2516 | 0.21% | 2/1 |
| C_minus_A | goal_entry_abs_y | 0.3034 | 0.1458 | 334.74% | 3/0 |
| C_minus_A | max_abs_y_pre_goal | 0.2085 | 0.2904 | 204.30% | 2/1 |
| C_minus_A | rough_abs_dy | 0.1142 | 0.1494 | 4218.48% | 2/1 |
| C_minus_A | goal_work_abs_J_proxy | 468.0096 | 81.4902 | 10.29% | 3/0 |
| C_minus_A | goal_work_positive_J_proxy | 264.9723 | 76.9825 | 7.63% | 3/0 |
| C_minus_A | goal_tau_sq_integral | 147.8685 | 216.9205 | 0.94% | 2/1 |
| C_minus_A | goal_dq_sq_integral | 1497.6565 | 173.4907 | 10.45% | 3/0 |
| C_minus_A | work_abs_per_m_J_proxy | 58.3289 | 10.2221 | 10.25% | 3/0 |
| C_minus_A | rough_work_abs_J_proxy | 173.2273 | 134.4506 | 15.39% | 3/0 |
| C_minus_A | nonrough_work_abs_J_proxy | 294.6843 | 198.1827 | 8.74% | 3/0 |
| B_minus_C | goal_time_sim_s | -0.0267 | 0.6376 | -0.05% | 1/2 |
| B_minus_C | goal_entry_abs_y | 0.0156 | 0.2224 | 7.94% | 2/1 |
| B_minus_C | max_abs_y_pre_goal | 0.0568 | 0.1580 | 16.27% | 2/1 |
| B_minus_C | rough_abs_dy | -0.0401 | 0.1994 | 34.68% | 2/1 |
| B_minus_C | goal_work_abs_J_proxy | -410.3435 | 208.9813 | -8.12% | 0/3 |
| B_minus_C | goal_work_positive_J_proxy | -222.6322 | 208.5142 | -5.84% | 1/2 |
| B_minus_C | goal_tau_sq_integral | 5.9182 | 266.6927 | 0.04% | 2/1 |
| B_minus_C | goal_dq_sq_integral | -1319.4464 | 225.0469 | -8.33% | 0/3 |
| B_minus_C | work_abs_per_m_J_proxy | -51.2576 | 26.1354 | -8.11% | 0/3 |
| B_minus_C | rough_work_abs_J_proxy | -69.2434 | 122.4941 | -4.85% | 1/2 |
| B_minus_C | nonrough_work_abs_J_proxy | -341.1905 | 194.6338 | -9.13% | 0/3 |

## Interpretation guard

- n=3 per condition is balanced descriptive screening evidence, not final statistical confirmation.
- B must preserve rough-terrain behavior while reducing non-rough work relative to C to support terrain-specific scheduling.
- Similar B and C results support a generic high-clearance effect rather than Objective Selector value.
- Large block-to-block sign reversals require additional paired repeats before promotion.

