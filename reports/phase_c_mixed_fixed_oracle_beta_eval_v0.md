# TRACER Phase-C Mixed Course Log Evaluation

| run | success | stopped | time_to_goal_s | max_abs_y | final_x | final_y | log_dir |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed | True | True | 240.00 | 0.711 | 8.410 | -0.690 | `logs/phase_c_mixed_fixed_20260707_181010` |
| oracle | True | True | 294.00 | 0.321 | 8.021 | 0.310 | `logs/phase_c_mixed_oracle_20260707_181741` |
| beta | True | True | 348.00 | 0.190 | 8.010 | 0.096 | `logs/phase_c_mixed_beta_20260707_182834` |

## fixed — `logs/phase_c_mixed_fixed_20260707_181010`

Segment threshold times:

| segment | t_rel_s | x | y | seq |
|---|---:|---:|---:|---:|
| upslope_start | 62.00 | 2.052 | -0.192 | 620 |
| rough_start | 116.00 | 4.002 | -0.441 | 1160 |
| downslope_start | 176.00 | 6.051 | -0.384 | 1760 |
| goal_flat_start | 240.00 | 8.056 | -0.497 | 2400 |

Context switches:

| label | t_rel_s | x | y |
|---|---:|---:|---:|
| flat | 0.96 | 0.000 | 0.000 |
| upslope | 60.70 | 2.000 | -0.187 |
| rough | 115.92 | 4.000 | -0.441 |
| downslope | 174.41 | 6.000 | -0.381 |
| goal_flat | 238.47 | 8.000 | -0.501 |

## oracle — `logs/phase_c_mixed_oracle_20260707_181741`

Segment threshold times:

| segment | t_rel_s | x | y | seq |
|---|---:|---:|---:|---:|
| upslope_start | 48.00 | 2.075 | -0.171 | 480 |
| rough_start | 100.00 | 4.022 | -0.192 | 1000 |
| downslope_start | 174.00 | 6.020 | 0.015 | 1740 |
| goal_flat_start | 294.00 | 8.004 | 0.314 | 2940 |

Context switches:

| label | t_rel_s | x | y |
|---|---:|---:|---:|
| flat | 1.08 | 0.000 | 0.000 |
| upslope | 46.44 | 2.000 | -0.177 |
| rough | 99.50 | 4.000 | -0.196 |
| downslope | 172.89 | 6.000 | 0.013 |
| goal_flat | 288.72 | 8.000 | 0.206 |
| downslope | 289.11 | 8.000 | 0.213 |
| goal_flat | 289.37 | 8.000 | 0.216 |
| downslope | 289.63 | 8.000 | 0.219 |
| goal_flat | 290.94 | 8.000 | 0.257 |
| downslope | 291.07 | 8.000 | 0.260 |
| goal_flat | 292.99 | 8.000 | 0.301 |
| downslope | 293.10 | 8.000 | 0.301 |
| goal_flat | 293.71 | 8.000 | 0.312 |
| downslope | 294.98 | 8.000 | 0.313 |
| goal_flat | 295.32 | 8.000 | 0.317 |
| downslope | 299.23 | 8.000 | 0.319 |
| goal_flat | 299.40 | 8.000 | 0.317 |
| downslope | 301.53 | 8.000 | 0.303 |
| goal_flat | 301.97 | 8.000 | 0.302 |
| downslope | 305.09 | 8.000 | 0.291 |
| goal_flat | 305.22 | 8.000 | 0.293 |
| downslope | 305.83 | 8.000 | 0.292 |
| goal_flat | 306.04 | 8.000 | 0.290 |
| downslope | 308.98 | 8.000 | 0.291 |
| goal_flat | 309.18 | 8.000 | 0.291 |
| downslope | 309.86 | 8.000 | 0.292 |
| goal_flat | 310.20 | 8.000 | 0.300 |
| downslope | 313.72 | 8.000 | 0.314 |
| goal_flat | 314.12 | 8.000 | 0.313 |
| downslope | 315.44 | 8.000 | 0.310 |
| goal_flat | 315.70 | 8.000 | 0.309 |

## beta — `logs/phase_c_mixed_beta_20260707_182834`

Segment threshold times:

| segment | t_rel_s | x | y | seq |
|---|---:|---:|---:|---:|
| upslope_start | 52.00 | 2.047 | 0.154 | 520 |
| rough_start | 114.00 | 4.044 | 0.041 | 1140 |
| downslope_start | 212.00 | 6.029 | -0.068 | 2120 |
| goal_flat_start | 348.00 | 8.002 | 0.054 | 3480 |

Context switches:

| label | t_rel_s | x | y |
|---|---:|---:|---:|
| flat | 0.98 | 0.000 | 0.000 |
| upslope | 50.74 | 2.000 | 0.160 |
| rough | 112.24 | 4.000 | 0.056 |
| downslope | 210.67 | 6.000 | -0.047 |
| goal_flat | 347.36 | 8.000 | 0.051 |
| downslope | 349.57 | 8.000 | 0.050 |
| goal_flat | 349.64 | 8.000 | 0.050 |
| downslope | 350.88 | 8.000 | 0.046 |
| goal_flat | 351.33 | 8.000 | 0.052 |
| downslope | 353.43 | 8.000 | 0.057 |
| goal_flat | 353.60 | 8.000 | 0.057 |
| downslope | 354.44 | 8.000 | 0.055 |
| goal_flat | 354.60 | 8.000 | 0.054 |
| downslope | 354.97 | 8.000 | 0.053 |
| goal_flat | 355.37 | 8.000 | 0.049 |
| downslope | 355.84 | 8.000 | 0.052 |
| goal_flat | 356.27 | 8.000 | 0.057 |
| downslope | 356.45 | 8.000 | 0.061 |
| goal_flat | 356.68 | 8.000 | 0.064 |
| downslope | 356.87 | 8.000 | 0.067 |
| goal_flat | 357.03 | 8.000 | 0.071 |
| downslope | 357.39 | 8.000 | 0.076 |
| goal_flat | 357.84 | 8.000 | 0.087 |
| downslope | 358.86 | 8.000 | 0.099 |
| goal_flat | 359.57 | 8.000 | 0.107 |
| downslope | 361.01 | 8.000 | 0.099 |
| goal_flat | 361.15 | 8.000 | 0.099 |
| downslope | 361.78 | 8.000 | 0.093 |
| goal_flat | 361.92 | 8.000 | 0.091 |
| downslope | 364.25 | 8.000 | 0.077 |
| goal_flat | 364.43 | 8.000 | 0.078 |
| downslope | 365.68 | 8.000 | 0.083 |
| goal_flat | 366.09 | 8.000 | 0.088 |
| downslope | 366.57 | 8.000 | 0.093 |
| goal_flat | 366.83 | 8.000 | 0.094 |
| downslope | 370.62 | 8.000 | 0.097 |
| goal_flat | 371.29 | 8.000 | 0.108 |
| downslope | 371.60 | 8.000 | 0.110 |
| goal_flat | 371.79 | 8.000 | 0.111 |
| downslope | 376.48 | 8.000 | 0.105 |
| goal_flat | 376.64 | 8.000 | 0.103 |
