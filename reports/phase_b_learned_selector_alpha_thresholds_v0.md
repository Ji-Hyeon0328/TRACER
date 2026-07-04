# Phase-B Learned Selector Alpha Threshold Report

This report estimates the α needed for the best learned-score challenger to beat the raw PPO action.

| world | raw action | raw p | raw z | best challenger | challenger p | challenger z | required α | current α | changed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| earth | trot_mid | 0.921 | 0.373 | trot_solid_fast | 0.004 | 1.457 | 4.935 | 0.150 | False |
| earth | trot_mid | 0.921 | 0.373 | trot_solid_fast | 0.004 | 1.457 | 4.935 | 0.150 | False |
| earth | trot_mid | 0.921 | 0.373 | trot_solid_fast | 0.004 | 1.457 | 4.935 | 0.150 | False |
| stairs_single | trot_solid_fast | 0.639 | -0.599 | trot_mid | 0.239 | 1.485 | 0.473 | 0.150 | False |
| stairs_single | trot_solid_fast | 0.639 | -0.599 | trot_mid | 0.239 | 1.485 | 0.473 | 0.150 | False |
| stairs_single | trot_solid_fast | 0.639 | -0.599 | trot_mid | 0.239 | 1.485 | 0.473 | 0.150 | False |
| tracer_sponge_firm_flat | trot_soft_mid_clear | 0.280 | 0.805 | trot_cautious | 0.131 | 1.123 | 2.384 | 0.150 | False |
| tracer_sponge_firm_flat | trot_soft_mid_clear | 0.280 | 0.805 | trot_cautious | 0.131 | 1.123 | 2.384 | 0.150 | False |
| tracer_sponge_firm_flat | trot_soft_mid_clear | 0.280 | 0.805 | trot_cautious | 0.131 | 1.123 | 2.384 | 0.150 | False |

## Interpretation

- If required α is around 0.3–0.6, a stronger but still moderate intervention can be tested.
- If required α is above 1.0, a pure additive score is probably not the right mechanism.
- For high-risk sponge behavior, a RAM risk gate/top-k gate may be safer than large α.
