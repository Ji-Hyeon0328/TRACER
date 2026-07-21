# TRACER Phase-I4 Damped RAM Shadow v0

This applies an offline damped/safe postprocess to Phase-I2 learned RAM shadow outputs.

- input csv: `logs/phase_i/i2_ram_shadow_active_20260721_160236/ram_shadow_i2_v0.csv`
- output csv: `datasets/phase_i/i4_damped_ram_shadow_20260721_160236_v0.csv`
- postprocess json: `models/phase_i/i4_damped_ram_postprocess_v0.json`
- rows: `3036`
- gain: `0.35`
- legacy_blend: `0.35`
- floor/ceil: `0.02` / `0.98`

## Overall summary

| group | rows | learned L1 | damped L1 | safe L1 | learned sat slip/rough/sigma | damped sat slip/rough/sigma | safe sat slip/rough/sigma |
|---|---:|---:|---:|---:|---|---|---|
| ALL | 3036 | 0.963292 | 0.913200 | 0.319620 | 0.925560/0.880435/0.582016 | 0.000000/0.000000/0.000000 | 0.000000/0.000000/0.000000 |
| KNOWN_CONTEXT_ONLY | 2307 | 0.963292 | 0.913200 | 0.319620 | 0.902037/0.842653/0.449935 | 0.000000/0.000000/0.000000 | 0.000000/0.000000/0.000000 |

## By context

| context | rows | safe slip | safe rough | safe sigma | safe L1 | safe sat slip | safe sat rough | safe sat sigma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| downslope | 513 | 0.384541 | 0.324420 | 0.434982 | 0.443943 | 0.000000 | 0.000000 | 0.000000 |
| flat | 457 | 0.114686 | 0.177447 | 0.213671 | 0.357147 | 0.000000 | 0.000000 | 0.000000 |
| goal_flat | 500 | 0.111881 | 0.177666 | 0.162758 | 0.352304 | 0.000000 | 0.000000 | 0.000000 |
| rough | 422 | 0.174249 | 0.472163 | 0.424267 | 0.228443 | 0.000000 | 0.000000 | 0.000000 |
| unknown | 729 | 0.295275 | 0.648342 | 0.337082 | NA | 0.000000 | 0.000000 | 0.000000 |
| upslope | 415 | 0.168346 | 0.234420 | 0.277625 | 0.180391 | 0.000000 | 0.000000 | 0.000000 |

## Safe interpretation

- `learned` is the original I2 RAM output.
- `damped` keeps the learned direction but pulls it toward the I0 prior.
- `safe` additionally blends with legacy RAM when available.
- If `safe` sharply reduces saturation and L1 from legacy, it can be used as the next shadow runtime output. It should still not replace RAM in active control yet.
