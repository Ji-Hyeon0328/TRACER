# TRACER Phase-L3 Matched Action Effect Summary v0

- source: `datasets/phase_l/l2_window_action_evidence_from_k6_v0.csv`
- context: `flat`
- baseline flat windows: 16
- no-op control windows: 14
- intervention windows: 18
- detailed effect rows: 32
- match k: 5

## Interpretation rule

- Positive signed effect means improvement.
- Progress-rate effect: candidate minus matched control.
- Lateral/stability effects: matched control minus candidate.
- No-op windows are matched against baseline-only windows.
- Intervention windows are matched against baseline + no-op empirical-equivalent controls.
- The 95th percentile absolute no-op effect defines the placebo variability band.
- This is matched observational evidence, not a causal-effect claim.

## Placebo variability thresholds

- `progress_rate`: 0.0015615557723042967
- `mean_abs_y_window`: 0.06223645781083801
- `max_abs_y_window`: 0.10319996982078503
- `end_abs_y`: 0.11703032843916919
- `lateral_growth`: 0.11617067503654269

## Group summary

### flat_clear03

- n: 6
- evidence role: `rollout_associated_negative_unpaired`
- `progress_rate` effect mean: 0.00023987357790609837 | bootstrap 95% CI [8.187950494581199e-05, 0.0004297388367124448] | `within_placebo_band`
- `mean_abs_y_window` effect mean: -0.021359720064090697 | bootstrap 95% CI [-0.03592945428622049, 4.0960215732381795e-05] | `within_placebo_band`
- `max_abs_y_window` effect mean: -0.03975616464824506 | bootstrap 95% CI [-0.07214853202158923, 0.0012100447392472656] | `within_placebo_band`
- `end_abs_y` effect mean: -0.043491825258117144 | bootstrap 95% CI [-0.09512512025467242, 0.0112367410609504] | `within_placebo_band`
- `lateral_growth` effect mean: -0.043595338277345505 | bootstrap 95% CI [-0.0952120809921457, 0.012080094385381922] | `within_placebo_band`

### flat_noop

- n: 5
- evidence role: `control_equivalent_noop`
- `progress_rate` effect mean: -4.266740455998036e-05 | bootstrap 95% CI [-0.0006338628405203148, 0.0005485280314003541] | `within_placebo_band`
- `mean_abs_y_window` effect mean: 0.0033405663563125256 | bootstrap 95% CI [-0.026480257771866248, 0.02803253835285209] | `within_placebo_band`
- `max_abs_y_window` effect mean: 0.023621333195335895 | bootstrap 95% CI [-0.01685744708166491, 0.05731653319533589] | `within_placebo_band`
- `end_abs_y` effect mean: 0.035643028781495056 | bootstrap 95% CI [0.0030093962721969526, 0.06618446129079317] | `within_placebo_band`
- `lateral_growth` effect mean: 0.03765242558703794 | bootstrap 95% CI [0.004673278381284858, 0.06842617279279103] | `within_placebo_band`

### flat_slow03

- n: 6
- evidence role: `near_miss_exposure`
- `progress_rate` effect mean: -0.0006179743461423096 | bootstrap 95% CI [-0.0013159018164252956, -3.976532897112528e-05] | `within_placebo_band`
- `mean_abs_y_window` effect mean: 0.010468015434989174 | bootstrap 95% CI [-0.004994538925332766, 0.022709882745081508] | `within_placebo_band`
- `max_abs_y_window` effect mean: 0.025044323780424285 | bootstrap 95% CI [0.008430127854401802, 0.0414555743638773] | `within_placebo_band`
- `end_abs_y` effect mean: 0.02963935313686664 | bootstrap 95% CI [-0.002806550686689814, 0.06380802918547492] | `within_placebo_band`
- `lateral_growth` effect mean: 0.029632034890520173 | bootstrap 95% CI [-0.0039413659316322345, 0.06366408112594474] | `within_placebo_band`

### flat_slow03_clear03

- n: 6
- evidence role: `rollout_associated_negative_unpaired`
- `progress_rate` effect mean: -0.00084225694795149 | bootstrap 95% CI [-0.001496676301077124, -0.0002503707786899433] | `within_placebo_band`
- `mean_abs_y_window` effect mean: 0.009990193948456834 | bootstrap 95% CI [-0.012208756888697428, 0.029623986289651773] | `within_placebo_band`
- `max_abs_y_window` effect mean: 0.025633677497772085 | bootstrap 95% CI [-0.01671377624879503, 0.06772031012055862] | `within_placebo_band`
- `end_abs_y` effect mean: 0.024489317431803367 | bootstrap 95% CI [-0.028609407243628662, 0.07397303129428828] | `within_placebo_band`
- `lateral_growth` effect mean: 0.024184319424295764 | bootstrap 95% CI [-0.026600649374954283, 0.07119517127811964] | `within_placebo_band`

### j19_profile

- n: 9
- evidence role: `control_equivalent_noop`
- `progress_rate` effect mean: 3.9259788969946296e-08 | bootstrap 95% CI [-0.0007550854260153727, 0.0007275357700769592] | `within_placebo_band`
- `mean_abs_y_window` effect mean: -0.009870857435377143 | bootstrap 95% CI [-0.03155462738519485, 0.009402768188704934] | `within_placebo_band`
- `max_abs_y_window` effect mean: -0.011132133978820279 | bootstrap 95% CI [-0.05293838735683754, 0.02565577793609131] | `within_placebo_band`
- `end_abs_y` effect mean: -0.002792554447937614 | bootstrap 95% CI [-0.05551396395042618, 0.04547413128788227] | `within_placebo_band`
- `lateral_growth` effect mean: -0.0003597990461361421 | bootstrap 95% CI [-0.05307487202683962, 0.04872657359277665] | `within_placebo_band`

## Safety / deployment

- No candidate is promoted to active deployment in L3.
- Empirical/default control remains the runtime policy.
- L3 outputs are intended to guide later selector supervision and broader candidate collection.
