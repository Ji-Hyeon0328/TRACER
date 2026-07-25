# TRACER Phase-L5G Route-Matched Body-Height Effect Summary v0

- source: `datasets/phase_l/l5g_fresh_body_height_validity_flat_evidence_v1.csv`
- repeats: [1, 2, 3, 4, 5, 6]
- primary comparison: candidate vs same-repeat flat_noop
- secondary comparison: candidate vs same-repeat baseline
- routing diagnostic: flat_noop vs baseline
- positive signed effect means improvement
- repeatable classification requires a non-crossing bootstrap CI and at least 5/6 pairs with the same sign

## routing_placebo: flat_noop vs baseline

- primary class: `unresolved`
- `progress_rate`: mean -0.00040399931031546987 | CI [-0.0018628253551359099, 0.0011156712536110763] | `unresolved` | +pairs 3/6 | sign p=1.0
- `mean_abs_y_window`: mean 0.016214243770797488 | CI [-0.011234323238564242, 0.0424092900816116] | `unresolved` | +pairs 3/6 | sign p=1.0
- `max_abs_y_window`: mean 0.035642478768000765 | CI [-0.013217266902719769, 0.07851975216568988] | `unresolved` | +pairs 4/6 | sign p=0.6875
- `end_abs_y`: mean 0.04531367117823527 | CI [-0.011499914157093857, 0.08459019615073454] | `unresolved` | +pairs 5/6 | sign p=0.21875
- `lateral_growth`: mean 0.04448108550349106 | CI [-0.014301420828343408, 0.08371954919992125] | `unresolved` | +pairs 5/6 | sign p=0.21875

## candidate_vs_noop: flat_body_low05 vs flat_noop

- primary class: `unresolved`
- `progress_rate`: mean 0.00033113221615029806 | CI [-0.0007270547304497358, 0.0013460754671062887] | `unresolved` | +pairs 3/6 | sign p=1.0
- `mean_abs_y_window`: mean 0.0025672946039907783 | CI [-0.013443269803340631, 0.01863965787893731] | `unresolved` | +pairs 3/6 | sign p=1.0
- `max_abs_y_window`: mean 0.005406833333333329 | CI [-0.017246683333333335, 0.03314033333333333] | `unresolved` | +pairs 2/6 | sign p=0.6875
- `end_abs_y`: mean 0.008469999999999998 | CI [-0.030076833333333337, 0.047988166666666665] | `unresolved` | +pairs 3/6 | sign p=1.0
- `lateral_growth`: mean 0.009662833333333336 | CI [-0.028615666666666668, 0.05047983333333334] | `unresolved` | +pairs 3/6 | sign p=1.0

## candidate_vs_baseline: flat_body_low05 vs baseline

- primary class: `stability_gain_without_progress_cost`
- `progress_rate`: mean -7.28670941651718e-05 | CI [-0.0012000416348190862, 0.001084199507895339] | `unresolved` | +pairs 3/6 | sign p=1.0
- `mean_abs_y_window`: mean 0.018781538374788267 | CI [-0.006614186703090551, 0.04703207325563773] | `unresolved` | +pairs 4/6 | sign p=0.6875
- `max_abs_y_window`: mean 0.0410493121013341 | CI [0.0024175387036449816, 0.08249467731556408] | `repeatable_positive` | +pairs 5/6 | sign p=0.21875
- `end_abs_y`: mean 0.05378367117823527 | CI [0.018091413994722597, 0.09422490434469415] | `repeatable_positive` | +pairs 5/6 | sign p=0.21875
- `lateral_growth`: mean 0.05414391883682439 | CI [0.01804848169832357, 0.09662266107503442] | `repeatable_positive` | +pairs 5/6 | sign p=0.21875

## candidate_vs_noop: flat_body_high05 vs flat_noop

- primary class: `unresolved`
- `progress_rate`: mean -0.0002987512554782461 | CI [-0.001443421519645888, 0.0007047058090100614] | `unresolved` | +pairs 3/6 | sign p=1.0
- `mean_abs_y_window`: mean 0.010171848501487715 | CI [-0.01003631892104048, 0.029539245282365373] | `unresolved` | +pairs 4/6 | sign p=0.6875
- `max_abs_y_window`: mean -0.0017388333333333388 | CI [-0.05152466666666668, 0.03468466666666666] | `unresolved` | +pairs 4/6 | sign p=0.6875
- `end_abs_y`: mean -0.016776833333333335 | CI [-0.06602916666666667, 0.019572333333333334] | `unresolved` | +pairs 3/6 | sign p=1.0
- `lateral_growth`: mean -0.0158275 | CI [-0.06470922916666666, 0.020936500000000007] | `unresolved` | +pairs 3/6 | sign p=1.0

## candidate_vs_baseline: flat_body_high05 vs baseline

- primary class: `unresolved`
- `progress_rate`: mean -0.000702750565793716 | CI [-0.0016913963561838556, 0.00022074004880762976] | `unresolved` | +pairs 1/6 | sign p=0.21875
- `mean_abs_y_window`: mean 0.0263860922722852 | CI [-0.0020668068031077743, 0.05188954807926174] | `unresolved` | +pairs 5/6 | sign p=0.21875
- `max_abs_y_window`: mean 0.03390364543466743 | CI [-0.00419919049869337, 0.06833249688052834] | `unresolved` | +pairs 4/6 | sign p=0.6875
- `end_abs_y`: mean 0.028536837844901932 | CI [-0.016114348196352577, 0.07212598524095563] | `unresolved` | +pairs 4/6 | sign p=0.6875
- `lateral_growth`: mean 0.02865358550349106 | CI [-0.01543816950370106, 0.07189241074100092] | `unresolved` | +pairs 4/6 | sign p=0.6875

## Safety

- No route-matched body-height result automatically promotes a candidate.
- Empirical/default remains active.
- Candidate labels require agreement between primary route-matched and secondary baseline sensitivity analyses.
