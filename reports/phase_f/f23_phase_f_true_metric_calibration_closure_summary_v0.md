# TRACER Phase-F True-Metric Calibration Closure Summary v0

## What Phase-F accomplished

Phase-F successfully added a true-metric diagnostic pipeline on top of the TRACER high-level runtime stack.

The pipeline now collects rollout-level physical proxies from the ROS1/Gazebo side, including:

- forward motion proxy: base forward velocity
- lateral/stability proxy: lateral velocity, lateral drift, IMU angular velocity norm
- energy proxy: joint absolute power, approximated by `sum(abs(effort_i * velocity_i))`
- contact/load proxy: summed foot contact force

These metrics were aggregated into true-metric beta target seeds for motion / stability / energy, then attached back to D7 runtime logs for calibration diagnostics.

## Key Phase-F outputs

### F16 balanced lateral sweep

The final balanced sweep contains:

- 7 lateral/reset conditions
- 3 rollouts per condition
- 21 rollout summaries
- complete D7 timestep attachment with no missing sources

Conditions:

- clean
- m015
- m030
- m060
- p015
- p030
- p060

### F17 outlier / variance audit

F17 showed that some grouped true-metric beta targets are sensitive to individual rollouts.

The strongest outlier was:

- `p060_r2`
- very low forward velocity
- unusually low contact force
- large leave-one-out beta shift

This justified building a robust target instead of relying only on mean aggregation.

### F18 robust grouped beta targets

F18 replaced mean metric aggregation with group median metric aggregation.

The robust target helped reduce sensitivity to abnormal rollouts. The largest mean-vs-robust beta shifts occurred in:

- m015
- m060
- m030

p060 remained stability-heavy even under robust median aggregation, meaning p060 is not only a mean-outlier artifact; it represents a difficult condition for the current runtime feature set.

### F19/F20 robust beta calibration

F19 successfully attached robust median true-metric beta targets to D7 timestep logs.

F20 trained a runtime-safe robust beta calibrator. It improved some conditions but remained diagnostic, not deployable.

Notable pattern:

- p015 is relatively well predicted.
- p060, p030, m060, and m030 remain difficult.
- p060 remains the hardest held-out condition.

### F21 mean vs robust comparison

Robust targets improved 5 out of 7 held-out conditions compared with mean targets.

Strongest improvement:

- m015

Main degradation:

- m030

This suggests robust targets are useful, but target robustness alone does not solve the problem.

### F22/F22b feature basis ablation

F22/F22b tested whether the remaining error comes from weak feature basis or missing runtime information.

Result:

- small feature suites such as RAM/ref, D7 beta only, score proxy, and lateral runtime all had similar mean LOTO error.
- all-runtime-safe features did not improve.
- adding nonlinear lateral interaction basis did not improve and slightly worsened the result.

Therefore, the remaining bottleneck is likely not the ridge model or simple nonlinear basis. It is the lack of sufficiently informative runtime context/RAM features.

## Final interpretation

Phase-F should be treated as a successful diagnostic/calibration phase, not as a completed deployable Objective Selector.

Safe claims:

1. The true-metric logging pipeline works.
2. D7 runtime beta predictions can be compared against physical rollout-derived beta targets.
3. Mean target construction is sensitive to outlier rollouts.
4. Robust median target construction reduces some target noise.
5. Current runtime-safe features are insufficient to infer difficult true-metric beta targets, especially p060/p030/m060.
6. The next research step should be RAM/context feature strengthening, not simply training a bigger beta regressor.

## What not to claim

Do not claim:

- completed IRL Objective Selector
- deployable beta calibrator
- learned RAM
- perception-based terrain context
- RL high-level meta-gait planner

Current system remains:

- oracle/segment context
- proxy/static RAM features
- supervised D6 selector
- diagnostic true-metric beta calibration

## Next phase

Proceed to Phase-G:

**Phase-G: RAM/context feature strengthening for true-metric objective prediction**

Initial goal:

Build a richer runtime feature table that can explain why conditions such as p060 and p030 require very different motion/stability/energy beta targets.

Candidate additions:

- rollout-level slip proxy from odom/reference mismatch
- commanded-vs-real velocity tracking error
- yaw tracking error
- lateral recovery trend
- contact asymmetry
- contact loss ratio
- energy-per-distance
- terrain segment transition history
- RAM-like temporal mismatch summary over short windows

The next model should not yet be deployed. It should first be evaluated with the same leave-one-tag-out diagnostic protocol used in Phase-F.
