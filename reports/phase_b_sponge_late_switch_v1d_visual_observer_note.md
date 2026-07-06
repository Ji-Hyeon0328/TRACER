# Phase-B Sponge Late-Switch v1d Visual Observer Note

## Context

- Policy: `tracer_sponge_late_switch_v1d`
- World: `tracer_sponge_firm_flat`
- Controller: frozen low-level controller
- Evaluation report: `reports/phase_b_reach_terminated_eval_sponge_late_switch_v1d.md`

## Quantitative result

The v1d late-switch profiles improved post-reach stability metrics:

- mean final distance: approximately `0.863`
- mean post-reach drift: approximately `0.520`
- mean R_s: approximately `+1.025`

However, reach success remained low:

- reach rate: approximately `0.178`

## Visual observation

During the run, the robot often showed visually degenerate locomotion:

- right-front / left-hind pair appeared folded or inactive
- left-front / right-hind pair appeared to provide most of the motion
- motion looked like crawling/dragging rather than normal trot/crawl locomotion

## Interpretation

This should not be treated as a valid deployable gait, even if final-distance or stability metrics look favorable.

The result is evidence that theta-lite high-level tuning with the frozen low-level controller can create metric-favorable but visually invalid locomotion on sponge terrain.

## Decision

Stop manual sponge theta tuning beyond v1d.

Next steps should focus on:

1. invalid gait labeling/detection
2. RAM/GMS recovery-needed or invalid-locomotion labeling
3. excluding visually invalid theta profiles from future training targets
4. later implementing a true low-level recovery/hold/crawl primitive if needed
