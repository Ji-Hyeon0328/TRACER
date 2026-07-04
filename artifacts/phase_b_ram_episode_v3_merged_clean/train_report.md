# Phase-B RAM Episode Teacher-Student Training v2 Clean

- Rows: `105`
- Input dim: `15`
- Excluded leakage features: `7`
- Best epoch: `473`
- Train bool acc: `0.848`
- Train risk MSE: `0.0495`
- Val bool acc: `0.933`
- Val risk MSE: `0.0369`

Inputs are restricted to world/terrain/theta features. Risk prediction is sigmoid-bounded to `[0,1]` during both train and eval.

This is an episode-level seed RAM. It should later be upgraded to window-level teacher-student RAM using time-series proprioception.
