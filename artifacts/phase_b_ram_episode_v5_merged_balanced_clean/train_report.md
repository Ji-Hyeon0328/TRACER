# Phase-B RAM Episode Teacher-Student Training v2 Clean

- Rows: `173`
- Input dim: `15`
- Excluded leakage features: `7`
- Best epoch: `295`
- Train bool acc: `0.791`
- Train risk MSE: `0.0655`
- Val bool acc: `0.888`
- Val risk MSE: `0.0409`

Inputs are restricted to world/terrain/theta features. Risk prediction is sigmoid-bounded to `[0,1]` during both train and eval.

This is an episode-level seed RAM. It should later be upgraded to window-level teacher-student RAM using time-series proprioception.
