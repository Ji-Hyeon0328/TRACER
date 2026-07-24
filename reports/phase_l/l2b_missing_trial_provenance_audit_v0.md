# Phase-L2B Missing Trial Provenance Audit v0

## Purpose

Phase-L2 reconstructed window-level state/action evidence from Phase-K
candidate trials.

Two candidate trials could not be reconstructed because their D5 policy-input
state streams were not recorded.

This audit verifies whether those trials could be recovered from alternate
paths or timestamp-matched runtime directories.

---

## Reconstruction Coverage

```text
K6 candidate trials:       34
successfully reconstructed: 32
unavailable:                2
The unavailable trials are:

K1  j19_profile  idx=8
K4C flat_noop    idx=4
K1 j19_profile idx=8

K6 references:

run_dir:
logs/phase_k/k1_long_baseline_vs_j19_profile_20260722_195542/8_j19_profile

manifest:
reports/phase_d4_context_meta_repeat_20260722_210801_manifest.tsv

Findings:

K6 d5_log_dir metadata is empty.
run_dir/d5_log_dir.txt exists but is empty.
Timestamp-matched directory exists:
logs/phase_d5_shadow_20260722_210801_trial_1
policy_inputs_v0.csv is absent.
No alternate policy_inputs_v0.csv exists inside the Phase-K run directory.

Conclusion:

state/context/beta/RAM input stream unavailable

The rollout remains valid as a K6 rollout-level outcome, but it cannot be used
as a Phase-L2 state/action window sample without inventing missing features.

K4C flat_noop idx=4

K6 references:

run_dir:
logs/phase_k/k4c_flat_micro_candidates_fixed_20260723_164401/4_flat_noop

manifest:
reports/phase_d4_context_meta_repeat_20260723_180107_manifest.tsv

Findings:

K6 d5_log_dir metadata is empty.
run_dir/d5_log_dir.txt exists but is empty.
Timestamp-matched directory exists:
logs/phase_d5_shadow_20260723_180107_trial_1
policy_inputs_v0.csv is absent.
No alternate policy_inputs_v0.csv exists inside the Phase-K run directory.

Conclusion:

state/context/beta/RAM input stream unavailable

The rollout remains available as rollout-level K6 evidence but is excluded from
the Phase-L2 window-level evidence dataset.

Missing-Data Policy

Phase-L does not infer or reconstruct missing state/context/objective features
when the corresponding runtime log was not recorded.

Therefore:

32 / 34 candidate trials

are retained in the Phase-L2 window evidence dataset.

The two missing trials remain visible through explicit skip reporting.

No synthetic values are inserted.

Impact

The missing-data rate among candidate trials is:

2 / 34

The remaining reconstructed candidate evidence includes:

14 control-equivalent no-op windows
 6 near-miss slow03 windows
12 non-noop rollout-associated negative windows

This is sufficient for the next Phase-L step: matched empirical/action-effect
analysis.

The two unavailable trials should not block Phase-L3.

Decision

Phase-L2 reconstruction is considered complete.

The next step is Phase-L3:

matched empirical/action-effect estimation

The objective is to compare candidate exposure windows against appropriate
empirical reference windows rather than treating K6 rollout decisions as direct
causal action labels.
