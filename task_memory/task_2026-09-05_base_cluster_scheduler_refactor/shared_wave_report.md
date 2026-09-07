## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-06 | Added pure shared EP wave phase prediction and barrier timing helpers. |

## Motivation

The PREFILL and DECODE shared EP handlers contained identical lane prediction and
dispatch/combine barrier arithmetic. This sub-step provides immutable utility
results while retaining scheduler-owned state and callback behavior in Base.

## Changes

- Added `EPWavePhaseTimes` and `EPWaveTiming` immutable NamedTuple results.
- Added `predict_ep_wave_phase_times`, preserving participant order, one predictor
  call per lane, five-phase callback validation, workload trace order, and
  sequential phase additions.
- Added `calculate_ep_wave_timing`, preserving max-based barriers, seconds
  conversion, and dispatch/combine/wave trace callback order.

## Validation

Command: `python -m pytest tests/unit/test_expert_parallel.py -q -p no:cacheprovider`

Environment: repository worktree; Python interpreter from active environment.

Result: PASS, 6 passed in 2.66s. Existing tests exercise expert-parallel helper
contracts; no runtime handler migration was performed in this concurrent sub-step.

Expected timing relationships implemented exactly:

- dispatch end = start + (max pre-dispatch + max dispatch) ms;
- combine end = dispatch end + (max routed + max combine) ms;
- wave end = combine end + max post-combine ms.

## Concerns

Base handler call-site replacement remains for the parent task because Base is
under concurrent ownership. The new functions require callbacks for lane builder,
phase getter, workload trace, barrier trace, and wave trace.
