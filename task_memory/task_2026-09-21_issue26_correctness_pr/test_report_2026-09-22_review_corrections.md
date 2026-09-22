# Test Report 2026-09-22 — External review corrections (packages B, C, D, E)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | C4 executed: corrected FP8 native case passed on H800 (`exp-0922-202645-561899`). Limits updated. |
| 2026-09-22 | Created: verification of the review corrections applied on `fix/issue26-correctness-pr`. Package A is reported in the PR34 task directory (`test_report_2026-09-22_cache_eligibility_correction.md`). |

## Scope

Review document: `.local-draft/Frontier_PR34_PR35_Current_Code_and_PP_Extension_Review_2026-09-22.md`
(local, not committed). Packages adopted: A (PR34), B, C, D, E. Package F (the
W9 implementation) was excluded by the user and is not started. Finding-level
dispositions are in `review.md`, "External review 2026-09-22 — findings
disposition".

Environment: simulator interpreter `/data/ycfeng/envs/frontier-py310/bin/python`
(Python 3.10, no torch); torch interpreter
`/data/ycfeng/envs/openmopd-py312/bin/python` (Python 3.12, torch 2.8.0, vLLM
0.11.0). `PYTHONPATH` at the worktree root, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, run from
`/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr` after merging the PR34
correction (`0d025f8`).

## Package B — C35-01: decode credit at a dense layer for a mixed source

### Change

| File | Change |
| --- | --- |
| `frontier/scheduler/utils/collective_timing.py` | `advance_decode_layer(requests, total_layers)`: validate the whole selection, then credit one layer to each request. |
| `frontier/scheduler/utils/forward_collective.py`, `decode_collective.py` | Use the helper in place of the duplicated validate-then-increment loop (behavior-preserving). |
| `frontier/scheduler/utils/dense_metrics.py` | `complete_dense_layer(phase="prefill")` credits the source batch's decoding members (`collect_active_requests` filtered by `is_prefill_complete`) before delegating to the prefill handler. A dense layer completes per source, outside the shared forward completion that credits routed layers, and the prefill handler credits nothing. |

No double credit: the shared forward path credits at `handle_forward_sync_collective` and enters the per-phase helpers with the credit already done; the dense path credits once per source (prefill-mode source here, decode-mode source in `handle_decode_sync_collective`). A PREFILL-role batch holds no request with `is_prefill_complete` set during its layers (`Request` sets it at prefill completion), so the new call is a no-op for the disaggregated roles.

### Verification

| # | Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- | --- |
| B1 | Mixed-forward unit tests, including the three new ones | `python -m pytest tests/unit/test_monolithic_mixed_forward_sync.py -q -p no:cacheprovider` | 23 existing + 3 new pass | 26 passed in 1.63 s | PASS |
| B2 | Dense layer executed for a mixed source (`test_a_dense_layer_credits_only_the_requests_that_are_decoding[mixed]`) | in B1 | decoding member `completed_layer_count == 1`, prefilling member `0`, decode-lane peers `1` | as expected | PASS |
| B3 | Pure-phase control (`[prefill]` parametrization) | in B1 | pure-prefill source credits nothing; decode lane `1` each | as expected | PASS |
| B4 | `MoE -> dense -> MoE` (`test_a_mixed_source_is_credited_once_per_layer_across_routed_and_dense`) | in B1 | decoding member counts `[1, 2, 3]`; prefiller `0`; peers `layer + 1` | `[1, 2, 3]` | PASS |
| B5 | Real loop, hybrid layers (`moe_layers_enum="0,2,3"`, `dense_mlp_hidden_dim=64`, chunked prefill, `attn_dp=2`, `moe_ep=2`, trained predictor on constant synthetic rows) | `python -m pytest tests/integration/test_monolithic_mixed_forward_runtime.py -q -p no:cacheprovider` | both tests pass; a mixed batch actually crosses the dense layer; every decode token credited exactly `num_layers` | 2 passed in 7.17 s; child evidence: `mixed_dense_completions: 4`, `decode_tokens_credited: 10`, `layer_credit_peaks: {"4": 10}`, `mixed_phase_cohorts: 4`, 4 of 4 requests complete, waiting rooms drained | PASS |
| B6 | Negative control: same hybrid case on the pre-fix `dense_metrics.py` (file restored from `HEAD`, then the fix copied back) | scratch driver calling `run_case(..., moe_layers_enum="0,2,3")` | the credit assertion fails | `AssertionError`; evidence `layer_credit_peaks: {"3": 4, "4": 6}` — the four decode tokens carried through the dense layer inside a mixed batch peaked one layer short | PASS (detects the defect) |
| B7 | Neighbors of the changed helpers | `python -m pytest tests/unit/test_monolithic_mixed_forward_sync.py tests/unit/test_collective_timing.py tests/unit/test_execution_time_metrics_ownership.py tests/unit/test_moe_routing_conservation.py tests/unit/test_stage_reporting_contract.py -q` | all pass | 72 passed in 2.00 s (before the three new tests were added; 26 of the 23 above included) | PASS |
| B8 | Whole unit suite versus the Step 8 baseline | `python -m pytest tests/unit -q -p no:cacheprovider -rfE --continue-on-collection-errors` | `FAILED` set identical to `base_failed.txt`; count deltas explained | 84 failed / 3789 passed / 50 skipped / 10 errors in 109.9 s (Step 8: 84 / 3782 / 49 / 11). `diff <(sort base) <(sort now)` empty. +7 passes = 3 new mixed-forward tests + 4 gate tests merged from PR34; +1 skip and −1 error = the optional-torch module (package C) | PASS |

Observation versus inference: B5 and B6 are observed runs; the claim that the
disaggregated roles are untouched is by construction (B8 shows the unchanged
failure set, but the suite has no PREFILL-role hybrid-layer case).

## Package C — C35-03 / C35-04: FP8 test wiring and optional-torch collection

### Change

| File | Change |
| --- | --- |
| `tests/integration/test_moe_fused_expert_numerical_parity.py` | FP8 case passes `block_shape=block_shape` to `_run_fused_moe_iteration`, matching `profile_fused_moe_kernel`. |
| `tests/unit/test_moe_fused_expert_arithmetic.py` | `torch = pytest.importorskip("torch", ...)` before importing the profiler module; the `_invoke_kernel` stub records `block_shape`; two new tests pin that both GEMM invocations receive `[128, 64]` under FP8 and `None` when omitted; `_run` accepts overrides. |

### Verification

| # | Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- | --- |
| C1 | Arithmetic boundary tests under torch | `openmopd-py312 python -m pytest tests/unit/test_moe_fused_expert_arithmetic.py -q -p no:cacheprovider` | 7 existing + 2 new pass | 9 passed in 6.31 s | PASS |
| C2 | Minimal environment collection | `frontier-py310 python -m pytest tests/unit/test_moe_fused_expert_arithmetic.py -q -p no:cacheprovider` | module skips instead of erroring | 1 skipped in 0.06 s | PASS |
| C3 | Unit-suite collection errors | B8 | 10 (the base's count), not 11 | 10 errors | PASS |
| C4 | Corrected FP8 native check on the approved worker | StepMind `RJobBackend`, `codesign` / H800, image `vllm/vllm-openai:v0.10.2`, `pytest -q -rA tests/integration/test_moe_fused_expert_numerical_parity.py` | 8 passed, exit 0; FP8 case runs with `block_shape=[128, 64]` | `exp-0922-202645-561899`, `gpu-h800-0095`, **8 passed in 14.27 s**, exit 0; details in the W6 report §8 | PASS |

The native run recorded in the W6 report (`exp-0922-145047-660565`) stands as
evidence for the arithmetic repair: seven reference comparisons at
`rtol=0, atol=0` plus one FP8 structural check that, as run, exercised the
per-tensor scale path. FP8 numerical equivalence is not established.

## Packages D and E — records only

No production code changed. Files: W6 report §5 (scope table) and §8 (seven
plus one, `block_shape` omission), `docs/profiling/README.md` (backend
envelopes differ), `summary.md`, `review.md` (D2 clause superseded; disposition
table), `progress.md` (status table, step rows, package table),
`requirements.md` (this request), `plan.md` (§17 → §18 renumber; §18.1, §18.3,
§18.4, §18.5, §18.6, §18.8 corrected; §18.11 added), `design.md` W9 (steady
state, gap table, key verdicts with the reproduced counterexample, conditional
witness, planned edits, expectations). PR34 and PR35 bodies updated through the
API; PR35 stays draft.

## Limits

- The corrected FP8 native check passed on a GPU; it remains a structural check (shape and finiteness), so FP8 numerical equivalence is still not established.
- The real-loop hybrid case uses constant synthetic profiling rows and a
  four-layer synthetic MoE model, as the W3 runtime test does; it proves the
  accounting, not latency fidelity.
- The disaggregated-role control is by construction plus the unchanged unit
  failure set; no PREFILL-role hybrid-layer run was added.
- Package E changes plans and designs only; none of its statements is a test
  result for W9.
