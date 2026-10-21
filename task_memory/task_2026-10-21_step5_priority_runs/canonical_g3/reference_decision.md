# Selected independent native functional reference

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded one source-backed runtime/reference choice before dependent implementation. |
| 2026-10-21 | Parent selected existing-D full-local-prefill; implemented continuation under existing authority. |

## Parent implementation disposition

The parent selected existing-D full-local-prefill with no transfer metadata, unchanged native image/model/connector/topology and disabled prefix caching. Separate reference request state follows observed idle. This records a parent implementation disposition under D20/D21 authority, not a new user approval, a reference replacement, or D-S5-REF activation. The conditional native-repair/image/connector changes do not apply. Direct native D requests, full resident identity and exact logical state/output comparisons are now implemented; native evidence has not yet been collected. Numerical differences have no invented tolerance or batch-drift waiver and remain UNVERIFIED.

## Historical proposal (superseded by that disposition)

**May an independent, complete local-prefill execution on the existing D engine be the native functional reference, without any kv_transfer_params or external KV, rather than starting another native instance or introducing another implementation?** Recommended: yes, subject to observed full parameter/buffer/scales identity and explicit reference-state/output checks. This is a proposal, not approval, and is not implemented in this slice.

The reference would use the same actual P0 TP8/DP2/EP16/PP1 D model, the original canonical6144 prompt IDs, full local6144 computation plus1024 output, prefix caching off and separate request state after idle. It is an independent execution path that does not consume P's transferred state, not the intra-run registration census. It would not supply clean performance timing or permit relaxed content/lifetime checks.

## Evidence

Frozen source root: `/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`.

- `distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:34–106`: remote token admission requires explicit do_remote_prefill/do_remote_decode metadata; paramsNone returns(0,False). Therefore the existing D API can schedule full local prefill without a new kernel or a synthetic transfer.
- `model_executor/model_loader/dummy_loader.py:35–66`: dummy initialization excludes nonpersistent buffers on the normal path and performs quantization processing. `model_executor/model_loader/weight_utils.py:1525–1577`: default dummy seed1234 is per tensor; nonfloating CUDA tensors are not initialized there. Equal seed/shape alone is not complete model/buffer/scales identity.
- `tests/integration/step5_environment/role_state_observation.py:240–281`: existing resident inventory and chunk-layout primitives are reusable, but existing parameter_samples contain only one element per view. A census or sample cannot replace complete content identity. Do not allocate/copy another entire model; use those existing bounded chunk primitives if the reference is approved.
- `tests/integration/step5_pd/r2_observation.py`: captures complete transferred rows, tail accumulators and journal bytes; no existing independent reference execution/comparison. `r2_verdict.py`'s registration census is only cross-worker consistency.

## Alternatives and cost

| Choice | Consequence / cost |
| --- | --- |
| Recommended existing-D full-local-prefill cohort | No additional GPUs; additional diagnostic execution and complete resident identity scan. Native EP/FP8/graph are retained. Must record independent requests, idle/reset and compare actual state/output; numerical comparison policy is not silently invented. |
| Separate native monolithic instance | New lifecycle, residency/identity verification and at least16 additional H800s to preserve TP8/DP2/EP16 concurrently, or a restart of the existing D role; neither is currently selected. Does not itself provide algorithmic independence. |
| Different functional/known-state implementation | Requires an explicitly chosen compatible Step sparse/FP8 implementation and its comparison contract; cannot be substituted by an arbitrary toy model, CPU transfer or equal seeds. |

No numeric tolerance, routing, dummy initialization, FP8 exclusion, clean timing definition or native retention policy is changed by this proposal. Bit-exact P→D wire-row integrity remains separate from any functional comparison across differing compute batches.

## Prior slice limitations and continuation

The delivered capture caller/consumer and restored P1 checks are usable code, not complete G3. A diagnostic proxy-native-D-client deadline now attempts pending-READ cancellation without native READ delays or fences; the consumer additionally requires a same-host observed READ-submit < native abort < READ completion interval, native pending status, all eight completions/readiness, source notification and page/pool conservation. It rejects first-token cancellation, late/early deadlines, observer fences and incomplete cleanup as pending-READ proof. CPU tests cover the caller and these negative cases, but actual native overlap and cleanup remain UNVERIFIED until parent execution.

The previous slice lacked the reference-state/reset/reuse/pressure consumer. The continuation implements that consumer and an explicit128-request6144/1024 native PD pressure wave, retaining frozen7168/8192/128 and unmodified capacity. Actual failed native allocation and observed physical ownership/free/reset records are required; offered concurrency alone cannot PASS. Native reset (`sparse_summary_cache.py:347–355`) clears mean/last-slot/journal validity but deliberately does not zero accumulators; native fresh-region use determines their live status. Existing G1 `compare_attention_state.py:27–43,65–87,204–207` supplies error reporting, not an unsupported compiled-kernel numerical bound. Exact wire integrity remains mandatory; a differing independent execution remains UNVERIFIED until an evidenced comparison rule exists.
