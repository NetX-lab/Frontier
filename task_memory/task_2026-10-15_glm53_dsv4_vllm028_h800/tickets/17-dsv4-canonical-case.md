# 17: DeepSeek-V4-Flash case bindings for the four priority tasks

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D-DSV4-H800-MAINLINE-CORRECTION: added the current route block; stated which checklist applies to P0, mapped the two open P0 items to H4a, and deferred the H200 rows. |
| 2026-10-23 | W128-V4-P0-CASE-BINDING: closed explicit P0 workload/client subitems from unchanged native06 R4; retained all-case checkboxes, memory-budget and10%/30% gates open. |
| 2026-10-22 | D-DSV4-ONLY-GLM-PAUSED: DSV4 directly reuses accepted shared68f891b workload; removed unfinished GLM16 completion dependency and retained all four V4 cases/mean gates. |
| 2026-10-21 | Bound V4 scope to the four priority cases, explicit32/64 layouts and promotion/resource rules; retained hardware-specific qualification limits. |
| 2026-10-21 | Accepted and published H200 storage admission6d88700; both remaining V4 Q10 scopes are approved, with grouped NIXL implementation started. |
| 2026-10-21 | Corrected stale G0-B status to its scoped acceptance; linked active H200 admission and remaining graph/measured/grouped-NIXL work. |
| 2026-10-21 | Recorded accepted shared-artifact checks and V4 G0-A, submitted full-model G0-B, retained native fp8 KV input and corrected successful NIXL context endpoints. |
| 2026-10-21 | Retargeted the V4 case to the shared fixed 128-request, 6144/1024-token, QPS-1.0 artifact; removed stale SharedGPT-fetch/tokenizer blockers and duplicated-manifest requirement. Native length, graph, memory and runtime parity remain open. |
| 2026-10-21 | Created from the canonical H200 calibration contract (`calibration.md`, decisions Q1–Q12). Replaces the DeepSeek-V4-Flash half of ticket 10's 36-cell matrix. |
| 2026-10-21 | Second round: the shared trace, retiming rule and 256-request selection are inherited from ticket 16; only the V4 tokenizer re-count is model-specific. THPT is total prompt+output. |

**What to build:** V4 bindings for P0-H800-32, P1-H800-64, P2-H200-32 and
P2-H200-64, using the exact accepted shared68f891b workload directly,with own
hardware,32/64 layout,resource class,purpose,source/runtime and native evidence.
Keep actual V4 configuration/C26 evidence, hardware/topology, resource class,
purpose and generation explicit. The V4 promotion receipts are independent
of GLM; GLM results cannot unlock V4 formal64-GPU calibration.

**Blocked by:** this ticket's DSV4 binding of the already accepted shared68f891b
artifact,matching01 runtime qualification and18's DSV4 canonical runtime/graph/
quantization/memory evidence. D-DSV4-ONLY-GLM-PAUSED sets all unfinished16/GLM
simulation/tests/calibration/packages PAUSED_PENDING_USER_APPROVAL until new explicit user approval;they are
not predecessors of current DSV4 acceptance. No new GLM work is needed to reuse
the shared artifact. Historical SharedGPT/tokenizer work is not a blocker.

**Status:** shared workload, scoped H200 G0-A/G0-B and storage admission remain
accepted only for their recorded scope. Grouped NIXL CPU structural/tracing is
published at2a84c58/2ffd2cb, not native/full-serving acceptance. Graph/measured
consumption and decode profiling now prioritize the H80032 P0 case. No formal
four-metric or30% promotion receipt exists for V4. H200 costs/capacity are not
H800 qualification, and earlier single-role G0-B does not prove all P/D cases.

## Current route — DSV4 H800 mainline (2026-10-25)

Authority: D-DSV4-H800-MAINLINE-CORRECTION in [requirements](../requirements.md#decision-2026-10-25--d-dsv4-h800-mainline-correction-current-route-authority). Route: [plan.md](../plan.md#current-execution-route--dsv4-h800-mainline-2026-10-25).

- For P0-H800-32, the `[x] P0 only` items below are the binding record; milestone H1 is done for P0. The original `## Milestones` checkboxes stay open because they cover all cases. They are closed per case only when that case has its own evidence.
- `P0 remaining: ... grouped54740-block allocation ...`: H4a-1 binds the native block count through Frontier's existing `explicit` memory mode for the DIAGNOSTIC run. The full non-KV memory reconciliation stays open for formal acceptance.
- `P0 remaining: qualified operator/graph/physicalTP8 costs ...`: H4a-2 to H4a-5 produce the first diagnostic prediction; formal closure stays with H4/H6 and ticket 19.
- The P1-H800-64 row is next after P0 promotion. The two P2-H200 rows are deferred (D-SCOPE-H800-DSV4).

## P0-scoped binding checkpoint — 2026-10-23

Evidence: [CPU report](../canonical_case_20261021/p0_binding/test_report_2026-10-23_p0_binding.md),
[exact inputs/commands](../canonical_case_20261021/p0_binding/receipt.json) and
`native06_acceptance_20261023/primary_acceptance_r4.json` ACCEPTED_NATIVE_REFERENCE_R4.
Root independently replayed the command; [root verification](../canonical_case_20261021/p0_binding/root_verification.json) accepts this P0 binding.
This is not all-case ticket completion; every existing milestone checkbox below remains open.

| Case | Actual workload/client binding | Native runtime reference | Memory/Frontier prerequisites | Numerical10% /30% |
| --- | --- | --- | --- | --- |
| P0-H800-32 | PASS scoped below | Accepted native06 R4 | OPEN | OPEN / OPEN |
| P1-H800-64 | Own binding pending | Own basic/runtime evidence pending | OPEN | OPEN / P0 promotion pending |
| P2-H200-32 | Own binding pending | Own runtime evidence pending | OPEN | OPEN / OPEN |
| P2-H200-64 | Own binding pending | Own basic/runtime evidence pending | OPEN | OPEN / H20032 promotion pending |

- [x] P0 only: all five shared68f891b workload files exactly match the original/R4 native06 copies; no second manifest.
- [x] P0 only:128 formal identities/arrivals0..127s/6144+1024 and786432/131072/917504 totals each round;432/432 complete records, zero missing/failed/unfinished.
- [x] P0 only:16 disjoint same-shape warmups before each of THREE repeats;144 stable templates. Frozen-source reconstructed prompt IDs100..129179 are inside actual model vocab129280; native usage confirms6144/1024. Original HTTP prompt arrays were not retained; native tokenizer guard execution is not claimed.
- [x] P0 only: independent mean client TTFT/TPOT `(E2E-TTFT)/1023`/E2E and total THPT917504 per completion window exactly reproduce existing aggregates; windows146.254631/146.259375/146.337699s, not127s. Existing seven-boundary idle helper PASS; no population filtering.
- [x] P0 only: admitted R4 binds P16+D16/TP8/DP2/EP16/PP1, non-preemptible H800/codesign, online P/D, dummy/quantization, thinking/prefix/chunking/MTP off,7168/8192/128 limits. Physical serving P NONE/D PIECEWISE+FULL and MXFP4/MARLIN/fp8_ds_mla are R4 scope, not new profiling admission.
- [ ] P0 remaining: reconcile retained native memory scalars/grouped54740-block allocation with complete non-KV and Frontier KV-capacity budget; zero native graph estimate is not proof of zero overhead.
- [ ] P0 remaining: qualified operator/graph/physicalTP8 costs and complete non-dummy Frontier prediction, then ticket19's independent10% final/30% promotion gates. Native reference metrics alone cannot close either numerical gate.

## Milestones

### canonical_manifest
- [ ] Bind V4 to `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/` and its checksums; reuse the shared artifact instead of creating a second byte-identical manifest.
- [ ] Verify all 128 formal identities, arrivals `0..127` seconds, exact 6144/1024 lengths, and totals 786432 input / 131072 output / 917504 combined tokens per repetition. Confirm native prompt token IDs are valid for V4 and preserve exact counts without tokenizer/chat expansion, clipping or early output termination.
- [ ] Bind the disjoint 16-request same-shape QPS-1.0 warmup pool to an independent warmup before every clean repetition. Prompt content uses stable template IDs, not repetition-qualified transport IDs. Preserve the earlier SharedGPT artifacts unchanged and do not inherit their selection or retiming rule.

### serving_configuration
- [ ] Verify the requested32-GPU P16+D16/TP8/DP2/EP16/PP1 and64-GPU
  P32+D32/TP8/DP4/EP32/PP1 layouts per role on the actual hardware. Record native
  refusal as BLOCKED; do not silently downgrade a priority task.
- [ ] Freeze the common native feature policy per case: online P/D, thinking,
  prefix and chunking off, graph/quantization on and dummy weights. Apply
  H800 non-preemptible, H20032 non-preemptible and H20064 preemptible classes.
- [ ] Require max_model_len>=7168 and max_num_batched_tokens>=max_model_len with chunking off; record native limits/graph/capture. Retarget05/06/09 to P6144/context0 and successful full-hit Dq1 cached6143..7166/attention6144..7167, including compression/page boundaries and first-tail phase. Returned7168 is distinct from computed KV7167.
- [ ] Preserve native-default MTP off and verify final-command/runtime parity and quantization via ticket18. Retain accepted kv_cache_dtype=fp8 resolving to fp8_ds_mla; do not substitute auto or conflate cache/group/kernel block sizes. Full-model effective evidence remains pending.
- [ ] Record the KV-transfer connector and configuration, and the memory and KV-capacity budget after ticket 18's non-KV overhead evidence.

### metric_binding
- [ ] Bind the four independent≤10% gates directly under19:mean client TTFT,
  mean request TPOT `(E2E-TTFT)/1023`,mean request E2E and total THPT
  `917504/(last_formal_last_token_completion-first_formal_submission)` per repeat.
  The127s arrival span is not THPT window. P90 remains recommendation,not approved
  implementation. Non-identical endpoints remain COVERAGE_MISS,not redefined.
- [ ] Bind 3 clean repetitions in one job, independent warmup/window for each, both roles idle between repetitions, and all 128 exact-length completions in every repetition. Record non-gating diagnostics and all failures/unfinished requests; no population filtering or warmup contribution to formal totals.

## Acceptance

Each DSV4 binding freezes independently when its workload,serving configuration
and metric binding have exact pins and resolved prerequisites;no paused GLM
completion or new16 activity is required. Tickets10/11 own priority
execution,18 native admission, and19 the independent30% promotion and10% final
gates. Basic64-GPU validation and preparation are not blocked by30%; formal
collection is. All four cases retain the same workload and V4-specific coverage.
