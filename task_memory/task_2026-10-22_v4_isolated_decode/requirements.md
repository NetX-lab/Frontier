# W128-V4-ISOLATED-DECODE-CONSUMER Requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded the delegated scope and accepted measurement constraints. |

## Original request
[Original Request] Implement the smallest existing-reader extension for exact measured ongoing q1 DECODE in `read_isolated_scope`, under D-Q10-V4-GRAPH-MEASURED-AND-NIXL approval. Start at 2c150a7 in clean branch/worktree `v4-isolated-decode`. Own only `frontier/execution_time_predictor/deepseek_v4.py`, focused existing V4 tests/fixtures and these records; maximum 500 changed lines. No GPU, platform, SSH, cloud, new diagnostics, GLM5.3 tests, merge or push.

[Original Request] Read only pinned native checkout `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69` at 2cf0a6915ce544dc493a0990f2ea38d81601128a. Distinguish ongoing decode from first-tail with existing batch metadata, or report a precise blocker; do not invent a flag/default.

[Original Request] Accepted warmed job exp-1022-003156-993689, producer d2878ed90ef4aecc1d2d80b597381b813420e2a8. Both accepted CSVs are under `/data/ycfeng/tmp/glm53_dsv4/v4_q1_warm_results/Q1_WARM_6143_7166/compute/h800/deepseek-ai/DeepSeek-V4-Flash/`. Each contains 6144 rows, C6143..7166, three variants and planning/reuse roles, physical TP1, target TP8/H800 DP2/EP16, eager six-layer reduced model. Native effective max_num_seqs1 differs from CSV planner128: match actual available metadata without relabeling. Kernel-only is a separate pass with no enclosing attention-layer scope.

[Original Request] Do not open constructor/family execution, serving_context PD, co-location serving, graph/compiled, first-tail, cached multiquery prefill or ordinary native admission. Every receipt retains serving_admission=false, physical TP1/eager and reduced-model limitations. No full-P0 equivalence, E2E, filtering, subtraction or repeated-CV claim. H200 cannot bind to H800.

[Original Request] Verify both methods, all variants/planning roles, first/last and compression boundaries against exact accepted latency/provenance. Retain wrong-device/layout/cache/graph/first-tail/unsupported-phase/missing-context/serving negatives and existing P6144/typed regressions. Run focused tests, audit strict77 inventory for GLM-specific runs before executing; pause prohibited cases only. Commit verified scoped changes and return compact task ID, diff path, commit, validation and remaining limits.

## Implementation decisions
- Reuse semantic `request_is_decoding` and scheduled `num_context_tokens`; q1 alone is insufficient. Compare context to existing `Request.num_prefill_tokens` to reject prompt-tail recomputation, including a contradictory decoding flag.
- Match complete existing flattened attention layout. Dynamic native max_seq_len is supplied explicitly for each measured context; nullable compressor fields remain null. No physical allocation/capacity field is guessed.
- Decode extension is attention-only; linear/MoE decode and mixed batches remain rejected. Exact CSV lookup, not fitting/extrapolation.

## R1 review request
[Original Request] W128-V4-ISOLATED-DECODE-R1: independent review requests P2 changes because631250b accepts historical cold b36c2b9 q1 rows using only self-consistent candidate provenance. Preserve all631250b receipts and strict77 identity. Reproduce old-table RED then GREEN. Bind independently supplied acceptance provenance through existing explicit isolated-scope argument contract, without hardcoded accepted path/commit/hash in core or new config flag/shared API. Accepted warmed methods pass exactly; old-table/source/fingerprint/input provenance negatives typed-reject. Keep all serving gates and legacy prefill. Commit scoped fix and supply full original-base code+record diff for rereview; run fixed-source strict comparison separately.

R1 decision: use existing `attention_layout` mapping for decode-only expected source_revision/runtime_fingerprint/model_config_sha256 and input_file_sha256. Expected values come from the independently accepted root warmup report/measurement02_primary receipt, captured in a focused V4 fixture; never from the candidate row. Existing RuntimeIdentity and file_sha256 mechanisms are reused. Legacy prefill layout stays unchanged.
