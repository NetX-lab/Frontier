# Grouped V4 structural NIXL CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Closed source/CPU acceptance with371 PASS and one clean strict77/77 identical comparison at1a65e49. |

## Environment and commands

Working directory: `/data/ycfeng/Frontier/.worktrees/v4-grouped-nixl`.
Base: `6d88700d5a01448041575ba5ae49940c80fd40b4`. Python executable:
`/data/ycfeng/envs/frontier-py310/bin/python`, **Python3.10.6**. No environment
changes, native checkout edits or external operations occurred.

```bash
PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python tests/integration/verify_dsv4_native_nixl_rules.py --native-repo /data/ycfeng/Frontier/.real-engine/vLLM-BS --output task_memory/task_2026-10-21_v4_grouped_nixl/native_rules.json > /data/ycfeng/tmp/v4_grouped_native_rules.log 2>&1
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_structural_pd.py tests/unit/test_glm53_structural_pd.py tests/unit/test_nixl_decode_context.py tests/unit/test_grouped_kv_cache_manager.py tests/unit/test_grouped_kv_cache_scheduler.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_source_execution.py tests/unit/test_dsv4_parameter_memory.py tests/unit/test_dsv4_cache_layout.py tests/unit/test_dsv4_cache_zero_execution.py tests/unit/test_dsv4_cache_zero_driver.py tests/unit/test_kv_cache_extract_pricing.py -q -p no:cacheprovider -o junit_family=legacy --junitxml=task_memory/task_2026-10-21_v4_grouped_nixl/cpu_tests.xml > /data/ycfeng/tmp/v4_grouped_nixl_focused_receipt.log 2>&1
git diff --check
```

## Expected versus observed

| Check | Passing condition | Actual | Result |
| --- | --- | --- | --- |
| Pinned native methods | Exact locked revision; execute pruning, scheduler finish, D external allocation, clip and worker suffix selection | Three prompts191/6144/6145; hashes and method line ranges in native_rules.json | PASS |
| Unchunked6144 source | Initial allocation remains distinct from finish pruning | Initial[24,96,96,1536,768]; P-held[24,2,2,2,16]; connector list including null[24,3,3,3,17] | PASS |
| Real D reservation | Existing physical pool; no runnable request before receive | D owns[24,2,2,2,16], is receiving, no pre-ready batch | PASS |
| Structural payload | Every group selected from P ownership and covered by D; sum component pages, not slab/ranks | [24,2,2,2,16],39521664 B per worker lane | PASS |
| Replay and output geometry | First context prompt-1; exactly1024 one-token outputs | 6143..7166 contexts; final computed7167; final request7168; also prompt191 and6145 boundary loops | PASS |
| Conservation | Per-pool live IDs unique; refcounts equal ownership each step; all pages returned on completion | Both pools3999 free +1 reserved null, zero request-owned pages | PASS |
| Capacity/FIFO | P remains held on D refusal; actual D batch completion retries one pending event and clears FIFO | Real allocator/end callback reserves at3.0; incoming one-output request completes and releases | PASS |
| Invalid state | Reject missing/empty/zero/wrong-length/duplicate/null-layout groups, changed metadata/actual ownership/bytes, duplicate completion, scalar shortcut | New negative tests reject before completion metrics or P release; failed sizing rolls D back | PASS |
| Existing GLM/context/P2P | Existing tests unchanged; whole prompt96 pages/569106432B;1024 outputs | Included in371 PASS | PASS |
| Unsupported execution | Non-NIXL grouped P/D rejects; dummy NIXL serving and non-dummy V4 pricing still reject | Exact NotImplementedError messages below | PASS |
| Focused regression total | No failed/skipped cases | **371 passed in16.95s**, exit0; cpu_tests.xml | PASS |
| Source whitespace | No patch errors | git diff --check exit0 | PASS |
| Strict77 comparison | Clean committed source, all artifacts/cache identities checked | 77/77 identical; no failed/missing/differing cases, zero provenance or cache filename differences | PASS |

The component lifecycle tests use real TP8/DP4/EP32 H200 model configurations,
Request/Batch objects, one-owner scheduling iterations, the actual grouped pool,
target selection, transfer start/end and release callbacks. The max length is7169
to include the6145+1024 boundary; the canonical6144+1024 request is unchanged.
Metrics callbacks and non-owned cluster context are fixtures. No EP compute,
four-owner busy-loop synchronization, native runtime process or GPU is executed.

## Preserved typed execution boundaries

- Non-dummy: `NotImplementedError: V4 measured/native execution remains unqualified`.
- Dummy with NIXL: `NotImplementedError: V4 structural serving requires eager co-location without PD`.
- Grouped PREFILL/DECODE with none/P2P: `NotImplementedError: Grouped KV cache allocation supports MONOLITHIC or NIXL PREFILL/DECODE only`.

These guards are tested at their actual consumer functions; no new full serving
CLI run or full-path PASS is claimed. Graph/timing/attention admission modules
are unchanged. `native_retention.md` separates source evidence, structural
payloads and the unmeasured native protocol boundaries.

## Scope and limits

Read the parent's accepted H200 G0-B report and V4 runtime_verification.json:
55/55 runtime checks, native V2,5 groups, block256, TP8/DP4/EP32, dummy weights.
Read H200 admission root_acceptance.json:6d88700 is tree-identical to1145b40;
storage-only approval,7,794,659,804 parameter bytes, no PD execution admission.
Those results are inherited evidence, not this task's GPU measurements.

No measured comparison is performed here, so native timing, absolute/relative
timing error, native wire bytes, contention, release-notification delay,128-request
three-repetition completion and four-metric <=10% acceptance remain **UNVERIFIED**.
Structural D-ready then P-release at one timestamp is the existing approximation.

All eight touched production modules remain below2000 lines. No capacity schema,
model graph consumer, profiling/H5 file, generic allocation owner or runtime
migration was added. The existing pool's new external-computed-token input is
needed to match native destination window boundaries with zero scheduled work.

## Clean strict77 comparison

Implementation commit: `1a65e4985091f5c85c14830f86385800a614e4e6`.
The command below was run **once**, after371 passing focused checks and commit.
The unchanged driver created a clean detached checkout at
`/data/ycfeng/Frontier/.worktrees/fidelity-candidate-1a65e49`.

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python tests/e2e/refactor_fidelity/measure_commit.py --sha 1a65e49 --output-root /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity --baseline-label baseline --label candidate_v4_grouped_nixl_1a65e49 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 > /data/ycfeng/tmp/v4_grouped_nixl_strict77.log 2>&1
```

Actual: **VERDICT: IDENTICAL**,77/77 cases produced artifacts and compared exactly.
No baseline/candidate failures, missing evidence, different definitions, skipped
comparisons or provenance findings. Predictor cache compared cleanly with0
baseline-only/0 candidate-only names. All77 per-case source stamps identify
1a65e49 with source_dirty=false; detached checkout remains clean afterwards.
Baseline is the required absolute
`/data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline` atbfa82de.

Durable copied comparator: `fidelity_comparison.json`; checked source/cleanliness
and result receipt: `fidelity_receipt.json`; CPU lifecycle observations:
`cpu_receipt.json`. Full manifest/results remain under the absolute candidate
label directory recorded in the fidelity receipt. The final records-only commit
does not change `frontier/` or `tests/` from the measured1a65e49 commit.
