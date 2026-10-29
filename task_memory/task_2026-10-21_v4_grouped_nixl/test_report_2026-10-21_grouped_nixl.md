# Grouped V4 structural NIXL CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Accepted merged2a84c58: strict77/77 identical with zero missing, failure, provenance or cache findings; primary evidence retained. |
| 2026-10-21 | Primary independently verified 470 cases, exact diffs and retained evidence; merged at2a84c58 after28 combined checks and began the required merged strict comparison. |
| 2026-10-21 | Closed source/CPU acceptance with371 PASS and one clean strict77/77 identical comparison at1a65e49. |
| 2026-10-21 | Closed P2 with actual MetricsStore RED/GREEN,470 focused PASS, retained native bodies and unique clean strict77/77. |

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
The original371 tests used fixture metrics callbacks; P2 below exercises actual
MetricsStore and TraceStore. Non-owned cluster context remains a fixture. No EP compute,
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

The original eight touched production modules remain below2000 lines. No capacity schema,
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
label directory recorded in the fidelity receipt. The original records-only
commit17e3b42 did not change `frontier/` or `tests/` from measured1a65e49. P2 is
separately measured below.

## P2: real MetricsStore/op-tracing callback repair

The reviewer found a real unsupported scalar metadata consumer missed by the
original _Metrics fixture. Before the fix, the real start callback rejected V4
mixed compression variants after allocation/claim/reservation, and therefore
returned no transfer-end event. This invalidates the original suite as evidence
for enabled op tracing; its allocation/native-rule results remain unchanged.

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest 'tests/unit/test_dsv4_structural_pd.py::test_real_grouped_handoff_and_all_outputs_conserve[6144-True]' -q -p no:cacheprovider -o junit_family=legacy --basetemp=/data/ycfeng/tmp/v4_grouped_nixl_p2_red --junitxml=task_memory/task_2026-10-21_v4_grouped_nixl/p2_red.xml > /data/ycfeng/tmp/v4_grouped_nixl_p2_red.log 2>&1
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest 'tests/unit/test_dsv4_structural_pd.py::test_real_grouped_handoff_and_all_outputs_conserve[6144-True]' -q -p no:cacheprovider -o junit_family=legacy --basetemp=/data/ycfeng/tmp/v4_grouped_nixl_p2_green --junitxml=task_memory/task_2026-10-21_v4_grouped_nixl/p2_green.xml > /data/ycfeng/tmp/v4_grouped_nixl_p2_green.log 2>&1
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_structural_pd.py tests/unit/test_glm53_structural_pd.py tests/unit/test_nixl_decode_context.py tests/unit/test_grouped_kv_cache_manager.py tests/unit/test_grouped_kv_cache_scheduler.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_source_execution.py tests/unit/test_dsv4_parameter_memory.py tests/unit/test_dsv4_cache_layout.py tests/unit/test_dsv4_cache_zero_execution.py tests/unit/test_dsv4_cache_zero_driver.py tests/unit/test_kv_cache_extract_pricing.py tests/unit/test_op_trace_utils.py tests/unit/test_transfer_metrics_contract.py tests/unit/test_metrics_stage_execution_time.py -q -p no:cacheprovider -o junit_family=legacy --basetemp=/data/ycfeng/tmp/v4_grouped_nixl_p2_focused --junitxml=task_memory/task_2026-10-21_v4_grouped_nixl/p2_focused.xml > /data/ycfeng/tmp/v4_grouped_nixl_p2_focused.log 2>&1
```

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| RED before code change | Fail in actual callback at whole-model binding | 1 failed in2.22s; ValueError: Mixed compression variants require an explicit global layer id; exact chain in p2_red.xml | EXPECTED FAIL |
| Same scenario after fix | Actual MetricsStore/TraceStore accept grouped metadata and return end event | 1 PASS in5.40s; no fallback or exception suppression | PASS |
| Callback metadata | Existing snapshots; component/group bytes sum to payload |39521664B; each component and group conserved; held/reserved/selected24/2/2/2/16; no fictitious dense tensor shape | PASS |
| Trace persistence | Real enabled TraceStore JSONL, not a log_event fixture | Header plus one transfer event; loaded metadata equals buffered event exactly | PASS |
| Completion/conservation |1024 outputs from6143; no leaked pages |6143..7166 contexts, final computed7167; both pools3999 free +1 null | PASS |
| GLM metadata | Every original metadata field unchanged through actual callback |65-token,2-page callback has main/indexer/scale bytes11501568/344064/10752; transfer completes | PASS |
| Focused regressions | Existing371 plus canonical real callback and existing shared metrics tests |470 PASS in21.45s,0 failures/skips | PASS |
| Native review evidence | Exact pinned bodies available without working checkout |12 complete methods and9 spec/callsite excerpts retained in native_retention.md; hashes match prior extraction | PASS |
| New strict77 | One committed, clean candidate; unchanged checks |77/77 identical at5428bd7; zero cache/provenance or other findings | PASS |

The repair only adds grouped transfer metadata in op_trace_utils.py and forwards
existing transfer_info/scheduler block size from MetricsStore. That caller is an
existing oversized module covered by docs/general/module-size-exceptions.md; its
two-line forwarding change adds no responsibility. No broad cleanup is needed;
the existing future ledger split remains outside this repair. No model_binding,
timing, graph/measured admission, native protocol or other-owner files changed.

P2 implementation commit: `5428bd7`. Started exactly one new clean candidate
after the passing focused checks and clean commit:

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python tests/e2e/refactor_fidelity/measure_commit.py --sha 5428bd7 --output-root /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity --baseline-label baseline --label candidate_v4_grouped_nixl_p2_5428bd7 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 > /data/ycfeng/tmp/v4_grouped_nixl_p2_strict77.log 2>&1
```

Actual: **VERDICT: IDENTICAL**. All77 cases produced artifacts and compared
identically. Baseline/candidate failures, missing evidence, differing definitions,
not-compared cases and provenance findings are all0. Predictor cache comparison
is complete, with0 baseline-only/0 candidate-only names. Every per-case source
stamp is5428bd7803137cac1de5c691409534af1e68888e with source_dirty=false;
the retained detached checkout at
`/data/ycfeng/Frontier/.worktrees/fidelity-candidate-5428bd7` is still clean.

The original exact comparator was copied to `p2_fidelity_comparison.json` before
another run could replace the shared comparison.json. `p2_receipt.json` records
the checked manifest/results paths and final verdict. `source.diff` matches the
cumulative6d88700->5428bd7 source/test delta; `p2_source.diff` matches
17e3b42->5428bd7. Final records do not change the measured frontier/tests trees.
No matrix or focused test was rerun solely for the records-only changes.

## Primary integration verification

P2 re-review returned no findings and closed the original callback defect.
Primary independently ran the same 15-file focused selection above: **470 PASS
in 21.19s**, no failures or skips. Result:
`/data/ycfeng/tmp/glm53_dsv4/root_v4_grouped_p2_470.xml`. Exact cumulative and
P2 source diffs match Git; `fc0918c` changes no `frontier/` or `tests/` tree
relative to `5428bd7`. Retained RED/GREEN and strict receipts were checked.

After merging into the published C26 source, primary ran:

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_attention_profiling.py 'tests/unit/test_dsv4_structural_pd.py::test_real_grouped_handoff_and_all_outputs_conserve[6144-True]' tests/unit/test_glm53_structural_pd.py::test_structural_transfer_trace_uses_dsa_cache_components -q -p no:cacheprovider -o junit_family=legacy --junitxml=/data/ycfeng/tmp/glm53_dsv4/root_v4_grouped_merged_28.xml
```

Actual: **28 PASS in 5.76s**. `git diff --check HEAD` passed. Primary documented
the exact two-line MetricsStore exception in the existing module-size record,
then committed merge `2a84c58`. The following comparison runs from that commit's
clean detached checkout, with a distinct output root to avoid another owner's
comparison report overwriting this result:

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python tests/e2e/refactor_fidelity/measure_commit.py --sha 2a84c58 --output-root /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/root_v4_grouped_2a84c58 --baseline-label /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline --label candidate_2a84c58 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 > /data/ycfeng/tmp/glm53_dsv4/root_v4_grouped_merged_strict77.log 2>&1
```

Merged strict result: **PASS, 77/77 IDENTICAL**. Both source stamps and the
detached checkout identify clean `2a84c58`. There are no failed cases, missing
evidence, differing definitions, unexamined cases, provenance findings or
predictor-cache filename differences. The original comparator and primary
JUnit artifacts are retained as `primary_fidelity_comparison.json`,
`primary_focused.xml` and `primary_merged.xml`; `primary_receipt.json` records
their hashes and exact external paths. Native/full-serving and four-metric
limitations remain unchanged. Record-only updates do not require another run.
