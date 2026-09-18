## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Reran all 58 cases at committed source c9f8f904; all 574 artifacts match the second campaign byte for byte, preserving raw failures and the independent arithmetic classification. |
| 2026-09-16 | Classified both completed 58-case campaigns without changing the comparator; isolated request/system, ledger, EP, dummy-scope and batching differences; checked final-source reproducibility. |

# W10 Fidelity Classification

## Result and scope

All three completed campaigns report **24 PASS / 34 FAIL** under the original comparator. The latest campaign is bound to committed source **c9f8f904e3550c11aad3cc5d851d75d648cef6e1**. All simulator subprocesses exit successfully and each side completes the expected requests: 122 requests across 58 scenarios per campaign. The 34 raw failures remain failures in the original `results.json`; each first reports `root: fields differ` because the candidate adds `frontier_ep_wave_lane_ledger.jsonl`. Comparing shared artifacts exposes the further differences below. This report does not relabel the raw harness result as an unconditional 58-case pass.

- **24 cases:** every original fidelity artifact matches exactly as parsed.
- **12 co-location MoE cases:** request/system artifacts match exactly; the stage ledger now includes all 32 physical layers, and the separate EP lane artifact is added. Independent stage arithmetic preserves actual stage duration.
- **22 disaggregated MoE cases:** request/system timing values differ. All 12 PDD cases expose the same dummy layer normalization defect in PREFILL and unified DECODE; all 10 PD-AF cases expose it in PREFILL. Per-case stage arithmetic, independent layer magnitudes and request role sums establish the scope of the correction. The two online chunked cases additionally change batch composition at a proven arrival boundary.

D01 retains the uniform layer/stage contract and requires independent arithmetic evidence. The PDD classification is based on the same inspected constructor/scaling defect and measured arithmetic, rather than a blanket exemption for PDD discrepancies. Request/throughput values that changed remain explicitly DIFFERENT. This is dummy-predictor structural and timing-contract evidence, not trained-model numerical parity or native GPU evidence.

## Execution and source binding

- First campaign: `/data/ycfeng/tmp/pr33-w10-final-fidelity`; log `/data/ycfeng/tmp/pr33-w10-final-fidelity.log`.
- Second campaign: `/data/ycfeng/tmp/pr33-w10-final-fidelity-2`; log `/data/ycfeng/tmp/pr33-w10-final-fidelity-2.log`.
- Latest campaign: `/data/ycfeng/tmp/pr33-w10-final-fidelity-3`; log `/data/ycfeng/tmp/pr33-w10-final-fidelity-3.log`; candidate manifest commit `c9f8f904e3550c11aad3cc5d851d75d648cef6e1`. `git diff --stat -- frontier` was empty at launch; production remained frozen through completion.
- Baseline: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`, worktree `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915`.
- Historical first/second candidate manifest HEAD: `5a1cc2af280e5ae2d51caae55fe016c80bc3e2d2`; the manifest HEAD alone does not identify uncommitted source. Bind the first run to `/data/ycfeng/tmp/pr33-w10-final-source.diff` and second run to `/data/ycfeng/tmp/pr33-w10-final-source-2.diff`. The third run uses the committed source above, including the final homogeneous same-object aggregation optimization.
- Python: `/usr/bin/python`, Python 3.12.3; no active conda environment. All run-specific commands and environment overrides are retained in `<campaign>/<case>/<side>/invocation.json`. Dummy operator value is 1 ms, with one pipeline stage and 32 model layers in the affected fixtures. CPU library threads are bounded to one by the harness.
- Analysis deliverable: `/data/ycfeng/tmp/pr33-w10-fidelity-classification.json`. It includes all three manifests, raw statuses/errors, every artifact path and comparison, exhaustive differing leaf-path groups with counts and numeric examples (absolute and relative difference), per-case oracles, batching evidence, both consecutive campaign byte comparisons and the executable classification source. `latest_campaign`, `latest_source_commit` and `third_campaign_recheck` identify the final result.

The classifier only reads existing simulator artifacts and writes its own JSON. It imports `compare` and `load_artifact` from `tests/integration/run_scheduler_refactor_fidelity.py`; it does not invoke that module's `main`. Reproduce the analysis from the candidate repository root with:

```bash
/usr/bin/python - <<'PY_ANALYSIS'
import json
from pathlib import Path
record = json.loads(Path('/data/ycfeng/tmp/pr33-w10-fidelity-classification.json').read_text())
source = record['classification_source']
exec(compile(source, '<w10-fidelity-classification>', 'exec'), {'classification_source': source})
PY_ANALYSIS
```

The first two campaigns were launched by the root agent; the classification owner launched the third. All use `--workers 2`. The exact outer commands were:

```bash
PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/integration/run_scheduler_refactor_fidelity.py --baseline ../pr33-r12-baseline-20260915 --candidate . --output /data/ycfeng/tmp/pr33-w10-final-fidelity --workers 2 > /data/ycfeng/tmp/pr33-w10-final-fidelity.log 2>&1
PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/integration/run_scheduler_refactor_fidelity.py --baseline ../pr33-r12-baseline-20260915 --candidate . --output /data/ycfeng/tmp/pr33-w10-final-fidelity-2 --workers 2 > /data/ycfeng/tmp/pr33-w10-final-fidelity-2.log 2>&1
PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/integration/run_scheduler_refactor_fidelity.py --baseline ../pr33-r12-baseline-20260915 --candidate . --output /data/ycfeng/tmp/pr33-w10-final-fidelity-3 --workers 2 > /data/ycfeng/tmp/pr33-w10-final-fidelity-3.log 2>&1
```

Full simulation invocation parameters are preserved in the per-run invocation files; the outer harness uses `--baseline`, `--candidate`, `--output` with the roots above and all 58 cases from `scenarios()`. This classifier did not rerun simulations. The scenario matrix covers three architectures, offline/online, dense/MoE, short/chunked/unchunked, plus speculative decoding, prefix caching, Thinking Mode, DP2/EP8 and PD-AF CUDA Graph/EP cases. No configuration/golden-data normalization is applied by the comparator.

## Comparator criteria and shared artifact results

The comparator is unchanged: identical mapping fields, list lengths and order; scalar equality first, then `math.isclose(rel_tol=1e-12, abs_tol=1e-9)`, preserving IDs and timestamps. The table's EXACT refers to parsed-value equality, stricter than numeric tolerance. Schema additions are classified independently and never silently dropped from the original comparison.

| Artifact | EXACT shared cases | DIFFERENT shared cases | Added / absent |
| --- | ---: | ---: | --- |
| `request_metrics.csv` | 36 | 22 | None |
| `system_metrics.json` | 36 | 22 | None |
| `frontier_stage_batch_ledger.jsonl` | 24 | 34 | None |
| `frontier_transfer_ledger.jsonl` | 16 | 22 | Absent on both sides in 20 cases |
| `op_precision_metadata.csv` | 58 | 0 | None |
| `frontier_ep_wave_lane_ledger.jsonl` | Not a baseline artifact | Not a baseline artifact | Added in all 34 MoE cases; absent in all 24 others |

Across all 58 cases, request IDs, token counts, restart/preemption/speculation outcomes and selected session/cohort fields match. The JSON records exact common columns for each case and separately lists every numerically changed column. There are 41 request columns with exact parsed values across every case; more columns match per case or under the unchanged numeric comparator. Request row counts/order and completion outcomes are preserved.

All system differences are restricted to TTFT/TPOT/E2E statistics and throughput duration/rates. Simulation metadata, quantization, architecture, memory, speculation, preemption, KV transfer statistics/bytes and throughput token/request totals match. Request timing differences include computation/service, waiting, TTFT/TPOT, normalized latency, and KV transfer start/end timestamps. Transfer-ledger timestamp or batching identity differences are retained in the JSON, rather than calling those ledgers equivalent.

## Independent arithmetic and causal isolation

The checks use the recorded 1 ms dummy magnitude, model layer count 32, stage-owned host/PP terms, supported active attention components and separate physical EP lane records. They do not accept the candidate's reported stage total as the sole expected value. EP phase checks are conservation/synchronization checks over measured simulator ledger fields, not native kernel correctness tests.

For each affected candidate attention stage, each active per-layer attention component must be `32 * 1 = 32 ms`; inactive components remain zero. Each active stage-owned host/PP component remains 1 ms. Component sums must equal the reported attention-stage total. The EP ledger must cover global layers 0–31 for that source batch. Each phase's named operators sum to its local work; a physical wave lasts the sum of **peer phase maxima**, with shared phase start times. An idle EP lane is allowed to finish its own local work before the wave's barrier: comparing a local sum directly to the whole wave would be an invalid oracle.

The attention-stage contribution plus physical EP critical work over 32 layers must equal the observed stage end minus start. Every baseline attention stage is checked against the corresponding inspected baseline normalization, including rows whose batch IDs changed. Request-level PREFILL, unified DECODE and DECODE_ATTN computation fields are separately checked against the sum of that request's actual stage spans.

| Observed family | Independent stage arithmetic | Baseline elapsed | Candidate elapsed | Absolute / relative elapsed change |
| --- | --- | ---: | ---: | ---: |
| Co-location MoE, normal | `6 + 32 * (8 attention + 8 EP)` | 518 ms | 518 ms | 0 / 0% |
| Co-location MoE DP2/EP8 | `6 + 32 * (8 attention + 7 EP)` | 486 ms | 486 ms | 0 / 0% |
| PDD MoE PREFILL / DECODE | `6 + 32 * (7 attention + 7 EP)` | 20 ms | 454 ms | 434 ms / 2170% |
| PD-AF MoE PREFILL, EP1 | `6 + 32 * (6 attention + 5 EP)` | 17 ms | 358 ms | 341 ms / 2005.882352941% |
| PD-AF MoE PREFILL, EP2 | `6 + 32 * (7 attention + 7 EP)` | 20 ms | 454 ms | 434 ms / 2170% |

The six fixed milliseconds are five host terms plus the PP term in these exact dummy fixtures; they are not scaled with layer count. Baseline PDD/PD-AF elapsed values satisfy `(candidate_elapsed - 6) / 32 + 6`. The baseline `sklearn_disaggregation_execution_time_predictor.py` creates dummy PREFILL and DECODE payloads with `num_layers_per_pipeline_stage=self._num_layers_per_pipeline_stage`, then scales by `num_layers / dummy_exec_time.num_layers` (baseline line 1504). A physical one-layer request therefore divides the 1 ms operator by 32. The candidate creates one-layer dummy values before normal stage assembly (current PREFILL/DECODE constructors near lines 730/779). This source path proves the PDD extension of the same defect.

Stage-reporting changes are distinct from scheduling duration changes. For example, co-location's attention ledger component changes from 1 to 32, its total from 14 to 262 ms, while EP work is 256 ms and actual stage time remains 518 ms. PD-AF EP1's baseline PREFILL attention ledger reports `6 + 6/32 = 6.1875 ms`; candidate reports `6 + 6*32 = 198 ms`, plus `5*32 = 160 ms` EP work, producing 358 ms. PDD's candidate attention ledger is 230 ms plus 224 ms EP work, producing 454 ms.

Observed request examples (full precision and all differing fields are in the JSON):

- `pdd_offline_moe_model_basic_short`: request E2E 100.5419430400001 → 2270.541943039986 ms, an increase of 2170 ms from five stages each increasing by 434 ms; unchanged KV duration accounts for the fractional remainder.
- `pd-af-disagg_offline_moe_model_basic_short`: request E2E 1657.5943718399844 → 1998.594371839978 ms, an increase of 341 ms from corrected PREFILL; per-layer DECODE_ATTN/FFN work remains unchanged.

The completed checks cover **127 candidate attention-stage rows**, **130 baseline attention-stage rows**, **11,264 EP lane rows in 4,064 physical waves**, and **176 request/role stage sums**, with **zero mismatches**. These figures describe the independent checks, not additional pytest passes. The report does not claim to independently reconstruct every aggregate percentile or every downstream queueing timestamp; the exhaustive artifact differences remain available for audit.

## Online batch-count differences

Both changed-count cases have the same request arrivals: request 0 at 102.0060287274801 ms; request 1 at 104.538912631754 ms; request 2 at 136.70131903925054 ms. These follow from the initial idle first-stage start and the unchanged request inter-arrival fields. All recorded prefill starts occur after the relevant arrivals, and each request consumes exactly 80 prefill tokens on both sides.

- **PDD online chunked: 12 → 10 stage rows.** Baseline's first 20 ms stage finishes at 122.00602872748034 ms, before request 2 arrives. Its prefill batches are `[0]:64`, `[1,0]:[64,16]`, `[2,1]:[64,16]`, `[2]:16`. Candidate's first 454 ms stage finishes at 556.0060287274801 ms, after both waiting requests arrive. Its batches are `[0]:64`, `[1,2,0]:[64,64,16]`, `[1,2]:[16,16]`. Thus PREFILL has four versus three batches. Requests 1 and 2 then finish prefill together and arrive together at unified DECODE, giving seven versus eight decode batches. Each request still has six decode steps, 18 request decode steps in total; request computation sums validate both schedules.
- **PD-AF online chunked: 387 → 386 stage rows.** Baseline's first prefill finishes at 119.00602872747978 ms and its second at 136.00602872747966 ms, both before request 2 arrives. Candidate's first prefill finishes at 460.0060287274804 ms, so requests 1 and 2 share the second prefill. PREFILL has three versus two rows. DECODE_ATTN and DECODE_FFN each retain 192 rows on both sides and their per-layer execution payloads are unchanged. The arrival boundary and token conservation explain the one removed prefill batch; it is not a missing execution.

## Per-case disposition

`E` = exact shared parsed artifact; `D` = different under the unchanged comparator; `—` = absent on both sides. Request/system statuses are shown together because they agree in every case. Every row also has exact operator precision metadata. `A` = raw comparator pass; `L` = proven layer-reporting correction plus added EP artifact; `S` = source-proven dummy layer correction with retained request/system numeric differences. Stage arithmetic is baseline → candidate elapsed in milliseconds; `M`, `P`, `D` mean MONOLITHIC, PREFILL, DECODE. All L/S rows have the added EP artifact and zero arithmetic-oracle errors.

| Case | Raw | Request / system | Stage | Transfer | Stage elapsed oracle | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| `co-location_offline_dense_model_basic_chunked` | PASS | E | E | — | Exact artifacts | A |
| `co-location_offline_dense_model_basic_short` | PASS | E | E | — | Exact artifacts | A |
| `co-location_offline_dense_model_basic_unchunked` | PASS | E | E | — | Exact artifacts | A |
| `co-location_offline_moe_model_basic_chunked` | FAIL | E | D | — | M 518→518 | L |
| `co-location_offline_moe_model_basic_dp2_ep8` | FAIL | E | D | — | M 486→486 | L |
| `co-location_offline_moe_model_basic_short` | FAIL | E | D | — | M 518→518 | L |
| `co-location_offline_moe_model_basic_unchunked` | FAIL | E | D | — | M 518→518 | L |
| `co-location_offline_moe_prefix_caching_feature` | FAIL | E | D | — | M 518→518 | L |
| `co-location_offline_moe_spec_dec_feature` | FAIL | E | D | — | M 518→518 | L |
| `co-location_offline_thinking_mode_basic_feature` | PASS | E | E | — | Exact artifacts | A |
| `co-location_online_dense_model_basic_chunked` | PASS | E | E | — | Exact artifacts | A |
| `co-location_online_dense_model_basic_short` | PASS | E | E | — | Exact artifacts | A |
| `co-location_online_dense_model_basic_unchunked` | PASS | E | E | — | Exact artifacts | A |
| `co-location_online_moe_model_basic_chunked` | FAIL | E | D | — | M 518→518 | L |
| `co-location_online_moe_model_basic_dp2_ep8` | FAIL | E | D | — | M 486→486 | L |
| `co-location_online_moe_model_basic_short` | FAIL | E | D | — | M 518→518 | L |
| `co-location_online_moe_model_basic_unchunked` | FAIL | E | D | — | M 518→518 | L |
| `co-location_online_moe_prefix_caching_feature` | FAIL | E | D | — | M 518→518 | L |
| `co-location_online_moe_spec_dec_feature` | FAIL | E | D | — | M 518→518 | L |
| `co-location_online_thinking_mode_basic_feature` | PASS | E | E | — | Exact artifacts | A |
| `pd-af-disagg_offline_dense_cuda_graph_feature` | PASS | E | E | E | Exact artifacts | A |
| `pd-af-disagg_offline_dense_model_basic_chunked` | PASS | E | E | E | Exact artifacts | A |
| `pd-af-disagg_offline_dense_model_basic_short` | PASS | E | E | E | Exact artifacts | A |
| `pd-af-disagg_offline_dense_model_basic_unchunked` | PASS | E | E | E | Exact artifacts | A |
| `pd-af-disagg_offline_moe_cuda_graph_feature` | FAIL | D | D | D | P 17→358 | S |
| `pd-af-disagg_offline_moe_model_basic_chunked` | FAIL | D | D | D | P 17→358 | S |
| `pd-af-disagg_offline_moe_model_basic_short` | FAIL | D | D | D | P 17→358 | S |
| `pd-af-disagg_offline_moe_model_basic_unchunked` | FAIL | D | D | D | P 17→358 | S |
| `pd-af-disagg_offline_moe_model_ep_feature` | FAIL | D | D | D | P 20→454 | S |
| `pd-af-disagg_online_dense_cuda_graph_feature` | PASS | E | E | E | Exact artifacts | A |
| `pd-af-disagg_online_dense_model_basic_chunked` | PASS | E | E | E | Exact artifacts | A |
| `pd-af-disagg_online_dense_model_basic_short` | PASS | E | E | E | Exact artifacts | A |
| `pd-af-disagg_online_dense_model_basic_unchunked` | PASS | E | E | E | Exact artifacts | A |
| `pd-af-disagg_online_moe_cuda_graph_feature` | FAIL | D | D | D | P 17→358 | S |
| `pd-af-disagg_online_moe_model_basic_chunked` | FAIL | D | D | D | P 17→358 | S |
| `pd-af-disagg_online_moe_model_basic_short` | FAIL | D | D | D | P 17→358 | S |
| `pd-af-disagg_online_moe_model_basic_unchunked` | FAIL | D | D | D | P 17→358 | S |
| `pd-af-disagg_online_moe_model_ep_feature` | FAIL | D | D | D | P 20→454 | S |
| `pdd_offline_dense_model_basic_chunked` | PASS | E | E | E | Exact artifacts | A |
| `pdd_offline_dense_model_basic_short` | PASS | E | E | E | Exact artifacts | A |
| `pdd_offline_dense_model_basic_unchunked` | PASS | E | E | E | Exact artifacts | A |
| `pdd_offline_moe_model_basic_chunked` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_offline_moe_model_basic_dp2_ep8` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_offline_moe_model_basic_short` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_offline_moe_model_basic_unchunked` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_offline_moe_prefix_caching_feature` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_offline_moe_spec_dec_feature` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_offline_thinking_mode_basic_feature` | PASS | E | E | E | Exact artifacts | A |
| `pdd_online_dense_model_basic_chunked` | PASS | E | E | E | Exact artifacts | A |
| `pdd_online_dense_model_basic_short` | PASS | E | E | E | Exact artifacts | A |
| `pdd_online_dense_model_basic_unchunked` | PASS | E | E | E | Exact artifacts | A |
| `pdd_online_moe_model_basic_chunked` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_online_moe_model_basic_dp2_ep8` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_online_moe_model_basic_short` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_online_moe_model_basic_unchunked` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_online_moe_prefix_caching_feature` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_online_moe_spec_dec_feature` | FAIL | D | D | D | D 20→454, P 20→454 | S |
| `pdd_online_thinking_mode_basic_feature` | PASS | E | E | E | Exact artifacts | A |

## Final source repeatability and remaining limits

The third campaign at **c9f8f904e3550c11aad3cc5d851d75d648cef6e1** completes all 58 cases with the same **24 PASS / 34 FAIL** statuses. The outer harness exits 1 because it retains the 34 comparison failures; all 116 simulator subprocesses exit 0 and complete the expected requests. Comparing second versus third campaign finds identical artifact sets and **574/574 byte-identical files: 270 baseline and 304 candidate**. Parsed values and the established comparator also match for every file. There is no new discrepancy to investigate. Because every input artifact to the prior per-case arithmetic checks is unchanged, all 34 D01 classifications and their zero-mismatch oracle evidence remain valid. `third_campaign_recheck` retains every file pair, both campaigns' paths, the final manifest and raw third-campaign results. The comparison process exits 0; CPU campaign and analysis work were complete before the exclusive performance window.

The second campaign completes all 58 cases with the same 24 PASS / 34 FAIL statuses. Comparing first versus second campaign finds identical artifact sets and **574/574 byte-identical files: 270 baseline and 304 candidate**. This includes every harness-selected CSV, JSONL and `system_metrics.json`; the newly added EP ledger is included. Config files, invocation paths and logs contain run-directory identities and are outside the established fidelity artifact set. The exact file pairs and byte/value/comparator results are retained under `second_campaign_recheck` in the JSON. Therefore the final source changes represented by `pr33-w10-final-source-2.diff` preserve this entire observed fidelity output set.

No unresolved arithmetic mismatch or unexplained batch-count loss was found in this classification. The 34 raw comparator failures and 22 request/system numerical differences remain visible. Hardware/native acceptance, trained homogeneous parity, hybrid GDN scheduler acceptance and performance gates belong to their separate W10 reports; this report supplies no substitute evidence for those requirements. No production source, simulator test, comparator, tolerance, scenario configuration or golden artifact was modified by this classification task.
