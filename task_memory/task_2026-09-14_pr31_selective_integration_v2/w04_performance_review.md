## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Completed 21-process controlled ablation; recorded exact output equivalence and separated measured attention reuse from inactive EP-cache controls. |
| 2026-09-16 | Completed the isolated mutable-snapshot supplement on frozen production commit f9099f85. |

## Execution and criteria

Script: `tests/performance/measure_pr33_cache_ablation.py`.

Executed command:

```bash
/usr/bin/python tests/performance/measure_pr33_cache_ablation.py --output /data/ycfeng/tmp/pr33-w04-ablation-20260916 --repetitions 3
```

The coordinator launches fresh subprocesses with one numerical-library thread each and alternates normal/bypass order by repetition. Twenty-one subprocesses cover three repetitions of:

1. Unchanged W00 `representative_moe`: sequential PDD, Qwen3-235B-A22B, 32 simulated GPUs, TP8/EP8/PP2, two online requests, QPS5, prefill32/decode8, dummy1ms. Its existing disabled-reporting flags are preserved in every arm.
2. Existing `test_hybrid_gdn_production_constructor_cpu_e2e(..., num_requests=3)` with its original synthetic CPU profiles, real trained/loaded models, normal Simulator/predictor constructors, layer scheduling, and complete requested outputs.
3. Separate component microbenchmarks of model-owned attention-key formation, cloned numerical payload, eight-layer stage snapshot/finalization, and immutable EP workload hit/miss.

Each Simulator scenario runs normal, attention-cache bypass, and EP-workload-cache bypass. The numerical bypass calls the existing uncached helper path; the workload bypass uses zero capacity only inside the test wrapper, preserving materialization and returned workload. Production source/configuration has no new ablation flags. Counter wrappers are identical across arms, with only the selected bypass differing.

Acceptance checks compare completed request count, event count, every request's arrival/completion time and processed tokens. For hybrid, compare all final CSV/JSON/JSONL contents, including operation rows, request metrics, traces, stage ledger/summary, system metrics, and config. Normalize the per-attempt output-root path and the trace metadata header's real wall-clock creation timestamp; all simulated times remain exact. The script aborts on any substantive output difference; all raw outputs remain in the attempt directories.

## Causal limits established before timing

The W00 dummy PDD scenario made zero calls to either target predictor cache. It is a negative control. The unchanged trained hybrid scenario made 336 attention queries per run with 90 cache hits in the normal/EP-bypass arms and zero hits in the attention-bypass arm. The hybrid scenario made zero calls to the target predictor EP-workload cache. The experiment does not alter stage size or request parameters to manufacture hits.

Component microbenchmarks use the normal-constructor deterministic timing fixture. They isolate operation cost and do not establish trained E2E speedup. Stage finalization remains required semantic work; the experiment measures its cost without dropping identities, snapshots, layers, diagnostics, or logical events. EP miss microtiming explicitly includes a dictionary clear before rematerialization and must be labeled accordingly.

## Observed results

All **21/21 subprocesses succeeded**: 18 real Simulator attempts and three component-microbenchmark workers. Source revision was `c1e924db2e337653f9a8f914948cee1f4e1d6350` plus the recorded working diff. Python was `/usr/bin/python` 3.12.3, no conda environment, numerical library threads fixed at one. The root agent held other CPU checks and source edits during the campaign. No profiler was active; the established minimal event counter remained identical in each arm.

| Scenario | Arm | Run 1 s | Run 2 s | Run 3 s | Median s | Median paired change vs normal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| w00 | normal | 9.326830 | 9.113952 | 9.331369 | 9.326830 | +0.000% |
| w00 | attention_bypass | 9.358806 | 9.317917 | 9.346946 | 9.346946 | +0.343% |
| w00 | ep_bypass | 9.190087 | 9.416437 | 9.280273 | 9.280273 | -0.548% |
| hybrid | normal | 1.816960 | 1.879852 | 1.848976 | 1.848976 | +0.000% |
| hybrid | attention_bypass | 2.293946 | 2.257655 | 2.262040 | 2.262040 | +22.340% |
| hybrid | ep_bypass | 1.815694 | 1.955996 | 1.920818 | 1.920818 | +3.885% |

On the trained synthetic-CPU hybrid fixture, disabling numerical attention reuse increases paired `Simulator.run()` time by 20.10%, 22.34%, and 26.25% (median 22.34%). Equivalently, normal caching reduces time versus its paired bypass by a median 18.26%. This establishes a benefit on this exact fixture, with unchanged simulated work; it does not establish hardware latency fidelity, a universal performance budget, or resolution of the final W11/D02 baseline gate.

The W00 and hybrid EP-bypass timings are negative controls: there were zero calls to that cache in either real scenario. Their small timing differences cannot be attributed to EP caching.

Every W00 attempt retained 3,560 events, two completed requests, and 40 processed tokens per request. Completion timestamps remained exactly 12.156758472815124 and 12.820758472815177 seconds. Every hybrid attempt retained 342 events, three completed requests, and 18 processed tokens per request. Completion timestamps remained exactly 0.029131020060287286, 0.05826102006028736, and 0.08739102006028725 seconds. Arrival timestamps also matched exactly.

All ten hybrid output files compared equal after the two explicitly scoped metadata normalizations: `config.json`, `frontier_stage_batch_ledger.jsonl`, `frontier_stage_batch_ledger_summary.json`, `monolithic_batch_metrics.csv`, `monolithic_cpu_operation_metrics.csv`, `monolithic_operation_metrics.csv`, `op_precision_metadata.csv`, `op_traces.jsonl`, `request_metrics.csv`, and `system_metrics.json`. No metric tolerance was introduced.

Initial component results combine 15 samples per operation (three fresh processes, five repetitions each, 1,000 calls per sample):

| Operation | Median microseconds/call | Min | Max |
| --- | ---: | ---: | ---: |
| attention_key | 0.333311 | 0.331914 | 0.384676 |
| attention_payload_copy | 5.284160 | 5.101569 | 5.374858 |
| ep_workload_hit | 2.153831 | 2.123978 | 2.249740 |
| ep_workload_miss_with_clear | 80.743009 | 80.255958 | 84.447224 |
| stage from eight already-finalized layers | 5.244521 | 5.119165 | 5.530127 |

The initial stage operation measures container creation/identity checks with already-finalized layer inputs. It excludes mutable-component snapshot construction; no claim that 5.24 microseconds covers all layer construction is made. The completed supplement below measures that omitted cost explicitly.

## Mutable-snapshot supplement on final frozen production source

Root authorized an exclusive CPU performance window after correctness/fidelity jobs finished. On frozen production commit `f9099f85`, one fresh micro worker completed successfully; no other tests or benchmarks were launched by this agent afterward. The window was immediately handed back for root's 18 paired processes.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python tests/performance/measure_pr33_cache_ablation.py --worker --scenario micro --output /data/ycfeng/tmp/pr33-w04-mutable-micro-20260916
```

Environment: `/usr/bin/python` 3.12.3, no conda environment. Five samples per operation, 1,000 calls per sample, one process, no profiler. Raw per-sample nanoseconds and query counters are in `/data/ycfeng/tmp/pr33-w04-mutable-micro-20260916/measurement.json`; status is `success` and the worker exited 0.

| Operation | Median microseconds/call | Min | Max |
| --- | ---: | ---: | ---: |
| attention_key | 0.333420 | 0.332871 | 0.449781 |
| attention_payload_copy | 5.217642 | 5.156045 | 5.222533 |
| ep_workload_hit | 2.141942 | 2.119774 | 2.179169 |
| ep_workload_miss_with_clear | 79.277333 | 79.120481 | 79.413649 |
| one_layer_snapshot | 17.412628 | 17.121105 | 17.558048 |
| stage_8_finalized_layers | 5.185594 | 5.169283 | 5.462549 |
| stage_8_mutable_layers | 137.845563 | 136.312447 | 138.416835 |

The observed cost separates already-finalized stage container construction (5.19 microseconds) from eight mutable-source snapshots plus stage construction (137.85 microseconds); one mutable snapshot costs 17.41 microseconds in this fixture. This supports the source-profile attribution that repeated numerical snapshots are materially more expensive than building a stage over finalized records. It does not assign these micro timings directly to Simulator wall time or establish an E2E speedup. The optimized homogeneous producers do not use the eight-mutable-snapshot path; it remains an explicit component-cost control. The final paired run owns the whole-simulator conclusion.

Fixture counters remained explicit: 8 attention queries / 6 hits; 10,001 EP-workload queries / 5,001 materializations. The EP miss benchmark includes its per-iteration cache clear. These counters establish the micro operations exercised and do not claim EP cache use in the real W00/hybrid scenarios, whose zero-call observations above remain unchanged.

## Failure visibility and artifacts

The first coordinator stopped after its fifth worker because the trace header's `meta.timestamp` differs across real process start times. A direct diff found that sole difference in `op_traces.jsonl`; event counts, all request values, and all business rows were exact. The comparator now removes only that header field. `--resume` retained the five immutable worker results and ran the remaining sixteen; no timing samples were discarded. The original failure log remains `/data/ycfeng/tmp/pr33-w04-ablation-coordinator.log`; resumed output is `/data/ycfeng/tmp/pr33-w04-ablation-resume.log`.

Full raw evidence root: `/data/ycfeng/tmp/pr33-w04-ablation-20260916/`. `manifest.json`, `resume_manifest.json`, `source_revision.txt`, and `working.diff` preserve provenance. `results.json` contains all twenty-one full result rows, including initialization/total-process time and RSS for W00. Each attempt retains `measurement.json`, logs, and simulator outputs. The hybrid fixture's outer `fixture_total_s` includes training and output checks; its `sim_wallclock_s` measures only the existing `Simulator.run()` boundary. It does not split hybrid initialization into an additional acceptance metric.

The script remains under `tests/performance/`; no production ablation flag or semantic shortcut was introduced. W11 final baseline pairing and D02 acceptance remain the root task's responsibility.
