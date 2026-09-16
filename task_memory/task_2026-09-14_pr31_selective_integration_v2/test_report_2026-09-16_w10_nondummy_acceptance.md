## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Final c9f8f904 rerun: 8 PASS in 35.43 s; 106 artifacts and 7 baseline metric pairs PASS. |
| 2026-09-16 | Latest-source rerun: 8 PASS in 37.57 s; 106 prior-final artifacts PASS under the established comparator. |
| 2026-09-16 | Recorded the final eight-case non-dummy CPU campaign, seven fixed-baseline comparisons, independent numerical oracles, and explicit evidence limits. |

# W10 synthetic non-dummy acceptance test report

**Latest PASS: 8 tests in 35.43 s** on `c9f8f904e3550c11aad3cc5d851d75d648cef6e1`; earlier 37.57 s and 36.78 s runs are preserved below. **PASS: 7/7 common request and 7/7 common system artifacts** under the existing comparator. This is synthetic-profile correctness, not production-data or native GPU timing parity.

## Execution and environment

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.
- Baseline: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915`, verified `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`.
- Python `/usr/bin/python` 3.12.3; no conda environment. CPU only; each simulator subprocess sets OMP and OpenBLAS threads to one and uses its isolated case directory as TMPDIR.
- Script: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/integration/test_pr33_nondummy_acceptance.py`.
- Candidate is the shared current worktree after the parent's W08 and predictor freeze, not an asserted final commit SHA. The parent owns the final source revision audit.

Exact final command, from candidate worktree:

```bash
python -m pytest tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-w10-nondummy-final > /data/ycfeng/tmp/pr33-w10-nondummy-final.log 2>&1
```

Observed output:

```text
........                                                                 [100%]
8 passed in 36.78s
```

Baseline controls used the same absolute script path with the baseline PYTHONPATH, without `--verify` because historical output ownership intentionally differs. The following command reproduces the seven controls from the baseline worktree; output directories must be absent before running:

```bash
for case_index in 0 1 2 3 4 5 6; do
    case "$case_index" in
        0) case_name=dense-coloc ;;
        3) case_name=mla ;;
        *) case_name=case$case_index ;;
    esac
    PYTHONPATH=/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp python /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/integration/test_pr33_nondummy_acceptance.py --case "$case_index" --output "/data/ycfeng/tmp/pr33-w10-baseline-$case_name" > "/data/ycfeng/tmp/pr33-w10-baseline-$case_name.log" 2>&1
done
```

All seven baseline processes returned zero and their `invocation.json` confirms the imported simulator path is the baseline worktree.

Exact comparison recipe from the candidate worktree (reuses the established comparator, `rel_tol=1e-12`, `abs_tol=1e-9`):

```bash
python - <<'PYCOMPARE'
from pathlib import Path
import json
from tests.integration.test_pr33_nondummy_acceptance import compare_baseline
results = {}
for i in range(7):
    name = 'dense-coloc' if i == 0 else 'mla' if i == 3 else f'case{i}'
    results[str(i)] = compare_baseline(
        Path('/data/ycfeng/tmp') / f'pr33-w10-baseline-{name}',
        Path('/data/ycfeng/tmp/pr33-w10-nondummy-final') / f'test_nondummy_simulator_accept{i}' / 'run',
    )
    assert results[str(i)]['request_metrics.csv']['status'] == 'PASS'
    assert results[str(i)]['system_metrics.json']['status'] == 'PASS'
Path('/data/ycfeng/tmp/pr33-w10-comparison-final.json').write_text(json.dumps(results, indent=2) + '\n')
print('Baseline comparison: 7/7 request metrics PASS; 7/7 system metrics PASS')
PYCOMPARE
```

## Criteria and independent numerical evidence

Every case constructs the real Simulator, loads/trains non-dummy predictors, executes requests, exports nonempty request/system/operation/trace/stage outputs, and completes two requests with 18 total tokens each. The oracle uses explicit physical layers, EP widths, architecture passes, and literal synthetic timing targets rather than comparing the predictor against its own aggregation. See `w10_nondummy_report.md` for the complete eight-case matrix and per-case cardinalities.

| Numerical check | Expected | Observed candidate | Absolute error | Relative error |
| --- | --- | --- | --- | --- |
| Family-native expanded layer operator | 0.01 ms | 0.01 ms | 0 | 0 |
| Four-layer dense/MLA co-location operation total | 0.04 ms | 0.04 ms | 0 | 0 |
| Nonempty EP lane grouped GEMM | 0.12 ms | 0.12 ms | 0 | 0 |
| Hybrid per-layer GDN prefill core | 0.22 ms | 0.22 ms within floating representation | <1e-15 ms | <1e-14 |
| Hybrid per-layer GDN decode core | 0.05 ms | 0.05 ms | 0 | 0 |
| Seven homogeneous request E2E vectors | Baseline values retained in campaign report | Same vectors | 0 | 0 |

Exact trace membership is checked for every physical layer, including aggregate membership for layers outside the per-layer quota. Hybrid GDN layers are `{0,1,2,4,5,6}`; dense layers are `{3,7}`. Both hybrid requests complete with slot capacity one, no active owners, empty allocation map, and zero allocated KV blocks.

## Failures retained and resolution

- Initial harness errors concerned valid fixture construction: process-global model isolation, required PD-AF microbatch/Orca fields, profile metadata and KERNEL_ONLY datasets, and namespace-package import provenance. They were fixed in the harness, not suppressed by skips or fake constructors.
- Concurrent intermediate production revisions failed on `on_prefill_sync(metrics_store=...)` and absent `MetricsStore.ep_wave_reporting_enabled`; reported to the parent and resolved by its W08 integration. Final complete runs passed.
- A strengthened provisional lane count failed PDD/PD-AF. Corrected the oracle to established architecture pass/ledger ownership: PDD24, PD-AF16 new PREFILL lanes plus16 existing FFN lanes. The ninth run passed8/36.50s; the freeze-following final run passed8/36.78s.
- No final failures, skips, missing requested artifacts, or unexplained request/system differences remain within these eight cases.

## Stable baseline differences

The following table preserves every final non-PASS artifact result. PASS request/system artifacts are asserted separately above. Operation schema removals concern inactive-family keys; common-column comparisons remain visible. Full raw comparison is retained at `/data/ycfeng/tmp/pr33-w10-comparison-final.json`.

| Case | Artifact | Status | First difference / common-column comparison |
| --- | --- | --- | --- |
| 0 | `monolithic_operation_metrics.csv` | DIFFERENCE | root[0]: fields differ; shared: root[0].attn_kv_cache_save: '0.16' != '0.04' |
| 0 | `op_traces.jsonl` | DIFFERENCE | root[0].meta: fields differ |
| 1 | `decode_operation_metrics.csv` | DIFFERENCE | root[0]: fields differ; shared: root[0].attn_decode: '0.8' != '0.2' |
| 1 | `op_traces.jsonl` | DIFFERENCE | root[0].meta: fields differ |
| 1 | `prefill_operation_metrics.csv` | DIFFERENCE | root[0]: fields differ; shared: root[0].attn_kv_cache_save: '0.16' != '0.04' |
| 2 | `decode_attn_operation_metrics.csv` | DIFFERENCE | root[0]: fields differ; shared: PASS |
| 2 | `decode_ffn_operation_metrics.csv` | DIFFERENCE | root[0]: fields differ; shared: PASS |
| 2 | `op_traces.jsonl` | DIFFERENCE | root[0].meta: fields differ |
| 2 | `prefill_operation_metrics.csv` | DIFFERENCE | root[0]: fields differ; shared: root[0].attn_kv_cache_save: '0.16' != '0.04' |
| 3 | `monolithic_operation_metrics.csv` | DIFFERENCE | root[0]: fields differ; shared: root[0].attn_mla_kv_cache_save: '0.08' != '0.04' |
| 3 | `op_traces.jsonl` | DIFFERENCE | root: lengths 78 != 92 |
| 4 | `frontier_ep_wave_lane_ledger.jsonl` | ADDED_ARTIFACT | New actual EP schedule artifact |
| 4 | `frontier_stage_batch_ledger.jsonl` | DIFFERENCE | root[0].execution_time.component_ledger_ms.attention_all_reduce_time: 0.002109227 != 0.008436908 |
| 4 | `monolithic_operation_metrics.csv` | DIFFERENCE | root: lengths 4 != 36; shared: root: lengths 4 != 36 |
| 4 | `op_traces.jsonl` | DIFFERENCE | root: lengths 28 != 320 |
| 5 | `decode_operation_metrics.csv` | DIFFERENCE | root: lengths 4 != 20; shared: root: lengths 4 != 20 |
| 5 | `frontier_ep_wave_lane_ledger.jsonl` | ADDED_ARTIFACT | New actual EP schedule artifact |
| 5 | `frontier_stage_batch_ledger.jsonl` | DIFFERENCE | root[0].execution_time.component_ledger_ms.attention_kv_cache_save_execution_time: 0.01 != 0.04 |
| 5 | `op_traces.jsonl` | DIFFERENCE | root: lengths 74 != 266 |
| 5 | `prefill_operation_metrics.csv` | DIFFERENCE | root: lengths 2 != 10; shared: root: lengths 2 != 10 |
| 6 | `decode_attn_operation_metrics.csv` | DIFFERENCE | root[0]: fields differ; shared: PASS |
| 6 | `decode_ffn_operation_metrics.csv` | DIFFERENCE | root[0]: fields differ; shared: PASS |
| 6 | `frontier_ep_wave_lane_ledger.jsonl` | ADDED_ARTIFACT | New actual EP schedule artifact |
| 6 | `frontier_stage_batch_ledger.jsonl` | DIFFERENCE | root[0].execution_time.component_ledger_ms.attention_all_reduce_time: 0.002109227 != 0.008436908 |
| 6 | `op_traces.jsonl` | DIFFERENCE | root: lengths 152 != 306 |
| 6 | `prefill_operation_metrics.csv` | DIFFERENCE | root: lengths 2 != 18; shared: root: lengths 2 != 18 |

Layer ownership corrections have independent arithmetic support under W03/W08. D01 accepted only its two isolated corrections; the parent must classify these additional reporting differences in the final decision audit without inferring blanket authorization. Added EP rows expose actual lane work. Trace differences carry family metadata and valid physical-layer expansion. No historical golden values were rewritten to hide them.

## Practical limits and remaining ownership

Synthetic CSV targets and labels do not prove native BF16 timing, GPU/AMD execution, production-data prediction accuracy, or vLLM parity. Generic MLA norm/model-projection names are absent from both baseline and candidate traces; this case validates its registered native attention/cache operators, not coverage of those absent scopes. This was reported to the parent without an unrelated production change.

The parent owns the full CPU suite, 58-case fidelity control, final clean-source/commit audit, and any W10 closure statement. This agent changed only its assigned test and task reports for this sub-step, with no production edits and no commit.

## Latest-source final rerun

After the parent changed homogeneous disaggregation to predict/finalize once, removed discarded ExecutionTime copies, and added direct-query GDN missing-artifact fail-fast, the entire eight-case campaign was rerun on that source. **PASS: 8 tests in 37.57 s**, no skips. This supersedes the earlier 36.78 s run as the latest-source execution evidence.

Exact command:

```bash
python -m pytest tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-w10-nondummy-final-2 > /data/ycfeng/tmp/pr33-w10-nondummy-final-2.log 2>&1
```

All eight new case outputs were compared with the prior final run using the same established comparator and trace-header normalization. **PASS: 106 artifacts**, per-case counts `[10,14,17,10,11,15,18,11]`. Selected artifact names are exactly identical. The comparison includes all CSV/JSONL outputs, system metrics, stage-ledger summaries, and exact acceptance cardinalities. No new numerical/discrete differences occurred. Previous baseline differences remain explicitly documented; this comparison does not change their decision status.

Evidence: `/data/ycfeng/tmp/pr33-w10-nondummy-final-2.log`, `/data/ycfeng/tmp/pr33-w10-final-2-vs-final.json`, and case roots under `/data/ycfeng/tmp/pr33-w10-nondummy-final-2`. No production or test code changed during this rerun.

Reproducible prior-final comparison command, from the candidate worktree:

```bash
python - <<'PYCOMPARE'
from pathlib import Path
import json
from tests.integration.test_pr33_nondummy_acceptance import compare_baseline
from tests.integration.run_scheduler_refactor_fidelity import compare, load_artifact
results = {}
for i in range(8):
    suffix = Path(f'test_nondummy_simulator_accept{i}/run')
    prior = Path('/data/ycfeng/tmp/pr33-w10-nondummy-final') / suffix
    current = Path('/data/ycfeng/tmp/pr33-w10-nondummy-final-2') / suffix
    results[str(i)] = compare_baseline(prior, current)
    old_metrics = next(prior.rglob('request_metrics.csv')).parent
    new_metrics = next(current.rglob('request_metrics.csv')).parent
    selected = lambda p: {x.name for x in p.iterdir() if x.suffix in {'.csv', '.jsonl'} or x.name == 'system_metrics.json'}
    assert selected(old_metrics) == selected(new_metrics)
    compare(load_artifact(prior / 'acceptance_evidence.json'), load_artifact(current / 'acceptance_evidence.json'))
    results[str(i)]['acceptance_evidence.json'] = {'status': 'PASS'}
    for name in ('frontier_stage_batch_ledger_summary.json',):
        compare(load_artifact(old_metrics / name), load_artifact(new_metrics / name))
        results[str(i)][name] = {'status': 'PASS'}
    assert all(row['status'] == 'PASS' for row in results[str(i)].values()), results[str(i)]
Path('/data/ycfeng/tmp/pr33-w10-final-2-vs-final.json').write_text(json.dumps(results, indent=2) + '\n')
print('8/8 cases PASS; all selected artifact names identical; per-case artifact counts:', [len(v) for v in results.values()])
print('Total artifacts compared:', sum(map(len, results.values())))
PYCOMPARE
```

## Final revision c9f8f904 verification

**Latest-source PASS: 8 tests in 35.43 s** on commit `c9f8f904e3550c11aad3cc5d851d75d648cef6e1`. This revision reuses a homogeneous sum only when the finalized timing payload is shared safely. This run supersedes final-2 as the latest-source evidence; all previous observations are preserved above.

Environment remains `/usr/bin/python` 3.12.3, no conda environment, CPU only; isolated simulator processes use OMP/OpenBLAS threads one and TMPDIR under the case root. The exact executed command was:

```bash
python -m pytest tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-w10-nondummy-final-3 > /data/ycfeng/tmp/pr33-w10-nondummy-final-3.log 2>&1
```

Observed: `8 passed in 35.43s`, no skips or collection changes. Compared all **106 artifacts** with final-2 using the established comparator (`rel_tol=1e-12`, `abs_tol=1e-9`) and existing trace-header normalization: **106/106 PASS**, artifact-name sets identical, per-case counts `[10,14,17,10,11,15,18,11]`. Rechecked baseline `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2` request and system metrics for the seven homogeneous cases: **7/7 request PASS, 7/7 system PASS**. The independent physical-layer, lane-count, timing, and capacity assertions all passed. There is no new numerical/discrete difference to classify.

Evidence: `/data/ycfeng/tmp/pr33-w10-nondummy-final-3.log`, `/data/ycfeng/tmp/pr33-w10-final-3-vs-final-2.json`, `/data/ycfeng/tmp/pr33-w10-final-3-baseline-metrics.json`, and isolated output roots below `/data/ycfeng/tmp/pr33-w10-nondummy-final-3`. No production or test edits accompanied this rerun. CPU correctness and comparison work is complete; this agent holds further CPU executions for the parent's exclusive paired window.

Exact final-3 comparison recipe from the candidate worktree:

```bash
python - <<'PYCOMPARE'
from pathlib import Path
import json, os, sys
from tests.integration.test_pr33_nondummy_acceptance import compare_baseline
from tests.integration.run_scheduler_refactor_fidelity import compare, load_artifact
results = {}; baseline_results = {}
for i in range(8):
    suffix = Path(f'test_nondummy_simulator_accept{i}/run')
    prior = Path('/data/ycfeng/tmp/pr33-w10-nondummy-final-2') / suffix
    current = Path('/data/ycfeng/tmp/pr33-w10-nondummy-final-3') / suffix
    results[str(i)] = compare_baseline(prior, current)
    old_metrics = next(prior.rglob('request_metrics.csv')).parent
    new_metrics = next(current.rglob('request_metrics.csv')).parent
    selected = lambda p: {x.name for x in p.iterdir() if x.suffix in {'.csv', '.jsonl'} or x.name == 'system_metrics.json'}
    assert selected(old_metrics) == selected(new_metrics)
    compare(load_artifact(prior / 'acceptance_evidence.json'), load_artifact(current / 'acceptance_evidence.json'))
    results[str(i)]['acceptance_evidence.json'] = {'status': 'PASS'}
    name = 'frontier_stage_batch_ledger_summary.json'
    compare(load_artifact(old_metrics / name), load_artifact(new_metrics / name))
    results[str(i)][name] = {'status': 'PASS'}
    assert all(row['status'] == 'PASS' for row in results[str(i)].values()), results[str(i)]
    if i < 7:
        base_name = 'dense-coloc' if i == 0 else 'mla' if i == 3 else f'case{i}'
        baseline = next((Path('/data/ycfeng/tmp') / f'pr33-w10-baseline-{base_name}').rglob('request_metrics.csv')).parent
        baseline_results[str(i)] = {}
        for name in ('request_metrics.csv', 'system_metrics.json'):
            compare(load_artifact(baseline / name), load_artifact(new_metrics / name))
            baseline_results[str(i)][name] = {'status': 'PASS'}
Path('/data/ycfeng/tmp/pr33-w10-final-3-vs-final-2.json').write_text(json.dumps(results, indent=2) + '\n')
Path('/data/ycfeng/tmp/pr33-w10-final-3-baseline-metrics.json').write_text(json.dumps(baseline_results, indent=2) + '\n')
print('Python:', sys.executable, sys.version.split()[0], 'Conda:', os.environ.get('CONDA_DEFAULT_ENV', 'none'))
print('Prior-final comparison: 8/8 cases PASS; artifacts:', sum(map(len, results.values())))
print('Per-case artifact counts:', [len(v) for v in results.values()])
print('Baseline comparison: 7/7 request metrics PASS; 7/7 system metrics PASS')
PYCOMPARE
```
