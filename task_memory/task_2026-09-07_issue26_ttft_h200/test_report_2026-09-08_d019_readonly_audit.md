## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified D019 operator inventory and route example arithmetic using existing artifacts; no new performance test executed. |

# D019 read-only artifact verification

Execution directory: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907.
Conda: dev-vidur-v03-hopper-e2e; Python3.13.13. No GPU, profiler, simulator, model training or CPU-overhead suite was executed in this plan-review turn.

Criteria: all15Frontier non-comm trace labels are represented exactly once; numeric event gaps reproduce the displayed rounded percentages; all38recorded device families are inventoried, including32comp/mem and3221activities; the common-clock route/enqueue example preserves the actual measured inversion. These checks detect omitted aliases/rows, invalid gap arithmetic and a misleading timeline. They do not validate numerical model accuracy.

Exact successful command:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import csv, json, math, sys
from pathlib import Path
p = Path('task_memory/task_2026-09-07_issue26_ttft_h200/analysis')
a = p / 'first-batch-op-rca'
with (a/'coverage_audit_d019_comp_mapping.csv').open() as f:
    mapping = list(csv.DictReader(f))
with (a/'coverage_audit_d019_kernel_inventory.csv').open() as f:
    inventory = list(csv.DictReader(f))
pred = json.loads((a/'frontier_summary.json').read_text())['op_totals_ms']
comm = {'attn_tp_allreduce', 'moe_tp_allreduce', 'share_expert_tp_allreduce'}
assert len(mapping) == 15 and {r['frontier_op'] for r in mapping} == set(pred)-comm
assert len({r['frontier_op'] for r in mapping}) == 15
for r in mapping:
    if r['event_gap_pct'].endswith('%'):
        v = float(r['vllm_event_ms'])
        assert math.isfinite(v) and v > 0
        gap = 100*(float(r['frontier_ms'])-v)/v
        assert abs(gap-float(r['event_gap_pct'].rstrip('%'))) <= 0.00051
assert len(inventory) == len({r['family_id'] for r in inventory}) == 38
assert sum(r['category']=='computation_memory' for r in inventory) == 32
assert sum(int(r['count']) for r in inventory) == 3221
assert all(math.isfinite(float(r['vllm_kernel_ms'])) and float(r['vllm_kernel_ms']) > 0 for r in inventory)
e = json.loads((p/'dp-workflow-rca/fresh_same_run_evidence.json').read_text())
pair = e['first_pair']; a5,a6 = e['first_formal_routes'][5:7]
enqueue6 = pair['route_6_minus_5_ms'] + a6['route_to_enqueue_ms']
enqueue5 = a5['route_to_enqueue_ms']
assert 0 < pair['route_6_minus_5_ms'] < enqueue6 < enqueue5
assert abs((enqueue5-enqueue6)-pair['enqueue_5_minus_6_ms']) < 1e-9
print(json.dumps({'status':'PASS_READ_ONLY_ARTIFACT_CHECK','python':sys.version.split()[0], 'frontier_comp_labels':len(mapping),'device_families':len(inventory),'comp_memory_families':32,'device_activities':3221,'pair_common_origin_ms':{'route5':0,'route6':pair['route_6_minus_5_ms'],'enqueue6':enqueue6,'enqueue5':enqueue5},'limits':'Inventory and arithmetic validation only; no new measurements, prediction calls or calibration acceptance.'}, indent=2))
PY
```

Observed exit code: 0.

```json
{
  "status": "PASS_READ_ONLY_ARTIFACT_CHECK",
  "python": "3.13.13",
  "frontier_comp_labels": 15,
  "device_families": 38,
  "comp_memory_families": 32,
  "device_activities": 3221,
  "pair_common_origin_ms": {
    "route5": 0,
    "route6": 2.023212146013975,
    "enqueue6": 18.767940811812878,
    "enqueue5": 18.81647016853094
  },
  "limits": "Inventory and arithmetic validation only; no new measurements, prediction calls or calibration acceptance."
}
```

An initial version of this check used `vllm_event_ms != 'missing'` to identify numeric rows. It failed with `ValueError: could not convert string to float: 'missing independent non-additive scope'` on a correctly annotated fused residual row. The corrected check only recalculates rows with a numeric percentage, preserving missing/alias/fused rows without zero substitution. No evidence table was changed to make the check pass.

Full source/coverage and CPU component review records: analysis/first-batch-op-rca/coverage_audit_d019.md, analysis/profile_coverage_plan_d019.md, analysis/cpu_overhead_reuse_d019.md. These include source paths, existing sample values and unexecuted proposals. The canonical next plan is plan.md/D019; implementation awaits YC review.

