## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-15 | Recorded independent completion-audit commands, exact-SHA evidence, result limits, and hardware boundaries. The filename is retained from the task handoff. |

# PR33 v1.1 Independent Completion-Audit Test Report

## Scope

This report records verification performed while auditing
`Frontier_PR33_review_revision_plan_2026-09-15_v1.1_en.md`. It does not implement
remaining fixes and does not treat prior progress reports as ground truth.

Source candidate:

```text
b8cecf53f8b81ea8380238971277ba94c6fe4961
```

Pinned clean baseline:

```text
0515589ac7f49ac5288a5f55b0ce38b0ede29bb2
```

Remote PR #33 was at `69d09305ca4d881ced4c7d1653ddc9c39f9d9701`, so the
remote PR does not contain the complete local candidate audited here.

## 1. Focused audit test execution

### Command

```bash
env \
PYTHONPATH=/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn:/home/i-fengyicheng/.local/lib/python3.12/site-packages \
WANDB_DISABLED=true \
VIDUR_DISABLE_WANDB=1 \
FRONTIER_LOG_LEVEL=ERROR \
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
-m pytest \
tests/unit/test_hybrid_runtime_family.py \
tests/unit/test_gdn_scheduler_slots.py \
tests/unit/test_gdn_profiler_cpu_increment7.py \
tests/unit/test_device_timer_contract.py \
tests/unit/test_attention_query_cache.py \
tests/unit/test_measurement_family_selector.py \
tests/unit/test_gdn_hybrid_e2e_increment14ab.py \
tests/unit/test_metrics_stage_execution_time.py \
tests/unit/test_dense_execution_time_layer_scaling.py \
-q -p no:cacheprovider \
--basetemp /data/ycfeng/tmp/pr33-completion-audit-20260916-focused-final
```

### Environment and criteria

- Python: `3.13.13` from the command's interpreter.
- CPU-only execution environment; `torch` is not installed in this environment.
- The selected tests cover GDN construction/phase/slots, cache identity, measurement-family paths, synthetic hybrid construction, stage/metrics behavior, and timer contracts.
- A passing CPU/fake-runtime test establishes the tested Python contract only. It does not establish CUDA, ROCm, AITER, RCCL, HIP graph, NVIDIA, or AMD behavior.

### Result

**PASS — 69 passed in 12.26s.**

This is focused evidence for existing branches. It does not close any R item by
itself because several plan acceptance conditions require real production
configuration paths, Simulator lifecycle coverage, non-dummy/golden artifacts,
resource measurements, or hardware execution.

## 2. R11 clean baseline/candidate matrix evidence

### Command retained from the exact-HEAD run

```bash
PYTHONPATH=$PWD \
WANDB_DISABLED=true \
VIDUR_DISABLE_WANDB=1 \
FRONTIER_TMP_ROOT=/data/ycfeng/tmp \
/usr/bin/python tests/integration/run_scheduler_refactor_fidelity.py \
  --baseline /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915 \
  --candidate /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn \
  --output /data/ycfeng/tmp/pr33-r11-fidelity-20260915-b8cecf53 \
  --workers 8
```

### Result and limits

**58 passed, 0 failed.** The manifest and results record baseline
`0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`, candidate `b8cecf53`, original
comparison tolerances `rel_tol=1e-12` and `abs_tol=1e-9`, matching request
counts, key/order checks, IDs, and discrete-event checks.

The result is not the complete R11 gate:

- `operation_metrics.csv` count in the matrix artifact is `0`.
- `op_traces.jsonl` count in the matrix artifact is `0`.
- The retained 197-pass “non-dummy/golden” group is a unit-test group, not the
  required real non-dummy/golden E2E/operator matrix.
- A full auditable baseline-vs-candidate unit failure-node/cause manifest is
  still required; equal reported failure counts do not prove equivalence.

## 3. R12 performance evidence

### Historical observation being audited

The retained historical exact workload reported:

```text
baseline Simulator.run(): 0.009217162 s
old candidate Simulator.run(): 0.086823318 s
ratio: 9.419745x
requests/events: 2 / 104
```

This is historical evidence, not a new run at the final candidate.

### Final exact-workload comparison

Artifact:

```text
/data/ycfeng/tmp/pr33-completion-audit-exact104-restored-env-b8cecf53
```

Five paired unprofiled runs completed successfully on both revisions. Each side
processed 2 requests and 104 events with the same case fingerprint and runner
SHA256 (`10209894e2cc66fc0efa929ce39a06014aa23130b165b11ac002a6db393ccc6c`).
Median values were:

| Metric | Main | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| `Simulator.run()` | `0.008927742 s` | `0.012872720 s` | `1.441878584x` |
| `total_proc_s` | `2.190006329 s` | `2.170492185 s` | `0.991089458x` |
| `init_s` | `2.178224750 s` | `2.155132932 s` | `0.989398790x` |

The historical approximately 9.42x result therefore does not reproduce under
the fixed final comparison, but the run-phase residual is still material.

### Broader final paired workloads

Artifact:

```text
/data/ycfeng/tmp/pr33-r12-paired-20260915-final-b8cecf53
```

All 18/18 runs succeeded. Median `sim_wallclock_s` ratios were:

| Workload | Ratio |
| --- | ---: |
| small dense | `1.504303x` |
| longer dense | `1.407252x` |
| representative MoE | `1.011424x` |

These measurements are evidence of current behavior, not an approved budget.

### Hotspot evidence and limitation

Independent profiles:

```text
/data/ycfeng/tmp/pr33-completion-audit-exact104-cprofile-b8cecf53
```

Approximate cumulative `Simulator.run()` profile time was `0.021 s` on main
and `0.032 s` on the candidate. The candidate profile includes approximately
800 additional `ExecutionTime.as_single_layer()` calls (about `0.007 s`
cumulative) and 20 `StageExecutionTime.from_execution_time()` calls (about
`0.010 s` cumulative); disaggregation stage prediction was approximately
`0.005 s` on main versus `0.015 s` on candidate.

This supports a hotspot direction around per-layer expansion/record
construction. It is not a causal ablation: no allocation/RSS measurement or
narrow change-and-remeasure proof isolates the removable share. The repeated
stage aggregation overhead was reduced in `b8cecf53`, but the remaining dense
residual is still open under D02.

## 4. CPU unit and broad-group evidence retained from the prior session

The exact-HEAD retained full-unit log reports:

```text
3309 passed, 19 failed, 25 skipped, 576 warnings in 76.98s
```

The reported 19 failure node IDs and causes match the baseline inventory, with
no candidate-only failure claimed. The broad corrected groups report 1234
passes and 19 skips across attention, GDN, scheduler/memory, stage/metrics,
MoE/parallel, timer/measurement, SGLang, and non-dummy/golden test groups.

These are retained prior-session results inspected during this audit, not a
replacement for the 69-test focused command above and not proof that every
R11/R09 lane passed. The initial two broad commands referenced nonexistent test
paths and were corrected; those collection errors are not test failures.

## 5. Hardware and dependency boundary

Observed hardware evidence:

- NVIDIA: `nvidia-smi -L` returned no device rows and `/dev/nvidia*` is absent.
  Required status: **SKIP: no visible NVIDIA device**.
- AMD/MI355X/ROCm/AITER/RCCL/HIP graph: no compatible worker/device was
  available. Required status: **SKIP: AMD/MI355X hardware unavailable**.

The CPU environment's Python inventory reported Torch `2.5.1+cu124` in the
broader prior-session environment, while `vllm`, `sglang`, `aiter`, and
`flashinfer` were unavailable there. The focused audit interpreter itself had
no `torch`. No CPU fake-torch result is promoted to GPU PASS.

## 6. Final acceptance interpretation

- Focused CPU contracts: PASS for the exact command and scope above.
- 58-case dummy/fidelity comparator: PASS, but insufficient for complete R11.
- Historical 9.42x regression: not reproduced at the final exact workload.
- Current dense run-phase residual and RCA: OPEN; no D02 acceptance.
- Real hybrid lifecycle, operator ownership seam, full non-dummy/golden lanes,
  R13 cleanup analysis, and GPU test-entry/timer evidence: OPEN.
- No source/test files were changed by this audit. Documentation-only edits
  after `b8cecf53` do not imply a source rerun.
