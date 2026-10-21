# Measured-root metrics phase review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record parent verification of initial root repair and independent P1 owner finding. |

## Scope and evidence

Reviewer: `ses_e9f8b6cffffeSZiBfBvZkmXBQa`; task `D23-ROOT-METRICS`.
Target `3a189ba`, baseline `61cbac9`; full diff
`/data/ycfeng/tmp/d23_root_metrics_3a189ba.diff`. The reviewer inspected the
task's requirements/plan/progress/report/summary, D23 authority, originating
CLI blocker and related predictor/entity/metrics/trace code. Read-only review;
the reviewer did not execute tests.

Primary independently ran161 tests successfully,1 explicitly deselected, in
25.61s. XML/log: `/data/ycfeng/tmp/d23_root_parent01.{xml,log}`. Exact command,
cwd=`/data/ycfeng/Frontier/.worktrees/step5-root-metrics`:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_measured_attention_root_metrics.py tests/unit/test_attention_trace_mapping.py tests/unit/test_metrics_stage_execution_time.py tests/unit/test_stage_execution_time.py tests/unit/test_execution_time_op_times.py tests/unit/test_op_trace_utils.py tests/unit/test_step5_measured_cost.py tests/unit/test_step5_measured_replay.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_measured_public_predictor.py -q --tb=short -p no:cacheprovider -k 'not structured_attention_legacy_coverage_uses_family_execution_time_attrs' --junitxml=/data/ycfeng/tmp/d23_root_parent01.xml > /data/ycfeng/tmp/d23_root_parent01.log 2>&1
```

The deselected
cache-order test was run separately on pre-fix and candidate trees; both have
the identical testcase/error `assert0.0 ==1.7`. Comparison evidence:
`/data/ycfeng/tmp/d23_root_cache_{baseline,candidate}01.{xml,log}`. The old GLM
fixture issue is recorded in the author report; no global all-tests PASS.

Primary merged initial repair at `f237362`; all seven source/test/exception
files match `3a189ba`. The original minimal root/Stage repro is GREEN:
root=.6244ms, ledger sum=.6244ms, runtime=.6244ms and23 kernels retained;
`/data/ycfeng/tmp/d23_ledger_parent_green01.log`.

## P1: stage-only owner attention projection — OPEN

`metrics_store.py:4370-4372` recursively builds the owner ledger. The newly
moved attention projection at4498-4506 also validates that owner's attention
payload even though only stage-owned fields are authoritative there.

The real measured predictor's first/last boundary path applies
`_replace_operator_time_values()` for input_cast/final_residual
(`sklearn_moe_execution_time_predictor.py:1469-1488`). The reconstructed owner
retains root duration but not OperatorCost (`execution_time.py:620-622`). The
authoritative layers retain their proofs. Thus a reachable owner can fail
`Inclusive attention root step5_layer_e2e requires OperatorCost` before its
non-stage fields would be discarded. This is source-confirmed by the reviewer;
public-call-site RED is assigned to the implementation owner, not yet claimed.

Required correction: project only stage-owned fields from the owner. Preserve
strict root validation for authoritative layers and add real public-predictor
first/last results directly entering the ledger as regressions. No fabricated
owner proof, disabled ledger or weakened layer validation.

Implementation follow-up: `ses_e9f9fd0bdffeZH9Q2Ux5NbCYko`, same scoped worktree.
Source phase is not fully accepted. Full CLI and candidate legacy matrix have
not started on this initial source. The existing generic trace gap-span limit
and inherited test fixture issues are not new findings.

## Independent work in progress

The new clean legacy baseline is `fidelity-baseline-0cdd6cb`. Both sides will
use the unchanged driver from `fidelity-candidate-32fb934`, so harness identity
stays fixed while the owner correction completes. Baseline command:

```bash
# cwd=/data/ycfeng/Frontier/.worktrees/fidelity-candidate-32fb934
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/fidelity-baseline-0cdd6cb --label baseline --output-root /data/ycfeng/tmp/step5_d23_root_fidelity_20261022 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 --clean-cache --continue-on-failure > /data/ycfeng/tmp/d23_root_fidelity_baseline01.log 2>&1
```

Baseline is in progress; candidate/comparison wait for the final repair.
Main worktree is prepared at
f237362 but no new actual subprocess has launched; it will be fast-forwarded
again before the unchanged-bound full6144/1024 verification.
