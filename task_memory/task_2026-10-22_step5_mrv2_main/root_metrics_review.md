# Measured-root metrics phase review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record parent verification of initial root repair and independent P1 owner finding. |
| 2026-10-22 | Clean pre-fix baseline completes77/77 cases; final candidate/comparison still pending. |
| 2026-10-22 | P1 corrected and independently accepted; parent167 PASS; merged26db654; paired CLI and candidate matrix launched. |

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

## P1: stage-only owner attention projection — RESOLVED

`metrics_store.py:4370-4372` recursively builds the owner ledger. The newly
moved attention projection at4498-4506 also validates that owner's attention
payload even though only stage-owned fields are authoritative there.

The real measured predictor's first/last boundary path applies
`_replace_operator_time_values()` for input_cast/final_residual
(`sklearn_moe_execution_time_predictor.py:1469-1488`). The reconstructed owner
retains root duration but not OperatorCost (`execution_time.py:620-622`). The
authoritative layers retain their proofs. Thus a reachable owner can fail
`Inclusive attention root step5_layer_e2e requires OperatorCost` before its
non-stage fields would be discarded. The reviewer initially confirmed this from
source; the author subsequently reproduced the real public-call-site failure.

Required correction: project only stage-owned fields from the owner. Preserve
strict root validation for authoritative layers and add real public-predictor
first/last results directly entering the ledger as regressions. No fabricated
owner proof, disabled ledger or weakened layer validation.

Implementation follow-up: `ses_e9f9fd0bdffeZH9Q2Ux5NbCYko`, same scoped worktree.
The initial source was not fully accepted and did not launch full CLI or a
candidate matrix. The existing generic trace gap-span limit
and inherited test fixture issues are not new findings.

## Final source acceptance

Code `7c70461`, records `85552ed`, parent merge `26db654`. Parent verified the
entire `frontier/` tree and changed tests/exception document match reviewed code.
The same reviewer checked the complete `61cbac9..7c70461` diff at
`/data/ycfeng/tmp/d23_root_metrics_7c70461.diff`, updated task records, source,
RED logs and parent log/XML. Verdict: **P1 resolved; No findings**.

The owner now contributes only its stage-owned scalar/canonical fields before
aggregation. Field names, millisecond units and rounding are retained; layer
aggregation excludes stage-owned fields. Authoritative layers still require
valid root proof and contribute their own gap. Non-authoritative owner attention
and its99ms test gap are ignored. Real public first/last outputs enter schedule,
operation series, trace and pending ledger with root/boundary each once;
dummy-first is verified at its real Stage ledger.

Primary reran the exact ten-file command above on `step5-root-metrics`, changing
only the output paths to `/data/ycfeng/tmp/d23_root_p1_parent01.{xml,log}`:
**167 passed,1 deselected in46.71s; zero failures/errors**. The exclusion remains
the previously compared inherited cache fixture. The author public-call-site
RED and expanded RED are retained in its test report; no remaining source
review blocker. This acceptance does not establish full CLI or native accuracy.

## Clean baseline execution: PASS

The new clean legacy baseline is `fidelity-baseline-0cdd6cb`. Both sides will
use the unchanged driver from `fidelity-candidate-32fb934`, so harness identity
stays fixed while the owner correction completes. Baseline command:

```bash
# cwd=/data/ycfeng/Frontier/.worktrees/fidelity-candidate-32fb934
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/fidelity-baseline-0cdd6cb --label baseline --output-root /data/ycfeng/tmp/step5_d23_root_fidelity_20261022 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 --clean-cache --continue-on-failure > /data/ycfeng/tmp/d23_root_fidelity_baseline01.log 2>&1
```

Baseline completed: **77/77 cases produced artifacts**, all case return codes0.
The log confirms source0cdd6cb, dirty=False and harness32fb934. Retained log:
`/data/ycfeng/tmp/d23_root_fidelity_baseline01.log`; results:
`/data/ycfeng/tmp/step5_d23_root_fidelity_20261022/baseline/results.jsonl`.
This is baseline execution only. The final candidate was launched from clean
`fidelity-candidate-26db654`, with the same driver/environment and command above
except `--label candidate`, the candidate repo path, and log
`/data/ycfeng/tmp/d23_root_fidelity_candidate01.log`. Comparison is pending.

The stable main worktree was fast-forwarded to26db654 before actual05 launch.
Both modes reuse fixture02/cache at unchanged6144/1024 and2400s child bounds;
see the exact commands and acceptance criteria in the actual-main test report.
