# Request-free sequential event logging

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Reproduce the actual-main logging blocker and verify the caller-precondition repair. |

## Scope and cause

D20 actual-main run08 reported `AssertionError: Batch has no requests` from
`Simulator._extract_sequential_event_details`. Its AF layer lookup evaluates
`Batch.af_inflight_layer_count`, which requires nonempty requests.

Source inspection distinguishes the possible causes:

1. The logger applies a request-owned getter to request-free work: confirmed.
2. Every such event should carry a layer ID: disproved by the real
   `dp_dummy_forward.advance_dummy_forward` stage-end constructor. Sync events
   carry an explicit layer ID; stage completion does not.
3. The empty request list is invalid: disproved by the existing DP dummy-forward
   constructor and lifecycle, which intentionally carry no request.

The repair guards only this request-owned lookup. It preserves actual event or
transfer layer IDs, omits unknown layer IDs, and leaves AF consistency assertions
active for request-owned batches. It changes no simulated cost or scheduling.
Both inspected production modules are below the documented 2000-line limit
(`simulator.py`: 1481, `entities/batch.py`: 1580; only the former is modified).

## Environment and commands

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-preview`.
Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6,
GCC 11.3.0, Linux. Shared environment:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp
PYTHON=/data/ycfeng/envs/frontier-py310/bin/python
```

Reproduction:

```bash
"$PYTHON" -m pytest tests/unit/test_simulator_event_logging.py \
  -q --tb=short -p no:cacheprovider \
  > /data/ycfeng/tmp/d21_event_logging_red02.log 2>&1
```

Expected RED: actual P/D request-free stage-end events fail at the AF getter;
explicit-layer sync events succeed. Observed: **2 FAIL / 2 PASS, 1.74 s**, exact
`Batch has no requests` at `simulator.py:787` -> `batch.py:723`.
The earlier red01 used the same extractor with a constructed schedule-event
carrier; red02 is the retained real-lifecycle regression seam.

After repair:

```bash
"$PYTHON" -m pytest tests/unit/test_simulator_event_logging.py \
  tests/unit/test_cluster_logger_cache.py \
  tests/unit/test_cluster_simulator_parallel_ordering.py \
  tests/unit/test_simulator_transfer_predictor_lifecycle.py \
  -q --tb=short -p no:cacheprovider \
  --junitxml=/data/ycfeng/tmp/d21_event_logging_green01.xml \
  > /data/ycfeng/tmp/d21_event_logging_green01.log 2>&1
```

Expected: empty request identity and actual token count preserved, explicit layer
IDs unchanged, unknown layer IDs omitted, adjacent lifecycle/order tests pass.
Observed: **36 PASS, 2.46 s**.

Required legacy matrix:

```bash
env PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode \
  "$PYTHON" -B tests/e2e/refactor_fidelity/run_matrix.py run \
  --repo-root /data/ycfeng/Frontier/.worktrees/step5-preview \
  --label candidate --output-root /data/ycfeng/tmp/step5_event_logging_fidelity_20261021 \
  --python-bin "$PYTHON" --jobs 4 --clean-cache --continue-on-failure \
  > /data/ycfeng/tmp/step5_event_logging_fidelity_20261021.log 2>&1
```

Observed: **77/77 cases ran successfully; 77 compared artifact sets are
identical; zero differing predictor cache names, missing cases, new failures,
or definition changes**. Baseline is the already completed
`/data/ycfeng/tmp/step5_d20_final_fidelity_20261021/candidate`.

```bash
"$PYTHON" -B tests/e2e/refactor_fidelity/run_matrix.py compare \
  --output-root /data/ycfeng/tmp/step5_event_logging_fidelity_20261021 \
  --baseline-label /data/ycfeng/tmp/step5_d20_final_fidelity_20261021/candidate \
  --candidate-label /data/ycfeng/tmp/step5_event_logging_fidelity_20261021/candidate \
  > /data/ycfeng/tmp/step5_event_logging_fidelity_compare01.log 2>&1
```

The strict comparator exits1 solely because both inputs recorded dirty working
trees. Preserve that failure: artifact identity does not fix commit provenance.
The baseline revisions bc8ae27 and5af7638 have no diff under `frontier`, `tests`
or `examples`, but this does not rewrite the recorded dirty flags. Create clean
detached before/after checkouts of the committed source and repeat the matrix
there before declaring strict fidelity acceptance. This necessary rerun
addresses the remaining source-provenance failure; it is not another numerical
or quality retry.

### Clean-commit completion

The source repair is committed as `b862e57906862e1e21aa6b21e8f862368ed96923`.
Detached checkouts were verified clean before running:

- Baseline: `5af7638ddb73374c2ba2ca055687dd36a10dc98c` at
  `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-5af7638`.
- Candidate: `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-b862e57`.

Both executed the existing candidate-commit `run_matrix.py` with `--jobs 4
--clean-cache --continue-on-failure`, the same Python3.10 environment and
`--output-root /data/ycfeng/tmp/step5_event_logging_clean_fidelity_20261021`.
Each used its own `--repo-root` and `--label baseline` or `--label candidate`;
`PYTHONPATH` was the clean candidate checkout. They ran concurrently in separate
directories. The unchanged comparator then ran:

```bash
PYTHONPATH=/data/ycfeng/Frontier/.worktrees/fidelity-candidate-b862e57 \
  "$PYTHON" -B /data/ycfeng/Frontier/.worktrees/fidelity-candidate-b862e57/tests/e2e/refactor_fidelity/run_matrix.py \
  compare --output-root /data/ycfeng/tmp/step5_event_logging_clean_fidelity_20261021
```

**PASS:** both executions exit0;strict compare exits0;77/77 cases are identical;
zero mismatches,new failures,missing evidence,definition changes or cache-name
differences. The old dirty-tree receipt is preserved unchanged. Raw
`baseline.log`,`candidate.log`,`compare.log` and both manifests remain under the
new output root. Durable comparison:
`evidence/request_free_event_logging_fidelity.json`.

## Verification limits and next action

This is the real event-extraction regression, not a full measured request PASS.
The actual-main owner must finish offline/online 6144/1024 execution and inspect
READ readiness, source release and final metrics. Native MRv2 dummy metadata,
FULL_AND_PIECEWISE mapping, G3 and formal four-metric acceptance remain separate
open work. No GPU operation, new profile, dependency install, runtime mutation,
or threshold change occurred in this repair.
