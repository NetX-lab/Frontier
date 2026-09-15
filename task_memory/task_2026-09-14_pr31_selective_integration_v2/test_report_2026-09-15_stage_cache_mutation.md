## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-15 | Recorded the stage aggregate cache mutation-invalidation regression and verification. |

# Stage Aggregate Cache Mutation Verification

## Execution

- Environment: Python 3.12.3 in the active Frontier environment.
- Initial reproduction:

  ```bash
  python -m pytest -q tests/unit/test_stage_execution_time.py::test_stage_model_time_refreshes_after_layer_mutation
  ```

- Focused regression:

  ```bash
  python -m pytest -q \
    tests/unit/test_stage_execution_time.py \
    tests/unit/test_metrics_stage_execution_time.py \
    tests/unit/test_execution_time_op_times.py
  ```

- Syntax and whitespace checks:

  ```bash
  python -m compileall -q frontier/entities/execution_time.py \
    frontier/entities/stage_execution_time.py tests/unit/test_stage_execution_time.py
  git diff --check
  ```

## Criteria

- A stage aggregate must refresh after an existing layer timing mutation.
- Repeated reads without a mutation must continue to reuse the cached aggregate.
- Existing stage, metrics, and operator timing behavior must remain passing.

## Evidence

- **FAIL before the fix:** the new regression observed `249.0 ms` after mutation while the expected value was `252.0 ms`; this confirmed stale cache state through the production mutation helper.
- **PASS after the fix:** `52 passed in 2.75s` across the focused stage, metrics, and operator timing suite.
- **PASS:** targeted `compileall` exited `0`.
- **PASS:** `git diff --check` exited `0`.

The regression covers timing payload mutation through the existing operator replacement path. Direct writes to private scalar fields remain outside the supported mutation contract.
