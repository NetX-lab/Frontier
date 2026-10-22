# D30 approved canonical-source assertion amendment

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record D30-SCOPE-8-750, isolated RED, affected65 outcomes and unchanged unit failure/error sets. |

## Analysis and decision

Full candidate713e6a8 ran8124 tests in1043.142s:7646 passed,268 failed,31 errors,
179 skipped. Compared with b8be287, exactly one failure was new:
`test_actual_canonical_capture_stage_preserves_native_case_and_original_prompts`.
The approved observer adds `vllm/v1/engine/core.py` to its pinned source inventory
at `r2_observation.py:45`; the stage now correctly has13 source hashes. The old
test at `test_step5_canonical_g3.py:109` expected12. Its isolated RED reproduces
`AssertionError: assert13 ==12`. Other267 failure IDs and31 error IDs are unchanged.

YC approves D30-SCOPE-8-750 in requirements.md: add only the eighth test path,
retain the750-line total ceiling. Amend the expected count to13 and require
`vllm.v1.engine.core` membership. This is +2/-1, making complete D30 scope
530insertions+73deletions=603 across eight files. Production code is unchanged
from the independently reviewed713e6a8 candidate.

## Commands and observed checks

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-finish-sync` for GREEN;
`/data/ycfeng/Frontier/.worktrees/d30-unit-713e6a8` for RED. Environment:
`PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1
TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

```bash
timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_canonical_g3.py::test_actual_canonical_capture_stage_preserves_native_case_and_original_prompts -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_canonical_count_red01 --junitxml=/data/ycfeng/tmp/d30_canonical_count_red01.xml > /data/ycfeng/tmp/d30_canonical_count_red01.log 2>&1
timeout 180s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_canonical_g3.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_canonical_count_green01 --junitxml=/data/ycfeng/tmp/d30_canonical_count_green01.xml > /data/ycfeng/tmp/d30_canonical_count_green01.log 2>&1
timeout 180s env STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_canonical_g3.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_canonical_fullenv01 --junitxml=/data/ycfeng/tmp/d30_canonical_fullenv01.xml > /data/ycfeng/tmp/d30_canonical_fullenv01.log 2>&1
```

- Python3.10.6 RED: one intended failure.
- Same-environment affected module:63 PASS, two existing `aiohttp` import failures,
  3.48s. The repaired case passes; both remaining failure IDs occur in the baseline.
- Existing complete Python3.12.13 environment:65/65 PASS,2.56s; no dependency edit.
- Parent parses all four XML sets, confirms the affected65 IDs are exactly a
  subset of the full run, and replaces only their previous outcomes. Latest
  evidenced outcomes:7647 PASS,267 FAIL,31 ERROR,179 SKIP. Failure/error sets
  equal the baseline exactly. This is full capture plus affected rerun, **not a
  second full-unit execution or a claim that the unit suite is entirely green**.
- Complete eight-file scope603<=750, AST/module<=2000 and whitespace PASS.
  No production/fidelity input changed since713e6a8: retain parent420 affected
  checks and77/77 strict comparison with902 artifact hash checks.

Persistent receipts/XML: `evidence/canonical_source_amendment/`, especially
`outcomes_and_budget.json`. Full-unit logs/XML remain at
`/data/ycfeng/tmp/d30_unit_{baseline,candidate}01{.log,.xml}`; parent integration
report records the original full commands and strict77 evidence.

## Limits

Source/CPU acceptance is satisfied. Merge4ec5b23 and exact parity with accepted
3d6744f are verified in `evidence/parent_integration/merged_source.json`.
Fresh native counter observation and full Step CLI qualification remain open;
this assertion-only amendment does not provide either.
