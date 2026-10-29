# D30 parent integration verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Candidate713e6a8: parent420 affected checks, strict77 and902 artifact rehashes PASS; unit delta pending. |
| 2026-10-22 | Baseline completes:7617 passed,267 failed,178 skipped,31 errors; retain IDs for candidate comparison. |
| 2026-10-22 | Start source-pinned unit baseline and validate retained strict77 baseline. |

Decision: D30-SCOPE-7-750. Parent owns this report; the implementation agent owns
its separate affected-test report. Source/test scope is the seven paths in
`requirements.md`, at most750 total added/deleted lines against non-D30 parent
`b8be287`. Native counter execution and complete Step CLI are separate open
acceptance items.

## Unit regression baseline — recorded, suite FAIL

Create isolated baseline from `/data/ycfeng/Frontier`:

```bash
git worktree add -b verify/d30-baseline-b8be287 /data/ycfeng/Frontier/.worktrees/d30-baseline-b8be287 b8be287
```

Run in that clean checkout:

```bash
timeout 1800s env PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit -q -p no:cacheprovider --continue-on-collection-errors --basetemp=/data/ycfeng/tmp/d30_unit_baseline01 --junitxml=/data/ycfeng/tmp/d30_unit_baseline01.xml > /data/ycfeng/tmp/d30_unit_baseline01.log 2>&1
```

Expected: finish baseline collection/execution and retain failure/error IDs;
candidate must have no unexplained new failures under the same environment.
Actual: exit1,810.248s;7617 passed,267 failed,178 skipped,31 errors (14 collection
errors and17 setup errors),8093 JUnit records. Baseline is not green. Failures
include missing torch/aiohttp, an unavailable historical reference checkout,
and existing code/test expectation mismatches; they are not attributed to D30.
Full details and IDs are retained in
`/data/ycfeng/tmp/d30_unit_baseline01_results.json` and the original log/XML.
Candidate failure/error IDs must be compared under the same environment; all
new failures require investigation. No unrelated fixes were made.

The receipt was extracted with Python3.10.6 using ElementTree over every
`testcase`, recording `classname::name`, failure/error messages and full text;
suite attributes retain the exact counters. This is ordinary simulator
regression, not a native CPU-gap measurement or performance campaign.

## Strict77 baseline reuse — PASS

Existing baseline:
`/data/ycfeng/tmp/step5_d23_perf_fidelity_20261022/candidate`, source/harness
`7e8bea3cd2362f8e46c4b7cd0e712d2f2fe392fb`. Parent reuses the prior read-only audit
function without modifying the baseline or auditor:

```bash
env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
import importlib.util
import json
from pathlib import Path
from tests.e2e.refactor_fidelity.run_matrix import _package_versions
path = Path('/data/ycfeng/tmp/step5_workflow_fidelity_20261022/audit_evidence.py')
spec = importlib.util.spec_from_file_location('retained_fidelity_audit', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
receipt = module.load_side(module.BASELINE, module.BASELINE_REVISION)
manifest = json.loads((module.BASELINE / 'manifest.json').read_text())
receipt['current_package_versions'] = _package_versions('/data/ycfeng/envs/frontier-py310/bin/python')
receipt['package_versions_match'] = receipt['current_package_versions'] == manifest['package_versions']
assert receipt['case_records'] == receipt['unique_case_records'] == 77
assert receipt['full_clean_capture'] and receipt['package_versions_match']
for key in ('case_failures', 'missing_cases', 'unexpected_cases', 'definition_mismatches', 'provenance_findings', 'record_provenance_errors', 'evidence_errors', 'manifest_dirty_paths', 'source_dirty', 'cache_inventory_changed'):
    assert not receipt[key], (key, receipt[key])
output = Path('/data/ycfeng/tmp/d30_fidelity_preflight01.json')
with output.open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
git diff --name-only 7e8bea3cd2362f8e46c4b7cd0e712d2f2fe392fb b8be287 -- tests/e2e/refactor_fidelity examples data
```

The audit exits0:77 unique cases,451 artifacts rehashed,519 cache names,
zero failures/evidence/provenance/definition findings, unchanged cache inventory,
full clean capture and matching package versions. The Git diff is empty.
Python3.10.6; numpy2.2.6, pandas2.3.3, sklearn1.7.2, scipy1.15.3, plotly7.1.0,
fasteners0.20, ddsketch3.0.1. Receipt:
`/data/ycfeng/tmp/d30_fidelity_preflight01.json`.

This proves baseline reuse only. Candidate strict77, candidate unit comparison,
parent affected checks and integration are still pending.

## Candidate and parent affected checks — PASS

Code `dc166cb74d5b506814a5560d07235be781e17803`; complete candidate
`713e6a81708e87b86cf91fd89f3b1f954bdcbead`. Parent independently compares all
seven source/test paths to the approval receipt, checks528 insertions+72
deletions=600, parses each AST, checks modules<=2000 and whitespace. Receipt:
`/data/ycfeng/tmp/d30_parent_budget01.json`. The current parent changes since
b8be287 contain no frontier/tests/data/examples/pyproject/.gitmodules changes.

Commands run in `/data/ycfeng/Frontier/.worktrees/step5-finish-sync`, with
`PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src`:

```bash
timeout 600s /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_pd_r2_stage.py tests/unit/test_step5_native_forward_census.py tests/unit/test_step5_native_pd_r2_verdict.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_parent_r2_01 --junitxml=/data/ycfeng/tmp/d30_parent_r2_01.xml > /data/ycfeng/tmp/d30_parent_r2_01.log 2>&1
timeout 300s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_finish_sync.py tests/integration/test_vllm_dp_dummy_pass_runtime.py -k 'not test_vllm_dp_dummy_pass_runtime or after_32_iterations or attention_dp_one' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_parent_boundaries01 --junitxml=/data/ycfeng/tmp/d30_parent_boundaries01.xml > /data/ycfeng/tmp/d30_parent_boundaries01.log 2>&1
```

Observed:388 PASS/10.09s and32 PASS/11 deselected/86.79s,420 distinct checks.
Python versions3.12.13 and3.10.6 respectively; existing environments unchanged.
Independent reviewer `ses_e9cb9d9a8ffeOycbC32aMulBAz`, task
`D30-FINAL-REVIEW`, reports No findings for all seven files and their production
call paths. This review did not execute tests or qualify live native behavior.

## Candidate strict77 — PASS

Created clean source-isolated verification checkout:

```bash
git worktree add -b verify/d30-fidelity-713e6a8 /data/ycfeng/Frontier/.worktrees/d30-fidelity-713e6a8 713e6a81708e87b86cf91fd89f3b1f954bdcbead
```

From that checkout, with `PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root "$PWD" --python-bin /data/ycfeng/envs/frontier-py310/bin/python --label candidate --output-root /data/ycfeng/tmp/d30_workflow_fidelity_20261022 --jobs 4 --clean-cache --continue-on-failure > /data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate_run.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/d30_workflow_fidelity_20261022 --baseline-label /data/ycfeng/tmp/step5_d23_perf_fidelity_20261022/candidate --candidate-label candidate > /data/ycfeng/tmp/d30_workflow_fidelity_20261022/comparison.log 2>&1
```

Capture and strict comparator exit0. Parent repeats `load_side` above on the
candidate and baseline, verifies77 unique clean cases per side, source revision,
definitions/cache inventories, complete strict comparison and all13 mismatch
lists empty; rehashes451+451=902 artifacts. Observed77/77 identical, zero
metric/artifact/failure/missing/provenance/cache differences. Both source checkout
HEAD and clean status match713e6a8 after capture. No thresholds or expectations
were modified. Audit: `parent_audit01.json` in the output root. Comparison SHA256:
`715e523e837a9696a6bc74c709b618787745d22d2187ba0ce255b44d1a56ca36`.

## Candidate unit delta — complete with approved assertion amendment

Created separate `verify/d30-unit-713e6a8` checkout at the same revision under
`/data/ycfeng/Frontier/.worktrees/d30-unit-713e6a8`, so its cache cannot affect
strict77. Run the baseline command unchanged there except basetemp/XML/log names
become `/data/ycfeng/tmp/d30_unit_candidate01{,.xml,.log}`. Actual full result:
7646 PASS/268 FAIL/31 ERROR/179 SKIP across8124 cases,1043.142s. The sole added
failure is canonical source-count12 versus13. All267 baseline failures and31
errors are retained. D30-SCOPE-8-750 authorizes the eighth test file; amendment
3d6744f changes only that unit assertion outside task records.

Affected rerun in the same environment:63 PASS and two baseline aiohttp import
failures. Existing complete Python3.12 environment:65/65 PASS. Parent replaces
only these65 outcomes in the full capture; latest evidenced7647 PASS,267 FAIL,
31 ERROR,179 SKIP, with failure/error sets exactly equal to baseline. This is
an affected rerun, not a second full-unit capture or an entirely green suite.
Exact commands and receipts: `test_report_2026-10-23_canonical_source_assertion.md`
and `evidence/canonical_source_amendment/`. Final D30 scope is eight files,
530 additions+73 deletions=603<=750. Production/fidelity inputs and the completed
independent source review are unchanged.

## Final integration — PASS (2026-10-23)

Merge `4ec5b23bb71a92e505083296912f3556cad5ba6a` integrates accepted3d6744f into
feat/step5-preview. `git diff --exit-code 3d6744f HEAD -- frontier tests examples
data pyproject.toml .gitmodules` and merge whitespace checks pass. Receipt:
`evidence/parent_integration/merged_source.json`. Existing unrelated `.cortexkit/`
and `.pi/` remain untouched. No repeated runtime tests are needed for identical
accepted bytes. Fresh native counter observation and full CLI remain open.
