# D30 parent integration verification

## Modification History
| Date | Change |
| --- | --- |
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
