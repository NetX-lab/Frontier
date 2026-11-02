# Groundtruth fixture CPU verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded original RED, exhaustive GREEN, immutable/source/matrix bindings. |

Worktree `/data/ycfeng/Frontier/.worktrees/v4-groundtruth-fixture`; source `6abd2d8e`, base `c4b59b65`. Python `/data/ycfeng/envs/asyncopd-py312/bin/python` (3.12.13), no conda activation or dependency installation.
Every test argv uses `sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s runuser -u brainpp -- env ... python -B`; full exact argv/cwd/environment are in the named `*_command.json` receipts, with no omitted arguments.
Environment: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=`, BLAS/OMP/MKL threads1, `PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site`, `TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-groundtruth-fixture`. Pytest cache disabled; each invocation has a distinct local basetemp/XML.
Native root is `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`. Legacy input is the existing main-worktree `h200_glm_canonical_integration_20261021/check_groundtruth_run.before.py`; `NATIVE_CHECKER_C26_FIXTURE` is its existing `v4_p0_native_20261021/candidate_h800_p0_02` alias, resolving to the retained profile02 package. No alias/package/input was created or modified.

| Receipt | Expected/observed | Result |
| --- | --- | --- |
| original_red2 |1FAIL/0ERROR/0SKIP;32 dp_utils setup errors,11hooks|expected RED|
| positive_green |1PASS;32ranks/six phases/14hooks;23.73s|PASS|
| groundtruth_non_v2 |80PASS/0FAIL/0ERROR/0SKIP;43.01s|PASS|
| groundtruth_v2_negative_a |9PASS/0FAIL/0ERROR/0SKIP;22.00s|PASS|
| groundtruth_v2_negative_b |9PASS/0FAIL/0ERROR/0SKIP;21.33s|PASS|
| test_native_v2_serving_inventory |46PASS/0FAIL/0ERROR/0SKIP;3.72s|PASS|

Groundtruth accepted coverage is exactly the99 collected original node IDs, no duplicates/omissions; total accepted145PASS. Positive serving counts: prefill3456 forwards/0replays, decode3456forwards/3456replays. All32 snapshots have zero observation errors/drops and complete dispatch operands. Retained negative controls and frozen-oracle rejection controls remain unchanged.
Preserved unsuccessful auxiliary attempts: initial system Python3.10 coordinator `hashlib.file_digest` error before execution; first root-selection attempt1SKIP due pytest module reimport; one-process99-test attempt exit124 at the fixed60-second ceiling. It produced no XML verdict and is not counted as a passing suite. Exhaustive disjoint node runs preserve that ceiling rather than increasing it.

Exact historical argv arrays contain every test selector and environment setting. Safe independent replay below changes only scratch paths, preserving the archived evidence:
```python
from pathlib import Path
import json, subprocess, tempfile
root = Path('/data/ycfeng/Frontier/.worktrees/v4-groundtruth-fixture')
records = root / 'task_memory/task_2026-10-23_v4_remaining_measurements/profile01/groundtruth_fixture'
scratch = tempfile.mkdtemp(prefix='groundtruth_root_rerun_', dir='/data/ycfeng/tmp/opencode-v2-tmp/opencode')
for name in ('positive_green', 'groundtruth_non_v2', 'groundtruth_v2_negative_a', 'groundtruth_v2_negative_b', 'test_native_v2_serving_inventory'):
    receipt = json.loads((records / f'{name}_command.json').read_text())
    argv = [arg.replace(receipt['environment']['TMPDIR'], scratch) for arg in receipt['argv']]
    subprocess.run(argv, cwd=receipt['cwd'], check=True)
```
Invoke the replay with `/data/ycfeng/envs/asyncopd-py312/bin/python -B -` and the exact stdin above. Original RED argv and its root-only plugin are separately retained in `original_red2_command.json`; original function/source bytes were unchanged and the main collector bytes were verified equal to this base.

Immutable PASS:189files rehashed before/after, including frozen checker/legacy/R2/R3,13runtime modules,87native source files,45sealed profile04 inputs and original failure evidence. Runtime equals1437ace6 Git bytes; native equals pinned2cf Git bytes; profile04 manifest remains `25bc61be...d58f`. Image binaries were not executed or rehashed; their existing package/source pins remain byte-identical.
Source PASS: `git diff --check c4b59b65 -- tests/unit/test_native_groundtruth_check.py`;32additions+5deletions=37/60gross; all17test-function ASTs unchanged. Exact source, diff and environment hashes are in `source_budget.json`/`source_binding.json`/`verification.json`.
Matrix PASS: existing `/data/ycfeng/tmp/glm53_dsv4/root_attention_input_4a1b53e6_compare/comparison.json`, SHA `7a8728ecc0851373827d9cdc7dfb68c552299e6c54b5fae75bd8cc5429d36ad6`,77identical cases and no mismatches/failures; unchanged domain tree equality is explicitly bound to the fixed source (`matrix_binding.json`). No new capture or GPU/runtime/cost/reference acceptance.
Artifacts: compact receipts/diff/logs/XML are in this directory; all32positive/RED rank snapshots remain under the recorded local TMPDIR, hash-bound in `verification.json`. Root independent review/rerun/integration and note #9 closure remain pending.
Archive whitespace checks: standard single-space blank context lines in raw `source.diff` and its diagnostic log are reported. Exact raw bytes are preserved; the final check is scoped to maintained Markdown/JSON and passes. This is not source whitespace damage or a test failure.
