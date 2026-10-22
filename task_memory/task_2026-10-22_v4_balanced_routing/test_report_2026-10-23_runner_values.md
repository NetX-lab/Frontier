# R1 runner-value rejection verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded actual-entry-point RED/GREEN, preserved controls and net routing count. |

Task: W128-V4-BALANCED-PREFILL-R1. Worktree: `/data/ycfeng/Frontier/.worktrees/v4-h800-prefill`, initially clean at59d370d. Python: `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6. No dependency, native/B/shared, builder, GPU, Docker, platform, SSH or cloud changes. No persistent harness/test file was created.

## Exact RED/GREEN command

The same command below ran before and after the single source correction. The stored receipt and existing fixtures are source/consumer controls, not GPU measurements.

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python - <<'PY'
import copy
import json
import tempfile
from pathlib import Path
import pandas as pd
from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.execution_time_predictor.deepseek_v4 import read_isolated_scope
from frontier.profiling.attention.dsv4_attention_rows import routing_columns
from frontier.profiling.common.runtime_identity import canonical_json
from tests.unit import test_dsv4_graph_measured as measured
from tests.unit import test_dsv4_source_execution as source
from tests.unit.test_dsv4_attention_workload import source_batch

identity = json.loads(Path('task_memory/task_2026-10-22_v4_balanced_routing/cpu_identity_receipt.json').read_text())['observed']
bad = copy.deepcopy(identity)
bad['runners'] = {name: 'builtins.object' for name in bad['routers']}
result = {}
try:
    routing_columns(bad)
except ValueError:
    result['routing_columns_rejects_wrong_runner'] = True
else:
    result['routing_columns_rejects_wrong_runner'] = False
result['balanced_receipt_positive'] = routing_columns(identity)['moe_routing'] == canonical_json(identity)
with tempfile.TemporaryDirectory(dir='/data/ycfeng/tmp', prefix='prefill-r1-') as temp:
    p, row = measured.measured.__wrapped__(source.predictor.__wrapped__(), Path(temp))
    original_layout = dict(p._isolated_attention_layout)
    columns = routing_columns(identity)
    p._replica_config.moe_gating_routing_runtime_path = 'balanced_topk'
    p._isolated_attention_layout = {**original_layout, **columns}
    pd.DataFrame([{**row, **columns}]).to_csv(p._attention_input_file, index=False)
    def consume():
        return read_isolated_scope(p, source_batch([6144], [0]), 0,
            'dsv4_sparse_attention', attention_layout=p._isolated_attention_layout)
    result['balanced_consumer_positive'] = consume()['time_ms'] == 0.125
    bad_columns = {**columns, 'moe_routing': canonical_json(bad)}
    p._isolated_attention_layout = {**original_layout, **bad_columns}
    pd.DataFrame([{**row, **bad_columns}]).to_csv(p._attention_input_file, index=False)
    try:
        consume()
    except DeepSeekV4CoverageError:
        result['isolated_consumer_rejects_wrong_runner'] = True
    else:
        result['isolated_consumer_rejects_wrong_runner'] = False
    p._replica_config.moe_gating_routing_runtime_path = None
    p._isolated_attention_layout = original_layout
    pd.DataFrame([row]).to_csv(p._attention_input_file, index=False)
    result['legacy_consumer_positive'] = consume()['time_ms'] == 0.125
print(json.dumps(result, indent=2))
assert all(result.values()), 'P2: matching runner keys with builtins.object values must be rejected at both real entry points'
PY
```

| Assertion | RED before change | GREEN after change |
| --- | --- | --- |
| Actual routing_columns rejects same-key/wrong-class receipt |false |true |
| Actual read_isolated_scope raises typed rejection for same altered row/layout |false |true |
| Valid balanced receipt |true |true |
| Valid balanced consumer, fixture0.125ms |true |true |
| Absent-selector legacy consumer, fixture0.125ms |true |true |

RED exited1 with `AssertionError: P2: matching runner keys with builtins.object values must be rejected at both real entry points`; GREEN exited0 with all five values true. The one-variable negative retained identical router names and all other valid receipt fields. Root cause: runner key-only comparison discarded class values. Fix: exact expected names-to-MoERunner mapping at `frontier/profiling/attention/dsv4_attention_rows.py:66-67`. Consumer uses the unchanged ValueError-to-DeepSeekV4CoverageError translation at `frontier/execution_time_predictor/deepseek_v4.py:659-661`.

## Preserved controls and commands

- Same Python stdin environment, reusing the prior direct fixture controls:23PASS. Both actual entry points reject wrong/missing/extra runners, bad environment, missing routers/native sources; consumer rejects missing routing columns, balanced/unbalanced mixture and bad strategy SHA. Valid balanced result retains its identity. Both methods retain six rows; historical raw rows are not relabeled; bad runner raw receipt and mixed raw identities reject. No support code is imported into production.
- `env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_attention_profiling.py tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider -k 'exact_prefill or exact_isolated or attention_rejects_unqualified or attention_does_not_median or attention_layout'`: expected affected regressions pass; actual22PASS,162deselected,2.23s.
- `git diff --check`: expected no whitespace errors; actual exit0/no output.
- `git diff --numstat a71fcd3 -- tests/comparison/calibration/profile_dsv4_prefill.py frontier/profiling/attention/dsv4_attention_runner.py frontier/profiling/attention/dsv4_attention_rows.py frontier/execution_time_predictor/deepseek_v4.py task_memory/task_2026-10-22_v4_h800_prefill/configured_capacity/prepare_package.py`: five maintained files,183added+7deleted=190net routing changed lines. R1 patch alone2added+1deleted=3touched lines; it adds one line to the effective baseline diff. Root's reserved12builder lines project202total. Task records/evidence are separate from the maintained source allocation.
- Builder SHA256 remains `c9a993833abd8ceb167bc2763adbd6f43ec2f7a6a52ada87c0848e4f47f9d11e`. Other four maintained C files match the prior source commit. Fixed strategy SHA/native bytes are untouched.

## Remaining qualification

Source-only review and CPU fixtures do not admit the image or runtime. Root owns joint tests/review, final allocation, current-clean-HEAD snapshot builder mechanism, h800_prefill_balanced generation, sealing and image/runtime evidence. This agent did not modify or execute the builder and did not perform GPU work.
