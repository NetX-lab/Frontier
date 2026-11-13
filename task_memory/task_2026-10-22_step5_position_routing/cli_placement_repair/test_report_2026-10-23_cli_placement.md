# CLI placement repair validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record real placement RED/GREEN, whole-file checks, retained fixture probe and exact ownership accounting. |
| 2026-10-23 | Seal verified source/tests/records commit ae1fb721. |

TaskID: **D29-CLI-PLACEMENT-REPAIR**. Worktree `/data/ycfeng/Frontier/.worktrees/step5-cli-placement-repair`, branch `feat/step5-cli-placement-repair`, base `1be91bad431d3bd886de7f5d80c19bd7e6c54a25`. Python `/data/ycfeng/envs/frontier-py310/bin/python`, version 3.10.6. Local pinned submodule checkout `3dac53c544e546574cb6e8078cd14053d0916113`. No package installation, full simulation CLI, full-unit, GPU, network or other platform actions.

## Expected and actual results

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Regression before runner edit | Actual command/config/Cluster/factory/CPU-plan chain reproduces original placement guard for P/D | 4 FAIL in 2.32s, all at exact original guard | Expected RED |
| Regression after one-line selected placement | Same four cases return the exact CPU plan cost, not just pass the guard | 4 cases PASS; cost0.03ms, keys and count receipt match | PASS |
| Regression plus old command/capacity/READ check | Both new and legacy cases pass | 5 PASS in 2.18s | PASS |
| Entire affected files | No new failure/exclusion | Final57 PASS/6 SKIP in59.24s,0FAIL/ERROR | PASS |
| Retained fixture03 constructor chain | Both modes/roles price exact supplied plan; all outputs relocated before parse | Offline/online P/D all0.03ms/keymatch;input hashes unchanged | PASS |
| Scope/AST/whitespace | Two files<=65gross,driver<=66,test<=283,CORE<=4404/global<=10000 | +40/-0=40/65;47/66,260/283;3873/4404,9116/10000 | PASS |

Suites overlap and are not summed. Six unchanged full-file skips concern inapplicable source prepare and missing torch; no test is excluded or new skip added. The driver is included as an affected module but contains no pytest tests. The whole integration test file is collected and run without `-k` or individual test exclusions.

## Isolation and cause

Created a new original-root worktree:

```bash
git worktree add -b feat/step5-cli-placement-repair /data/ycfeng/Frontier/.worktrees/step5-cli-placement-repair 1be91bad431d3bd886de7f5d80c19bd7e6c54a25
```

The existing local submodule repository was read to create a separate detached checkout at the already-recorded gitlink, with no checkout/reset/write to its existing source:

```bash
git worktree add --detach /data/ycfeng/Frontier/.worktrees/step5-cli-placement-repair/frontier/cc_backend/backends/collective-sim 3dac53c544e546574cb6e8078cd14053d0916113
```

This command ran from the existing submodule repository. The new parent gitlink stays clean; no submodule source edit or network access occurred.

Parent already tested three hypotheses before this implementation: missing placement, wrong parallel dimensions, and a downstream native-path rejection. Real dimensions are TP8/CP1/DP2/EP1; the generic placement remains TP,CP,DP,EP. Parent changing only the existing global placement option yielded real CPU cost0.03ms. This task therefore skips another design/minimization round and implements the approved one-line input selection, while establishing its own persistent RED before the change.

The regression uses `runner.command` directly, parses its actual argv through `SimulationConfig.create_from_cli_args`, constructs actual `Cluster` and lazy CC backend factory, reconstructs a `CollectivePlan` from the current-repository CPU writer's exact count key, and calls `predict_collective_plan`. It does not manually preset config/backend placement or catch downstream exceptions. Its outputs/cache are under pytest scratch before the parser writes config.json; global configuration state is restored by the existing monkeypatch fixture. No external worktree import or host-specific retained input is in the persistent test.

## Exact RED/GREEN commands

All commands below execute from the isolated worktree. `PYTHONDONTWRITEBYTECODE=1`, OMP and OPENBLAS threads1, scratchTMPDIR and existing test-module PYTHONPATH are explicit. Foreground tool limit120s; the simulator's original2400s per-mode bound is unchanged and no simulation loop is launched here.

Before editing the driver:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-cli-placement-red01 tests/integration/test_step5_measured_main.py::test_position_command_prices_exact_plan_through_real_cluster > /data/ycfeng/tmp/d29-cli-placement-red01.log 2>&1
```

Raw result: `4 failed in 2.32s`. Cases: offline-PREFILL,offline-DECODE,online-PREFILL,online-DECODE. All raise `ValueError: Step plans require explicitly configured TP8/DP2 placement` at unchanged backend line809; the dimension assertion succeeds first.

After the one-line position-only addition:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-cli-placement-green01 tests/integration/test_step5_measured_main.py::test_position_command_prices_exact_plan_through_real_cluster tests/integration/test_step5_measured_main.py::test_actual_main_command_retains_capacity_graph_topology_and_measured_read > /data/ycfeng/tmp/d29-cli-placement-green01.log 2>&1
```

Raw result: `5 passed in 2.18s`. Duration0.03ms equals the exact writer entry; full plan key and serialized paired count receipt are equal. No downstream failure is converted into GREEN.

The initial whole-file command to `d29-cli-placement-whole01` was57PASS/6SKIP59.97s. A legacy-default assertion was then added: balanced runner argv does not emit the selected placement option. The exact final whole-file command:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-cli-placement-whole02 tests/integration/test_step5_measured_main.py tests/integration/run_step5_measured_main.py > /data/ycfeng/tmp/d29-cli-placement-whole02.log 2>&1
```

Raw result: `57 passed, 6 skipped in 59.24s`. No new failure, error or exclusion.

## Retained fixture03 actual command/config/backend check

This independent short chain uses the unchanged retained manifest, its real profile paths and collective artifact, with current-repository imports. It relocates metrics, CC cache, predictor cache and generated READ fixture before any CLI parse. It does not import the scout's fixed external worktree or execute its permissive exception handling. The parent's stored commands/logs/config and active integrated04 are untouched.

Exact command, run once with120s tool limit:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY' > /data/ycfeng/tmp/d29-cli-placement-retained-green01.log 2>&1
import json,hashlib,sys
from pathlib import Path
from frontier.config import SimulationConfig
from frontier.entities import Cluster
from frontier.cc_backend.collective_plan import CollectivePlan,CollectiveTensor
from tests.integration.run_step5_measured_main import command,read_profile,write_read_cost
source=Path('/data/ycfeng/tmp/d29-parent-position-fixture03')
root=Path('/data/ycfeng/tmp/d29-cli-placement-retained01');root.mkdir(exist_ok=False)
manifest=source/'fixture_manifest.json'
fixture=read_profile(manifest)
fixture['cache_dir']=str(root/'predictor_cache')
path=Path(fixture['collective_plan_input_file'])
fingerprint={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest,path)}
read=root/'read.json';write_read_cost(read)
for mode in ('offline','online'):
 output=root/mode;output.mkdir();argv=command(mode,output,fixture,read)
 (output/'command.json').write_text(json.dumps(argv,indent=2)+'\n')
 sys.argv=argv[3:]
 config=SimulationConfig.create_from_cli_args()
 assert Path(config.metrics_config.output_dir).is_relative_to(root)
 assert Path(config.metrics_config.cache_dir).is_relative_to(root)
 artifact=json.loads(path.read_text());key=next(k for k in sorted(artifact['counts']) if json.loads(k)['phase']=='dispatch')
 value=json.loads(key)
 plan=CollectivePlan(value['phase'],tuple(value['sizes']),tuple(CollectiveTensor(**t) for t in value['tensors']),tuple(tuple(g) for g in value['groups']),value['graph_mode'],value['context_json'],value['buffer_json'])
 assert plan.cache_key()==key
 results=[]
 for role,cc in config.get_clusters().items():
  cc.replica_config.cluster_num_replicas=int(cc.num_replicas)
  backend=Cluster(cc,config.metrics_config,config.request_generator_config).cc_backend
  cost=backend.predict_collective_plan(plan)
  assert cost.duration_ms==artifact['counts'][key]['cost']['duration_ms']==.03 and cost.plan_key==key
  results.append(dict(role=role.name,dims=backend._get_attention_parallel_dim_sizes(),placement=list(backend._placement_order),duration_ms=cost.duration_ms,plan_key_matches=True))
 (output/'result.json').write_text(json.dumps(results,indent=2)+'\n')
 print('RESULT',mode,json.dumps(results))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in fingerprint.items())
(root/'immutable_input_sha256.json').write_text(json.dumps(fingerprint,indent=2)+'\n')
print('PASS retained fixture03 immutable manifest/collective hashes; outputs/caches under',root)
PY
```

Results: both offline and online have PREFILL andDECODE with dimensionsTP8/CP1/DP2/EP1,placementTP,DP,duration0.03ms and matching exact key. Exit0; no exception catch. ManifestSHA256`ed0491872ea1b726b3d0aeb5a6dd45d54c5ccab5af53f0f79160bb867c32e9d2`; collectiveSHA256`447e697340d77c205e7457eb50c7b63c47ab103ba77c7f52e7c5fe844498ca90`, before/after equal. Original inputs remain at `/data/ycfeng/tmp/d29-parent-position-fixture03`. Selected results/argv/generated config are under `/data/ycfeng/tmp/d29-cli-placement-retained01`; no original-run output is rewritten.

## Accounting and limits

Exact final patch: `placement_repair.diff`; machine ledger: `budget.json`. Commands: `git diff --numstat 1be91bad -- <two paths>`, `git diff --numstat 1aee0af -- <two paths>`, `git diff --check`; PythonAST parse and numeric cap assertions pass. Final source delta+40/-0=40/65. Driver47/66,test260/283,donor unchanged. Root final ledger3833CORE/9076global plus40 =3873/4404 and9116/10000;111paths/PRICING77 unchanged. No root overlay/document edits or new source paths.

This closes only the reproduced selected placement wiring at actual constructors. Root independently reviews/integrates/freezes and runs fresh full6144/1024offline/online within2400s. Strict77/native/source/GPU/gap/T2 restaging are not performed or claimed. Inert role-scoped overrides are explicitly deferred. The scout's previously overwritten retained main05offlineconfig is not restored, relabelled or used as original evidence. No debug instrumentation, retries, fallback or downstream exceptions-as-GREEN remain.

Immutable verified source/tests/records commit:ae1fb721. Final tested source matches this commit exactly. Worktree/submodule remain clean; no root-owned or retained-run path is modified.
