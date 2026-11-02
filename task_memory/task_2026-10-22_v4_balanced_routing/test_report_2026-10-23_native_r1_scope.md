# Native R1 review reproductions and complete-scope estimate

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded two targeted REDs, real isolated reconstruction/source check and no-source-edit scope blocker. |
| 2026-10-23 | Approved23/950 implementation: two source files, persistentRED/currentGREEN,133PASS, actual77-source roles and archive preserved. |

Task W128-V4-BALANCED-NATIVE-R1, steers integration negatives. Source commit65743ea, Python3.12.13 at `/data/ycfeng/envs/asyncopd-py312/bin/python`, no conda activation. Source remains unedited; no builder/package/GPU/platform/cloud/SSH/Docker/delegation.

## Targeted RED evidence

Both invocations use importlib.spec_from_file_location to load actual B R3, actual root775-field file copied to a local temporary `case/inputs/frontier.json`, and digest of those exact bytes in retained_files, tool_hashes and manifest.inputs_sha256. Engine remains actual target DeepSeek-V4-Flash/TP8/DP2/PP1/H800/balanced_topk.

1. Ambient-source RED ran the same invocation in two child interpreters: `[sys.executable, '-I', '-B', '-c', code, ambient_root, checker_path, input_path]`, `cwd=/data/ycfeng/tmp/opencode-v2-tmp/opencode`. Child explicitly preloaded SimulationConfig/create_flat_dataclass from ambient_root. Declared harness was A in both cases. B preload rejects complete format, A preload accepts. Actual schema/flat origins were `{B,A}/frontier/config/{config,flat_dataclass}.py`. Expected both invocations must bind declared A and have identical verdict. **RED**.
2. DP RED imported actual A config, copied the actual complete file, changed only `replica_config_attn_dp` from2 to1 and recomputed all three existing hashes. Direct `checker.verify_balanced_frontier(config, engine, manifest)` returned normally. NativeDP2 vs FrontierDP1 should reject. **RED**.

Input reference: root `task_memory/task_2026-10-22_v4_balanced_routing/balanced_frontier_input/frontier.json`, SHA747a08e5da6759f8189d29b2c8419ab48eb215ac2db4a2596af4f1d15ef34897. No abbreviated replacement or source modification.

## Exact isolated reconstruction/source substrate command

Executed at `/data/ycfeng/Frontier/.worktrees/v4-balanced-callers`:

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -I -B - <<'PY'
import sys, pathlib, json, subprocess, hashlib
from unittest.mock import patch
root=pathlib.Path('/data/ycfeng/Frontier/.worktrees/v4-balanced-callers')
sys.path.insert(0,str(root))
from frontier.config import SimulationConfig
from frontier.config.flat_dataclass import create_flat_dataclass
from frontier.types import ClusterType
path=pathlib.Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/balanced_frontier_input/frontier.json')
flat=json.loads(path.read_bytes())
output=pathlib.Path(flat['metrics_config_output_dir'])/'deepseek_ai_deepseek_v4_flash'/'online_serving'/flat['metrics_config_run_id']/'config.json'
before=hashlib.sha256(output.read_bytes()).hexdigest()
with patch.object(SimulationConfig,'write_config_to_file',return_value=None):
    config=create_flat_dataclass(SimulationConfig)(**flat).reconstruct_original_dataclass()
roles=config.cluster_config.get_cluster_configs_for_disaggregation()
commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
checked=[]
for name,module in tuple(sys.modules.items()):
    if name.startswith('frontier.') and getattr(module,'__file__',None):
        actual=pathlib.Path(module.__file__).resolve()
        relative=actual.relative_to(root)
        pinned=subprocess.check_output(['git','-C',str(root),'show',f'{commit}:{relative}'])
        assert actual.read_bytes()==pinned,(name,actual)
        checked.append(str(relative))
observed={str(role):dict(model=cluster.replica_config.model_name,tp=cluster.replica_config.attn_tensor_parallel_size,dp=cluster.replica_config.attn_dp,ep=cluster.replica_config.moe_expert_parallel_size,pp=cluster.replica_config.num_pipeline_stages,routing_distribution=cluster.replica_config.moe_routing_distribution_type,routing_runtime=cluster.replica_config.moe_gating_routing_runtime_path,world_size=cluster.num_replicas*cluster.replica_config.world_size) for role,cluster in roles.items()}
assert set(roles)=={ClusterType.PREFILL,ClusterType.DECODE}
assert all(x['tp']==8 and x['dp']==2 and x['ep']==16 and x['world_size']==16 and x['routing_runtime']=='balanced_topk' for x in observed.values())
assert hashlib.sha256(output.read_bytes()).hexdigest()==before
print(json.dumps(dict(status='PASS_SUBSTRATE_ONLY_NOT_R3_GREEN',source_commit=commit,schema_origin=sys.modules[SimulationConfig.__module__].__file__,flat_origin=sys.modules[create_flat_dataclass.__module__].__file__,source_modules_bytes_checked=len(checked),effective_roles=observed,existing_config_archive_unchanged=True),indent=2))
PY
```

Observed source186dfef5503ecb5329bd0fa96f9e978cc813276a,77 modules origin/bytes checked, P/D each TP8/DP2/EP16/PP1/world16, balanced/balanced_topk. **PASS substrate only; R3 still RED**. Actual current model experimental warnings and monolithic-num_replicas ignored warning were emitted; no native/graph/price/Simulator qualification claimed. Only config archive write suppressed to preserve root's existing file; actual constructors and materialization ran. Its hash remained unchanged.

Initial substrate invocation outside harness cwd failed relative `data/config/models/...json` lookup; correct production fresh interpreter needs cwd=harness. Next receipt referenced nonexistent cluster.world_size; corrected the observation to existing num_replicas*replica_config.world_size. No fallback or runtime behavior edit.

## Scope and unresolved decision

Smallest current ordinary proposal: existing R3 isolated-interpreter code, complete schema from pinned imports, actual git-source origin/bytes, real reconstruction and effective P/D comparisons, explicit DP; existing tests gain DP/ambient controls and harness/commit context. R3 incremental+41/-6=47 but cumulative from9d44675 routing70->105=net35. Tests incremental+21/-10=31, cumulative99->110=net11. Total net46. No source or proposed diff was executed.

Actual874 + reserved C12 generation/C1 runner/B4 timeout + repair46 = **937**, still23 maintained source files; B253+46+4=303 vs maximum266. Need37 extra aggregate lines or legitimate reductions elsewhere. No waived checks/formatting compression/generated hidden code offered. Optional C10-line persisted missing-row consumer test was previewed only and superseded; C source-backed direct5+9+16 controls remain separately recorded.

**Unfinished:** both P2 fixes/GREEN; root scope choice; original R2 predicates must stay identical; native05 remains UNVERIFIED; common review/freeze/new packages/actual runtime proof all root-owned.

## Approved R1 final verification (supersedes the scope pause)

YC approved23/950 and root assignedA432/B316/C202. B312 before four reserved root timeout lines. R3/test source edits only; no native/C26/R2/strategy/old-package edits, builder generation, package sealing, GPU/platform/SSH/cloud/Docker or delegation.

### Persistent red-capable regressions against65743ea

The existing real-input fixture now contains `dp`, `ambient` and `source_bytes` controls. Actual pre-fix source was recovered by `git show`, not reset/reverted in the worktree. Run at B worktree:

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
from pathlib import Path
import subprocess,tempfile,sys
sys.path.insert(0,'/data/ycfeng/Frontier/.worktrees/v4-balanced-callers')
import frontier.config
sys.path.insert(0,'/data/ycfeng/Frontier/.worktrees/v4-balanced-binding')
import pytest
with tempfile.TemporaryDirectory(dir='/data/ycfeng/tmp/opencode-v2-tmp/opencode',prefix='native-r1-persistent-red-') as directory:
    old=Path(directory)/'check_groundtruth_run_successor_r3.py'
    old.write_bytes(subprocess.check_output(['git','show','65743ea:task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py']))
    class OldSource:
        def pytest_runtest_setup(self,item):
            original=item.module.load
            item.module.load=lambda path: original(old if path.name=='check_groundtruth_run_successor_r3.py' else path)
    result=pytest.main(['tests/unit/test_native_reader_binding.py','-k','balanced_real_frontier and (dp or ambient)','-q','-p','no:cacheprovider','--basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-balanced-r1-persistent-red02'],plugins=[OldSource()])
    assert result==pytest.ExitCode.TESTS_FAILED
    print('EXPECTED_RED_PERSISTENT_REGRESSIONS_AGAINST_65743ea',result)
PY
```

Observed2FAIL in0.67s: DP `Failed: DID NOT RAISE ... (KeyError, ValueError)`; ambient `ModuleNotFoundError: import of frontier.config halted; None in sys.modules`. This is expected RED. The first experimental injection patched a separate imported test module instead of pytest's collected module, producing2PASS and outer AssertionError; corrected injection targets item.module during setup. Source was not modified for old-source runs.

Actual unmodified full-input A/B preload RED results and rehashedDP acceptance are retained in `evidence/native_r1_red.json`: same input SHA747a08e5.../declaredA, B774 rejects and A775 accepts; rehashedDP1 SHA9efcff9b... accepts. These direct child calls used `[sys.executable, '-I', '-B', '-c', code, ambient_root, checker_path, staged_input]` in approved temp cwd.

### Exact current-source GREEN commands

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_reader_binding.py -k 'balanced_real_frontier and (dp or ambient or source_bytes)' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-balanced-r1-persistent-green
env PYTHONDONTWRITEBYTECODE=1 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 NATIVE_READER_BINDING_CHECKER=/data/ycfeng/Frontier/.worktrees/v4-balanced-binding/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_serving_inventory.py tests/unit/test_native_v2_serving_inventory.py tests/unit/test_native_reader_binding.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-balanced-r1-green05 --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-balanced-r1-green05.xml
```

Observed targeted3PASS in2.46s before final source-origin assertion; final affected133PASS in50.34s includes that assertion and all10 actual input variants. No failure/skip. No A preload is needed now; B774 checker process uses declared acceptedA186dfef in its isolated child. This settles the original worktree-dependent bug, not merely an import workaround.

### Actual imports/roles/archive and rejected identities

`evidence/native_r1_green.json` retains two complete actual receipts (ambientA/B identical) and exact rejection errors. Current direct verifier uses declared manifest.harness_worktree/harness_commit, not candidate-inferred schema or current HEAD. Without manifest, explicit config.harness_worktree/harness_commit is required. Immutable revision must resolve to the same full commit object ID; missing revision raises KeyError, `HEAD` raises `Frontier schema requires the declared immutable full harness revision`.

Child command is exactly `[sys.executable, '-I', '-B', '-c', code, harness, staged_input, revision, json.dumps(required)]`, with `cwd=harness`, captured stdout/stderr. Existing complete flat factory and constructors run; only archive-write observer is suppressed. Existing architecture warning/cluster summaries precede final JSON; parser consumes the final structured line, never interprets warnings as a receipt. On failure it raises ValueError with the child's actual stderr traceback.

| Condition | Actual observation | Verdict |
| --- | --- | --- |
| Complete real775-field input independent of A/B preloads | Both direct receipts exactly equal, declared revision186dfef5503ecb5329bd0fa96f9e978cc813276a | PASS |
| Source schema origin is actual declared path/bytes | config.py at A, SHA f03aec143e22fc5132288e54be4ecb759e2d08b662379b3ff58b680144d63d86; flat_dataclass.py at A, SHA24481c37deea901058b3da6612b162f6287e3ad045eb4cb721fe28c363ab9645 | PASS |
| All imported Frontier code bound to immutable revision |77 actual frontier.* module origins within A; current bytes equal git-show declared revision; receipt records actual origins/hashes | PASS |
| Real effective roles, not inferred role labels | existing reconstruct_original_dataclass and get_cluster_configs_for_disaggregation materialize exactlyP/D; observed eachTP8/DP2/EP16/PP1/world16/DeepSeek-V4-Flash/h800/balanced/balanced_topk | PASS |
| RehashedDP1, old source bytes and actual-origin shadow reject | DP1 rejected by canonical case check; different immutable9d44675 revision rejects actual model_architectures.py bytes; injected module origin `/unbound/flat_dataclass.py` rejects outside harness | PASS |
| Original archive preserved | SHA3808af195b09e32074922b2f068e4b670ba1a8ed80fe916a2527967ab3997223 before/after | PASS |
| Source shadow/native callable/missing env controls retained | existing reader/observer/strategy origin tests all in133PASS | PASS CPU |
| Original frozenR2 authority/non-graph predicates | hash29523700ac8492fa6e192ce272f8ab5a7a48d1528bd4064417e26e811533d10a; provenance/inputs/replay/workers/lifecycle/terminal AST unchanged | PASS |

Direct controls count8 includes standalone explicit pinned config; origin shadow uses the actual child string extracted from R3 with one injected module.__file__ to model a wrong actual import. It does not edit another checkout or run GPU code. Existing source-byte persistent case asserts the specific imported-source error, not arbitrary failure. Original model/cluster warnings retain closed runtime guards; no numerical/graph/FP8 proof is claimed.

### Exact cumulative accounting and root-only remainder

Source incremental diff from65743ea: R3+52/-6, reader tests+20/-7 =85 changed lines. Count from9d44675: R3+116/-0, reader tests+112/-0; original four B files84 lines. Maintained builder is compared with preserved pre-edit baseline SHA21b5a7232e31df74052856d933c003b5af9caca7ce2fc6362fc1dbca36593560 (routing29, separate dispatch3). **Six maintained files/B312 routing**, reserved root timeout4 givesB316, aggregateA432+C202+B316=950. Dispatch488+3=491 is distinct. `evidence/native_r1_scope.json` and `evidence/native_r1_source.diff` are evidence, not parallel implementation.

Root must replace the existing builder validation invocation before generation:

```python
checker.verify_balanced_frontier(config | {'harness_commit': commit}, engine)
```

It supplies the existing clean immutable builder snapshot without adding a persistent config field, preserves input hashes and replaces a routing-added line (incremental+1/-1, cumulative routing net0). Earlier suggestion to pass bare lock was corrected because it lacks input hashes. Builder remains unchanged in this step; it correctly cannot call the verifier without a declared revision. Root owns this call replacement, timeout4, final merged source review/freeze/package and actual runtime proof. Native05 remains UNVERIFIED. No source/CPU result seals or accepts a GPU package.
