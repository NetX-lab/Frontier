# Native R1 review reproductions and complete-scope estimate

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded two targeted REDs, real isolated reconstruction/source check and no-source-edit scope blocker. |

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
