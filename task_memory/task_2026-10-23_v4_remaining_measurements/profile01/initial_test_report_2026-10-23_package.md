# Native profile01 package: CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded actual generation, source/workload preservation, staged CPU controls and verification limits. |

Task: W128-V4-PROFILE01-PACKAGE. Worktree: /data/ycfeng/Frontier/.worktrees/v4-native-profile-run, branch feat/v4-native-profile-run. Baseline1dc93c6fa93210aa898fd10481ce91a05e6709ee; source commit69f92c553ff4058be613641bce0a42a9150a115b. Scope: builder90 additions/5 deletions + profile plan44 additions =139 gross lines/300. Runtime remains05952396b0b8e5a794c53e10f9a21213beb5193d.

## Environment and actual generation

Builder/frozen Frontier reconstruction: /data/ycfeng/envs/frontier-py310/bin/python, Python3.10.6 (GCC11.3.0). CPU package controls: /data/ycfeng/envs/asyncopd-py312/bin/python, Python3.12.13 (Clang22.1.3), PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site. All Python commands used -B; no dependency changes or conda activation. Native source was read locally, clean at2cf0a6915ce544dc493a0990f2ea38d81601128a. No GPU initialization, platform, SSH, cloud access, Docker, seal or launch.

Exact source commit and generation commands, from the worktree above:

```bash
git diff --check
git add -f task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile_plan.json task_memory/task_2026-10-23_v4_remaining_measurements/profile01/progress.md
git diff --cached --numstat
git diff --cached --check
git commit -m "Stage diagnostic V4 native profile packages through the existing builder"
git status --short --branch
/data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py --generation 1 --profile-plan task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile_plan.json
```

Expected: clean committed input, one new unsealed diagnostic package, no overwrite. Actual: source commit69f92c55; git status clean before generation; first generation succeeded, native_source_pins87, unsealed=true, actual_submission_or_native_admission=false. Package: task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_01. No failed generation or generation rerun. Package remained untouched after its successful build.

## Actual controls and results

| Criterion | Expected | Actual | Result |
| --- | --- | --- | --- |
| Runtime bytes | Every staged runtime module equals git05952396 and packaging commit | 13/13 exact; source_lock committed_files and config tool_hashes match | PASS |
| Native source pins | Original74 plus actual profiler/RPC/hook callsites | 87/87 hashes;13 additions; all11 hook modules covered (19 methods,10 functions,2 launch hooks) | PASS |
| Tool/retained hashes | Match staged files, including extension/bootstrap/overlay/launch/seal/plan | 23/23 tool hashes,10/10 retained hashes | PASS |
| Model metadata | Original4 files and accepted C26 leaf unchanged | 4/4 metadata; overlay_model.py d3556a47056bd52680b295f1585906149310397ec53ab90aa0008ff7d57e86ee | PASS |
| Workload | Original5 files match proposal_inputs and FINAL_FREEZE_R4 | Both copies5/5 match;128formal/16warmup templates,3repetitions =>384formal+48warmup rows | PASS |
| Configuration identity | Only profile selection differs from native06 engine | Removing profiler_config/worker_extension_cls makes engines equal | PASS |
| Topology/resources/timeouts | P16+D16,TP8/DP2/EP16/PP1;8GPU/node,4nodes,64CPU,512GiB;3600/1800/1800 | Exact configuration values; model/workload/layout preserved | PASS |
| Diagnostic restrictions | No E2E/reference admission and no four-metric acceptance transplant | config/case/readiness/lock/plan diagnostic_only=true,e2e_reference_admission=false; historical metric policy removed | PASS |
| Profiler config | Actual native field names,torch64,zero delay/warmup/wait,ignore_frontend,gzip; other features off | Native ProfilerConfig class/decorator/validators executed via AST with only logger/hash import stand-ins; negative count/unknown key/kind rejected3/3 | PASS |
| CPU command path | Both roles and DP heads/peers,extension,profiler arguments,balanced routing | node_plan + instance_env + existing server_command4/4; localDP1; peer start_rank1/headless; correct NIXL producer/consumer | PASS |
| Trace paths | Existing worker-derived role/DP directories | torch_traces/{prefill,decode}/{dp0,dp1},4distinct paths | PASS |
| Collector limits | Actual config/env keys and parser | serving_profile_limits => FRONTIER_NATIVE_PROFILE_LIMITS => records250000/bytes268435456,4/4 | PASS |
| Balanced Frontier | Frozen actual schema/role reconstruction | Existing R3 verify_balanced_frontier passed;84 imported sources pinned; each role world_size16 | PASS |
| Historical06 behavior | No unselected-path semantic change | AST after plan-branch elision matches baseline; only independent package-name/assert order normalized | PASS |
| Original preservation | Frozen originals and frozen R3/R2/R4 immutable | 123/123 original file hashes and3/3 checker hashes reverified before/after controls | PASS |
| Matrix | Explicit existing77/77 binding for byte-identical runtime/simulator domains | matrix_binding_final SHA e6d1b99ac31cd0c943c34f7ea4906aa1d73cc4abb0be7188309cca149b94346d; comparison SHA7a8728ecc0851373827d9cdc7dfb68c552299e6c54b5fae75bd8cc5429d36ad6; no new capture | PASS by binding |

The requested build_command path is the existing server_command function; no function named build_command exists. CPU controls aliased build_command=vllm_replay.server_command and exercised it rather than adding another command builder. Collector limit forwarding was executed from the actual staged worker AST; the hardware-dependent worker run itself was not invoked. Native Worker.profile source explicitly selects CPU+CUDA activities; no native profiler or GPU hook installation was claimed.

Package45files/6,980,280bytes. Full maps live in config.json, source_lock.json and delivery.json. Map digests use SHA256(json.dumps(map,sort_keys=True,separators=(',',':')).encode()). Package file-map digest5b6fc997102102f2f4ec300ec52c1edc91ffecd7324f93587af119c14b4fadcf.

Actual staged balanced reconstruction command:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
import importlib.util,json,pathlib
r=pathlib.Path.cwd();p=r/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_01';c=json.loads((p/'config.json').read_text());l=json.loads((p/'source_lock.json').read_text());s=importlib.util.spec_from_file_location('checker',r/'task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);result=m.verify_balanced_frontier(c|{'harness_commit':l['harness_commit']},c['engine']);print(json.dumps({'staged_balanced_frontier':'PASS','sources':len(result['sources']),'roles':result['roles']},indent=2))
PY
```

## Preserved failures and resolution

Exact final package/hash/CPU-path confirmation command (the in-memory names build_command and staged_limit_forwarding do not create files):

```bash
PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site /data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
import ast, hashlib, json, os, pathlib, subprocess, sys
from types import SimpleNamespace
from unittest.mock import patch
r=pathlib.Path.cwd();t=r/'task_memory/task_2026-10-23_v4_remaining_measurements';d=json.loads((t/'profile01/delivery.json').read_text());p=pathlib.Path(d['package']);c=json.loads((p/'config.json').read_text());lock=json.loads((p/'source_lock.json').read_text());e=c['engine'];n=pathlib.Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def mapsha(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
files={str(x.relative_to(p)):sha(x) for x in p.rglob('*') if x.is_file()}
assert files==d['package_hashes'] and len(files)==45
assert mapsha(files)==d['hash_maps']['package_map_sha256']
assert sum(x.stat().st_size for x in p.rglob('*') if x.is_file())==6980280
assert not (p/'manifest.json').exists() and not list(p.rglob('__pycache__'))
assert c==json.loads((p/'case/inputs/native_config.json').read_text())
assert e==json.loads((p/'engine.json').read_text())==json.loads((p/'case/inputs/engine.json').read_text())
for name,target in c['harness_files'].items():
    content=(p/target).read_bytes()
    for revision in (d['accepted_runtime_commit'],d['packaging_source_commit']):
        assert content==subprocess.check_output(['git','show',f'{revision}:{name}'])
    assert c['tool_hashes'][str(p/target)]==sha(p/target)
for name,expected in c['expected_source_hashes'].items():
    assert sha(p/'overlay_model.py' if name=='vllm/models/deepseek_v4/nvidia/model.py' else n/name)==expected
for name,expected in c['tool_hashes'].items(): assert sha(pathlib.Path(name))==expected
for name,expected in c['retained_files'].items(): assert sha(p/name)==expected
for name,expected in d['model_hashes'].items(): assert sha(p/'model'/name)==expected
inputs=json.loads((t/'driver/proposal_inputs.json').read_text());freeze=json.loads((r/inputs['freeze_path']).read_text())
for name,expected in d['workload_hashes'].items():
    row=inputs['files']['trace/'+name]
    assert sha(p/'trace'/name)==sha(p/'case/inputs/trace'/name)==expected==row['sha256']==freeze['original_files'][row['source_path']]
for name,expected in freeze['original_files'].items(): assert sha(pathlib.Path(name))==expected
for name,expected in dict(freeze['dependencies'],**{freeze['checker_path']:freeze['checker_sha256']}).items(): assert sha(pathlib.Path(name))==expected
sys.path[:0]=[str(p),str(p/'tests/comparison/calibration'),str(p/'tests/comparison/dp_placement_pp')]
import native_pd_worker, pd_replay, vllm_replay, native_serve_profile
build_command=vllm_replay.server_command
peers=[dict(host=f'cpu-node-{i}',ip=f'192.0.2.{i+1}') for i in range(4)];observed=[]
for peer in peers:
    plan=native_pd_worker.node_plan(c,peers[::-1],peer['host'],[str(i) for i in range(8)])
    role,lane,engine=plan['role'],plan['lane'],plan['engine']
    env,changed=pd_replay.instance_env('clean',engine,p,{'NCCL_DEBUG':'WARN'},'0,1,2,3,4,5,6,7',observer=None,role=role)
    command=build_command(engine,p/'model',c['http_ports'][role])
    expected=e['profiler_config']|{'torch_profiler_dir':str(pathlib.Path(e['profiler_config']['torch_profiler_dir'])/role/f'dp{lane}')}
    assert json.loads(command[command.index('--profiler-config')+1])==expected
    assert command[command.index('--worker-extension-cls')+1]=='native_serve_profile.NativeServeProfile'
    assert ('--headless' in command)==(lane==1) and ('--data-parallel-start-rank' in command)==(lane==1)
    assert env['VLLM_MOE_ROUTING_SIMULATION_STRATEGY']=='balanced_topk' and env['NCCL_DEBUG']=='WARN'
    assert env['PYTHONPATH'].split(os.pathsep)[0]==str(p/'tests/comparison/calibration')
    statement=next(x for x in ast.walk(ast.parse((p/'native_pd_worker.py').read_text())) if isinstance(x,ast.If) and ast.unparse(x.test)=="'serving_profile_limits' in config")
    exec(compile(ast.Module(body=[statement],type_ignores=[]),'staged_limit_forwarding','exec'),dict(config=c,env=env,changed=changed,json=json))
    with patch.dict(os.environ,{'FRONTIER_NATIVE_PROFILE_LIMITS':env['FRONTIER_NATIVE_PROFILE_LIMITS']}):
        assert native_serve_profile.ServingProfile(SimpleNamespace()).limits=={'records':250000,'bytes':268435456}
    observed.append((role,lane))
assert observed==[('prefill',0),('prefill',1),('decode',0),('decode',1)]
manifest,formal,warmup=pd_replay.canonical_workload(p/'trace');assert (len(formal),len(warmup),manifest['repetitions']['clean_repetitions'])==(128,16,3)
print(json.dumps({'package_hashes':'PASS','runtime_bytes_05952396':'PASS','original_preservation':'PASS','role_dp_paths':observed,'workload_templates':[128,16,3]}))
PY
```

Environment-only exploratory probe, exit1:

```text
$ /data/ycfeng/envs/asyncopd-py312/bin/python -I -B -c 'from frontier.config import SimulationConfig'
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'frontier'
```

Cause: -I omits cwd/PYTHONPATH. The existing frozen checker explicitly inserts the worktree; its real reconstruction succeeded under existing py310. No environment or staged output changed.

First inline CPU package control, exit1 (generation had already succeeded):

```text
Traceback (most recent call last):
  File "<stdin>", line 46, in <module>
  File "/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/config/utils.py", line 51, in <module>
    @dataclass_transform(field_specifiers=(PydanticField,))
                                           ^^^^^^^^^^^^^
NameError: name 'PydanticField' is not defined
```

Cause: inline AST verification namespace omitted the native decorator's imported pydantic Field name. Retried only CPU verification with PydanticField=Field; actual native dataclass/validator positive and3negative controls passed. No package regeneration or mutation. An inspection-only out-of-range read also returned `Offset 420 is out of range for this file (415 lines)`; no code failure or source change resulted.

## Limits and root-owned work

No GPU/native/reference qualification, useful-batch coverage, numerical costs or four-metric claims. Runtime hook resolution, actual stream/event/PDL/graph/CUPTI/trace coverage and all32rank receipts remain unverified. Image digest is pinned, but installed image/binary equality remains root-owned; cloud destination unused status and quota were not inspected by this CPU owner. Root independently inspects and seals; do not regenerate into this package. Final record-only commit does not alter staged runtime or packaging lineage.
