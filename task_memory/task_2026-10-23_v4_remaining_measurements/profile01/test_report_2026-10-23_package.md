# Native profile01 package: CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded actual generation, source/workload preservation, staged CPU controls and verification limits. |
| 2026-10-23 | Appended reproduced profile01 P1 RED, packaging-only repair d87f4c74 and fresh profile02 launcher/bootstrap-child GREEN. |

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

## Reopened P1: historical profile01 FAIL, fresh profile02 GREEN

This section supersedes the original package-entry/completion claim above. The original commands supplied sys.path and/or imported pd_replay before native_pd_worker, so they established source/command integrity but missed the actual entry defect. Historical report/progress bytes at record commit62092da3 are archived as initial_test_report_2026-10-23_package.md and initial_progress.md. initial_delivery.json and initial_source.diff preserve the original delivery/diff bytes exactly. Original candidate01 remains45files/6,980,280bytes, unsealed/unsubmitted and unchanged.

### Root reproduction and owner RED

Root evidence: /data/ycfeng/tmp/glm53_dsv4/profile01_entry_red_acbrv5ib/{receipt.json,stderr.log}. Owner reproduced the exact command twice, no preload:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -I -B -c 'import sys; sys.path.insert(0,sys.argv[1]); import native_pd_worker' /data/ycfeng/Frontier/.worktrees/v4-native-profile-run/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_01
```

Actual exit1, preserved verbatim:

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/data/ycfeng/Frontier/.worktrees/v4-native-profile-run/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_01/native_pd_worker.py", line 19, in <module>
    from vllm_replay import NATIVE_WORKER_EXTENSIONS
ModuleNotFoundError: No module named 'vllm_replay'
```

The first actual generated01 launcher on a disposable CPU copy with test-only manifest also failed before write_run_manifest, exit1. Exact command/traceback are old_launcher_red.{receipt.json,stderr.log} under /data/ycfeng/tmp/glm53_dsv4/profile02_owner_entry_nUFzTw. Neither original package was sealed, launched or submitted.

### Bounded repair and single clean generation

Repair/source commitd87f4c74b9d5c0eec0958ba2b7351e76b04acf14. Only maintained source/config paths are the allocated build_candidate.py and profile01/profile_plan.json. Incremental diff62092da3..d87f4c74: builder14add/3del=17, plan2add/2del=4, total21gross. Cumulative1dc93c6f..d87f4c74: builder102add/6del=108, plan44add/0del=44, total152/300gross. No accepted runtime/native/C26 source changes.

The profile-only generated launcher inserts package root and its actual tests/comparison/dp_placement_pp before native_pd_worker import. Generated bootstrap replaces root-only PYTHONPATH with root plus that actual staged dependency directory; it does not append inherited external paths. Retained source_lock distinguishes original source_sha256 and staged_sha256; config.retained_files/tool_hashes bind staged bytes. Generation02 is narrowly admitted; cloud root/resources/layout/window unchanged. Repair AST equality to69f92c55 after profile-branch elision passed, preserving its prior baseline06 equivalence.

Exact source/generation commands from the allocated worktree:

```bash
git add -f task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile_plan.json task_memory/task_2026-10-23_v4_remaining_measurements/profile01/progress.md task_memory/task_2026-10-23_v4_remaining_measurements/profile01/initial_delivery.json task_memory/task_2026-10-23_v4_remaining_measurements/profile01/initial_source.diff
git diff --cached --check
git diff --cached --stat
git commit -m "Fix flattened profile package entry paths and select fresh generation02"
/data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py --generation 2 --profile-plan task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile_plan.json
```

Builder's clean-tree assertion passed. Fresh02 staging succeeded once, no generation failure/rerun. Actual package task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_02;45files/6,980,696bytes;87native pins;unsealed=true;actual_submission_or_native_admission=false.

### Actual fresh entry and affected package controls

All checks below used existing Python3.10.6 unless explicitly py312; no conda/dependency changes. Fresh-entry subprocesses remove inherited PYTHONPATH/PYTHONHOME and real authorization/access secrets. Only a literal CPU-only sentinel authorization exists; no real credentials are read. New launcher process uses -I and -B, runpy and tracing only, with no test-added sys.path/import preload. A test-only manifest is written solely in a disposable copy. Trace stops at the call event before write_run_manifest body; no cloud reads/writes, real run files, credential reads, submission or platform code follows. The generated launcher, not the test, resolves the worker/dependencies.

Exact commands:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -I -B /data/ycfeng/tmp/glm53_dsv4/profile02_owner_entry_nUFzTw/entry_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -I -B /data/ycfeng/tmp/glm53_dsv4/profile02_owner_entry_nUFzTw/package_controls.py
/data/ycfeng/envs/asyncopd-py312/bin/python -I -B /data/ycfeng/tmp/glm53_dsv4/profile02_owner_entry_nUFzTw/native_profiler_controls.py
```

Script SHA256s: entry_controls4b34a18186a677019ddac8c6316ac85fcb5cc0e8c00cbc6278804ab38954a8e0; package_controlsd53235745cf180e1cd4964ffd8afd5db4260dd5c1cb955774cc4c5374a03dc23; native_profiler_controls5526340115fe9b48eaf8a340ed8cabd6c8c484091c23d5bb1dbb4f6b53357a44. They are explicitly disposable CPU verification, not new maintained test machinery. Per-subprocess receipt.json files record complete actual argv; delivery.json records all evidence hashes.

| Control | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Historical01 actual launcher | Same missing dependency as root | Exit1, ModuleNotFoundError vllm_replay at worker19 | RED reproduced |
| Fresh02 actual launcher | Generated entry alone imports worker; sentinel before body | Exit0, FRESH_LAUNCHER_GREEN_BEFORE_WRITE_RUN_MANIFEST_BODY; pd_replay absent from sys.modules; vllm_replay origin in copied staged dp_placement_pp | GREEN |
| Generated bootstrap env/child | Actual generated bootstrap verifies retained tools and produces sufficient env | Real bootstrap function with explicitly labelled installed source/binary/PathFinder stand-ins; env exactly stage:stage/tests/comparison/dp_placement_pp; fresh child import exit0, BOOTSTRAP_CHILD_IMPORT_GREEN, no preload/test path insert | GREEN wiring only |
| Actual worker pre-output imports | Imports through generated entry and real run() statements, not assisted paths | Sentinel at root=Path(config['output_dir']) after own imports, before any output-path access | PASS |
| Runtime/native/C26 | Accepted bytes unchanged | 13/13 runtime equals05952396 and d87f4c74;87/87native source hashes including unchanged C26 leaf | PASS |
| Tool/retained provenance | New staged launch/bootstrap hashes included | 23/23tools,10/10retained; original source_sha256 and staged_sha256 rehashed; source_lock/config agree | PASS |
| Model/workload/resources | Exact original values | 4/4model files,5/5workload files/both copies,128formal/16warmup/3repetitions;3600/1800/1800 and all original resources unchanged | PASS |
| Four role/DP CPU commands | Balanced routing/extension/profiler/headless path retained | prefill0/1 and decode0/1;TP8/DP2/localDP1;64step/zero delay/warmup/wait/ignore_frontend/gzip;4derived trace directories;records250000/bytes268435456 | PASS |
| Native ProfilerConfig | Native class/decorator/validators and actual names | Positive config PASS;3negative controls rejected using AST with logger/hash stand-ins only | PASS CPU config |
| Frozen original files/checkers | Nothing modified | Original123/123 and checker3/3 hashes; original package45/45; all initial archives byte-exact | PASS |
| Matrix/default06 | Existing accepted binding suffices for unchanged runtime | Same explicit77/77 binding hashes; repair default06 AST equality | PASS by binding |
| Real candidates | No actual seal/run output/mutation | Both45files unchanged through entry controls, no manifest.json/pycache/case/runs | PASS |

Balanced Frontier command (same actual checker, not a stand-in):

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
import importlib.util,json,pathlib
r=pathlib.Path.cwd();p=r/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_02';c=json.loads((p/'config.json').read_text());l=json.loads((p/'source_lock.json').read_text());s=importlib.util.spec_from_file_location('checker',r/'task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);result=m.verify_balanced_frontier(c|{'harness_commit':l['harness_commit']},c['engine']);assert len(result['sources'])==84;assert all(role['world_size']==16 for role in result['roles'].values());e=pathlib.Path('/data/ycfeng/tmp/glm53_dsv4/profile02_owner_entry_nUFzTw');(e/'balanced_frontier.receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'staged_balanced_frontier':'PASS','sources':len(result['sources']),'roles':result['roles']}))
PY
```

Actual PASS84sources; each roleworld_size16 and balanced/balanced_topk. Fresh02 map digests (same canonical JSON serialization as original): native1737349f185dea21aea168d128bb2aa51b5a914cc7dda0d845a284dfcf11659e; runtime61e9a9a23cf023e283f20b769ef74af4c8fb247b2f8ef848827a1b47cc446cbd; tools59968ed79868b1ee4dd8d179b4f397cb827156d99da413972ab707d3dae5fad0; model64e57c243f12c48826131cbad6223508277f41ea0dc19048e8e59aa3780803fe; workload79a45bf08dde1f764829eb55201ca3d65824b46d86cbf166de968d2d406587b7; package7768ddd67c024b7c08942d1cecc2dce8e2c620eefae8d631e9703171e1411dee.

Generated bootstrap original source SHA8571fb6b3d7ae888d48eceb9a0f05f4da0b3259e9f3a2601792572bea7942dfc, staged SHA74cf0dec1c0b4d271abf2291252b9ca3cd6f6ce889a9f1f6485983eecb4393ec; launcher original source6bbd1cf8ee8aec27a23ce64dbbb2e72751475c4b326351b122a742ac4786e934, stagedbee2ca673e613e4acf4a83b04feb2194509a838dfd516f9ebc6a040fd05e907b. Only11of45file hashes differ between independently retained01/02, all attributable to generated entry/config/generation/provenance bindings; originals are not updated.

### Preserved auxiliary verifier failures (no staged-output changes)

The optional repeat native-config AST check initially omitted the native ConfigT annotation binding, then os import. Both exit1 traces are retained under the evidence directory; the final script restores only those actual imported/bound names, no package mutation or generation rerun.

```text
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "native_config_decorator", line 53, in <module>
NameError: name 'ConfigT' is not defined
```

```text
Traceback (most recent call last):
  File "/data/ycfeng/tmp/glm53_dsv4/profile02_owner_entry_nUFzTw/native_profiler_controls.py", line 33, in <module>
    config = namespace['ProfilerConfig'](**engine['profiler_config'])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/ycfeng/envs/asyncopd-py312/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in __init__
    s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
  File "native_profiler_config", line 187, in _validate_profiler_config
NameError: name 'os' is not defined. Did you forget to import 'os'?
```

### Remaining limits

No CPU packaging blocker remains. Root must independently re-review the repaired source/package and repeat the locked-image87sources/23tools/20binary verification with the new staged bytes, including actual bootstrap environment/import; the stand-ins prove generated wiring only. No actual candidate seal/launch, platform, SSH, cloud, GPU or Docker operation was performed. Diagnostic_only remains true; no native/reference/E2E qualification or four-hook runtime acceptance claim. All runtime coverage/32rank/stream/event/PDL/graph/CUPTI gates remain root-owned.

### Final artifact checks

Final delivery/archives validation PASS: both actual45file maps/counts/bytes match delivery/initial_delivery; four initial archives equal git62092da3 originals exactly; all evidence hashes/source.diff bytes match; frozen123originals/3checkers rehashed unchanged. git diff1dc93c6f..d87f4c74 --check for the two maintained sources PASS, and record diff check excluding source.diff PASS.

Plain git diff --check reported two context-only whitespace lines inside the byte-exact diff artifact (exit2, not source whitespace; trailing spaces are displayed as <SPACE> here):

```text
task_memory/task_2026-10-23_v4_remaining_measurements/profile01/source.diff:7: trailing whitespace.
+<SPACE>
task_memory/task_2026-10-23_v4_remaining_measurements/profile01/source.diff:8: trailing whitespace.
+<SPACE>
```

source.diff lines7/8 are git-generated blank context lines (one space), preserved as exact diff evidence rather than silently rewritten. Final checks:

The first report copy retained those literal spaces and its own record diff check flagged report lines271/273 (exit2). Only the human-readable log rendering above was changed to <SPACE>; source.diff remains exact.

```bash
git diff --check -- . ':(exclude)task_memory/task_2026-10-23_v4_remaining_measurements/profile01/source.diff'
git diff 1dc93c6f d87f4c74 --check -- task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile_plan.json
```
