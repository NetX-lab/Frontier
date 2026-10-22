# Native19 integration verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record owned CPU, D30, source/image-context and static graph results. |
| 2026-10-23 | Record reviewer repair red/green/affected checks, exact quota and narrower graph evidence. |
| 2026-10-23 | Append documentation/input-only T2 static conformance and resource proposal evidence. |
| 2026-10-23 | Begin two-file engine-limit regression; record red constructor-flow and preserved old controls. |
| 2026-10-23 | Verify engine-limit actual-entrypoint/helper-body flow and affected175 PASS/15 SKIP; refresh input provenance and exact native budget. |

## Environment and exact commands
All commands ran in /data/ycfeng/Frontier/.worktrees/step5-position-native.
Prefix: `env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
Python /data/ycfeng/envs/asyncopd-py312/bin/python,3.12.13; Torch2.9.1+cu128.
No CUDA/GPU operation was run. Baseline251 PASS is in the interface test report.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_position_cyclic.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_reference.py tests/unit/test_step5_native_image.py tests/unit/test_step5_native_forward_census.py tests/unit/test_step5_http_capture.py tests/unit/test_step5_native_source_contract.py tests/integration/step5_environment/test_role_capacity_observation.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-final04 > /data/ycfeng/tmp/d29-native-final04.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29-native-derived-20261023-candidate01 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_finish_sync.py tests/unit/test_step5_native_forward_census.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-d30-final02 > /data/ycfeng/tmp/d29-native-d30-final02.log 2>&1
```

Actual:418 PASS/39.78s and48 PASS/67.40s. The latter used Python3.10.6.
Expected: retained old controls, actual client adapter propagation including
independent direct-D, HTTP fresh/reused/EOF semantics, conditional model/context/
factory/simulator wrong-root/spec/hash/missing module rejection, soft native
return/exception semantics, D30steps1/16 andengine/core source/counter coverage.
Actual derived staged observation retains13 canonical source hashes including
vllm.v1.engine.core. Static source pin agreement is not imported live evidence.

Earlier attempts: wrong capacity-suite path caused exit4/no tests;
asyncopd lacksplotly for Frontier scheduler imports(exit2). Existingfrontier-py310
was used instead, no dependency edits. One D30 test expects candidate parent root,
not candidate/vllm; final command uses its real convention, all48 PASS.
Logs are retained at /data/ycfeng/tmp/d29-native-final02.log,
d29-native-final03.log andd29-native-d30-final.log.

## Source/context/static checks
Builder directly executed `prepare(base, candidate, listing, output)` using:
- base /data/ycfeng/tmp/step5_profiling_20261020/image_src;
- candidate /data/ycfeng/tmp/d29-native-derived-20261023-candidate01;
- listing /data/ycfeng/tmp/d29-native-base-build-pins.txt;
- initial output /data/ycfeng/tmp/d29-native-build-context01;
- final output /data/ycfeng/tmp/d29-native-build-context02 (rerun after builder checks).
Result PASS: exact4 changed source files; all other candidate/base file bytes
agree; pin+manifest+patch hashes retained inbuild_context_receipt.json. Base
listing53 is50 existing sealed pins plus3 local base-export hashes, not actual
installed derived evidence. Container/image operations NOT_RUN.

Scoped follow-up `python -B -m pytest tests/unit/test_step5_native_image.py -q
-p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-image-final`:
1 PASS/0.15s after additional foreign-file/base-hash rejection controls.
Final builder iteration rejects all source symlinks before walking inventories;
same focused test1 PASS/0.15s atd29-native-image-final02 andactual context02 PASS.

AST parsed all owned Python files and4 actual native files. T2 static checks
require real native eager-break decorator, wrapper counters>0/>=2, stable tensor
addresses and capture counts, actual opaque-MoE seam, bound positions/mask,
FP8/BF16 existing layer loader and collective/expert observers. Result PASS
source-only; no runtime claim. Dockerfile's Python payload AST PASS,4 native
COPY paths only, checked-hash bytecode compilation, no forced ENV/pip/backend rebuild.
Receipt:t2_static_receipt.json. `git diff --check`:PASS.
Final baseline-to-HEAD whitespace check must exclude the retained unified patch
artifact: its blank context lines contain a required single-space diff prefix,
which Git's generic added-text checker reports as trailing whitespace. Command:
`git diff --check 1aee0af HEAD -- . ':(exclude)task_memory/task_2026-10-22_step5_d29_native_balanced/native_source.patch'`.
Actual native source line-ending checks,
all maintained repository files and the unchanged patch SHA256 PASS. The patch
was not stripped or rewritten to hide/change its evidence.

Manifest byte checks: capture02all56 entries PASS;
capture03all57 entries PASS and original manifest
cdaad24ecded6da6f2eea90ed4ef4a582432db2fd75c56c29aafb22f2f56a57d.
capture01not present at inspected parent root; no historical capture writes.
Scope receipt counts actual19 paths and1017 gross diff includingnative99, ceiling1582.

## Limits
GPU NONE/PW/FULL, actual CUDA selected-score arithmetic, auto FP8/BF16 dispatch,
all-rank imported model/source evidence, derived image/dependencies/library bytes,
full serving/G3 state and transfer/op-gap/E2E remain UNVERIFIED. Mocked host-runtime
provenance controls above are not native execution or native qualification.

## Reviewer repair continuation (current)

Environment: same worktree, asyncopd-py312 Python3.12.13 (Clang22.1.3),
Torch2.9.1+cu128,pytest9.0.2. No package install, GPU, Docker or platform operation.
Root's separate local image Python3.12.3/Torch2.13.0+cu130/pytest9.1.1 observations
are not the environment used for the checks below.

### Exact regression commands

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_position_cyclic.py tests/unit/test_step5_native_pd_r2_observation.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-repair-red -k 't2 or instance_forward or kernel or monolithic or quant' > /data/ycfeng/tmp/d29-native-repair-red.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_position_cyclic.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_canonical_g3.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-repair-green02 > /data/ycfeng/tmp/d29-native-repair-green02.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_position_cyclic.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_reference.py tests/unit/test_step5_native_image.py tests/unit/test_step5_native_forward_census.py tests/unit/test_step5_http_capture.py tests/unit/test_step5_native_source_contract.py tests/integration/step5_environment/test_role_capacity_observation.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-repair-affected02 > /data/ycfeng/tmp/d29-native-repair-affected02.log 2>&1
```

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Before-fix fault/invocation regression | Fail at keyword capture, stale dummy buffers, instance-forward and invalid kernel/monolithic provenance |14 FAIL,20 PASS,157 deselected/2.76s; original call count1 but captured0, dummy assertion at line142, provenance faults accepted | FAIL as intended |
| Focused repaired controls | All router capture/soft/native exception, forward and backend relationship controls pass |307 PASS/8.35s | PASS |
| Final affected combination | Preserve adjacent adapter/canonical/reference/stage/source/capacity checks; reject added missing backend schema pin |485 PASS/36.16s | PASS |

Quick iterations: initial green01 was275 PASS/11.45s; affected01 was484
PASS/38.12s. Final iteration added a missing-backend-pin stage negative and
executed actual pinned PW dispatch/replay bodies, replacing the shallower
post-capture-only seam. Positive backend fixtures now use actual modular
impl properties and FP8 fallback/BF16 object relationships. They remain host
fixtures, not evidence of imported native execution. Actual-source T1/T2 CPU
selection has no CUDA/collective/backend initialization.

### Source, AST, quota and native preservation

Exact inventory/availability check (run with the same Python/env prefix):

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
import ast, hashlib
from pathlib import Path
from tests.integration.step5_pd.r2_runtime import BACKEND_MODULES
base=Path('/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm')
candidate=Path('/data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm')
left={p.relative_to(base):p for p in base.rglob('*.py')}
right={p.relative_to(candidate):p for p in candidate.rglob('*.py')}
assert left.keys()==right.keys()
changed=[str(p) for p in left if left[p].read_bytes()!=right[p].read_bytes()]
assert set(changed)=={'forward_context.py','models/step5/nvidia/model.py','model_executor/layers/fused_moe/router/router_factory.py','model_executor/layers/fused_moe/router/routing_simulator_router.py'}
for name in BACKEND_MODULES:
    path=name.removeprefix('vllm.').replace('.','/')+'.py'
    assert (base/path).read_bytes()==(candidate/path).read_bytes()
    ast.parse((candidate/path).read_text())
    print(name,hashlib.sha256((candidate/path).read_bytes()).hexdigest())
print('PASS',len(left),'Python source files; exactly4 approved diffs; all12 backend source files available and unchanged')
PY
git diff --numstat 1aee0af -- tests
git diff --check 1aee0af -- . ':(exclude)task_memory/task_2026-10-22_step5_d29_native_balanced/native_source.patch'
```

Observed:2907 Python source files, exactly4 approved differences; all12 backend
sources unchanged and available. All owned Python paths AST parse. Native4 base,
derived and sparse snapshot hashes match interface_receipt.json; retained
native_source.patch matches its recorded SHA256. PASS whitespace, no native
changes since early interface and no image rebuild needed for these repairs.
Actual total:19 paths,1269 gross/1582, native99 unchanged;313 lines remain.
Repair diff changes7 owned existing paths; total gross increases252 from1017.
Per-path estimates remain guidance; no stream quota or file ownership borrowed.
Verified repair commit74d5c48. Exact code diff:
`git diff --stat 2c4cc9e 74d5c48 -- tests`:7 files,267 additions/23 deletions.
Final baseline budget is1269, not1017+290, because budget counts the actual
baseline-to-final diff, while repair290 counts changed lines against2c4cc9e.

The12 backend modules are added only to existing staged schema_source_hashes.
D30's canonical source_hashes count remains13 including vllm.v1.engine.core,
checked by the derived stage test. Previous D30 finish-sync48 PASS is historical,
not a new repair rerun; D30 implementation and counter semantics are untouched.

### Current qualification limits

The script tests thin actual outer setter -> opaque MoE -> router capture/replay
and separate eager actual FP8/BF16 modules only. It explicitly marks actual MRv2
serving inputs/attention/connector breaks and real backend graphs unqualified.
Dummy replay baseline is recorded after writes execute, not after capture alone.
FULL capture, CUDA weights, NONE/PW/FULL runtime behavior, actual auto-backend
execution, all-rank imported objects/source, G3/transfer/E2E remain UNVERIFIED.
No root local image reference was bound as a published immutable identity.

## D29-T2-INPUT-PREPARATION (static input review only)

Root authorized import9090407 after preserving owned edits. Exact commands:
`git status --short` (empty at6827507), `git merge --ff-only 9090407` (PASS).
Native source/CPU/image acceptance is complete and was not reinvestigated or
rerun. Newly generated artifacts are under this task's `t2_preparation/`; no
production/test code, central/peer task edits, GPU/platform/auth/cloud/SSH/Docker
or push operations. Actual source prefix is accepted published export, never
worker import precedence. Python3.12.13 asyncopd-py312, stdlib only, no torch/vllm
imports. No numerical state tolerance or changed G3 criterion.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Retained config copy | Complete original metadata revision/hash; no rewritten exclusions/bounds |78447 bytes, official revision364b3b03fed2477aa0953fe855a2c4c8c48ee3cc, e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63 | PASS |
| Existing case schema |graph exactly2 same-shape arrays;3 module frames each2 DP arrays |graph17/17 rows with16/4 live;modules physical[6129,1]/[1,17]/[4,4],live[6128,0]/[0,16]/[1,0] | PASS_STATIC |
| Position/mask types and bound |integer0<=position<7168, bool masks, positive physical rows |all supplied arrays conform, masked stale43/44/7167 etc intentional | PASS_STATIC |
| Existing static plan |actual helper probe_plan/topology bodies, no native imports |TP8/DP2/EP16,2nodes/16ranks, layers3FP8/88BF16, timing_rows0 | PASS_STATIC |
| Provenance and dimensions |hashes of retained/config/input/script/source/publication consistent; raw dimension formulas correct |hash agreement;FP8 routed415236096B,BF16830472192B, FP8 scales101376B; excludes runtime workspace/rounding | PASS_STATIC |
| Actual engine7168 |not fabricated from unmodified metadata |both existing EngineArgs omit max_model_len; config max_position_embeddings1048576 preserved | NOT_BOUND |
| Code/GPU scope |no code edits or native/GPU execution |no diff9090407 in tests/frontier; no torch/vllm import | PASS_SCOPE; GPU UNVERIFIED |

Exact static command (no persistent/new harness; executes only existing stdlib
source bodies from the helper for input planning):

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
import ast,hashlib,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
root=Path.cwd(); task=root/'task_memory/task_2026-10-22_step5_d29_native_balanced'; base=task/'t2_preparation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((base/'provenance.json').read_text()); cases=json.loads((base/'cases.json').read_text()); proposal=json.loads((base/'resource_proposal.json').read_text())
assert set(cases)=={'graph','modules'} and len(cases['graph'])==2 and len(cases['modules'])==3
assert [len(c['positions']) for c in cases['graph']]==[17,17]
frames=cases['graph']+[f for c in cases['modules'] for f in c['dp']]
for frame in frames:
    assert set(frame)=={'positions','mask'}
    assert 0<len(frame['positions'])==len(frame['mask'])
    assert all(type(v) is int and 0<=v<7168 for v in frame['positions'])
    assert all(type(v) is bool for v in frame['mask'])
for case in cases['modules']:
    assert set(case)=={'name','dp'} and len(case['dp'])==2
assert [[len(f['positions']) for f in c['dp']] for c in cases['modules']]==[[6129,1],[1,17],[4,4]]
assert [[sum(not v for v in f['mask']) for f in c['dp']] for c in cases['modules']]==[[6128,0],[0,16],[1,0]]
assert sha(base/'model_metadata/config.json')==p['model_metadata']['sha256']==sha(Path(p['model_metadata']['retained_original_path']))
assert sha(base/'cases.json')==p['inputs']['sha256']
for relative,row in p['script_sources'].items(): assert sha(root/relative)==row['sha256']
for relative,digest in p['publication_artifacts'].items(): assert sha(root/relative)==digest
for relative,digest in p['native_mechanism_sources'].items(): assert sha(Path(p['actual_source_root'])/relative)==digest
metadata=json.loads((base/'model_metadata/config.json').read_text()); text=metadata['text_config']
assert (text['moe_num_experts'],text['moe_top_k'],text['norm_expert_weight'],text['moe_router_scaling_factor'])==(352,8,True,3.0)
assert text['max_position_embeddings']==1048576 and text['num_nextn_predict_layers']==3
helper=ast.parse((root/'frontier/profiling/collectives/step5_dispatch_probe.py').read_text())
nodes=[n for n in helper.body if (isinstance(n,ast.FunctionDef) and n.name in {'expected_topology','validate_launcher','probe_plan'}) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LAYERS' for t in n.targets))]
namespace=dict(json=json)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<existing static-only probe plan>','exec'),namespace)
args=SimpleNamespace(model_path=base/'model_metadata',tensor_parallel_size=8,data_parallel_size=2,gpus_per_node=8,tokens_per_dp_rank=[6129,1],disable_custom_all_reduce=False)
plan=namespace['probe_plan'](args); namespace['validate_launcher'](plan['topology'],dict(WORLD_SIZE='16',LOCAL_WORLD_SIZE='8'))
assert plan['topology']['node_count']==2 and [c['layer_id'] for c in plan['cases']]==[3,88]
assert [c['precision'] for c in plan['cases']]==['fp8','bf16'] and plan['timing_rows']==0
for frame in frames:
    routes=[[int((v*8+s)%352) for s in range(8)] for v,masked in zip(frame['positions'],frame['mask']) if not masked]
    assert all(len(set(row))==8 and all(0<=v<352 for v in row) for row in routes)
assert (43*8)%352==344 and (44*8)%352==0 and (6128*8)%352==96 and (6144*8)%352==224
weights=proposal['raw_allocation_lower_bounds']; assert weights['fp8_routed_weight_bytes']==22*3*4096*1536
assert weights['bf16_routed_weight_bytes']==2*weights['fp8_routed_weight_bytes']
assert weights['fp8_block_scale_fp32_bytes']==22*((24*32)+(32*12))*4
assert not any(n=='torch' or n.startswith('vllm') for n in sys.modules)
changed=subprocess.check_output(['git','diff','--name-only','9090407'],text=True).splitlines()
assert all(f.startswith('task_memory/task_2026-10-22_step5_d29_native_balanced/') for f in changed)
assert not subprocess.check_output(['git','diff','9090407','--','tests','frontier'],text=True)
print('PASS STATIC_INPUT_ONLY; NO_NATIVE_IMPORT; NOT_ENGINE7168; NOT_GPU_RUNTIME_OR_LAUNCH_APPROVAL')
PY
git diff --check
```

Observed shell output:static schema/hash/topology/dimension checks PASS;
NO_TORCH_VLLM_IMPORT True;no production/test change. Integer cycle/conservation
checks describe controlled inputs only, not native execution or expert histograms.
Published root acceptance remains accepted evidence, not this stdlib check's
runtime result. Hash index and `static_review_receipt.json` bind prepared files.
Both execution-proposal bash blocks were passed as text to `bash -n` and parse
successfully; no bash block was executed. `sha256sum --check hashes.sha256`
from t2_preparation verifies all6 indexed artifacts. Hash index does not hash
itself; model copy is78447B and cases67312B, no large model/trace artifact added.

Root exact commands/mounts/env and CUDA compatibility recipe are in
`t2_preparation/execution_proposal.md`. Proposed minimum topology2×8H800,
TP8/DP2/EP16/noSP/noMTP/auto/skip-padding;32CPU/128GiB host/node,
single1800s finite unit/8GPUh maximum. CPU/RAM/time are proposed reserves, not
measured minima/fit or a new allocation. Root still independently seals inputs,
finishes readiness, chooses actual resources and secures run approval. Remaining
specific dependency: actual engine7168 if required is not expressible by the
current T2/helper. Node/mount/rendezvous/output/retention and all actual GPU/
backend/import/library/G3 witnesses remain unresolved; no substituted data or
tolerance is introduced. Thin graph/eager modules is not real MRv2/full backend
graph, state/transfer/E2E or CPU/gap qualification.

## D29-T2-ENGINE-LIMITS
Actual preserved engine source:
`/data/ycfeng/tmp/step5_d24_native_20261022/capture03/stage/engine.json`,
SHA2567947f58bf37ab0256756faf1ec0613f4dc92101e94ad4a7f596c50ae4a11655e.
Triple:max_model_len7168,max_num_batched_tokens8192,max_num_seqs128.
Published vllm/model_executor/layers/fused_moe/layer.py SHA256
ddc92bf76eabaf0465e2dc515b844293c16691bb6e66f92a45e47c4956b6e328,
lines274/361 bind MoE max_num_tokens to scheduler max_num_batched_tokens.
Existing full metadata SHAe535c5c9... remains unchanged.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_position_cyclic.py -k entrypoint -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-t2-limits-red > /data/ycfeng/tmp/d29-t2-limits-red.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_position_cyclic.py -k 'not entrypoint' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-t2-limits-existing > /data/ycfeng/tmp/d29-t2-limits-existing.log 2>&1
git diff --numstat 6ebb038 -- tests/integration/step5_environment/test_position_cyclic_graph.py tests/unit/test_step5_native_position_cyclic.py
git diff --check
```
Red:3 FAIL at real helper EngineArgs fixture KeyError:max_model_len, confirming
old path lacked constructor inputs. Old actual-source controls:13 PASS/1.97s,
3 new entrypoint cases deselected. AST2/whitespace PASS. Native two-file delta
87 incremental gross at draft (script17add+5del;test65add), within root100 reserve.
No helper edits/copied builder. The final result below supersedes draft labels.

### Verified helper delivery and final affected check
PRICING delivers3f54bc640392d2a029e371e85ab212f0e2ac20ee with verified
`create_native_config(args, *, engine_limits=None)`.
Source SHA2562a6da8cb8184e267aab32e8c1cdc45744c2acf7cefc1439acc6a692d500b7323.
Its mixed commit also changes producer behavior and central task records. Native
does not import that broad commit, copy a second production builder or edit a peer
file. Read-only AST execution of the actual delivered helper body supplies the
dependency to this host check; root must integrate it before a bare entrypoint run.

Environment: asyncopd-py312 Python3.12.13, existing CPU Torch2.9.1+cu128 controls,
OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,no pytest cache,
all scratch/logs under/data/ycfeng/tmp. No CUDA initialization/native import/image
operation. Exact final command (run from this worktree):

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY' > /data/ycfeng/tmp/d29-t2-limits-affected-final.log 2>&1
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from frontier.profiling.collectives import step5_dispatch_probe as probe
source = subprocess.check_output(['git', 'show', '3f54bc6:frontier/profiling/collectives/step5_dispatch_probe.py'], text=True)
print('verified_helper_commit=3f54bc640392d2a029e371e85ab212f0e2ac20ee sha256='+hashlib.sha256(source.encode()).hexdigest())
function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'create_native_config')
exec(compile(ast.Module(body=[function], type_ignores=[]), '3f54bc6:frontier/profiling/collectives/step5_dispatch_probe.py', 'exec'), probe.__dict__)
inputs = Path('task_memory/task_2026-10-22_step5_d29_native_balanced/t2_preparation')
cases = json.loads((inputs/'cases.json').read_text())
args = SimpleNamespace(model_path=inputs/'model_metadata', tensor_parallel_size=8, data_parallel_size=2, gpus_per_node=8, tokens_per_dp_rank=[len(row['positions']) for row in cases['modules'][0]['dp']], disable_custom_all_reduce=False, routing_input_file=None)
latest_plan = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'probe_plan')
ns = probe.__dict__.copy()
exec(compile(ast.Module(body=[latest_plan], type_ignores=[]), '3f54bc6:probe_plan', 'exec'), ns)
assert ns['probe_plan'](args) == probe.probe_plan(args)
print('actual current/peer controlled probe plans identical: PASS,2 layers/16 ranks')
raise SystemExit(pytest.main(['tests/unit/test_step5_native_position_cyclic.py', 'tests/unit/test_step5_dispatch_probe.py', '-q', '-p', 'no:cacheprovider', '--basetemp=/data/ycfeng/tmp/d29-t2-limits-affected-final']))
PY
git diff --numstat 6ebb038 a5e5189 -- tests/integration/step5_environment/test_position_cyclic_graph.py tests/unit/test_step5_native_position_cyclic.py
git diff --check
```

Focused previous run, identical read-only helper installation with only the
native_position_cyclic suite:/data/ycfeng/tmp/d29-t2-limits-helper-focused.log,
**16 PASS/1.96s**. Final affected command above:**175 PASS/15 SKIP/5.12s**;
skips are existing actual-vllm-import dependent tests in test_step5_dispatch_probe,
not skipped new constructor regressions. No accepted418/485 or CPU-gap suite rerun.

Expected/pass conditions and observations:
- Real T2 main invokes the actual PRICING helper body and direct EngineArgs;
  both constructor kwargs contain7168/8192/128 before create_engine_config. PASS.
- Both entered configurations are the post-with_hf_config returned objects;
  modules enforce_eager=True,graph/default does not set eager. PASS.
- Final report modules/graph values observed exactly match, or record deliberate
  max_num_seqs129 drift. Either drift remains soft UNVERIFIED and both paths run.
  No config value is assigned after creation by T2. PASS host flow.
- Missing cases.engine_limits raises KeyError before construction; original
  LookupError constructor instance propagates unchanged with one attempted call.
  PASS. Host dependency fixtures are not native/GPU config qualification.
- Current/PRICING controlled probe plans agree on actual retained metadata,
  TP8/DP2/EP16/2nodes/16ranks and layers3/88. PASS. T2 supplies existing
  routing_input_file=None; no observed profiling frame/substitute family added.
- AST2/whitespace/owned2 paths PASS. a5e5189 diff15add+2del script,68add test:
  **85 incremental gross<=100**. Native19 baseline1aee0af total**1350/1582**,
  +81 from accepted1269,232 remaining; native4 gross99 and bytes unchanged.
- cases graph/modules remain byte-value identical to aa872fa; only engine_limits
  added from the hash-bound actual sealed engine file. Metadata bytes unchanged.

Closed: supported pre-construction engine-limit interface gap. Still pending:
root dependency integration/source+input sealing/actual-image CPU config check,
resource/run authorization and GPU NONE/PW/FULL/native backend/full-serving/G3.
The16H800/30min8GPUh finite-unit proposal remains unapproved; no state tolerance
or numeric/qualification criterion changed.

### Final artifact/static check
Same capped Python3.12.13 environment, no torch/vllm import; exact command:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY' > /data/ycfeng/tmp/d29-t2-limits-static-final02.log 2>&1
import ast, hashlib, json, subprocess
from pathlib import Path
repo=Path.cwd();task=repo/'task_memory/task_2026-10-22_step5_d29_native_balanced';inputs=task/'t2_preparation'
cases=json.loads((inputs/'cases.json').read_text());provenance=json.loads((inputs/'provenance.json').read_text());receipt=json.loads((task/'scope_receipt.json').read_text())
old=json.loads(subprocess.check_output(['git','show','aa872fa:'+str((inputs/'cases.json').relative_to(repo))]))
engine_path=Path(provenance['frozen_serving_limits']['path']);engine=json.loads(engine_path.read_text())
assert cases['engine_limits']=={key:engine[key] for key in ('max_model_len','max_num_batched_tokens','max_num_seqs')}
assert hashlib.sha256(engine_path.read_bytes()).hexdigest()==provenance['frozen_serving_limits']['sha256']
assert {key:cases[key] for key in old}==old
assert (inputs/'model_metadata/config.json').read_bytes()==Path(provenance['model_metadata']['retained_original_path']).read_bytes()
for name in ('tests/integration/step5_environment/test_position_cyclic_graph.py','tests/unit/test_step5_native_position_cyclic.py'):ast.parse((repo/name).read_text())
interface=json.loads((task/'interface_receipt.json').read_text())
assert hashlib.sha256((task/'native_source.patch').read_bytes()).hexdigest()==interface['patch_sha256']
for row in interface['native_changes']:
    path=row['path'].removeprefix('native-image/')
    assert hashlib.sha256((Path(interface['candidate_root'])/path).read_bytes()).hexdigest()==row['derived_sha256']
    assert hashlib.sha256((task/'native_source'/path).read_bytes()).hexdigest()==row['derived_sha256']
assert receipt['gross']==sum(row['added']+row['deleted'] for row in receipt['rows'])==1350
for block in (inputs/'execution_proposal.md').read_text().split('```bash\n')[1:]:subprocess.run(['bash','-n'],input=block.split('```')[0],text=True,check=True)
subprocess.run(['sha256sum','-c','hashes.sha256'],cwd=inputs,check=True)
print('PASS: frozen engine hash/triple,unchanged metadata+graph/modules,AST2,native4+patch hashes,budget1350/1582,bash-n2,checksums6/6')
PY
```

Expected/observed: PASS unchanged graph/modules/metadata/native4, actual engine
triple/hash,AST2,total1350/1582,bash-n2,checksums6/6. First ad-hoc static command
failed FileNotFoundError on native_source/forward_context.py because it removed
vllm/ twice. Inspected actual native_source/vllm path; correcting only this read
path passed final02. No test/production/artifact byte modification for this error.
