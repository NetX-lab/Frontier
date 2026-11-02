# Exact bounded CPU command

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Captured inline use of existing pinned readers; not a production checker or new manifest. |

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Stdout is the machine receipt; stderr and exit status are retained by the command runner. No native code, network request, GPU operation or Git command is executed.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 - <<'PY'
import ast, hashlib, importlib.util, inspect, json, math, os, platform, statistics, struct, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path.cwd()
T = ROOT/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800'
OUT = T/'canonical_case_20261021/p0_binding'
A = T/'native06_acceptance_20261023'
C = Path('/data/ycfeng/tmp/glm53_dsv4/native06_case_20261023')
E = Path('/data/ycfeng/tmp/glm53_dsv4/native06_case_r4_20261023')
TAG = 'dsv4-h800-p0-canonical-20261021-06'
R = C/'runs/groundtruth_clean'/TAG
ER = E/'runs/groundtruth_clean'/TAG
P = Path('/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_06')
K = ROOT/'task_memory/task_2026-10-22_v4_native_binding/check_groundtruth_run_successor_r2.py'
start = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
sha = lambda b: hashlib.sha256(b).hexdigest()
pins = {}
def read(path):
    path = Path(path)
    b = path.read_bytes()
    st = path.stat()
    pins[str(path)] = dict(sha256=sha(b), bytes=len(b), mtime_ns=st.st_mtime_ns)
    return b
def load(path):
    return json.loads(read(path))
primary = load(A/'primary_acceptance_r4.json')
assert primary['status'] == 'ACCEPTED_NATIVE_REFERENCE_R4'
assert primary['run_tag'] == TAG and primary['evaluation_case'] == str(E)
result = load(A/'frozen_r4_result.json')
assert pins[str(A/'frozen_r4_result.json')]['sha256'] == primary['result_sha256']
assert result['runs'][TAG]['status'] == 'PASS' and len(result['runs'][TAG]['checks']) == 8
assert all(v.startswith('PASS:') for v in result['runs'][TAG]['checks'].values())
client = load(A/'client_receipt.json')
assert pins[str(A/'client_receipt.json')]['sha256'] == primary['client_receipt_sha256']
freeze = load(primary['freeze_path'])
assert pins[primary['freeze_path']]['sha256'] == primary['freeze_sha256']
assert sha(read(freeze['checker_path'])) == primary['checker_sha256']
for path, digest in freeze['dependencies'].items():
    assert sha(read(path)) == digest
M = load(R/'run_manifest.json')
assert read(ER/'run_manifest.json') == read(R/'run_manifest.json')
shared = ROOT/'data/processed_traces/canonical_fixed_6144_1024_req128_qps1'
workload_hashes = {}
for name in ('trace.csv','request_ids.json','warmup_pool.csv','manifest.json','SHA256SUMS'):
    original = read(C/'inputs/trace'/name)
    assert original == read(E/'inputs/trace'/name) == read(shared/name) == read(P/'trace'/name)
    digest = sha(original)
    assert digest == M['inputs_sha256'][f'inputs/trace/{name}']
    workload_hashes[name] = digest
prior = read(T/'test_report_2026-10-21_common_workload_identity.md').decode()
assert '68f891b' in prior and workload_hashes['manifest.json'] in prior
for name in ('client_metrics','e2e_metrics_gap','pd_replay','pd_metrics'):
    rel = f'tests/comparison/calibration/{name}.py'
    assert sha(read(P/rel)) == M['harness_sha256'][rel]
for rel in ('tests/comparison/dp_placement_pp/vllm_replay.py','tests/comparison/stage_admission_pp/vllm_burst_driver.py'):
    assert sha(read(P/rel)) == M['harness_sha256'][rel]
assert sha(read(K)) == freeze['dependencies'][str(K)]
sys.path.insert(0, str(P))
from tests.comparison.calibration.client_metrics import native_client_rows
from tests.comparison.calibration.pd_replay import canonical_workload, dedicated_services, formal_metrics
from tests.comparison.calibration.e2e_metrics_gap import metric_values
from tests.comparison.dp_placement_pp.vllm_replay import replay, prompt_token_ids
W = canonical_workload(C/'inputs/trace')
assert W[0]['repetitions']['repeat_indices'] == [1,2,3]
F = native_client_rows(R, W, load(C/'inputs/trace/request_ids.json'))
S = load(R/'run/replay_summary.json')
B = load(R/'run/canonical_repetitions.json')
records = [json.loads(line) for line in read(R/'run/client_requests.jsonl').splitlines()]
for name in ('client_requests.jsonl','replay_summary.json','canonical_repetitions.json'):
    assert read(ER/'run'/name) == read(R/'run'/name)
assert S['canonical_repetitions'] == B and S['status'] == 0 and not S['failed_request_ids']
assert len(records) == len({r['request_id'] for r in records}) == 432
assert load(R/'replay_finished.json')['returncode'] == 0
services = dedicated_services(R, {role:S['instances'][role]['http_url'] for role in ('prefill','decode')}, 2)
for path in R.glob('nodes/*/server_ready.json'):
    read(path)
spec = importlib.util.spec_from_file_location('existing_p0_boundary', K)
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)
idle = k.native_idle_evidence(B, records, W[0], 2, services)
assert idle[0]
model = load(P/'model/config.json')
assert pins[str(P/'model/config.json')]['sha256'] == M['inputs_sha256'][str(P/'model/config.json')]
assert S['model_config_sha256'] == pins[str(P/'model/config.json')]['sha256']
vocab = model['vocab_size']
assert vocab == 129280
tree = ast.parse(inspect.getsource(replay))
calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'prompt_token_ids']
assert len(calls) == 1 and ast.unparse(calls[0]) == "prompt_token_ids(row.get('template_request_id', row['request_id']), row['num_prefill_tokens'], vocab_size)"
templates = W[1] + W[2]
token_digest = hashlib.sha256()
minimum, maximum, count = vocab, -1, 0
for row in templates:
    tokens = prompt_token_ids(row['request_id'], row['num_prefill_tokens'], vocab)
    assert len(tokens) == 6144 and all(type(t) is int and 0 <= t < vocab for t in tokens)
    matching = [r for r in records if r['template_request_id'] == row['request_id']]
    assert {r['repeat'] for r in matching} == {1,2,3} and len(matching) == 3
    assert all(prompt_token_ids(r['template_request_id'], r['num_prefill_tokens'], vocab) == tokens for r in matching)
    minimum, maximum = min(minimum,min(tokens)), max(maximum,max(tokens))
    count += len(tokens)
    token_digest.update(row['request_id'].encode()+b'\0'+struct.pack('<6144I',*tokens))
metrics = []
for repeat in (1,2,3):
    rows = [r for r in records if r['repeat'] == repeat and r['segment'] == 'formal']
    assert len(rows) == 128 and sorted(r['arrived_at'] for r in rows) == list(range(128))
    assert sum(r['usage']['prompt_tokens'] for r in rows) == 786432
    assert sum(r['output_token_count'] for r in rows) == 131072
    m = formal_metrics(rows)
    assert m == B['repetitions'][repeat-1]['metrics']
    values = {name:float(n/d) for name,(n,d,_) in metric_values(F[f'repeat{repeat:02d}']).items()}
    target = dict(ttft_ms=m['ttft_ms']['mean'],tpot_ms=m['tpot_ms']['mean'],request_e2e_time_ms=m['e2e_ms']['mean'],token_throughput_tps=m['total_thpt_tokens_s'])
    assert all(math.isclose(values[n],v,rel_tol=0,abs_tol=1e-9) for n,v in target.items())
    assert m['total_tokens'] == 917504 and m['window_s'] > 127
    assert math.isclose(statistics.mean((r['e2e_ms']-r['ttft_ms'])/1023 for r in rows),target['tpot_ms'],rel_tol=0,abs_tol=1e-9)
    metrics.append(dict(repeat=repeat,**target,window_s=m['window_s'],first_submission=m['first_submission_monotonic'],last_token=m['last_token_monotonic']))
native = load(C/'inputs/native_config.json')
assert read(E/'inputs/native_config.json') == read(C/'inputs/native_config.json')
assert S['engine_config'] == native['engine']
runtime = []
for node in sorted((R/'nodes').iterdir()):
    n = load(node/'result.json')
    assert read(ER/'nodes'/node.name/'result.json') == read(node/'result.json')
    assert n['model_hashes']['config.json'] == S['model_config_sha256']
    assert n['server_env_set']['VLLM_MOE_ROUTING_SIMULATION_STRATEGY'] == 'balanced_topk'
    assert '--skip-tokenizer-init' in n['server_command'] and '--speculative-config' not in n['server_command']
    assert n['plan']['engine']['tensor_parallel_size'] == 8 and n['plan']['engine']['data_parallel_size'] == 2
    for path in sorted((node/'serving').glob('rank*.json')):
        assert read(ER/'nodes'/node.name/'serving'/path.name) == read(path)
        assert pins[str(path)]['sha256'] == freeze['original_files'][str(path)]
        inv = json.loads(read(path))['inventory']
        assert {g:inv['groups'][g]['world_size'] for g in ('tp','dp','ep')} == dict(tp=8,dp=2,ep=16)
        scalars = inv['memory']['native_scalars']
        runtime.append(dict(role=n['plan']['role'],rank=inv['worker_rank'],num_blocks=inv['cache']['num_blocks'],memory_native_scalars=scalars))
assert len(runtime) == 32
for path, expected in list(pins.items()):
    b = Path(path).read_bytes()
    assert dict(sha256=sha(b),bytes=len(b),mtime_ns=Path(path).stat().st_mtime_ns) == expected
print(json.dumps(dict(task_id='W128-V4-P0-CASE-BINDING',analysis_state='COMPLETE',status='PASS_P0_WORKLOAD_METRIC_BINDING_ONLY',started_at_utc=start,finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.perf_counter()-clock,environment=dict(executable=sys.executable,python=platform.python_version(),pandas=sys.modules['pandas'].__version__,CONDA_PREFIX=os.getenv('CONDA_PREFIX'),PYTHONDONTWRITEBYTECODE=os.getenv('PYTHONDONTWRITEBYTECODE'),cwd=str(ROOT),root_revision_supplied='4425bf8b',git_invoked=False),command_path=str(OUT/'command.md'),command_sha256=sha((OUT/'command.md').read_bytes()),native_primary_status=primary['status'],source_revision=M['harness_commit'],reader_revision=primary['reader_source_revision'],shared_revision_from_prior_acceptance='68f891b',workload_hashes=workload_hashes,counts=dict(total=432,formal=384,warmup=48,repetitions=3,formal_per_repeat=128,warmup_per_repeat=16,missing=0,failed=0,unfinished=0),prompt_evidence=dict(kind='pinned source deterministic reconstruction; not retained HTTP prompt arrays',vocab_size=vocab,native_input_processor_guard_claimed=False,template_count=144,template_prompt_tokens=count,three_repeat_prompt_tokens=count*3,min_token_id=minimum,max_token_id=maximum,aggregate_sha256=token_digest.hexdigest(),digest_encoding='formal then warmup template order; UTF8 template+NUL+6144 uint32 little-endian IDs',stable_template_binding=True),idle_helper=idle,metrics=metrics,runtime_native_scalars=runtime,native_graph_quantization_binding='Reuse hash-verified primary R4: P NONE; D PIECEWISE+FULL; MXFP4/MARLIN; fp8 KV resolves fp8_ds_mla. No new graph/quantization admission.',unverified=['Ticket17 all-four hardware cases','Non-KV stage-separated/Frontier KV-budget comparison','Qualified physical TP8/operator/graph costs and full Frontier prediction','Every numerical10% final gate and30% promotion gate','Independent retained HTTP prompt arrays/output-ID vocabulary audit','Native READ/device completion and observer-overhead claims'],inputs_unchanged=True,input_count=len(pins),inputs=pins,new_manifest=False,new_gpu_run=False,calibration_admission=False),indent=2))
PY
```
