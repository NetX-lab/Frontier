"""CPU evidence checks, provenance and arithmetic; no acceptance metric fitting."""
import csv
import hashlib
import json
import platform
from pathlib import Path

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
TASK=OUT.parents[1]
NATIVE=Path('/data/ycfeng/tmp/opencode/vllm-v0.28.0')
checks=[]
current=json.load(open(OUT/'current.json'))
payload=json.load(open(OUT/'payload_summary.json'))
gap=json.load(open(OUT/'gap_verify.json'))
manifest=json.load(open(TASK/'h200_mainline/p2_64_wave/package_02/config.json'))
hashes={}
for name in ('vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py',
             'vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py',
             'vllm/v1/worker/kv_connector_model_runner_mixin.py',
             'vllm/models/deepseek_v4/attention.py',
             'vllm/utils/multi_stream_utils.py',
             'vllm/distributed/device_communicators/all2all.py',
             'vllm/distributed/device_communicators/cuda_communicator.py'):
    actual=hashlib.sha256((NATIVE/name).read_bytes()).hexdigest()
    expected=manifest.get('expected_source_hashes',{}).get(name)
    hashes[name]=dict(actual=actual,expected=expected)
    if expected:
        assert actual==expected,(name,actual,expected)
checks.append('inspected native files match every supplied expected_source_hash')
assert payload['structural_bytes']==39521664 and payload['native_packed_bytes']==46103040
assert sum(g['selected_pages'] for g in payload['groups'])==46
checks.append('source capacity/payload calculation reproduces ledger and native packed bytes')
for name,r in payload['native'].items():
    assert r['bytes']==[46103040]
assert payload['native']['wave64']['handles_hist']=={'8':127,'16':1}
assert payload['native']['lifecycle32']['handles_hist']=={'8':384}
checks.append('wave64 has127 eight-sample windows and one ambiguous16-sample aggregate window; lifecycle32 has384 eight-sample windows; every sample has46,103,040B')
assert all(d['bytes']==39521664 for gens in current.values() for d in gens.values())
assert all(abs(d['extra_wire_ms']-0.41720291600633913)<1e-10 for gens in current.values() for d in gens.values())
checks.append('six current replay byte prices agree; packed minus component rate arithmetic = 0.417203 ms')
runner=list(csv.DictReader(open(REPO/'data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/model_runner_replay_kernel_only.csv')))
prep=[r for r in runner if r['operator']=='dsv4_input_preparation' and r.get('padded_rows')=='6144' and r.get('graph_runtime_mode')=='NONE']
sampling=[r for r in runner if r['operator']=='dsv4_sampling' and r.get('b')=='1' and r.get('needs_logits_processing')=='0']
assert len(prep)==1 and prep[0]['kernel_count']=='67' and prep[0]['memcpy_count']=='9'
assert {r['kernel_count'] for r in sampling}=={'9'}
checks.append('selected runner preparation row has 67 kernels / 9 copies and sampling rows have 9 kernels; native kernel-only preparation/post counts are checked separately')
for case in current:
    native=json.load(open(OUT/f'{case}_segments.json'))
    for fw in native['native']:
        names=fw['kernels_by_segment']['dispatch']
        counts={kind:sum(v['count'] for name,v in names.items() if kind in name) for kind in ('AllGatherV_RING','ReduceScatter_Sum','Reduce_Sum','AllGather_RING')}
        assert counts=={'AllGatherV_RING':129,'ReduceScatter_Sum':43,'Reduce_Sum':43,'AllGather_RING':44},counts
        assert fw['segments']['prepare']['count']==67 and fw['segments']['post']['count']==11
        post=fw['kernels_by_segment']['post']
        assert sum(v['count'] for name,v in post.items() if 'ncclDevKernel_AllGather_' in name)==1
        assert sum(v['count'] for name,v in post.items() if name.startswith('nvjet_'))==1
    for gen,d in current[case].items():
        s=d['dp0']['scope_ms']
        g=gap['gap_eager_ms']
        assert abs(s['dsv4_sp_attention_allgather']/(0.16160960197448732+g)-43)<1e-6
        assert abs(s['dsv4_sp_attention_reduce_scatter']/(0.15662400126457215+g)-43)<1e-6
checks.append('three native real forwards witness EP dispatch 129=43x3; SP executed scopes are 43, not 44 price queries')
assert abs(gap['direct_device_delta_ms']-86*gap['gap_eager_ms'])<1e-12
checks.append('isolated actual _operator_gaps ignores per-layer count3 and uses registry1; registry3 fixture adds 0.046836 ms/forward')
json.dump(dict(python=platform.python_version(),pass_count=len(checks),checks=[dict(status='PASS',description=c) for c in checks],native_source_hashes=hashes),open(OUT/'validation.json','w'),indent=2)
json.dump(dict(preparation=prep,sampling_count_hist=sorted({r['kernel_count'] for r in sampling}),sampling_rows=len(sampling)),open(OUT/'runner_rows.json','w'),indent=2)
print(f'{len(checks)}/{len(checks)} PASS; Python {platform.python_version()}')
