"""Read only two TP0 native traces and retained current segment/collective rows.

Native kernel membership is joined by launch correlation via the existing
census. P2-32's peer-delayed profile is structural, not an absolute clean price.
"""
import collections
import json
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
sys.path.insert(0, str(REPO / 'tests/analysis'))
from v4_runtime_census import forward_kernel_events, union_us

case = sys.argv[1]
roots = {'p2_64':'dsv4-h200-p2-64-profile-20261025-01', 'p2_32':'dsv4-h200-p2-profile-20261025-01'}
root = Path('/data/ycfeng/tmp/glm53_dsv4/h200/downloads') / roots[case] / 'artifacts'
trace = next((root/'torch_traces/prefill/dp0').glob('*_tp0_*.json.gz'))
selected, previous = [], None


def group(kernels):
    by = collections.defaultdict(lambda: dict(count=0, sum_ms=0))
    for k in kernels:
        by[k['name']]['count'] += 1
        by[k['name']]['sum_ms'] += k['dur']/1000
    return dict(sorted(by.items(), key=lambda item:-item[1]['sum_ms']))


def measure(ks):
    if not ks:
        return dict(count=0,sum_ms=0,union_ms=0,span_ms=0,idle_ms=0)
    intervals=[(k['ts'],k['ts']+k['dur']) for k in ks]
    union=union_us(intervals)/1000
    span=(max(b for a,b in intervals)-min(a for a,b in intervals))/1000
    return dict(count=len(ks),sum_ms=sum(k['dur'] for k in ks)/1000,
                union_ms=union,span_ms=span,idle_ms=span-union,
                overlap_ms=sum(k['dur'] for k in ks)/1000-union,
                streams=dict(collections.Counter(k['args'].get('stream') for k in ks)))


for fw in forward_kernel_events('prefill', trace):
    r=fw.record
    if r['mode']=='NONE' and r['real'] and r.get('step',{}).get('context_tokens')==6144:
        item=dict(record=r,host_segment_bounds_us=fw.segment_bounds,
                  segments={k:measure(v) for k,v in fw.kernels_by_segment.items()},
                  kernels_by_segment={k:group(v) for k,v in fw.kernels_by_segment.items()})
        if previous:
            item['previous_record']=previous.record
            item['previous_post']=measure(previous.kernels_by_segment['post'])
            item['previous_post_kernels']=group(previous.kernels_by_segment['post'])
        selected.append(item)
    previous=fw
assert selected, f'no matching real prefill in {trace}'
cur=json.load(open(OUT/'current.json'))[case]
collective=[]
for gen,d in cur.items():
    entries=json.load(open(Path(d['root'])/'collectives.json'))['rows']
    for e in entries:
        if e['role']=='prefill' and e['requested_bytes']>=6291456 and (e['curve_key'][0].startswith('sp_') or e['curve_key'][0].startswith('ep_')):
            collective.append(dict(generation=gen,curve_key=e['curve_key'],bytes=e['requested_bytes'],
                calls=e['call_count'],calls_per_real_forward=e['call_count']/128,price_ms=e['predicted_ms'],rows=e['unequal_rows']))
result=dict(case=case,trace=str(trace),native=selected,current_collective_counts=collective,
    current={gen:dict(segments=d['segments'],step_ms=d['step_ms'],scope_ms=d['scope_ms']) for gen,d in cur.items()},
    scope_trace_limit='Native full-model trace exposes kernel names/launches, not per-layer dsv4 scope-cost/gap inputs; current forward observer exposes scope sums, not each layer gap/count/start. Count/overlap discrepancies therefore require a matched scope/launch witness, not a total-step subtraction.')
json.dump(result,open(OUT/f'{case}_segments.json','w'),indent=2)
print(case, trace)
for r in selected:
    print('forward',r['record']['index'],r['segments'],'previous_post',r.get('previous_post'))
print('collectives',json.dumps(collective))
