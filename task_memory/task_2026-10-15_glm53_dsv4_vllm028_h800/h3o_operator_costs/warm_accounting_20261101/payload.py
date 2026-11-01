"""Exercise existing capacity/payload functions through CPU-only AST extraction.

Imports in extracted pure functions are replaced by their scoped definitions,
not behavioral substitutes. Enum labels are the actual recorded h800/h200
strings; no Torch/config/model execution is imported.
"""
import ast
import collections
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
TASK = OUT.parents[1]
NATIVE = Path('/data/ycfeng/tmp/opencode/vllm-v0.28.0')
spec = importlib.util.spec_from_file_location('warm_memory', REPO / 'frontier/attention/memory.py')
memory = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = memory
memory_tree = ast.parse((REPO / 'frontier/attention/memory.py').read_text())
memory_tree.body = [n for n in memory_tree.body if not (isinstance(n, ast.ImportFrom) and n.module and n.module.startswith('frontier.'))]
exec(compile(memory_tree, str(REPO / 'frontier/attention/memory.py'), 'exec'), memory.__dict__)
ns = dict(vars(memory), ceil=math.ceil, DeviceSKUType=SimpleNamespace(H800='h800', H200='h200'))


def extract(path, names, assignments=()):
    tree = ast.parse(path.read_text())
    kept = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in node.targets):
            kept.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            node.body = [s for s in node.body if not isinstance(s, (ast.Import, ast.ImportFrom))]
            kept.append(node)
    assert len(kept) == len(names) + len(assignments)
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), 'exec'), ns)


extract(REPO / 'frontier/attention/families.py', ('require_deepseek_v4_storage_device', 'deepseek_v4_main_kv_cache_dtype', '_deepseek_v4_kv_cache_capacity'), ('DEEPSEEK_V4_MAIN_KV_CACHE_DTYPE_ALIASES',))
extract(REPO / 'frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py', ('get_grouped_nixl_payload',))
current = json.load(open(OUT / 'current.json'))
mc = current['p2_64']['B29']['model_config']
cfg = SimpleNamespace(**mc)
ratios = mc['compress_ratios'][:mc['num_layers']]
cfg.get_head_dim = lambda: mc['head_dim']
cfg.get_layer_attention_specs = lambda: [SimpleNamespace(global_layer_id=i, family_id='deepseek_v4_attention', variant_id={0:'uncompressed',4:'csa',128:'hca'}[r]) for i, r in enumerate(ratios)]
selection = memory.AttentionKVCacheSelection(block_size=256, attention_backend='FLASHMLA_SPARSE_DSV4', kv_cache_dtype='fp8', indexer_kv_cache_dtype='fp8', device='h200', gpu_platform='cuda', decode_context_parallel_size=1)
capacity = ns['_deepseek_v4_kv_cache_capacity'](cfg, 256, selection)
tables, serial = {}, 1
for g in capacity.groups:
    n = math.ceil(6144 / g.logical_block_size)
    skipped = 0 if g.retention == 'full_history' else max(0,6144-g.window_tokens+1-g.extra_retained_tokens)//g.logical_block_size
    tables[g.group_id] = (None,) * skipped + tuple(range(serial, serial + n - skipped))
    serial += n - skipped
selected, structural = ns['get_grouped_nixl_payload'](capacity, 6144, tables, tables)
groups = [dict(group=g.group_id, selected_pages=len(selected[g.group_id]), logical_block=g.logical_block_size,
               real_page_bytes=capacity.get_group_size_bytes(g),
               structural_bytes=len(selected[g.group_id])*capacity.get_group_size_bytes(g),
               packed_bytes=len(selected[g.group_id])*capacity.physical_block_size_bytes) for g in capacity.groups]
packed = sum(g['packed_bytes'] for g in groups)
assert structural == current['p2_64']['B29']['bytes'] == 39521664
assert packed == 46103040 and capacity.physical_block_size_bytes == 1002240
all_native = {}
roots = dict(wave64='dsv4-h200-p2-64-wave-20261030-01', lifecycle32='dsv4-h200-p2-lifecycle-20261025-01')
for name, run in roots.items():
    root = Path('/data/ycfeng/tmp/glm53_dsv4/h200/downloads') / run / 'artifacts'
    registrations = []
    for p in root.glob('nodes/*/server.log'):
        for i, line in enumerate(open(p), 1):
            if 'Registering packed KV cache:' in line:
                registrations.append(dict(path=str(p), line=i, text=line.rstrip()))
    reqs = []
    for p in root.glob('nodes/*/scheduler_lifecycle/*.jsonl'):
        rows = [json.loads(l) for l in open(p)]
        start = next(r for r in rows if r['event'] == 'start')
        if start['role'] != 'decode':
            continue
        adds = {r['request_id']:r['t'] for r in rows if r['event']=='add' and 'formal' in r['request_id']}
        for r in rows:
            if r['event'] != 'update':
                continue
            for rid in r.get('finished_recving', []) or []:
                if rid not in adds:
                    continue
                key=rid.split('cmpl-',1)[1].rsplit('-0-',1)[0]
                updates=[u for u in rows if u['event']=='update' and adds[rid]<u['t']<=r['t'] and u.get('kv_stats',{}).get('transfer_duration')]
                stats=[u['kv_stats'] for u in updates]
                durations=[v*1000 for s in stats for v in s['transfer_duration']]
                reqs.append(dict(request=key, lane=start['dp_rank'], scheduler_add=adds[rid], receive_update=r['t'], receive_step=r.get('step'),
                    handles=len(durations), bytes=[v for s in stats for v in s['bytes_transferred']],
                    descriptors=[v for s in stats for v in s.get('num_descriptors',[])],
                    xfer_ms=durations, post_ms=[v*1000 for s in stats for v in s.get('post_duration',[])],
                    stats_update_times=[u['t'] for u in updates], stats_update_steps=[u.get('step') for u in updates]))
    json.dump(reqs, open(OUT / f'{name}_handle_stats.json','w'),indent=2)
    all_native[name]=dict(n=len(reqs), handles_hist=dict(collections.Counter(r['handles'] for r in reqs)),
        bytes=sorted({b for r in reqs for b in r['bytes']}), descriptors=sorted({d for r in reqs for d in r['descriptors']}),
        mean_max_xfer_ms=statistics.mean(max(r['xfer_ms']) for r in reqs),
        mean_handle_xfer_ms=statistics.mean(v for r in reqs for v in r['xfer_ms']),
        mean_post_ms=statistics.mean(v for r in reqs for v in r['post_ms']),
        registrations=registrations)
result=dict(groups=groups, physical_block_size_bytes=capacity.physical_block_size_bytes,
            structural_bytes=structural, native_packed_bytes=packed, missing_wire_bytes=packed-structural,
            missing_fraction_native=(packed-structural)/packed, native=all_native,
            lifecycle_timing_limit='All-request geometry/statistics only, not a warm timing reference. No per-worker start_load_kv, transfer(handle), check_xfer_state or get_finished timestamp/handle-ID is retained; scheduler stats update is not the wire completion endpoint.')
json.dump(result,open(OUT/'payload_summary.json','w'),indent=2)
print(json.dumps(result,indent=2))
