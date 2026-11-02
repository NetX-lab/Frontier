"""Q3 CPU evidence: native page bound, production price, retained trials and sinks.

Run from the project root with the assigned Python and environment. No native
runtime is imported. Extracted native methods execute against host fixtures;
this is source evidence, not an observed serving descriptor witness.
"""
from __future__ import annotations

import ast
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import statistics
import textwrap
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TASK = REPO / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800'
H3O = TASK / 'h3o_operator_costs'
Q1 = HERE.parent
NATIVE = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
NX = 'vllm/distributed/kv_transfer/kv_connector/v1/nixl/'
CORE = 'vllm/v1/core/'
INPUTS = {}
EXTRACTED = []


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def track(path):
    path = Path(path).resolve()
    INPUTS[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return path


def load(path):
    return json.loads(track(path).read_text())


def csv_rows(path):
    with track(path).open() as stream:
        return list(csv.DictReader(stream))


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method(relative, owner, name, globals_=None):
    path = track(NATIVE / relative)
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    body = textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))
    context = {'np': np, 'cdiv': lambda a, b: (a+b-1)//b, **(globals_ or {})}
    exec(compile('from __future__ import annotations\n' + body, str(path), 'exec'), context)
    EXTRACTED.append(dict(path=str(path), owner=owner, symbol=name,
                          start_line=node.lineno, end_line=node.end_lineno,
                          sha256=INPUTS[str(path)]['sha256']))
    return context[name]


class Pool:
    """Only host allocation IDs/refcounts needed by the extracted methods."""
    def __init__(self):
        self.next_id = 1
    def get_new_blocks(self, count):
        blocks = [NS(block_id=i, block_hash=None, is_null=False)
                  for i in range(self.next_id, self.next_id+count)]
        self.next_id += count
        return blocks
    def free_blocks(self, blocks):
        pass


def native_witness(prompt, group_rows):
    single = CORE + 'single_type_kv_cache_manager.py'
    add = method(single, 'SingleTypeKVCacheManager', 'add_local_computed_blocks')
    external = method(single, 'SingleTypeKVCacheManager', 'allocate_external_computed_blocks')
    remove = method(single, 'SingleTypeKVCacheManager', 'remove_skipped_blocks')
    remove_range = method(single, 'SingleTypeKVCacheManager', '_remove_blocks_in_range')
    skip = method(single, 'SlidingWindowManager', 'get_num_skipped_tokens')
    unhashed = method(CORE+'kv_cache_manager.py', 'KVCacheBlocks', 'get_unhashed_block_ids_all_groups')
    clip = method(NX+'base_scheduler.py', 'NixlBaseConnectorScheduler', 'get_exchange_clipped_blocks')
    prefix = method(NX+'base_worker.py', 'NixlBaseConnectorWorker', '_apply_prefix_caching')
    desc = method(NX+'base_worker.py', 'NixlBaseConnectorWorker', '_compute_desc_ids')
    null = NS(block_id=0, block_hash=None, is_null=True)
    ppool, dpool = Pool(), Pool()
    p_tables, d_tables = [], []
    for group in group_rows:
        bs, window = group['logical_block_size'], group['window_tokens']
        def manager(pool, table):
            obj = NS(block_size=bs, sliding_window=window, extra_retained_tokens=0,
                     enable_caching=False, block_pool=pool, _null_block=null,
                     _record_new_block_ids=False, req_to_blocks={'request': table},
                     num_cached_block={}, _has_partial_local_hit=lambda *args: False)
            obj.get_num_skipped_tokens = (lambda tokens: skip(obj, tokens)) if window else (lambda tokens: 0)
            obj._remove_blocks_in_range = lambda *args: remove_range(obj, *args)
            return obj
        pt = ppool.get_new_blocks(math.ceil(prompt/bs))
        pm = manager(ppool, pt)
        remove(pm, 'request', prompt)
        dt = []
        dm = manager(dpool, dt)
        add(dm, 'request', (), 0, prompt)
        external(dm, 'request', 0, prompt)
        p_tables.append(pt)
        d_tables.append(dt)
    limits = [math.ceil(g['window_tokens']/g['logical_block_size'])+1 if g['window_tokens'] else 0 for g in group_rows]
    connector = NS(_is_hma_required=True, blocks_per_sw=limits,
                   _ssm_spec_blocks=[None]*len(limits), _ssm_state_slots_are_positional=False)
    p_clip = clip(connector, tuple([b.block_id for b in t] for t in p_tables))
    d_live = unhashed(NS(blocks=tuple(d_tables)))
    d_clip = clip(connector, d_live)
    worker = NS(_has_mamba=False, num_regions=1)
    local, remote = prefix(worker, d_clip, p_clip, 1, 1)
    ids = desc(worker, remote, 109934, None, 1)
    assert len(ids) == sum(map(len, local))
    assert all(int(i) > 0 for i in ids) and len(set(map(int, ids))) == len(ids)
    return dict(prompt_tokens=prompt, initial_p_pages=[math.ceil(prompt/g['logical_block_size']) for g in group_rows],
                p_held=[sum(not b.is_null for b in t) for t in p_tables],
                p_clipped_including_null=list(map(len, p_clip)), d_live=list(map(len, d_live)),
                d_clipped=list(map(len, d_clip)), selected=list(map(len, remote)),
                descriptor_count=len(ids), null_descriptors=sum(int(i)==0 for i in ids))


def check1():
    rules = load(REPO / 'task_memory/task_2026-10-21_v4_grouped_nixl/native_rules.json')
    for key, expected in rules['source_sha256'].items():
        assert sha(track(NATIVE / key.split(':')[0])) == expected
    qualification = load(H3O / 'qualification_codex_20261031/evidence/qualification_evidence.json')
    registrations, native_nodes = [], []
    for case, c in qualification['cases'].items():
        for node in c['nodes']:
            entry = qualification['sources'][node['source']]
            raw = load(entry['path'])
            assert sha(entry['path']) == entry['sha256']
            assert node['engine']['enable_prefix_caching'] is False
            assert node['engine']['tensor_parallel_size'] == 8
            assert node['engine']['pipeline_parallel_size'] == 1
            command = raw['server_command']
            command = command if isinstance(command, str) else ' '.join(command)
            assert '--disable-hybrid-kv-cache-manager' not in command
            assert '--speculative-config' not in command
            native_nodes.append(dict(case=case, source=entry['path'], tp=8,
                                     prefix_caching=False, chunked_prefill=node['engine']['enable_chunked_prefill'],
                                     explicit_hma_disable=False, speculative_config_present=False))
        for log in c['startup_logs']:
            entry = qualification['sources'][log['source']]
            path = track(entry['path'])
            assert sha(path) == entry['sha256']
            lines = path.read_text().splitlines()
            assert not any('User-specified logical block size' in line for line in lines)
            for row in log['rows']:
                if 'Registering packed KV cache:' in row['text']:
                    assert lines[row['line']-1].endswith(row['text'])
                    assert 'block_stride=1002240' in row['text'] and 'num_regions=1' in row['text']
                    expected_blocks = 109934 if case == 'p2_32' else 112914
                    assert f'num_blocks={expected_blocks},' in row['text']
                    registrations.append(dict(case=case, path=str(path), line=row['line'], text=row['text']))
    assert collections.Counter(r['case'] for r in registrations) == {'p2_32': 32, 'p2_64': 64}
    write_json('registrations.json', registrations)
    workloads = []
    for scale in (32, 64):
        package = TASK / f'h200_mainline/p2_{scale}_reference/package_01'
        inp = load(package / 'case/inputs/frontier.json')
        rows = csv_rows(inp['trace_request_generator_config_trace_file'])
        trace_hash = sha(inp['trace_request_generator_config_trace_file'])
        assert len(rows) == 128 and {int(r['num_prefill_tokens']) for r in rows} == {6144}
        assert {int(r['num_decode_tokens']) for r in rows} == {1024}
        assert sha(package/'case/inputs/trace/trace.csv') == trace_hash
        assert inp['vllm_v1_scheduler_config_enable_prefix_caching'] is False
        assert inp['speculative_decoding_config_enabled'] is False
        model = load(package / 'model/config.json')
        assert model['sliding_window'] == 128
        assert model.get('attention_chunk_size') is None
        proxy = track(package / 'tests/comparison/calibration/pd_proxy.py')
        assert 'json.dumps(original | {"kv_transfer_params": params})' in proxy.read_text()
        workloads.append(dict(case=f'p2_{scale}', requests=128, prompt_tokens=6144,
                              decode_tokens=1024, prefix_caching=False, trace_sha256=trace_hash,
                              native_proxy=str(proxy), native_nodes=scale//8))
    from frontier.config import ReplicaConfig
    from frontier.attention.families import get_attention_family
    from frontier.attention.memory import AttentionKVCacheSelection, get_attention_kv_cache_capacity
    replica = ReplicaConfig(model_name='deepseek-ai/DeepSeek-V4-Flash', device='h200', network_device='h200_dgx',
                            attn_tensor_parallel_size=8, attn_dp=4, moe_tensor_parallel_size=1,
                            moe_expert_parallel_size=32, attention_backend='FLASHMLA_SPARSE_DSV4',
                            kv_cache_dtype='fp8', indexer_kv_cache_dtype='fp8')
    capacity = get_attention_kv_cache_capacity(get_attention_family('deepseek_v4_attention'), replica.model_config,
                    block_size=256, selection=AttentionKVCacheSelection.from_replica(replica, block_size=256))
    group_rows = [dict(group_id=g.group_id, logical_block_size=g.logical_block_size,
                       window_tokens=g.window_tokens or 0, extra_retained_tokens=g.extra_retained_tokens,
                       component_bytes_per_page=capacity.get_group_size_bytes(g)) for g in capacity.groups]
    assert all(g['extra_retained_tokens'] == 0 for g in group_rows)
    witness = native_witness(6144, group_rows)
    assert witness['selected'] == witness['d_live'] == witness['p_held'] == [24, 2, 2, 2, 16]
    assert witness['p_clipped_including_null'] == [24, 3, 3, 3, 17]
    for i, group in enumerate(group_rows):
        group.update(initial_p_pages=witness['initial_p_pages'][i], p_held=witness['p_held'][i],
                     p_exchange_including_null=witness['p_clipped_including_null'][i],
                     d_non_null_pages=witness['d_live'][i], maximum_read_pages=witness['selected'][i],
                     registered_stride_bytes=capacity.physical_block_size_bytes,
                     bound_bytes=witness['selected'][i]*capacity.physical_block_size_bytes,
                     structural_bytes=witness['selected'][i]*group['component_bytes_per_page'])
    bound = sum(g['bound_bytes'] for g in group_rows)
    assert bound == 46103040 and capacity.physical_block_size_bytes == 1002240
    structural = sum(g['structural_bytes'] for g in group_rows)
    assert structural == 39521664
    write_csv('per_group_bound.csv', group_rows)
    return dict(status='PASS', B_max=bound, pages=46, registered_stride_bytes=1002240,
                structural_bytes=structural, wire_minus_structural_bytes=bound-structural,
                clip_only_bound_pages=50, clip_only_bound_bytes=50*1002240,
                without_unconditional_reductions_pages=2520,
                without_unconditional_reductions_bytes=2520*1002240,
                registrations=len(registrations), native_source_pin=rules['native_revision'],
                per_group=group_rows, native_witness=witness, workloads=workloads,
                native_nodes=native_nodes, aliases_deduplicated=False)


def check2(bound):
    from frontier.config import AnalyticalKVCacheTransferConfig
    from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import AnalyticalKVCacheTransferPredictor
    from frontier.types import ClusterType
    model = load(Q1/'nixl_timing_q2/proposed_input.json')['model']
    predictor = AnalyticalKVCacheTransferPredictor(AnalyticalKVCacheTransferConfig(
        kv_connector='nixl', network_bandwidth_gbps=model['bandwidth_gbps'],
        network_latency_ms=model['intercept_ms'], enable_compression=False))
    price = predictor.get_transfer_time(ClusterType.PREFILL, ClusterType.DECODE, None, bound)
    formula = model['intercept_ms'] + bound*8/(model['bandwidth_gbps']*1e6)
    assert price == formula and 0 < price
    rows = csv_rows(Q1/'nixl_timing_q2/possible_post_windows.csv')
    assert len(rows) == 2034
    summary, outputs = [], []
    for name in sorted({row['run'] for row in rows}):
        selected = [r for r in rows if r['run']==name]
        for row in selected:
            gap = (float(row['next_start_s'])-float(row['possible_post_s']))*1000
            assert math.isclose(gap, float(row['next_gap_ms']), rel_tol=1e-12)
            outputs.append(dict(run=name, transfer_id=row['transfer_id'], lane=row['lane'],
                                possible_post_s=row['possible_post_s'], next_start_s=row['next_start_s'],
                                next_gap_ms=gap, B_max=bound, price_ms=price,
                                margin_ms=gap-price, status='PASS' if price < gap else 'FAIL'))
        summary.append(dict(run=name, possible_posts=len(selected),
                            min_gap_ms=min(float(r['next_gap_ms']) for r in selected),
                            price_ms=price, crossings=sum(r['status']=='FAIL' for r in outputs if r['run']==name)))
    assert all(r['status']=='PASS' for r in outputs)
    write_csv('model_windows.csv', outputs)
    return dict(status='PASS', model=model, production_price_ms=price, formula_price_ms=formula,
                difference_ms=price-formula, possible_posts=len(rows), runs=summary)


def check3(bound, gaps):
    raw_path = track('/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download/C_nixl/nixl_reads.jsonl')
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    summary = csv_rows(Q1/'nixl6144_run/case_summary.csv')
    flat = csv_rows(Q1/'nixl_timing_q2/scratch/reader_trials.csv')
    manifest = load(raw_path.parents[1]/'download_manifest.json')
    entry = next(e for e in manifest if e['path']=='C_nixl/nixl_reads.jsonl')
    assert sha(raw_path) == entry['sha256'] and raw_path.stat().st_size == entry['bytes']
    expected = {(r['case'],int(r['pass'])):r for r in summary}
    assert len(raw) == len(expected) == 90
    flat_map = {(r['case'],int(r['pass_index']),int(r['rank']),int(r['trial_index'])):
                float(r['complete_ms']) for r in flat}
    assert len(flat_map)==len(flat)==12150
    visited = set()
    for record in raw:
        row = expected[(record['case'],record['pass'])]
        assert int(row['bytes_per_reader']) == record['bytes_per_reader']
        assert record['bytes_per_reader'] == record['blocks_per_request']*record['requests_in_flight']*1002240
        assert len(record['readers'])==record['active_pairs']
        for reader in record['readers']:
            assert reader['data_verified'] is True and len(reader['timings'])==30
            for i, t in enumerate(reader['timings']):
                key = (record['case'],record['pass'],reader['rank'],i)
                assert key not in visited and flat_map[key]==t['complete_us']/1000
                visited.add(key)
    assert visited==set(flat_map)
    grid = sorted({r['bytes_per_reader'] for r in raw if r['active_pairs']==8})
    assert bound <= max(grid)
    coordinate = min(b for b in grid if b>=bound)
    selected = [r for r in raw if r['active_pairs']==8 and r['bytes_per_reader']==coordinate]
    trials, pass_rows = [], []
    for r in sorted(selected, key=lambda r:(r['case'],r['pass'])):
        values=[]
        for reader in r['readers']:
            for i, t in enumerate(reader['timings']):
                ms=t['complete_us']/1000
                values.append(ms)
                trials.append(dict(case=r['case'], pass_index=r['pass'], rank=reader['rank'], trial_index=i,
                    bytes_per_reader=coordinate, active_pairs=8, inflight=r['requests_in_flight'],
                    prep_ms=t['prep_us']/1000, post_ms=t['post_us']/1000, complete_ms=ms,
                    gap_p2_32_ms=gaps[32], gap_p2_64_ms=gaps[64],
                    below_p2_32=0<ms<gaps[32], below_p2_64=0<ms<gaps[64]))
        pass_rows.append(dict(case=r['case'],pass_index=r['pass'],trials=len(values),
                              min_ms=min(values),median_ms=statistics.median(values),max_ms=max(values),
                              p2_32_failures=sum(not(0<v<gaps[32]) for v in values),
                              p2_64_failures=sum(not(0<v<gaps[64]) for v in values)))
    assert len(trials)==1440 and all(t['below_p2_32'] and t['below_p2_64'] for t in trials)
    write_csv('selected_reader_trials.csv', trials)
    write_csv('selected_pass_summary.csv', pass_rows)
    return dict(status='PASS', raw_records=90, raw_and_flat_equal_trials=len(visited),
                eight_pair_grid_bytes=grid, largest_measured_bytes_per_reader=max(grid),
                selected_bytes_per_reader=coordinate, selected_cases=sorted({r['case'] for r in selected}),
                selected_case_passes=len(selected), selected_trials=len(trials), removed_trials=0,
                min_ms=min(t['complete_ms'] for t in trials), max_ms=max(t['complete_ms'] for t in trials),
                pass_results=pass_rows, min_margin_p2_32_ms=gaps[32]-max(t['complete_ms'] for t in trials),
                min_margin_p2_64_ms=gaps[64]-max(t['complete_ms'] for t in trials))


def check4():
    configurations = []
    for scale in (32,64):
        roots = [Path(f'/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_{scale}_warm_k19b29'),
                 Path(f'/data/ycfeng/tmp/glm53_dsv4/two_fix_20261101/runs/p2_{scale}')]
        for root in roots:
            receipt = load(root/'receipt.json')
            config = load(Path(receipt['metrics_dir'])/'config.json')
            assert config['metrics_config']['enable_op_level_tracing'] is False
            assert config['kv_cache_transfer_config']['enable_compression'] is False
            assert config['kv_cache_transfer_config']['nixl_read_completion']=='engine_step'
            assert config['cluster_config']['replica_scheduler_config']['num_blocks_mode']=='explicit'
            transfers = [r for r in (json.loads(line) for line in track(Path(receipt['metrics_dir'])/'frontier_transfer_ledger.jsonl').read_text().splitlines())
                         if r['transfer_kind']=='kv_cache']
            assert len(transfers)==128 and all(r['status']=='completed' for r in transfers)
            assert {r['bytes'] for r in transfers}=={39521664}
            ends=[]
            for r in transfers:
                ends.extend([(r['end_ts_s'],-1),(r['start_ts_s'],1)])
            occupancy=peak=0
            for _,delta in sorted(ends):
                occupancy+=delta
                peak=max(peak,occupancy)
                assert occupancy>=0
            assert occupancy==0 and peak==1
            forward_rows=zero=0
            with gzip.open(track(root/'forwards.jsonl.gz'),'rt') as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['role']=='decode':
                        forward_rows+=1
                        zero+=row['end_s']<=row['start_s']
            assert zero==0
            configurations.append(dict(root=str(root),op_level_tracing=False,completion='engine_step',
                                      compression=False,explicit_capacity=True,transfer_rows=len(transfers),
                                      ledger_structural_byte_count=39521664,peak_overlapping_transfers=peak,
                                      decode_forward_rows=forward_rows,zero_duration_decode_forwards=zero))
    # Complete call-site census for the wire function: pricing and optional tracing.
    calls=[]
    for path in sorted((REPO/'frontier').rglob('*.py')):
        text=path.read_text()
        if 'get_grouped_nixl_wire_bytes(' not in text:
            continue
        track(path)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='get_grouped_nixl_wire_bytes':
                calls.append(dict(path=str(path.relative_to(REPO)),line=node.lineno))
    assert calls==[dict(path='frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py',line=148),
                   dict(path='frontier/metrics/op_trace_utils.py',line=894)]
    return dict(status='PASS', wire_function_calls=calls, configurations=configurations,
                scope='retained P2 configurations; optional operator tracing is disabled',
                active_wire_sinks=['READ transfer_time_ms'],
                inactive_wire_sinks=['operator trace meta.wire_bytes; enable_op_level_tracing=False'],
                unchanged_non_wire_outputs=['structural byte_count=39521664; transfer ledger and total_data_transferred',
                    'component page IDs and bytes; allocator and completion ownership checks',
                    'physical registered stride=1002240; explicit block capacity'],
                reported_wire_bytes_on_p2=False,
                four_metrics=['TTFT','TPOT','E2E','THPT'])


def source_excerpts():
    spans = {
        NX+'base_scheduler.py':[(88,99),(128,150),(240,284),(372,377)],
        NX+'pull_scheduler.py':[(61,66),(84,108),(149,179),(261,281)],
        NX+'base_worker.py':[(99,130),(539,545),(564,582),(960,1018),(1395,1455),(2381,2410),(2437,2446)],
        NX+'pull_worker.py':[(143,171),(240,252),(287,329)],
        NX+'tp_mapping.py':[(75,114)],
        CORE+'kv_cache_manager.py':[(99,109),(449,464),(493,551),(783,803)],
        CORE+'single_type_kv_cache_manager.py':[(250,279),(306,325),(597,661),(1095,1098)],
        CORE+'sched/scheduler.py':[(918,921),(1008,1043),(1050,1080),(2677,2700)],
        'vllm/v1/kv_cache_interface.py':[(537,545),(595,615)],
        'vllm/models/deepseek_v4/compressor.py':[(61,71),(173,203)],
        'vllm/models/deepseek_v4/attention.py':[(716,738),(762,774)],
        'vllm/models/deepseek_v4/sparse_mla.py':[(51,55)],
        'vllm/v1/attention/backends/mla/sparse_swa.py':[(82,110),(117,125)],
        'vllm/v1/attention/backends/mla/indexer.py':[(202,210)],
        'vllm/v1/worker/utils.py':[(301,310),(323,345)],
        'vllm/config/vllm.py':[(1717,1789)],
        'vllm/config/model.py':[(619,625)],
        'vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py':[(79,93)],
        'vllm/platforms/cuda.py':[(660,663)],
    }
    frontier_spans={
        'frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py':[(54,74),(84,97),(139,149)],
        'frontier/attention/memory.py':[(211,224)],
        'frontier/events/kv_cache_transfer_start_event.py':[(165,186),(192,215),(228,285)],
        'frontier/events/kv_cache_transfer_end_event.py':[(12,44),(68,109),(111,136)],
        'frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py':[(39,87)],
        'frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py':[(353,380)],
        'frontier/entities/request.py':[(329,335),(953,1007)],
        'frontier/metrics/metrics_store.py':[(2424,2479),(2981,2982),(3501,3562),(5468,5518),(5520,5532),(5534,5580)],
        'frontier/metrics/op_trace_utils.py':[(841,850),(881,895)],
        'frontier/scheduler/utils/memory_planner.py':[(253,274)],
        'frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py':[(414,438)],
        'frontier/entities/kv_cache_transfer_info.py':[(19,25),(46,65),(96,100)],
    }
    chunks=['# Q3 source excerpts\n\nRead-only source; original one-based line numbers. SHA256 is in results.json.\n']
    for root, ranges in ((NATIVE,spans),(REPO,frontier_spans)):
        for relative, intervals in ranges.items():
            path=track(root/relative)
            lines=path.read_text().splitlines()
            chunks.append(f'\n## {path}\n')
            for first,last in intervals:
                chunks.append('```text\n'+'\n'.join(f'{i}: {lines[i-1]}' for i in range(first,min(last,len(lines))+1))+'\n```\n')
    (HERE/'source_excerpts.md').write_text(''.join(chunks))


def main():
    bound=check1()
    model=check2(bound['B_max'])
    gaps={scale:min(r['min_gap_ms'] for r in model['runs'] if r['run'].startswith(f'p2_{scale}')) for scale in (32,64)}
    measured=check3(bound['B_max'],gaps)
    sinks=check4()
    source_excerpts()
    for p in [TASK/'requirements.md',TASK/'plan.md',TASK/'progress.md',TASK/'summary.md',
              Q1/'ledger.md',Q1/'payload_observation_contract.md',H3O/'a10/a10_driver.py',
              H3O/'two_fix_20261101/run_replay.py',REPO/'AGENTS.md',NATIVE/'AGENTS.md',NATIVE/'CLAUDE.md',
              H3O/'qualification_codex_20261031/qualification_report.md',H3O/'b30b/kv_layout_evidence.json',
              H3O/'two_fix_20261101/nixl_boundaries.json',Q1/'nixl_timing_q2/transfer_windows.csv',
              Q1/'nixl_timing_q2/report.md',REPO/'task_memory/task_2026-10-21_v4_grouped_nixl/native_retention.md']:
        track(p)
    write_json('results.json',dict(status='PASS',criterion='D-P2-PAYLOAD-C; unchanged four checks',
        classification='source-derived, metric-invariant; proposed ledger disposition, not native observation',
        check1=bound,check2=model,check3=measured,check4=sinks,gaps_ms={str(k):v for k,v in gaps.items()},
        extracted_native_methods=EXTRACTED,evidence_hashes=INPUTS,
        environment=dict(python=platform.python_version(),executable=os.sys.executable,cwd=str(Path.cwd()),
                         env={k:os.environ.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','TMPDIR')})))
    print(json.dumps(dict(checks={str(i):'PASS' for i in range(1,5)},B_max=bound['B_max'],
                         model_ms=model['production_price_ms'],selected_trials=measured['selected_trials'],
                         observed_max_ms=measured['max_ms'],gaps_ms=gaps)))


if __name__=='__main__':
    main()
