#!/usr/bin/env python3
"""Qualify installed paged sparse decode with declared exact-content fixtures."""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import time
import traceback


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases(config, preparation, source):
    r = config['sparse_config']['region_block_size']
    k = config['sparse_config']['topk']
    w = config['sliding_window']
    b = preparation['derived_inputs']['paged_prefill_candidates_from_source_guard'][0]
    import ast
    tree = ast.parse(source.read_text())
    planner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sparse_decode_splits')
    namespace = {}
    exec(compile(ast.Module(body=[planner], type_ignores=[]), str(source), 'exec'), namespace)
    rows = 4
    splits = namespace['sparse_decode_splits'](rows)
    lengths = sorted({r-1, r, r+1, 2*r-1, 2*r, 2*r+1, b-1, b, b+1,
                      w-1, w, w+1, k*r-1, k*r, k*r+1, (k+1)*r+1})
    return [dict(id=f'L{n}_S{s}', length=n, splits=s, rows=rows,
                 heads=config['num_attention_heads']//8, dim=config['head_dim'],
                 block=b, region=r, topk=k, window=w, region_width=k+1)
            for n in lengths for s in (1, splits)]


def bf16(array):
    import numpy as np
    values = np.asarray(array, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded


def record(array, dtype):
    return dict(dtype='torch.'+dtype, shape=list(array.shape),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def fixture(case, phase):
    import numpy as np
    n, b, d, rows = (case[key] for key in ('length', 'block', 'dim', 'rows'))
    pages = (n+b-1)//b
    table = np.asarray([[pages+row*pages+(pages-1-col+phase) % pages
                         for col in range(pages)] for row in range(rows)], dtype=np.int32)
    current = (n-1)//case['region']
    history = list(range(current))
    if len(history) > case['topk']:
        history.remove(1)
    selected = history+[current]
    regions = np.full((rows, case['region_width']), -1, dtype=np.int64)
    regions[:, -len(selected):] = selected
    query = np.zeros((rows, case['heads'], d), dtype=np.float32)
    query[:, :, 0] = 32
    cache = np.zeros(((rows+1)*pages, 1, b, 2*d), dtype=np.float32)
    cache[:pages, :, :, 0] = 256
    cache[:pages, :, :, d:] = -64
    outputs = np.zeros_like(query)
    masks = ((1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 1), (0, 1, 1, 0))
    tokens = np.arange(pages*b)
    for row in range(rows):
        values = np.asarray([((-1)**(phase+col//64)) * (tokens % 128+128 if col % 3 == 0
                             else tokens//128+1 if col % 3 == 1 else np.full_like(tokens, row+1)) * 2**(phase%2)
                             for col in range(d)], dtype=np.float32).T
        keys = np.zeros((pages*b, d), dtype=np.float32)
        probe = n-1 if row % 2 == 0 else 0
        keys[probe, 0] = 128
        keys[n:, 0] = 256
        if current > case['topk']:
            keys[case['region']:2*case['region'], 0] = 256
        cache[table[row], 0] = np.concatenate((keys, values), axis=1).reshape(pages, b, 2*d)
        if masks[phase][row]:
            outputs[row] = values[probe]
    arrays = dict(query=bf16(query), cache=bf16(cache), regions=regions, table=table,
                  positions=np.full(rows, n-1, dtype=np.int64), active=np.asarray(masks[phase], dtype=np.int32))
    return arrays, bf16(outputs)


def expected(case, phase):
    return expected_frame(tuple(case.items()), phase)


@lru_cache(maxsize=104)
def expected_frame(case_items, phase):
    import base64
    case = dict(case_items)
    arrays, output = fixture(case, phase)
    return dict(phase=phase, output=dict(dtype='torch.bfloat16', shape=list(output.shape),
                                       bytes=base64.b64encode(output.tobytes()).decode()),
                inputs={name: record(value, 'bfloat16' if name in ('query', 'cache') else
                                    'int64' if value.dtype.itemsize == 8 else 'int32')
                        for name, value in arrays.items()}, addresses_unchanged=True, output_identity=True)


def fault_output(case, phase, fault):
    """Evaluate explicit wrong addressing or visibility with independent FP64 math."""
    import numpy as np
    arrays, _ = fixture(case, phase)
    cache = (arrays['cache'].astype(np.uint32) << 16).view(np.float32)
    output = np.zeros((case['rows'], case['heads'], case['dim']), dtype=np.float32)
    for row in range(case['rows']):
        if not arrays['active'][row]:
            continue
        regions = [v for v in arrays['regions'][row] if v >= 0]
        if fault == 'missing_current':
            regions = regions[:-1]
        if fault == 'whole_history':
            regions = list(range((case['length']-1)//case['region']+1))
        tokens = list(regions) if fault == 'region_as_token' else [r*case['region']+t for r in regions for t in range(case['region'])]
        tokens = [t for t in tokens if fault == 'future_read' or t < case['length']]
        if fault == 'swa_truncation':
            tokens = [t for t in tokens if t >= case['length']-case['window']]
        if not tokens:
            continue
        table = arrays['table'][0 if fault == 'cross_request' else row]
        pages = np.asarray(tokens)//case['block'] if fault == 'identity_mapping' else table[np.asarray(tokens)//case['block']]
        stored = cache[pages, 0, np.asarray(tokens) % case['block']]
        scores = stored[:, 0].astype(np.float64)*32/math.sqrt(case['dim'])
        weights = np.exp(scores-scores.max())
        output[row] = weights @ stored[:, case['dim']:].astype(np.float64)/weights.sum()
    return bf16(output)


def compare(case, row):
    if row.get('error') or row.get('case') != case:
        return ['case_or_native_execution']
    return [] if row.get('frames') == [expected(case, p) for p in range(4)] and row.get('graph_replays') == 3 else ['content_inputs_or_graph']


def execute(case, reference):
    import torch
    from vllm.models.step5.nvidia.ops.sparse_decode_attention import sparse_decode_attention, sparse_decode_splits
    assert case['splits'] in (1, sparse_decode_splits(case['rows']))
    arrays, _ = fixture(case, 0)
    def tensor(name, value):
        cpu = torch.from_numpy(value)
        return (cpu.view(torch.bfloat16) if name in ('query', 'cache') else cpu).cuda()
    tensors = {name: tensor(name, value) for name, value in arrays.items()}
    output = torch.empty_like(tensors['query'])
    addresses = {name: value.data_ptr() for name, value in dict(tensors, output=output).items()}
    def restore(phase):
        arrays, _ = fixture(case, phase)
        for name, value in arrays.items():
            tensors[name].copy_(tensor(name, value))
        output.fill_(16)
    def call():
        sparse_decode_attention(tensors['query'], tensors['cache'], tensors['regions'], tensors['table'],
                                tensors['positions'], tensors['active'], output,
                                scale=case['dim']**-0.5, query_len=1, splits=case['splits'],
                                clear_inactive=True, gate=None)
    def frame(phase):
        torch.cuda.synchronize()
        captures = {name: reference.capture(value) for name, value in tensors.items()}
        import base64
        return dict(phase=phase, output=reference.capture(output),
                    inputs={name: dict(dtype=c['dtype'], shape=c['shape'],
                                       sha256=hashlib.sha256(base64.b64decode(c['bytes'])).hexdigest())
                            for name, c in captures.items()},
                    addresses_unchanged=addresses == {name: value.data_ptr() for name, value in dict(tensors, output=output).items()},
                    output_identity=output.untyped_storage().data_ptr() not in
                                    [value.untyped_storage().data_ptr() for value in tensors.values()])
    restore(0)
    call()
    frames = [frame(0)]
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for phase in range(1, 4):
        restore(phase)
        graph.replay()
        frames.append(frame(phase))
    return dict(case=case, frames=frames, graph_replays=3)


def rejections(case):
    import torch
    from vllm.models.step5.nvidia.ops.sparse_decode_attention import sparse_decode_attention
    arrays, _ = fixture(case, 0)
    tensors = {name: torch.from_numpy(value).view(torch.bfloat16).cuda() if name in ('query', 'cache')
               else torch.from_numpy(value).cuda() for name, value in arrays.items()}
    output = torch.empty_like(tensors['query'])
    tests = {'query_width': dict(query=tensors['query'][:, :, :-1]),
             'cache_width': dict(cache=tensors['cache'][:, :, :, :-1]),
             'excess_heads': dict(query=torch.empty((case['rows'], 17, case['dim']), device='cuda')),
             'query_stride': dict(query=torch.empty((case['rows'], case['heads'], 2*case['dim']), device='cuda')[:, :, ::2]),
             'output_stride': dict(output=torch.empty((case['rows'], case['heads'], 2*case['dim']), device='cuda')[:, :, ::2]),
             'cache_stride': dict(cache=torch.empty((*tensors['cache'].shape[:-1], 4*case['dim']), device='cuda')[:, :, :, ::2]),
             'regions_stride': dict(regions=torch.empty((case['rows'], 2*case['region_width']), device='cuda', dtype=torch.int64)[:, ::2]),
             'regions_rows': dict(regions=tensors['regions'][:-1])}
    results = []
    for name, changed in tests.items():
        inputs = dict(tensors, output=output, **{})
        inputs.update(changed)
        try:
            sparse_decode_attention(inputs['query'], inputs['cache'], inputs['regions'], inputs['table'],
                                    inputs['positions'], inputs['active'], inputs['output'],
                                    scale=case['dim']**-0.5, splits=1)
        except AssertionError:
            results.append(dict(id=name, exception='AssertionError', passed=True))
        else:
            results.append(dict(id=name, passed=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'run', 'check', 'self-test'])
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.task_root = args.task_root.resolve()
    args.output_dir = args.output_dir.resolve()
    reference = load(args.task_root/'native_g1_joint/check_joint.py')
    assert args.task_root == reference.TASK
    reference.inputs()
    collector = load(args.task_root/'native_g1/run_g1_cases.py')
    preparation, manifest, _ = collector.verify_inputs(args.task_root/'g1_preparation.json')
    source = args.task_root/'native_source_export/observed/vllm/models/step5/nvidia/ops/sparse_decode_attention.py'
    source_record = next(row for row in manifest['files'] if row['relative_path'] == 'vllm/models/step5/nvidia/ops/sparse_decode_attention.py')
    assert collector.sha256_file(source) == source_record['source_sha256']
    assert source.stat().st_size == source_record['source_size_bytes']
    config = json.loads((args.task_root/'reference_inputs/config.json').read_text())['text_config']
    matrix = cases(config, preparation, source)
    if args.mode == 'submit':
        import os
        assert os.environ['STEPMIND_BACKEND'] == 'rjob' and not args.output_dir.exists()
        launcher = load(args.task_root/'native_source_export/submit_export.py')
        launcher.inspection.locked_inputs()
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        os.chdir(launcher.inspection.WORKTREE)
        os.environ['EXP_ID'] = 'step5-g1-sparse-decode'
        command = 'export VLLM_ENABLE_CUDA_COMPATIBILITY=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH; exec python3 -B '+shlex.join([str(Path(__file__).resolve()), 'run', '--task-root', str(args.task_root), '--output-dir', str(args.output_dir)])
        workers = spawn_tasks(ResourceConfig(**launcher.resource_config_arguments()), **launcher.spawn_arguments('timeout 2400 bash -c '+shlex.quote(command)))
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True)
        return
    if args.mode == 'self-test':
        assert len(matrix) == 26
        for case in matrix:
            row = dict(case=case, frames=[expected(case, p) for p in range(4)], graph_replays=3)
            assert not compare(case, row)
            import numpy as np
            for phase in range(4):
                _, output = fixture(case, phase)
                assert np.array_equal(fault_output(case, phase, 'none'), output)
            for key, value in [('error', 'injected'), ('graph_replays', 2), ('frames', [])]:
                assert compare(case, dict(row, **{key: value}))
            for field, value in [('output', {}), ('inputs', {}), ('phase', 8), ('addresses_unchanged', False), ('output_identity', False)]:
                changed = json.loads(json.dumps(row))
                changed['frames'][0][field] = value
                assert compare(case, changed)
        for fault in ('region_as_token', 'missing_current', 'whole_history', 'future_read',
                      'swa_truncation', 'identity_mapping', 'cross_request'):
            case = max(matrix, key=lambda c: (c['length'], c['splits']))
            import numpy as np
            _, output = fixture(case, 0)
            assert not np.array_equal(fault_output(case, 0, fault), output), fault
        print('SPARSE_DECODE_SELF_TEST PASS 26 cases, 208 rejected mutations, 7 distinguishable math faults')
        return
    if args.mode == 'check':
        summary = json.loads((args.output_dir/'summary.json').read_text())
        assert summary['cases'] == matrix and summary['worker_sha256'] == collector.sha256_file(Path(__file__))
        assert summary['config_sha256'] == reference.CONFIG_SHA and summary['manifest_sha256'] == reference.MANIFEST_SHA
        assert len(summary['rejections']) == 8 and all(r['passed'] for r in summary['rejections'])
        verdicts = [dict(id=c['id'], reasons=compare(c, json.loads((args.output_dir/(c['id']+'.json')).read_text()))) for c in matrix]
        collector.dump_json(args.output_dir.parent/'verdicts.json', verdicts)
        print('SPARSE_DECODE_VERDICTS', sum(not r['reasons'] for r in verdicts), '/', len(matrix))
        raise SystemExit(any(r['reasons'] for r in verdicts))
    args.output_dir.mkdir(exist_ok=False, parents=True)
    manifest = json.loads((args.task_root/'native_source_export/observed/source_manifest.json').read_text())
    collector.IMPORTED_FILES = [r['relative_path'] for r in manifest['files']]
    collector.verify_installed(manifest)
    environment = collector.record_environment(args.output_dir)
    assert environment['device_name'] == 'NVIDIA H800'
    assert environment['vllm'] == load(args.task_root/'native_source_export/submit_export.py').EXPECTED_VLLM_VERSION
    started = time.perf_counter()
    import torch
    for case in matrix:
        try:
            with torch.no_grad():
                row = execute(case, reference)
        except Exception:
            row = dict(case=case, error=traceback.format_exc())
        collector.dump_json(args.output_dir/(case['id']+'.json'), row)
        print('SPARSE_DECODE_CASE', case['id'], 'FAIL' if row.get('error') else 'EXECUTED', flush=True)
    import resource
    collector.dump_json(args.output_dir/'summary.json', dict(environment=environment, cases=matrix, rejections=rejections(matrix[0]),
        worker_sha256=collector.sha256_file(Path(__file__)), config_sha256=reference.CONFIG_SHA, manifest_sha256=reference.MANIFEST_SHA,
        wall_seconds=time.perf_counter()-started,
        host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, cuda_peak_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
