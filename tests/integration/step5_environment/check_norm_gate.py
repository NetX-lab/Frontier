#!/usr/bin/env python3
"""Qualify installed Step5 normalization and headwise gating on exact fixtures."""
import argparse
import ast
import itertools
import json
import math
import struct
from pathlib import Path
import shlex
import time
import traceback


def cases(config):
    sparse = config['sparse_config']
    norms = [dict(kind='norm', width=w, epsilon=config['rms_norm_eps'], input_dtype=i, weight_dtype=d, residual=r, rows=n)
             for w, i, d, r, n in itertools.product(
                 [config['hidden_size'], config['head_dim'], sparse['proxy_dim']],
                 ['bfloat16', 'float32'], ['float32', 'bfloat16'], [False, True], [1, 8])]
    gates = [dict(kind='gate', width=config['num_attention_heads'] // 8 * config['head_dim'],
                  heads=config['num_attention_heads'] // 8, gate_dtype=d, mask=m, strided=s, rows=n)
             for d, m, s, n in itertools.product(
                 ['bfloat16', 'float32'], ['none', 'active', 'slots'], [False, True], [1, 8])]
    return [dict(id=f'{c["kind"]}_{i:03d}', **c) for i, c in enumerate(norms + gates)]


def values(case, phase):
    return [(-1 if (col + row + phase) % 2 else 1) * (1 + row % 2)
            for row in range(case['rows']) for col in range(case['width'])]


def residual_values(case, phase):
    return [(-1 if v < 0 else 1) * (-1/32 if col % 2 == 0 else 1/256)
            for col, v in enumerate(values(case, phase))]


def wanted(case, phase, reference):
    width, rows = case['width'], case['rows']
    source = values(case, phase)
    if case['kind'] == 'norm':
        offsets = residual_values(case, phase) if case['residual'] else [0] * len(source)
        summed = [v + r for v, r in zip(source, offsets)]
        output = []
        for row in range(rows):
            chunk = summed[row*width:(row+1)*width]
            inverse = 1/math.sqrt(sum(v*v for v in chunk)/width + case['epsilon'])
            for col, v in enumerate(chunk):
                bits = struct.unpack('<I', struct.pack('<f', v * inverse * (1 + col % 2)))[0]
                bits += 0x7fff + ((bits >> 16) & 1)
                output.append(struct.unpack('<f', struct.pack('<I', (bits >> 16) << 16))[0])
        result = dict(output=reference.packed(output, 'bfloat16', [rows, width]))
        if case['residual']:
            result['residual'] = reference.packed(summed, 'float32', [rows, width])
        return result
    dimension = width // case['heads']
    output = [v * [0.5, 1.0, 0.0][(col // dimension) % 3]
              if case['mask'] == 'none' or row % 2 == 0 else 0
              for row in range(rows) for col, v in enumerate(source[row*width:(row+1)*width])]
    return dict(output=reference.packed(output, 'bfloat16', [rows, width]))


def compare(case, row, reference):
    if row.get('error') or row.get('case') != case or not row.get('native_supported'):
        return ['native_execution_or_identity']
    target = [dict(phase=p, tensors=wanted(case, p, reference), aliases=True) for p in range(4)]
    return [] if row.get('frames') == target and row.get('graph_replays') == 3 else ['content_alias_or_graph']


def execute(case, config, reference):
    import torch
    from vllm.models.step5.nvidia.layernorm import Step5RMSNorm
    from vllm.models.step5.nvidia.ops.rms_norm import supports_rms_norm
    from vllm.models.step5.nvidia.ops.attention_gate import headwise_sigmoid_gate
    rows, width = case['rows'], case['width']
    x = torch.empty((rows, width * (2 if case.get('strided') else 1)), device='cuda',
                    dtype=getattr(torch, case.get('input_dtype', 'bfloat16')))
    x = x[:, ::2] if case.get('strided') else x
    residual = torch.empty_like(x, dtype=torch.float32) if case.get('residual') else None
    if case['kind'] == 'norm':
        norm = Step5RMSNorm(width, config['rms_norm_eps'], config['zero_centered'],
                            dtype=getattr(torch, case['weight_dtype'])).cuda()
        norm.weight.data.copy_(torch.arange(width, device='cuda') % 2)
        supported = supports_rms_norm(x, norm.weight, residual)
        explicit = torch.empty_like(x, dtype=torch.bfloat16)
        def call():
            return norm(x, residual, out_dtype=torch.bfloat16) if residual is not None else norm(x, output=explicit, out_dtype=torch.bfloat16)
    else:
        gate = torch.empty((rows, case['heads'] * (2 if case['strided'] else 1)), device='cuda', dtype=getattr(torch, case['gate_dtype']))
        gate = gate[:, ::2] if case['strided'] else gate
        gate.copy_(torch.tensor([[0, 100, -100][h % 3] for h in range(case['heads'])], device='cuda'))
        mask = torch.tensor([row % 2 == 0 for row in range(rows)], device='cuda')
        slots = torch.tensor([row if row % 2 == 0 else -1 for row in range(rows)], device='cuda')
        supported = True
        def call():
            return headwise_sigmoid_gate(x, gate, active=mask if case['mask'] == 'active' else None,
                                         slot_mapping=slots if case['mask'] == 'slots' else None)
    if not supported:
        raise RuntimeError('Fixture is not supported by the native RMSNorm kernel')
    def restore(phase):
        x.copy_(torch.tensor(values(case, phase), device='cuda', dtype=x.dtype).reshape(rows, width))
        if residual is not None:
            residual.copy_(torch.tensor(residual_values(case, phase), device='cuda').reshape(rows, width))
    def frame(result, phase):
        output, returned = result if isinstance(result, tuple) else (result, None)
        tensors = dict(output=reference.capture(output))
        aliases = returned is residual if residual is not None else output is explicit if case['kind'] == 'norm' else output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        unchanged = reference.capture(x) == reference.packed(values(case, phase), str(x.dtype).split('.')[1], [rows, width])
        aliases = aliases and unchanged
        if returned is not None:
            tensors['residual'] = reference.capture(returned)
        return dict(phase=phase, tensors=tensors, aliases=aliases)
    restore(0)
    frames = [frame(call(), 0)]
    restore(0)
    call()
    torch.cuda.synchronize()
    restore(0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    addresses = (x.data_ptr(), residual.data_ptr() if residual is not None else None)
    for phase in range(1, 4):
        restore(phase)
        graph.replay()
        torch.cuda.synchronize()
        assert addresses == (x.data_ptr(), residual.data_ptr() if residual is not None else None)
        frames.append(frame(captured, phase))
    return dict(case=case, native_supported=supported, frames=frames, graph_replays=3)


def rejections(config):
    import torch
    from vllm.models.step5.nvidia.layernorm import Step5RMSNorm, get_step5_norm_dtype
    from vllm.models.step5.nvidia.ops.rms_norm import rms_norm
    from vllm.models.step5.nvidia.ops.attention_gate import headwise_sigmoid_gate
    from types import SimpleNamespace
    width = config['hidden_size']
    x = torch.ones((1, width), device='cuda', dtype=torch.bfloat16)
    weight = torch.zeros(width, device='cuda', dtype=torch.float32)
    norm = Step5RMSNorm(width, config['rms_norm_eps'], config['zero_centered'], dtype=torch.float32).cuda()
    tests = {
        'input_precision': (lambda: rms_norm(x.double(), weight, config['rms_norm_eps']), 'Unsupported RMSNorm inputs'),
        'input_width': (lambda: rms_norm(x[:, :-1], weight, config['rms_norm_eps']), 'Unsupported RMSNorm inputs'),
        'residual_precision': (lambda: rms_norm(x, weight, config['rms_norm_eps'], residual=x.clone()), 'Unsupported RMSNorm inputs'),
        'output_shape': (lambda: norm(x, output=torch.empty((1, width-1), device='cuda', dtype=x.dtype)), 'output buffer must match shape and dtype'),
        'output_precision': (lambda: norm(x, output=x.float()), 'output buffer must match shape and dtype'),
        'fp16_output': (lambda: norm(x, fp16_out=True), 'does not support fp16_out'),
        'gate_masks': (lambda: headwise_sigmoid_gate(x, weight[None, :1], active=torch.ones(1, device='cuda'), slot_mapping=torch.zeros(1, device='cuda')), 'Pass either active flags or decode slots'),
        'norm_setting': (lambda: get_step5_norm_dtype(SimpleNamespace(norm_dtype='float64')), 'Unknown norm_dtype'),
    }
    rows = []
    for name, (call, message) in tests.items():
        try:
            call()
        except ValueError as error:
            rows.append(dict(id=name, passed=message in str(error), error=str(error)))
        else:
            rows.append(dict(id=name, passed=False, error='Expected ValueError was absent'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'run', 'check', 'self-test'])
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import importlib.util
    path = args.task_root / 'native_g1_joint/check_joint.py'
    spec = importlib.util.spec_from_file_location('joint_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.inputs()
    config = json.loads((args.task_root / 'reference_inputs/config.json').read_text())['text_config']
    tree = ast.parse((args.task_root / 'native_source_export/observed/vllm/transformers_utils/configs/step5.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Step5Config')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    defaults = dict(zip([n.arg for n in init.args.args][-len(init.args.defaults):], init.args.defaults))
    config = dict(config, zero_centered=config.get('zero_centered', ast.literal_eval(defaults['zero_centered'])))
    matrix = cases(config)
    if args.mode == 'submit':
        import os
        assert os.environ['STEPMIND_BACKEND'] == 'rjob' and not args.output_dir.exists()
        launcher = reference.load(args.task_root / 'native_source_export/submit_export.py')
        launcher.inspection.locked_inputs()
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        os.chdir(launcher.inspection.WORKTREE)
        os.environ['EXP_ID'] = 'step5-g1-norm-gate'
        command = 'export VLLM_ENABLE_CUDA_COMPATIBILITY=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH; exec python3 -B ' + shlex.join([str(Path(__file__).resolve()), 'run', '--task-root', str(args.task_root), '--output-dir', str(args.output_dir)])
        workers = spawn_tasks(ResourceConfig(**launcher.resource_config_arguments()), **launcher.spawn_arguments('timeout 2400 bash -c ' + shlex.quote(command)))
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True)
        return
    if args.mode == 'self-test':
        assert len(matrix) == 72
        for case in matrix:
            row = dict(case=case, native_supported=True, frames=[dict(phase=p, tensors=wanted(case, p, reference), aliases=True) for p in range(4)], graph_replays=3)
            assert not compare(case, row, reference)
            for key, value in [('error', 'injected'), ('graph_replays', 2), ('frames', []), ('native_supported', False)]:
                assert compare(case, dict(row, **{key: value}), reference)
            for field, value in [('aliases', False), ('tensors', {}), ('phase', 8)]:
                changed = json.loads(json.dumps(row))
                changed['frames'][0][field] = value
                assert compare(case, changed, reference)
        print('NORM_GATE_SELF_TEST PASS 72 cases, 504 rejected mutations')
        return
    if args.mode == 'check':
        summary = json.loads((args.output_dir / 'summary.json').read_text())
        assert summary['cases'] == matrix and summary['worker_sha256'] == reference.hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        assert len(summary['rejections']) == 8 and all(r['passed'] for r in summary['rejections'])
        rows = [json.loads((args.output_dir / (c['id'] + '.json')).read_text()) for c in matrix]
        verdicts = [dict(id=c['id'], reasons=compare(c, r, reference)) for c, r in zip(matrix, rows)]
        reference.load(args.task_root / 'native_g1/run_g1_cases.py').dump_json(args.output_dir.parent / 'verdicts.json', verdicts)
        print('NORM_GATE_VERDICTS', sum(not r['reasons'] for r in verdicts), '/', len(matrix))
        raise SystemExit(any(r['reasons'] for r in verdicts))
    args.output_dir.mkdir(exist_ok=False, parents=True)
    collector = reference.load(args.task_root / 'native_g1/run_g1_cases.py')
    manifest = json.loads((args.task_root / 'native_source_export/observed/source_manifest.json').read_text())
    collector.IMPORTED_FILES = [r['relative_path'] for r in manifest['files']]
    collector.verify_installed(manifest)
    environment = collector.record_environment(args.output_dir)
    from vllm.transformers_utils.configs.step5 import Step5Config
    resolved = Step5Config(**config)
    assert resolved.zero_centered and resolved.norm_dtype == config['norm_dtype']
    from vllm.models.step5.nvidia.layernorm import get_step5_norm_dtype
    import torch
    assert get_step5_norm_dtype(resolved) == torch.float32
    assert resolved.fp32_residual_connection and resolved.use_head_wise_attn_gate
    collector.dump_json(args.output_dir / 'resolved_config.json', resolved.to_dict())
    started = time.perf_counter()
    for case in matrix:
        try:
            with torch.no_grad():
                row = execute(case, config, reference)
        except Exception:
            row = dict(case=case, error=traceback.format_exc())
        collector.dump_json(args.output_dir / (case['id'] + '.json'), row)
        print('NORM_GATE_CASE', case['id'], 'FAIL' if row.get('error') else 'EXECUTED', flush=True)
    import resource
    import torch
    collector.dump_json(args.output_dir / 'summary.json', dict(environment=environment, cases=matrix, rejections=rejections(config),
        worker_sha256=collector.sha256_file(Path(__file__)), wall_seconds=time.perf_counter()-started,
        host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, cuda_peak_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
