#!/usr/bin/env python3
"""Versioned host-operand oracle; all other frozen R2 predicates stay unchanged."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

R2_SHA256 = '29523700ac8492fa6e192ce272f8ab5a7a48d1528bd4064417e26e811533d10a'
R2_PATH = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/'
               'task_2026-10-22_v4_native_binding/check_groundtruth_run_successor_r2.py')
SOURCE_PINS = ('vllm/v1/worker/gpu/dp_utils.py', 'vllm/v1/worker/utils.py',
    'vllm/v1/worker/gpu/states.py',
    'vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py')
DISPATCH_HOOKS = {'vllm.v1.worker.gpu.model_runner.' + name for name in
    ('get_uniform_decode_token_count', 'dispatch_cg_and_sync_dp')}
DISPATCH_HOOKS.add('vllm.v1.worker.gpu.dp_utils.dist.all_reduce')


def native_dispatch_oracle(manifest, manager):
    """Execute unchanged pinned classifier/DP bodies; replace only CPU transport."""
    import torch
    source = Path(manifest['groundtruth_checkout_path'])
    namespace = dict(manager.dispatch.__func__.__globals__)
    namespace.update(torch=torch, get_dp_group=lambda: SimpleNamespace(cpu_group=None))
    names = {SOURCE_PINS[1]: ('is_uniform_query_len', 'get_uniform_decode_token_count'),
             SOURCE_PINS[0]: ('sync_cudagraph_and_dp_padding', 'dispatch_cg_and_sync_dp')}
    for relative, selected in names.items():
        tree = ast.parse((source / relative).read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
        if {n.name for n in nodes} != set(selected):
            raise ValueError('Missing pinned native dispatch functions')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source / relative), 'exec'), namespace)
    return namespace


def verify_dispatch_row(row, engine, manager, native, owner):
    evidence = row['dispatch_evidence']
    if (type(row['successful_forwards']) is not int or row['successful_forwards'] <= 0
            or type(row['successful_replays']) is not int or row['successful_replays'] < 0):
        raise ValueError('Dispatch evidence lacks a successful native forward')
    if set(evidence) != {'classifier', 'dispatch', 'dp_post_all_reduce'}:
        raise ValueError('Missing or ambiguous native dispatch evidence')
    classifier, observed = evidence['classifier'], evidence['dispatch']
    ci, inputs = classifier['inputs'], observed['inputs']
    if (set(classifier) != {'inputs', 'uniform_token_count'}
            or set(ci) != {'num_reqs', 'num_tokens', 'max_query_len', 'has_prefill'}
            or type(ci['has_prefill']) is not bool
            or set(observed) != {'inputs', 'descriptor', 'num_tokens_across_dp'}
            or set(inputs) != {'num_reqs', 'num_tokens', 'uniform_token_count', 'dp_size',
                'dp_rank', 'max_query_len', 'need_eager', 'num_active_loras'}):
        raise ValueError('Incomplete actual classifier/dispatch operands')
    for name in ('num_reqs', 'num_tokens', 'max_query_len'):
        if (type(ci[name]) is not int or type(inputs[name]) is not int
                or ci[name] <= 0 or ci[name] != inputs[name]):
            raise ValueError('Classifier/dispatch inputs do not belong to the same batch')
    if (ci['num_reqs'] != row['scheduled_num_requests']
            or ci['num_tokens'] != row['scheduled_num_tokens']
            or ci['num_reqs'] > ci['num_tokens'] or ci['max_query_len'] > ci['num_tokens']
            or ci['num_tokens'] > ci['num_reqs'] * ci['max_query_len']
            or type(inputs['need_eager']) is not bool or inputs['need_eager']
            or type(inputs['num_active_loras']) is not int or inputs['num_active_loras'] != 0
            or type(inputs['dp_size']) is not int or inputs['dp_size'] != engine['data_parallel_size']
            or type(inputs['dp_rank']) is not int or inputs['dp_rank'] != owner):
        raise ValueError('Unsupported or mismatched actual dispatch inputs')
    uniform = native['get_uniform_decode_token_count'](**ci)
    if any(type(v) is not int and v is not None for v in
           (classifier['uniform_token_count'], inputs['uniform_token_count'])):
        raise ValueError('Invalid uniform decode count')
    if uniform != classifier['uniform_token_count'] or uniform != inputs['uniform_token_count']:
        raise ValueError('Actual classifier result differs from pinned native classification')
    vector = evidence['dp_post_all_reduce']
    dp, rank = inputs['dp_size'], inputs['dp_rank']
    if (type(vector) is not list or len(vector) != 4 or any(type(r) is not list or len(r) != dp
            or any(type(n) is not int for n in r) for r in vector)
            or any(n < 0 or n > engine['max_num_batched_tokens'] for n in vector[0])
            or any(n not in (0, 1, 2) for n in vector[1])
            or any(n < 0 for n in vector[2]) or any(n < -1 or n == 0 for n in vector[3])):
        raise ValueError('Malformed pre-padding DP synchronization vector')
    import dataclasses
    import torch
    mode = native['CUDAGraphMode']
    desired = manager.dispatch(ci['num_reqs'], ci['num_tokens'], uniform, 0,
                               max_query_len=ci['max_query_len'])
    if [r[rank] for r in vector] != [ci['num_tokens'], desired.cg_mode.value,
                                   uniform or 0, ci['max_query_len'] or -1]:
        raise ValueError('DP vector is not associated with the actual local dispatch')

    def all_reduce(tensor, **kwargs):
        tensor.copy_(torch.tensor(vector, dtype=tensor.dtype))

    native['dist'] = SimpleNamespace(all_reduce=all_reduce)
    desc, padded = native['dispatch_cg_and_sync_dp'](manager, **inputs)
    expected = dataclasses.asdict(desc) | {'cg_mode': desc.cg_mode.name}
    if (observed['descriptor'] != expected
            or any(type(observed['descriptor'][k]) is not type(v) for k, v in expected.items())
            or observed['num_tokens_across_dp'] != padded.tolist()
            or any(type(n) is not int for n in observed['num_tokens_across_dp'])):
        raise ValueError('Actual dispatch return differs from pinned DP dispatch')
    expected_mode = desc.cg_mode.name
    descriptor = dict(num_tokens=desc.num_tokens,
        num_reqs=desc.num_reqs if desc.cg_mode == mode.FULL else None,
        uniform=None if desc.cg_mode == mode.FULL else False,
        has_lora=False, num_active_loras=0)
    if (row['graph_mode'] != expected_mode or row['batch_descriptor'] != descriptor
            or any(type(row['batch_descriptor'][k]) is not type(v) for k, v in descriptor.items())
            or (expected_mode == 'NONE' and row['successful_replays'])
            or (expected_mode != 'NONE' and not row['successful_replays'])
            or (expected_mode == 'FULL' and row['successful_replays'] != row['successful_forwards'])):
        raise ValueError('Successful forward/replay does not match the independently computed dispatch')


def successor_source(path=R2_PATH):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != R2_SHA256:
        raise ValueError('Frozen R2 dependency hash mismatch')
    text = content.decode()
    edits = [
        ('            native_v4_source_binding(manifest, engine, config)\n',
         '            source_files += SOURCE_PINS\n            native_v4_source_binding(manifest, engine, config)\n'),
        ('            dispatch, captured = native_v2_graph_dispatch(manifest, engine)\n',
         '            hooks |= DISPATCH_HOOKS\n            dispatch, captured = native_v2_graph_dispatch(manifest, engine)\n'
         '            native_operands = native_dispatch_oracle(manifest, dispatch)\n'),
    ]
    start = text.index('                        native_desc = dispatch.dispatch(num_requests, scheduled,')
    end = text.index('                    elif mode == "NONE":', start)
    edits.append((text[start:end], '                        verify_dispatch_row(row, engine, dispatch, native_operands, owner)\n'))
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError('Frozen R2 replacement boundary is not unique')
        text = text.replace(old, new)
    return text


exec(compile(successor_source(), __file__ + ':frozen_r2_with_dispatch_operands', 'exec'), globals())
