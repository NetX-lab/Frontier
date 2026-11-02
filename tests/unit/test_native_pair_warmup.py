"""Pair routing and receipt checks on CPU; no GPU qualification."""
import ast
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.comparison.calibration import native_pd_worker, pd_proxy, pd_replay
from vllm_replay import request_headers

NATIVE = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')


def templates(count, phase):
    return [dict(request_id=f'{phase}-r{i:03d}', role=phase, arrived_at=0.,
                 num_prefill_tokens=4, num_decode_tokens=2) for i in range(count)]


@pytest.mark.parametrize('p,d', [(1, 1), (2, 3), (4, 4)])
def test_pair_product_comes_from_dp_sizes(p, d):
    original = templates(p * d, 'warmup')
    targeted = pd_replay.warmup_pair_targets(original, dict(prefill=p, decode=d))
    assert {(r['intended_pair']['prefill'], r['intended_pair']['decode']) for r in targeted} == {
        (x, y) for x in range(p) for y in range(d)}
    assert all('intended_pair' not in row for row in original)
    with pytest.raises(ValueError, match=f'at least {p*d} requests'):
        pd_replay.warmup_pair_targets(original[:-1], dict(prefill=p, decode=d))


def test_formal_target_rejected_before_http():
    with pytest.raises(ValueError, match='Only warmup'):
        request_headers(dict(request_id='formal', segment='formal', intended_pair=dict(prefill=1, decode=0)))
    assert request_headers(dict(request_id='formal', segment='formal')) == {
        'Content-Type': 'application/json', 'X-Request-Id': 'formal'}
    with pytest.raises(ValueError, match='explicit warmup'):
        pd_proxy.warmup_pair({'X-Frontier-Warmup-Prefill-Rank': '1',
                              'X-Frontier-Warmup-Decode-Rank': '0'})


def test_cpu_http_routes_each_warmup_pair_and_keeps_formal_native(tmp_path, monkeypatch):
    web = pytest.importorskip('aiohttp.web')
    from aiohttp.test_utils import TestServer
    seen = []
    counts = dict(prefill=0, decode=0)

    def endpoint(role):
        async def complete(request):
            body = await request.json()
            rid = request.headers['X-Request-Id']
            target = request.headers.get('X-data-parallel-rank')
            seen.append((role, rid, target))
            counts[role] += 1
            if role == 'prefill':
                params = dict(do_remote_prefill=True, do_remote_decode=False, transfer_mode='pull',
                              remote_engine_id=f'cpu_prefill_dp{target or 0}', remote_request_id=f'cmpl-{rid}-0-id',
                              remote_host='127.0.0.1', remote_port=5557, remote_block_ids=[[1]])
                return web.json_response(dict(kv_transfer_params=params))
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            data = [dict(choices=[dict(index=0, text='', token_ids=[1])]),
                    dict(choices=[dict(index=0, text='', token_ids=[2], finish_reason='length')]),
                    dict(choices=[], usage=dict(prompt_tokens=4, completion_tokens=2, total_tokens=6))]
            for row in data:
                await response.write(('data: ' + json.dumps(row) + '\n\n').encode())
            await response.write(b'data: [DONE]\n\n')
            await response.write_eof()
            return response
        return complete

    async def idle(*args, **kwargs):
        return dict(verified=True, observations=[dict(roles={role: dict(completed_requests=count)
                                                           for role, count in counts.items()})])
    monkeypatch.setattr(pd_replay, 'observe_idle', idle)
    workload = (dict(repetitions=dict(repeat_indices=[1, 2, 3], runtime_request_id_rule='{request_id}-repeat{repeat:02d}')),
                templates(2, 'formal'), templates(4, 'warmup'))

    async def exercise():
        p, d = web.Application(), web.Application()
        p.router.add_post('/v1/completions', endpoint('prefill'))
        d.router.add_post('/v1/completions', endpoint('decode'))
        async with TestServer(p) as ps, TestServer(d) as ds:
            urls = dict(prefill=str(ps.make_url('')).rstrip('/'), decode=str(ds.make_url('')).rstrip('/'))
            app = pd_proxy.build_app(urls['prefill'], urls['decode'], '', '', tmp_path / 'proxy.jsonl',
                                     vllm_version='0.28.0')
            async with TestServer(app) as front:
                return await pd_replay.replay_repetitions(workload, front.port, 300, 0., urls, 2, 1., .01,
                                                         pair_warmup=True)
    result, records = asyncio.run(exercise())
    proxy = {r['request_id']: r for r in map(json.loads, (tmp_path / 'proxy.jsonl').read_text().splitlines())}
    assert len(records) == 18 and all(r['status'] == 'completed' for r in records)
    assert len(result['idle_boundaries']) == 7
    for repeat in (1, 2, 3):
        warm = [r for r in records if r['repeat'] == repeat and r['segment'] == 'warmup']
        assert {(r['intended_pair']['prefill'], r['intended_pair']['decode']) for r in warm} == {
            (0, 0), (0, 1), (1, 0), (1, 1)}
    for record in records:
        rid = record['request_id']
        upstream = {role: rank for role, request, rank in seen if request == rid}
        if record['segment'] == 'warmup':
            assert proxy[rid]['intended_pair'] == record['intended_pair']
            assert upstream == {role: str(rank) for role, rank in record['intended_pair'].items()}
        else:
            assert 'intended_pair' not in record and 'intended_pair' not in proxy[rid]
            assert upstream == dict(prefill=None, decode=None)


def native_method(relative, cls, name):
    tree = ast.parse((NATIVE / relative).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in owner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    method.decorator_list = []
    namespace = dict(Request=object, EngineCoreRequest=object, EngineIdentity=str, EngineCoreOutputs=object,
                     get_late_interaction_engine_index=lambda *args: None, sys=sys)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(NATIVE / relative), 'exec'), namespace)
    return namespace[name]


def test_pinned_native_header_and_single_client_idle_routing_state():
    header = native_method('vllm/entrypoints/generate/base/serving.py', 'GenerateBaseServing', '_get_data_parallel_rank')
    assert header(SimpleNamespace(headers={'X-data-parallel-rank': '3'})) == 3
    assert header(SimpleNamespace(headers={})) is None
    route = native_method('vllm/v1/engine/core_client.py', 'DPLBAsyncMPClient', 'get_core_engine_for_request')
    finish = native_method('vllm/v1/engine/core_client.py', 'DPLBAsyncMPClient', 'process_engine_outputs')
    states = []
    for targeted in (False, True):
        client = SimpleNamespace(core_engines=['e0', 'e1', 'e2', 'e3'], lb_engines=[[0, 0, 0.] for _ in range(4)],
                                 engine_inflight=Counter(), reqs_in_flight={}, client_count=1, eng_start_index=0)
        for i in range(16):
            request = SimpleNamespace(request_id=str(i), data_parallel_rank=i % 4 if targeted else None,
                                      pooling_params=None)
            assert route(client, request) == 'e' + str(i % 4)
            asyncio.run(finish(client, SimpleNamespace(finished_requests=[str(i)])))
            client.lb_engines = [[0, 0, 0.] for _ in range(4)]  # received native all-idle snapshot
        assert not any(client.engine_inflight.values()) and not client.reqs_in_flight
        states.append(client.eng_start_index)
        assert route(client, SimpleNamespace(request_id='formal', data_parallel_rank=None, pooling_params=None)) == 'e0'
    assert states == [0, 0]


@pytest.mark.parametrize('fault', ['missing', 'hash', 'equal', 'later', 'naive'])
def test_receipt_rejects_unlaunchable_binding(tmp_path, fault):
    built = datetime.now(timezone.utc) - timedelta(seconds=1)
    issued = built - timedelta(seconds=1)
    if fault == 'equal':
        issued = built
    elif fault == 'later':
        issued = built + timedelta(seconds=.1)
    elif fault == 'naive':
        issued = issued.replace(tzinfo=None)
    path = tmp_path / 'receipt.json'
    content = json.dumps(dict(issued_utc=issued.isoformat(), predicate=dict(result=True))).encode()
    if fault != 'missing':
        path.write_bytes(content)
    config = dict(promotion_receipt=dict(path=str(path), build_utc=built.isoformat(),
                                        sha256='wrong' if fault == 'hash' else hashlib.sha256(content).hexdigest()))
    with pytest.raises((ValueError, FileNotFoundError)):
        native_pd_worker.validate_promotion_receipt(config)


def test_pinned_native_multiple_api_clients_retain_different_scan_indices():
    route = native_method('vllm/v1/engine/core_client.py', 'DPLBAsyncMPClient', 'get_core_engine_for_request')
    finish = native_method('vllm/v1/engine/core_client.py', 'DPLBAsyncMPClient', 'process_engine_outputs')
    # Internal DP4 defaults to four API processes in this pinned deployment.
    # These plausible per-client counts total sixteen warmups, but do not
    # individually wrap modulo four. Idle metrics never reset the scan index.
    distributions = (5, 5, 3, 3)
    states = []
    for targeted in (False, True):
        starts = []
        for index, count in enumerate(distributions):
            client = SimpleNamespace(core_engines=['e0','e1','e2','e3'], lb_engines=[[0,0,0.] for _ in range(4)],
                engine_inflight=Counter(), reqs_in_flight={}, client_count=4, eng_start_index=index)
            for ordinal in range(count):
                request = SimpleNamespace(request_id=str(ordinal), data_parallel_rank=ordinal%4 if targeted else None,
                                          pooling_params=None)
                route(client, request)
                asyncio.run(finish(client, SimpleNamespace(finished_requests=[str(ordinal)])))
                client.lb_engines = [[0,0,0.] for _ in range(4)]
            starts.append(client.eng_start_index)
            assert route(client, SimpleNamespace(request_id='formal', data_parallel_rank=None, pooling_params=None)) == 'e'+str(starts[-1])
        states.append(starts)
    assert states == [[1,2,1,2], [0,1,2,3]]
