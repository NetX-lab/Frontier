"""Exercise independent-reference contents, HTTP routing and soft diagnostic failures."""

import asyncio
import importlib
import json
from types import SimpleNamespace as NS

import pytest

from tests.unit.test_step5_canonical_g3 import stage_capture
from tests.unit.test_step5_native_pd_r2_observation import PD, Owner, configuration, events, r2


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.syspath_prepend(str(PD))
    monkeypatch.syspath_prepend(str(PD.parent/'step5_environment'))
    return importlib.import_module('r2_reference')


@pytest.mark.parametrize('mutation', ['weight', 'buffer', 'attribute', 'quant_tensor', 'scalar', 'alias'])
def test_complete_resident_identity_detects_each_state_and_alias_mutation(tmp_path, reference, mutation):
    import torch

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(80, dtype=torch.float32).reshape(8, 10).t())
    model.alias = model.weight
    model.register_buffer('buffer', torch.arange(7, dtype=torch.float32), persistent=False)
    model.scale = torch.tensor([1.0])
    model.quant_method = NS(scale=torch.tensor([2.0]), clamp_limit=7.0)
    model._k_scale_float = 1.0
    worker = Owner()
    worker.vllm_config = NS(parallel_config=NS(data_parallel_rank=0),
                            compilation_config=NS(static_forward_context={}))
    worker.model_runner = NS(model=model)
    observer = r2.Observation(configuration(tmp_path, resident_identity={
        'chunk_bytes': 64, 'max_bytes': 4096, 'max_seconds': 60}))
    reference.resident_identity(observer, worker, None, 'startup', 0)
    first = events(observer)[-1]
    with torch.no_grad():
        if mutation == 'weight':
            model.weight[0, 0] += 1
        elif mutation == 'buffer':
            model.buffer[0] += 1
        elif mutation == 'attribute':
            model.scale += 1
        elif mutation == 'quant_tensor':
            model.quant_method.scale += 1
        elif mutation == 'scalar':
            model._k_scale_float = 2.0
        else:
            model.alias = torch.nn.Parameter(model.weight.clone())
    reference.resident_identity(observer, worker, None, 'reference', 0)
    second = events(observer)[-1]
    assert first['complete'] and first['peak_chunk_bytes'] <= 64
    assert {row['kind'] for row in first['tensors']} >= {'parameter', 'buffer', 'tensor_attribute', 'quant_tensor'}
    assert reference.canonical_identity(first) != reference.canonical_identity(second)


def test_logical_checkpoint_excludes_padding_and_physical_page_identity(tmp_path, monkeypatch, reference):
    import torch

    observer = r2.Observation(configuration(tmp_path, block_size=16))
    monkeypatch.setattr(observer, 'synchronize', lambda: None)
    raw = torch.arange(4*2*16*4, dtype=torch.float32).reshape(4, 2, 16, 4)
    mean = torch.arange(4*2*2*4, dtype=torch.float32).reshape(4, 2, 2, 4)
    connector = Owner()
    connector.tp_rank, connector.engine_id = 0, 'D'
    connector.vllm_config = NS(parallel_config=NS(data_parallel_rank=0),
        compilation_config=NS(static_forward_context={'mean': NS(_cache=NS(mean=mean))}))
    connector.kv_cache_config = NS(transfer_group_ids=[0])
    connector.device_kv_caches = {'raw': raw}
    observer.connector_states[connector] = {
        'group_specs': [{'sliding_window': None}],
        'registrations': [dict(group_id=0, layer='layer0', registration_name=name, component=component)
                          for name, component in (('raw', 'raw_kv'), ('mean', 'summary_mean'))]}
    request = {'computed': 18, 'blocks': [[1, 2]]}
    reference.checkpoint(observer, connector, 'case', request, 'first_local_tail_complete')
    first = events(observer)[-1]
    raw[2, :, 2:] += 1000
    raw[3].copy_(raw[2])
    mean[3].copy_(mean[2])
    request['blocks'] = [[1, 3]]
    reference.checkpoint(observer, connector, 'case', request, 'first_local_tail_complete')
    second = events(observer)[-1]
    controls = importlib.import_module('r2_controls')
    assert controls.logical_state(first) == controls.logical_state(second)
    assert all('raw_hex' not in row['sample'] for row in second['logical_components'])
    raw[3, 0, 0, 0] += 1
    reference.checkpoint(observer, connector, 'case', request, 'first_local_tail_complete')
    assert controls.logical_state(second) != controls.logical_state(events(observer)[-1])


@pytest.mark.parametrize('error_type', [RuntimeError, IndexError])
def test_reference_analysis_fault_cannot_escape_soft_check(tmp_path, monkeypatch, error_type):
    from tests.integration.step5_pd import r2_controls, r2_verdict

    _, _, stage = stage_capture(tmp_path)
    def broken(*args):
        raise error_type('injected reference diagnostic error')
    monkeypatch.setattr(r2_controls, 'reference', broken)
    result = r2_verdict.judge(stage, 'Succeeded')
    check = result['checks']['independent_reference']
    assert check['status'] == 'UNVERIFIED'
    assert any('injected reference diagnostic error' in value for value in check['missing'])


@pytest.mark.parametrize('reference_fails', [False, True])
def test_actual_reference_driver_uses_direct_d_and_continues_after_auxiliary_failure(
        tmp_path, monkeypatch, reference_fails):
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from tests.integration.step5_pd import r2_replay

    _, config, stage = stage_capture(tmp_path)
    definition = json.loads((stage/'r2_run.json').read_text())
    names = ['reference6144', 'canonical6144']
    definition['waves'] = [next(wave for wave in definition['waves'] if wave['name'] == name) for name in names]
    (stage/'r2_run.json').write_text(json.dumps(definition))
    monkeypatch.setattr(r2_replay.canonical_replay, 'local_model_config', lambda *args: {'vocab_size': 32000})
    async def idle(*args):
        return {'verified': True, 'observations': [{'pages': {'prefill': '', 'decode': ''}}]}
    monkeypatch.setattr(r2_replay.canonical_replay, 'observe_idle', idle)
    bodies = {'prefill': [], 'decode': []}
    async def run():
        async def prefill(request):
            body = await request.json()
            bodies['prefill'].append(body)
            return web.json_response({'kv_transfer_params': {
                'transfer_mode': 'pull', 'do_remote_prefill': True, 'do_remote_decode': False,
                'remote_engine_id': 'P', 'remote_request_id': 'p', 'remote_host': '127.0.0.1',
                'remote_port': 5600, 'remote_block_ids': [[1]], 'remote_num_tokens': 6128}})
        async def decode(request):
            body = await request.json()
            bodies['decode'].append(body)
            if reference_fails and 'kv_transfer_params' not in body:
                return web.json_response({'error': 'injected reference HTTP failure'}, status=500)
            payload = {'choices': [{'index': 0, 'prompt_token_ids': body['prompt'],
                'token_ids': [2]*body['max_tokens'], 'finish_reason': 'length'}],
                'usage': {'prompt_tokens': len(body['prompt']), 'completion_tokens': body['max_tokens']}}
            return web.Response(text='data: '+json.dumps(payload)+'\n\ndata: [DONE]\n\n',
                                content_type='text/event-stream')
        p, d = web.Application(), web.Application()
        p.router.add_post('/v1/completions', prefill)
        d.router.add_post('/v1/completions', decode)
        async with TestServer(p) as ps, TestServer(d) as ds:
            async with TestServer(web.Application()) as reserve:
                port = reserve.port
            args = NS(trace_dir=stage/'trace', model_dir=tmp_path, output_dir=tmp_path/'replay',
                prefill_url=str(ps.make_url('/')).rstrip('/'), decode_url=str(ds.make_url('/')).rstrip('/'),
                proxy_port=port, origin_lead_s=0)
            assert await r2_replay.run(args) == 0
    asyncio.run(run())
    assert len(bodies['prefill']) == 1 and len(bodies['decode']) == 2
    reference_body, pd_body = bodies['decode']
    assert 'kv_transfer_params' not in reference_body and pd_body['kv_transfer_params']['remote_num_tokens'] == 6128
    assert reference_body['prompt'] == pd_body['prompt'] and len(reference_body['prompt']) == 6144
    summary = json.loads((tmp_path/'replay/r2_replay_summary.json').read_text())
    assert summary['waves']['reference6144']['dispatch'] == 'direct_native_D_no_proxy_or_transfer_metadata'
    assert 'error' not in summary['waves']['canonical6144']
    assert bool(summary['errors']) is reference_fails
    assert summary['ready_for_formal'] is False
