"""Exercise independent-reference contents, HTTP routing and soft diagnostic failures."""

import asyncio
import copy
import hashlib
import importlib
import json
from pathlib import Path
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
    assert all('raw_hex' in row['sample'] for row in second['logical_components'])
    raw[3, 0, 0, 0] += 1
    reference.checkpoint(observer, connector, 'case', request, 'first_local_tail_complete')
    assert controls.logical_state(second) != controls.logical_state(events(observer)[-1])


def cache_observation(tmp_path, monkeypatch, reference, prompt, rank=0):
    """Use the real collector with CPU tensors in the locked raw/mean layouts."""
    import torch

    pages = (prompt+16-1)//16+2
    raw = torch.arange(pages*16*384, dtype=torch.float32).reshape(pages, 1, 16, 384).to(torch.bfloat16)
    mean = torch.arange(pages*2*256, dtype=torch.int32).reshape(pages, 2, 256).to(torch.uint8)
    cache = NS(mean=mean, numerator=torch.zeros(pages, 256), denominator=torch.ones(pages, 256),
               maximum=torch.zeros(pages, 1), last_slot=torch.full((pages,), -1, dtype=torch.int64),
               request_journal_width=0)
    connector = Owner()
    connector.tp_rank, connector.engine_id = rank, 'D'
    connector._physical_blocks_per_logical_kv_block = 1
    connector.vllm_config = NS(parallel_config=NS(data_parallel_rank=0),
        compilation_config=NS(static_forward_context={'mean': NS(_cache=cache)}))
    connector.kv_cache_config = NS(transfer_group_ids=[0, 1])
    connector.device_kv_caches = {'raw': raw, 'swa': raw, 'mean': mean}
    settings = configuration(tmp_path, selected_client_request_ids=['actual', 'reference'],
        content_client_request_ids=['actual', 'reference'], local_reference_client_request_ids=['reference'],
        resident_identity={'chunk_bytes': 64, 'max_bytes': 4096, 'max_seconds': 60},
        request_specs={client: {'num_decode_tokens': 16} for client in ('actual', 'reference')},
        reference_helper_sha256=hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest())
    observer = r2.Observation(settings)
    monkeypatch.setattr(observer, 'reference_module', lambda: reference)
    registrations = [dict(group_id=group, layer=layer, registration_name=name, component=kind)
                     for group, layer, name, kind in ((0, 'full', 'raw', 'raw_kv'),
                         (1, 'swa', 'swa', 'raw_kv'), (0, 'full', 'mean', 'summary_mean'))]
    observer.connector_states[connector] = {
        'group_specs': [{'group_id': 0, 'sliding_window': None}, {'group_id': 1, 'sliding_window': 32}],
        'registrations': registrations, 'tail_checkpoints': {}}
    observer.connectors.append(r2.weakref.ref(connector))
    worker = Owner()
    worker.vllm_config = connector.vllm_config
    worker.model_runner = NS(model=torch.nn.Linear(1, 1))
    observer.runner_reported.add(worker)
    blocks = [list(range(1, pages)), list(range(1, pages))]
    return observer, connector, worker, blocks


def verify_checkpoint(row, connector):
    import torch
    from tests.integration.step5_pd import r2_controls, r2_verdict

    computed = row['num_computed_tokens']
    last = (computed-1)//16
    expected_bytes, kept_bytes = 0, 0
    for item in row['logical_components']:
        name = {'raw_kv': 'swa' if item['layer'] == 'swa' else 'raw', 'summary_mean': 'mean'}[item['component']]
        tensor = connector.device_kv_caches[name]
        if item['component'] == 'raw_kv':
            tensor = tensor.transpose(1, 2)
        lo, hi = item['valid_units']
        raw = tensor[item['physical_page'], lo:hi].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        sample = item['sample']
        assert sample['sha256'] == hashlib.sha256(raw).hexdigest()
        assert sample['shape'] == [hi-lo]+item['unit_shape']
        keep = row['boundary'] == 'first_local_tail_complete' and last-1 <= item['logical_page'] <= last
        assert ('raw_hex' in sample) is keep
        if keep:
            assert bytes.fromhex(sample['raw_hex']) == raw
            kept_bytes += len(raw)
        expected_bytes += len(raw)
    registration = dict(group_specs=[{'group_id': 0, 'sliding_window': None},
                                     {'group_id': 1, 'sliding_window': 32}],
        expected_layers=[{'group_id': 0, 'layer': 'full', 'components': ['raw_kv', 'summary_mean']},
                         {'group_id': 1, 'layer': 'swa', 'components': ['raw_kv']}])
    check = r2_verdict.Check()
    r2_controls.checkpoint_census(row, registration, 16, 8, check)
    assert check.result()['status'] == 'PASS', check.result()
    hashes_only = copy.deepcopy(row)
    for item in hashes_only['logical_components']:
        item['sample'].pop('raw_hex', None)
    assert r2_controls.logical_state(row) == r2_controls.logical_state(hashes_only)
    return expected_bytes, kept_bytes


@pytest.mark.parametrize('prompt', [6143, 6144, 6145])
@pytest.mark.parametrize('rank', range(8))
@pytest.mark.parametrize('client', ['actual', 'reference'])
def test_first_tail_values_use_actual_and_full_local_worker_callers(
        tmp_path, monkeypatch, reference, prompt, rank, client):
    import torch
    from unittest.mock import Mock

    observer, connector, worker, blocks = cache_observation(tmp_path, monkeypatch, reference, prompt, rank)
    sync = Mock()
    monkeypatch.setattr(observer, 'synchronize', sync)
    sample = Mock(wraps=observer.tensor_sample)
    reserve = Mock(wraps=observer.reserve_capture)
    cpu = Mock(side_effect=torch.Tensor.cpu)
    monkeypatch.setattr(observer, 'tensor_sample', sample)
    monkeypatch.setattr(observer, 'reserve_capture', reserve)
    monkeypatch.setattr(torch.Tensor, 'cpu', lambda tensor: cpu(tensor))
    external = 16*((prompt-16)//16) if client == 'actual' else 0
    output = NS(finished_req_ids=set(), scheduled_cached_reqs=NS(req_ids=[]),
        scheduled_new_reqs=[NS(req_id=client, block_ids=blocks,
            num_computed_tokens=external, prompt_token_ids=[1]*prompt)],
        num_scheduled_tokens={client: prompt-external})
    scheduled = observer.worker_before(worker, {'scheduler_output': output})
    identity_samples = sample.call_count
    identity_copies = cpu.call_count
    identity_reserves = reserve.call_count
    sync.reset_mock()
    observer.worker_after(worker, None, {}, scheduled)
    first = [row for row in events(observer) if row['event'] == 'state_checkpoint'][-1]
    expected_bytes, kept_bytes = verify_checkpoint(first, connector)
    assert first['tp_rank'] == rank and first['request_id'] == client
    assert first['num_computed_tokens'] == prompt
    assert {item['component'] for item in first['logical_components'] if 'raw_hex' in item['sample']} == {'raw_kv', 'summary_mean'}
    assert kept_bytes < expected_bytes
    # Existing local_tail + checkpoint have two synchronizations. Each sample
    # has exactly one reservation and one host copy, including mean wire rows.
    assert sync.call_count == 2
    assert sample.call_count-identity_samples == len(first['logical_components'])+10
    assert reserve.call_count-identity_reserves == sample.call_count-identity_samples
    assert sum(call.args[1] for call in reserve.call_args_list[identity_reserves:]) == expected_bytes+5144
    assert cpu.call_count-identity_copies == sample.call_count-identity_samples
    assert {row['event'] for row in events(observer) if row['event'] == 'reference_start'} == (
        {'reference_start'} if client == 'reference' else set())
    tail = [row for row in events(observer) if row['event'] == 'tail_state'][-1]
    assert tail['page_selection'] == 'computed_token_page_and_preceding_page'
    assert tail['continuity_verdict'] == 'UNVERIFIED_REQUIRES_REFERENCE_REVIEW'
    assert len(tail['pages']) == 2
    for page in tail['pages']:
        payload = bytes.fromhex(page['summary_mean_wire_row']['raw_hex'])
        assert len(payload) == 512
        assert hashlib.sha256(payload).hexdigest() == page['summary_mean_wire_row']['sha256']
        assert page['summary_mean_wire_row']['dtype'] == 'torch.uint8'
    # Repeating the same completed forward does not add a first-tail capture.
    before = len([row for row in events(observer) if row['event'] == 'state_checkpoint'])
    observer.worker_after(worker, None, {}, scheduled)
    assert len([row for row in events(observer) if row['event'] == 'state_checkpoint']) == before
    for computed in (prompt+9, prompt+15):
        output.scheduled_new_reqs = []
        previous = observer.worker_states[worker][client]['computed']
        output.scheduled_cached_reqs = NS(req_ids=[client], new_block_ids=[None],
            resumed_req_ids=set(), num_computed_tokens=[previous])
        output.num_scheduled_tokens = {client: computed-previous}
        scheduled = observer.worker_before(worker, {'scheduler_output': output})
        observer.worker_after(worker, None, {}, scheduled)
    checkpoints = [row for row in events(observer) if row['event'] == 'state_checkpoint']
    assert [row['boundary'] for row in checkpoints] == [
        'first_local_tail_complete', 'first_decode_region_crossing', 'final_decode_input_complete']
    for row in checkpoints:
        verify_checkpoint(row, connector)
    tails = [row for row in events(observer) if row['event'] == 'tail_state']
    assert all('raw_hex' not in page['summary_mean_wire_row'] for row in tails[1:] for page in row['pages'])


def test_retained_values_do_not_change_independent_verdict_or_inactive_tail_comparison(
        tmp_path, monkeypatch, reference):
    from tests.integration.step5_pd import r2_controls, r2_verdict

    observer, connector, _, blocks = cache_observation(tmp_path/'observer', monkeypatch, reference, 80)
    monkeypatch.setattr(observer, 'synchronize', lambda: None)
    boundaries = [('first_local_tail_complete', 65), ('first_decode_region_crossing', 73),
                  ('final_decode_input_complete', 80)]
    for boundary, computed in boundaries:
        observer.local_tail(connector, 'actual', {'computed': computed, 'blocks': blocks}, boundary)
        reference.checkpoint(observer, connector, 'actual', {'computed': computed, 'blocks': blocks}, boundary)
    captures = [row for row in events(observer) if row['event'] in ('state_checkpoint', 'tail_state')]
    records = []
    for rank in range(8):
        common = dict(host='worker', dp_rank=0, tp_rank=rank)
        records.append(dict(event='state_registration', **common,
            group_specs=[{'group_id': 0, 'sliding_window': None}, {'group_id': 1, 'sliding_window': 32}],
            expected_layers=[{'group_id': 0, 'layer': 'full', 'components': ['raw_kv', 'summary_mean']},
                             {'group_id': 1, 'layer': 'swa', 'components': ['raw_kv']}]))
        records.extend([
            dict(event='reference_start', client_request_id='reference', **common,
                 initial_computed_tokens=0, actual_prompt_tokens=65, actual_prompt_sha256='a'*64),
            dict(event='resident_identity', client_request_id='reference', phase='reference:reference', **common),
            dict(event='target_compute', client_request_id='reference', num_computed_tokens=0,
                 num_scheduled_tokens=80, monotonic_ns=1, **common)])
        for client in ('actual', 'reference'):
            for original in captures:
                row = copy.deepcopy(original)
                row.update(common, client_request_id=client)
                # These stale predecessors differ, but last_slot=-1 makes them
                # inactive. New wire bytes must not trigger positional comparison.
                if row['event'] == 'tail_state' and client == 'actual':
                    assert row['pages'][0]['last_slot'] == -1
                    row['pages'][0]['exact_tensors']['numerator']['sha256'] = 'b'*64
                records.append(row)
    records.append(dict(event='block_allocate', client_request_id='reference',
        kv_transfer_params_present=False, num_external_tokens=0, num_local_cache_tokens=0))
    sources = {key: {'total': value} for key, value in
               (('local_compute', 65), ('local_cache_hit', 0), ('external_kv_transfer', 0))}
    replay = tmp_path/'evidence/replay'
    replay.mkdir(parents=True)
    (replay/'r2_replay_summary.json').write_text(json.dumps({'waves': {'reference': {
        'before': {'verified': True}, 'after': {'verified': True},
        'dispatch': 'direct_native_D_no_proxy_or_transfer_metadata',
        'transfer_delta': {'decode': {'prompt_tokens_by_source': sources},
                           'prefill': {'prompt_tokens_by_source': {'local_compute': {'total': 0}}}}}}}))
    config = {'engine': {'block_size': 16}, 'pd_state_observation': {
        'local_reference_client_request_ids': ['reference'], 'content_client_request_ids': ['actual', 'reference'],
        'region_size': 8}}
    expected = {client: {'wave': client, 'num_prefill_tokens': 65, 'num_decode_tokens': 16}
                for client in ('actual', 'reference')}
    clients = {(client, client): [{'prompt_sha256': 'a'*64, 'token_id_deltas': [{'token_ids': [2]*16}]}]
               for client in ('actual', 'reference')}

    def verdict():
        check = r2_verdict.Check()
        r2_controls.reference(tmp_path, records, config, expected, clients, check)
        return check.result()

    assert verdict()['status'] == 'PASS'
    checkpoint = next(row for row in records if row.get('client_request_id') == 'actual'
        and row['event'] == 'state_checkpoint' and row['boundary'] == 'first_local_tail_complete')
    sample = next(item['sample'] for item in checkpoint['logical_components']
                  if item['component'] == 'raw_kv' and 'raw_hex' in item['sample'])
    payload = bytearray.fromhex(sample['raw_hex'])
    payload[0] ^= 1
    sample.update(raw_hex=payload.hex(), sha256=hashlib.sha256(payload).hexdigest())
    result = verdict()
    assert result['status'] == 'UNVERIFIED' and not result['failures']
    assert len(result['missing']) == 1 and 'no cross-batch error bound' in result['missing'][0]
    assert result['compared_state_pairs'] == 24
    assert result['numerical_policy'] == 'Exact logical bytes/output; mismatches UNVERIFIED, never presumed batch drift'
    for row in records:
        if row['event'] == 'state_checkpoint':
            for item in row['logical_components']:
                item['sample'].pop('raw_hex', None)
    assert verdict() == result


@pytest.mark.parametrize('computed', [1, 16, 17, 48, 63, 64, 65])
@pytest.mark.parametrize('boundary', ['first_local_tail_complete', 'first_decode_region_crossing', 'final_decode_input_complete'])
def test_checkpoint_value_selection_is_two_logical_pages_at_first_tail_only(
        tmp_path, monkeypatch, reference, computed, boundary):
    observer, connector, _, blocks = cache_observation(tmp_path, monkeypatch, reference, computed)
    monkeypatch.setattr(observer, 'synchronize', lambda: None)
    reference.checkpoint(observer, connector, 'actual', {'computed': computed, 'blocks': blocks}, boundary)
    checkpoint = events(observer)[-1]
    verify_checkpoint(checkpoint, connector)
    observer.local_tail(connector, 'actual', {'computed': computed, 'blocks': blocks}, boundary)
    tail = events(observer)[-1]
    assert len(tail['pages']) == min(2, (computed+15)//16)
    for page in tail['pages']:
        assert ('raw_hex' in page['summary_mean_wire_row']) is (boundary == 'first_local_tail_complete')


@pytest.mark.parametrize('cap', ['request', 'record', 'time'])
@pytest.mark.parametrize('client', ['actual', 'reference'])
def test_value_capture_cap_failures_are_soft_after_native_return(tmp_path, monkeypatch, reference, cap, client):
    observer, connector, worker, blocks = cache_observation(tmp_path, monkeypatch, reference, 65)
    monkeypatch.setattr(observer, 'synchronize', lambda: None)
    observer.worker_states[worker] = {client: {'computed': 65, 'prompt_tokens': 65, 'blocks': blocks}}
    observer.frozen_helpers()  # Pin provenance outside the injected cap.
    if cap == 'request':
        observer.config['max_capture_bytes_per_request'] = 1
    elif cap == 'record':
        observer.config['max_record_bytes'] = observer.written+4096
    else:
        observer.config['max_capture_seconds_per_request'] = 1e-9
    calls = []

    class Native:
        def run(self):
            calls.append('native result')
            return 41

    r2.wrap(Native, 'run', observer,
        after=lambda *args: observer.worker_after(worker, None, {}, [client]))
    assert Native().run() == Native().run() == 41
    assert calls == ['native result', 'native result']
    assert observer.disabled
    assert events(observer)[-1]['event'] == 'observer_budget_exceeded'


def test_existing_stage_builder_pins_bounded_value_collector_without_policy_changes(tmp_path):
    from tests.integration.step5_pd import verify_r2_stage

    run, config, stage = stage_capture(tmp_path)
    for name in ('r2_reference.py', 'r2_observation.py'):
        assert (stage/name).read_bytes() == (PD/name).read_bytes()
    assert config['pd_state_observation']['reference_helper_sha256'] == hashlib.sha256(
        (stage/'r2_reference.py').read_bytes()).hexdigest()
    assert run['pd_state_observation']['max_record_bytes'] == 1073741824
    assert run['pd_state_observation']['max_capture_bytes_per_request'] == 1073741824
    assert run['pd_state_observation']['max_capture_seconds_per_request'] == 600
    assert (config['engine']['max_model_len'], config['engine']['max_num_batched_tokens'],
            config['engine']['max_num_seqs']) == (7168, 8192, 128)
    assert len(config['pd_state_observation']['content_client_request_ids']) == 8
    receipt = verify_r2_stage.verify(stage)
    assert receipt['g3_qualified'] is receipt['clean_timing_eligible'] is False
    assert config['clean_timing_eligible'] is False


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
            args = NS(trace_dir=stage/'trace', engine_config=stage/'engine.json',
                model_dir=tmp_path, output_dir=tmp_path/'replay',
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
