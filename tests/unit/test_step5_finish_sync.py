"""Focused engine-wave boundaries; CPU evidence is not native qualification."""

import hashlib
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.entities import DummyForwardBatch, Request
from frontier.config import BaseModelConfig, ReplicaConfig
from frontier.types import ClusterType
from tests.integration.test_vllm_dp_placement_runtime import _model
from tests.unit.test_step5_pd_placement import _scheduler
from tests.unit.test_step5_structural_pd import simulation_globals
from tests.unit.test_step5_native_pd_r2_observation import configuration, events, r2
from tests.integration.step5_pd import r2_verdict
from tests.unit.test_step5_native_pd_r2_verdict import observations, write_stage


def _step_lanes(role):
    cluster = _scheduler(role)
    lanes = [cluster.get_replica_scheduler(cluster._serving_replica_id, lane)
             for lane in range(cluster._replica_dp_size)]
    for lane in lanes:
        lane._dp_wave_running = True  # A native START_DP_WAVE wakes all owners.
    return lanes


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
def test_idle_step_wave_finishes_after_one_request_free_dummy_per_lane(role):
    lanes = _step_lanes(role)
    for lane in lanes:
        [dummy] = lane.on_schedule(0)
        assert isinstance(dummy, DummyForwardBatch)
        assert dummy.requests == [] and not lane.has_unfinished_requests
        assert lane.on_dummy_forward_end(.01)
        assert lane.on_schedule(.01) == []
    assert all(not lane._dp_wave_running and lane._dp_wave_steps == 0 for lane in lanes)
    assert all(not lane._waits_at_dp_sync for lane in lanes)


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
@pytest.mark.parametrize("boundary", [1, 16, 32])
def test_step_sync_waits_for_every_peer_and_keeps_an_active_wave(role, boundary):
    lanes = _step_lanes(role)
    request = Request(0, 16, 2)
    lanes[-1]._running_requests.append(request)
    for lane in lanes:
        lane._dp_wave_steps = boundary - 1
    for lane in lanes[:-1]:
        assert lane._end_iteration()
        assert lane._waits_at_dp_sync
    assert not lanes[-1]._end_iteration()
    assert lanes[-1].consume_dp_sync_release() == [0, 1, 2]
    assert all(lane._dp_wave_running and lane._dp_wave_steps == boundary for lane in lanes)
    assert all(not lane._waits_at_dp_sync for lane in lanes)
    # A request-free peer must still supply the next forward.
    [dummy] = lanes[0].on_schedule(.01)
    assert isinstance(dummy, DummyForwardBatch) and dummy.requests == []
    assert lanes[-1]._running_requests == [request]


@pytest.mark.parametrize("boundary", [1, 16, 32])
def test_step_idle_consensus_resets_each_lane_and_the_next_wave_syncs_at_one(boundary):
    lanes = _step_lanes(ClusterType.DECODE)
    for lane in lanes:
        lane._dp_wave_steps = boundary - 1
    for lane in lanes[:-1]:
        assert lane._end_iteration()
    assert not lanes[-1]._end_iteration()
    assert all(not lane._dp_wave_running and lane._dp_wave_steps == 0 for lane in lanes)
    for lane in lanes:
        lane._dp_wave_running = True
    for lane in lanes[:-1]:
        assert lane._end_iteration()
    assert not lanes[-1]._end_iteration()
    assert all(lane._dp_wave_steps == 0 for lane in lanes)


@pytest.mark.parametrize("step", [2, 15, 17, 31, 33])
def test_step_does_not_stop_between_native_finish_boundaries(step):
    lanes = _step_lanes(ClusterType.PREFILL)
    for lane in lanes:
        lane._dp_wave_steps = step - 1
        assert not lane._end_iteration()
        assert lane._dp_wave_steps == step and lane._dp_wave_running


def test_unaffected_moe_family_still_syncs_only_after_32_iterations(monkeypatch):
    model = _model(is_moe=True, num_layers=4)
    original = BaseModelConfig.create_from_name
    monkeypatch.setattr(BaseModelConfig, 'create_from_name', classmethod(
        lambda cls, name: model if name == model._model_name else original(name)))
    replica = ReplicaConfig(
        model_name=model._model_name, device='a100', network_device='a100_pairwise_nvlink',
        num_pipeline_stages=1, attn_tensor_parallel_size=1, attn_dp=2,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=2,
        total_expert_num=8, router_topk=2,
    )
    cluster = _scheduler(ClusterType.MONOLITHIC, replica=replica)
    lanes = [cluster.get_replica_scheduler(cluster._serving_replica_id, lane) for lane in range(2)]
    for lane in lanes:
        lane._dp_wave_running = True
    for step in range(1, 32):
        for lane in lanes:
            assert not lane._end_iteration() and lane._dp_wave_steps == step
    assert lanes[0]._end_iteration()
    assert not lanes[1]._end_iteration()
    assert all(not lane._dp_wave_running and lane._dp_wave_steps == 0 for lane in lanes)


@pytest.mark.parametrize('role', [ClusterType.MONOLITHIC, ClusterType.PREFILL, ClusterType.DECODE])
def test_step_dp_one_never_enters_the_dummy_wave_path(role):
    replica = ReplicaConfig(
        model_name='stepfun-ai/Step-5-Preview', device='h800', network_device='h800_dgx',
        num_pipeline_stages=1, attn_tensor_parallel_size=16, attn_dp=1,
        moe_expert_parallel_size=1 if role == ClusterType.MONOLITHIC else 16,
        moe_tensor_parallel_size=16 if role == ClusterType.MONOLITHIC else 1,
        total_expert_num=352, router_topk=8, attention_backend='STEP5_DSA',
        kv_cache_dtype='auto',
    )
    cluster = _scheduler(role, replica=replica)
    lane = cluster.get_replica_scheduler(cluster._serving_replica_id, 0)
    assert not lane._runs_dp_dummy_passes
    assert lane.on_schedule(0) == [] and lane.on_schedule(.01) == []
    assert not lane._dp_wave_running and lane._dp_wave_steps == 0


def _native_counter(observer):
    class Engine:
        def __init__(self):
            self.dp_rank = 2
            self.current_wave = self.step_counter = 0
            self.dp_sync_interval = 16
            self.engines_running = True
            self.pending_pause = self.ignore_start_dp_wave = False
            self.calls = self.collectives = self.inputs = 0
            self.fail = None

        def _has_global_unfinished_reqs(self, local_unfinished: bool):
            self.calls += 1
            if self.fail:
                raise self.fail
            self.step_counter += 1
            if self.step_counter != 1 and self.step_counter % self.dp_sync_interval != 0:
                return True
            self.collectives += 1
            return local_unfinished

        def _process_input_queue(self):
            self.inputs += 1
            return self.inputs

    module = SimpleNamespace(__name__='vllm.v1.engine.core', __file__=__file__,
                             DPEngineCoreProc=Engine)
    observer.config['source_hashes'][module.__name__] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    r2.instrument(module, observer)
    return Engine()


def test_native_counter_observer_records_actual_steps_native_returns_and_wave_reset(tmp_path):
    observer = r2.Observation(configuration(tmp_path))
    engine = _native_counter(observer)
    for step in range(1, 18):
        assert engine._has_global_unfinished_reqs(local_unfinished=False) is (step not in (1, 16))
    assert engine.calls == 17 and engine.collectives == 2
    checks = [row for row in events(observer) if row['event'] == 'native_dp_finish_check']
    assert [row['step_counter'] for row in checks] == list(range(1, 18))
    assert [row['step_before'] for row in checks] == list(range(17))
    assert [row['step_counter'] for row in checks if row['source_sync_boundary']] == [1, 16]
    assert all(row['dp_rank'] == 2 and row['dp_sync_interval'] == 16
               and row['local_unfinished'] is False and row['observer_fence'] is False
               for row in checks)
    engine.current_wave = 1
    engine.step_counter = 0
    engine.engines_running = False
    assert engine._process_input_queue() == 1
    assert engine._process_input_queue() == 2
    resets = [row for row in events(observer) if row['event'] == 'native_dp_wave_counter_zero']
    assert len(resets) == 1 and resets[0]['current_wave'] == 1
    assert resets[0]['step_counter'] == 0 and resets[0]['engines_running'] is False
    engine.engines_running = True
    assert engine._has_global_unfinished_reqs(False) is False
    assert [row for row in events(observer) if row['event'] == 'native_dp_finish_check'][-1]['step_before'] == 0


@pytest.mark.parametrize('role', ['prefill', 'decode'])
def test_emitted_native_counters_are_retained_metadata_not_request_evidence(tmp_path, role):
    observer = r2.Observation(configuration(tmp_path / 'capture', role=role))
    engine = _native_counter(observer)
    assert engine._has_global_unfinished_reqs(False) is False
    engine.engines_running = False
    engine.step_counter = 0
    assert engine._process_input_queue() == 1
    emitted = [row for row in events(observer) if row['event'].startswith('native_dp_')]
    assert {row['event'] for row in emitted} == {'native_dp_finish_check', 'native_dp_wave_counter_zero'}
    assert all(not row.get(key) for row in emitted for key in ('request_id', 'engine_id', 'tp_rank'))
    rows, trace = observations()
    stage = write_stage(tmp_path, rows, trace)
    baseline = r2_verdict.judge(stage, 'Succeeded')
    path = stage / 'evidence/nodes' / emitted[0]['host'] / 'pd_state/counters.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in emitted))
    instrumentation, content = r2_verdict.Check(), r2_verdict.Check()
    records, errors = r2_verdict.load(stage, instrumentation, {}, {}, content)
    assert not instrumentation.missing and not errors
    assert len(records) == len(rows) and not any(row['event'].startswith('native_dp_') for row in records)
    retained = instrumentation.details['diagnostic_metadata']
    assert retained == [{'location': f'{path}:{i}', 'raw': row} for i, row in enumerate(emitted, 1)]
    verdict = r2_verdict.judge(stage, 'Succeeded')
    assert verdict['status'] == baseline['status'] == 'PASS'
    assert verdict['transfers'] == baseline['transfers']
    assert verdict['ticket08_accepted'] is False and verdict['clean_timing_eligible'] is False
    (tmp_path / 'metadata_only').mkdir()
    metadata_stage = write_stage(tmp_path / 'metadata_only', emitted, trace)
    verdict = r2_verdict.judge(metadata_stage, 'Succeeded')
    assert verdict['status'] == 'UNVERIFIED' and not verdict['scope_pass']
    assert verdict['transfers'] == [] and verdict['ticket08_accepted'] is False
    assert verdict['checks']['instrumentation']['diagnostic_metadata']
    for rejected in ('unknown_counter', 'observer_error'):
        path.write_text(json.dumps({'event': rejected, 'message': 'capture failed'}) + '\n')
        verdict = r2_verdict.judge(stage, 'Succeeded')
        assert verdict['checks']['instrumentation']['status'] == 'UNVERIFIED'
        assert any(('Unknown event' if rejected == 'unknown_counter' else 'Observer error') in message
                   for message in verdict['checks']['instrumentation']['missing'])


@pytest.mark.parametrize('failure', ['before', 'after', 'native', 'reset', 'budget'])
def test_counter_capture_failure_cannot_replace_a_native_call_return_or_exception(tmp_path, failure):
    observer = r2.Observation(configuration(tmp_path))
    engine = _native_counter(observer)
    error = LookupError('native counter failure')
    if failure == 'before':
        engine.dp_rank = object()  # Not a native scalar; capture alone must fail.
    elif failure == 'after':
        original = observer.emit

        def emit(event, *args, **kwargs):
            if event == 'native_dp_finish_check':
                raise OSError('counter capture write failed')
            return original(event, *args, **kwargs)

        observer.emit = emit
    elif failure == 'native':
        engine.fail = error
    elif failure == 'budget':
        observer.config['max_events'] = observer.events + 1
    else:
        engine.engines_running = False
        engine.dp_rank = object()
        assert engine._process_input_queue() == 1 and engine.inputs == 1
    if failure == 'native':
        with pytest.raises(LookupError) as caught:
            engine._has_global_unfinished_reqs(False)
        assert caught.value is error and engine.calls == 1
        assert not any(row['event'] == 'native_dp_finish_check' for row in events(observer))
    else:
        assert engine._has_global_unfinished_reqs(False) is False
        assert engine.calls == 1 and engine.collectives == 1 and observer.disabled
        assert any(row['event'] in ('observer_error', 'observer_budget_exceeded')
                   for row in events(observer))


def test_counter_observer_accepts_the_pinned_native_method_without_importing_vllm(tmp_path):
    root = os.environ.get('STEP5_NATIVE_SOURCE_ROOT')
    if not root:
        pytest.skip('Requires the locked native source root')
    source = Path(root) / 'vllm/v1/engine/core.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        '5a0344942c5af99cd85eb68d9eded7f856df40ff0986bda8de6d2ff22f6d357e')
    engine_class = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.ClassDef) and node.name == 'DPEngineCoreProc')
    method = next(node for node in engine_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_has_global_unfinished_reqs')
    calls = []

    def sync(group, *, has_unfinished, pending_pause):
        calls.append((group, has_unfinished, pending_pause))
        return has_unfinished, pending_pause

    namespace = {'ParallelConfig': SimpleNamespace(sync_dp_state=sync),
                 'logger': SimpleNamespace(debug=lambda *args: None)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    observer = r2.Observation(configuration(tmp_path))
    engine = _native_counter(observer)
    type(engine)._has_global_unfinished_reqs = namespace['_has_global_unfinished_reqs']
    module = SimpleNamespace(__name__='vllm.v1.engine.core', __file__=str(source),
                             DPEngineCoreProc=type(engine))
    observer.config['source_hashes'][module.__name__] = hashlib.sha256(source.read_bytes()).hexdigest()
    r2.instrument(module, observer)
    engine.dp_group = object()
    for step in range(1, 18):
        assert engine._has_global_unfinished_reqs(False) is (step not in (1, 16))
    assert calls == [(engine.dp_group, False, False)] * 2
    engine.step_counter = 31
    engine.pending_pause = True
    assert engine._has_global_unfinished_reqs(False) is False
    assert engine.ignore_start_dp_wave is True and engine.pending_pause is False
    last = [row for row in events(observer) if row['event'] == 'native_dp_finish_check'][-1]
    assert last['step_counter'] == 32 and last['source_sync_boundary'] is True
    assert last['ignore_start_dp_wave'] is True and last['pending_pause'] is False
