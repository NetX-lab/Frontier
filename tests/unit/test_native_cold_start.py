"""Captured native responses at the real replay seam; no GPU qualification."""

import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from tests.comparison.calibration import pd_replay
from tests.unit.test_native_pd_canonical_http import FixtureAsyncio, FixtureClock
from tests.unit.test_native_streaming_replay import PARAMS, event, token_event, usage_event
from tests.unit.test_native_groundtruth_check import archived_harness, native_ready_exports, native_case


CAPTURE = Path('/data/ycfeng/tmp/glm53_dsv4/v4_p0_native03_snapshot_20261021a')
ACTUAL = Path('/data/ycfeng/tmp/glm53_dsv4/v4_p0_native03_results')
TRACE = Path(__file__).resolve().parents[2] / 'data/processed_traces/canonical_fixed_6144_1024_req128_qps1'


def service_files(root, urls):
    """Explicit CPU process receipts, following the production worker layout."""
    manifest = dict(run_generation='CPU-only-cold-start', mode='clean',
                    written_before_launch_utc='2026-10-21T00:00:00+00:00', resources={'replica': 2})
    (root / 'run_manifest.json').write_text(json.dumps(manifest))
    digest = hashlib.sha256((root / 'run_manifest.json').read_bytes()).hexdigest()
    for role in pd_replay.ROLES:
        node = root / 'nodes' / role
        node.mkdir(parents=True)
        launched = dict(server_pid=101 if role == 'prefill' else 102,
                        server_started_at_utc='2026-10-21T00:00:01+00:00',
                        run_manifest_sha256=digest, run_generation=manifest['run_generation'])
        plan = dict(role=role, lane=0, engine={'data_parallel_size': 2, 'data_parallel_size_local': 2},
                    urls=urls)
        (node / 'result.json').write_text(json.dumps(launched | {'plan': plan}))
        (node / 'server_ready.json').write_text(json.dumps(launched | dict(
            role=role, url=urls[role], ready_at_utc='2026-10-21T00:00:02+00:00', health_status=200)))
    pd_replay.dedicated_services(root, urls, 2)
    return root


@pytest.mark.parametrize('fault', [None, 'missing_lane', 'missing_reason', 'wrong_lane', 'nonzero', 'existing_client',
                                  'stale_completion', 'failed_warmup', 'missing_baseline'])
def test_captured_first_boundary_replays_all_canonical_phases(tmp_path, monkeypatch, fault):
    if not CAPTURE.is_dir():
        pytest.skip('Retained native response is not available')
    workload = pd_replay.canonical_workload(TRACE)
    captures = {role: (CAPTURE / f'{role}_metrics.txt').read_text() for role in pd_replay.ROLES}
    clock = FixtureClock()
    replay_module = __import__('sys').modules[pd_replay.replay.__module__]
    for module in (pd_replay, replay_module):
        monkeypatch.setattr(module, 'time', clock)
        monkeypatch.setattr(module, 'asyncio', FixtureAsyncio(clock))
    submitted = {role: 0 for role in pd_replay.ROLES}
    if fault == 'missing_lane':
        captures['decode'] = '\n'.join(line for line in captures['decode'].splitlines()
            if not (line.startswith('vllm:request_success_total') and 'engine="1"' in line))
    if fault == 'wrong_lane':
        captures['decode'] = captures['decode'].replace('engine="1"', 'engine="2"')
    if fault == 'missing_reason':
        captures['decode'] = '\n'.join(line for line in captures['decode'].splitlines()
            if not (line.startswith('vllm:request_success_total') and 'finished_reason="abort"' in line))
    if fault == 'nonzero':
        captures['decode'] = captures['decode'].replace('finished_reason="stop",model_name="dp_pp_case"} 0.0',
                                                        'finished_reason="stop",model_name="dp_pp_case"} 1.0')

    class Response:
        def __init__(self, text):
            self.body = text
            self.status = 200
            self.content = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        async def text(self):
            return self.body

        async def iter_any(self):
            yield self.body

    urls = dict(prefill='http://192.0.2.1:8100', decode='http://192.0.2.2:8200')
    proof = service_files(tmp_path, urls)

    class Session(Response):
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url):
            role = next(role for role in urls if url.startswith(urls[role]))
            text = captures[role]
            if submitted[role]:
                if fault != 'stale_completion':
                    text = text.replace('finished_reason="length",model_name="dp_pp_case"} 0.0',
                                        f'finished_reason="length",model_name="dp_pp_case"}} {submitted[role] / 2}')
                text += '\n' + '\n'.join(f'vllm:{name}{{engine="{lane}",model_name="dp_pp_case"}} 0'
                    for lane in range(2) for name in ('num_requests_running', 'num_requests_waiting',
                                                     'kv_cache_usage_perc'))
            return Response('missing gauges and counters' if fault == 'missing_baseline' and not submitted[role] else text)

        def post(self, url, *, data, **kwargs):
            # Exercise the actual client request/stream path at the HTTP seam.
            body = json.loads(data)
            assert body['max_tokens'] == 1024 and len(body['prompt']) == 6144
            for role in submitted:
                submitted[role] += 1
            if fault == 'failed_warmup' and submitted['decode'] == 145:
                response = Response('retained warmup HTTP502 pattern')
                response.status = 502
                return response
            return Response(event(token_event([1] * 1024, 'length'))
                            + event(usage_event(6144, 1024)) + event('[DONE]'))

    import aiohttp
    monkeypatch.setattr(aiohttp, 'ClientSession', Session)
    existing = [{'dispatch_monotonic': 1.0}] if fault == 'existing_client' else []
    result, records = asyncio.run(pd_replay.replay_repetitions(
        workload, 8000, 154880, 0., urls, 2, .01, .001, fresh_services=proof, records=existing))
    assert result['client_workload_verified'] is (fault is None)
    first, *later = result['idle_boundaries']
    assert len(later) == 6 and len(records) == 432 + (fault == 'existing_client')
    if fault in (None, 'failed_warmup'):
        assert first['proof'] == 'dedicated_services_ready_client_zero_native_counters_zero'
        assert first['client_dispatched_requests'] == 0
        assert all(first['observations'][-1]['roles'][role]['lanes'] == {'0': {}, '1': {}}
                   for role in pd_replay.ROLES)
        assert all(boundary['verified'] for boundary in later)
        assert [boundary['completed_after']['decode'] for boundary in later] == [16, 144, 160, 288, 304, 432]
    else:
        assert not any(boundary['verified'] for boundary in later)
    if fault == 'stale_completion':
        assert first['verified']
        assert [boundary['completed_after']['decode'] for boundary in later] == [16, 144, 160, 288, 304, 432]
        assert all(boundary['observations'][-1]['roles']['decode']['idle'] for boundary in later)
        assert all(boundary['observations'][-1]['roles']['decode']['completed_requests'] == 0 for boundary in later)
    if fault == 'failed_warmup':
        assert sum(row['status'] == 'completed' for row in records) == 431
        assert not result['repetitions'][1]['client_workload_verified']
        assert result['repetitions'][2]['client_workload_verified']
    assert result['reference_admission'] == 'UNVERIFIED'
    (tmp_path / 'cold_start_replay.json').write_text(json.dumps(result))
    if fault is None:
        path = TRACE.parents[2] / 'task_memory/task_2026-10-21_native_cold_start/check_groundtruth_run_successor.py'
        spec = importlib.util.spec_from_file_location('cold_start_successor', path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        bound = pd_replay.dedicated_services(proof, urls, 2)
        assert checker.native_idle_evidence(result, records, workload[0], 2, bound)[0]
        for mutation in ('unbound', 'counter', 'client', 'later_gauge', 'stale', 'first_after_dispatch'):
            changed = copy.deepcopy(result)
            first_sample = changed['idle_boundaries'][0]['observations'][-1]
            if mutation == 'counter':
                first_sample['roles']['decode']['completion_counters']['1'].pop('stop')
            elif mutation == 'client':
                changed['idle_boundaries'][0]['client_dispatched_requests'] = 1
            elif mutation == 'later_gauge':
                changed['idle_boundaries'][1]['observations'][-1]['roles']['decode']['lanes']['1'].pop('kv_cache_usage_perc')
            elif mutation == 'stale':
                changed['idle_boundaries'][1]['observations'][-1]['roles']['decode']['completed_requests'] = 0
            elif mutation == 'first_after_dispatch':
                first_sample['monotonic'] = records[0]['dispatch_monotonic']
            assert not checker.native_idle_evidence(changed, records, workload[0], 2,
                                                    None if mutation == 'unbound' else bound)[0]
        frozen = Path('/data/ycfeng/Frontier/task_memory/task_2026-09-27_pr_stack_vllm_e2e_accuracy_gate/calibration/check_groundtruth_run.py')
        spec = importlib.util.spec_from_file_location('cold_start_frozen_checker', frozen)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        assert not previous.native_idle_evidence(result, records, workload[0], 2)[0]
        legacy = copy.deepcopy(result)
        initial = legacy['idle_boundaries'][0]['observations'][-1]
        for observed in initial['roles'].values():
            observed['lanes'] = {str(lane): dict(num_requests_running=0, num_requests_waiting=0,
                                                kv_cache_usage_perc=0) for lane in range(2)}
            observed['idle'] = True
        legacy['idle_boundaries'][0].pop('proof')
        assert checker.native_idle_evidence(legacy, records, workload[0], 2) == previous.native_idle_evidence(legacy, records, workload[0], 2)


@pytest.mark.parametrize('fault', ['missing', 'pid', 'url', 'generation', 'manifest', 'before_launch', 'lanes'])
def test_dedicated_services_reject_unbound_readiness(tmp_path, fault):
    urls = dict(prefill='http://192.0.2.1:8100', decode='http://192.0.2.2:8200')
    service_files(tmp_path, urls)
    path = tmp_path / 'nodes/decode/server_ready.json'
    ready = json.loads(path.read_text())
    if fault == 'missing':
        ready.pop('server_pid')
    else:
        key, value = {'pid': ('server_pid', 999), 'url': ('url', urls['prefill']),
            'generation': ('run_generation', 'old'), 'manifest': ('run_manifest_sha256', '0' * 64),
            'before_launch': ('ready_at_utc', '2026-10-20T00:00:00+00:00'),
            'lanes': ('role', 'prefill')}[fault]
        ready[key] = value
    path.write_text(json.dumps(ready))
    with pytest.raises((ValueError, KeyError)):
        pd_replay.dedicated_services(tmp_path, urls, 2)


def test_old_actual_v4_run_remains_unverified():
    if not ACTUAL.is_dir():
        pytest.skip('Actual V4 output not retained')
    result = json.loads((ACTUAL / 'run/canonical_repetitions.json').read_text())
    assert [boundary['verified'] for boundary in result['idle_boundaries']] == [False] + [True] * 6
    assert not result['client_workload_verified']
    records = [json.loads(line) for line in (ACTUAL / 'run/client_requests.jsonl').read_text().splitlines()]
    assert len(records) == 432 and sum(row['status'] == 'completed' for row in records) == 431
    assert next(row for row in records if row['status'] != 'completed')['http_status'] == 502
    with pytest.raises(KeyError, match='server_started_at_utc'):
        urls = {role: json.loads((ACTUAL / 'run/replay_summary.json').read_text())['instances'][role]['http_url']
                for role in pd_replay.ROLES}
        pd_replay.dedicated_services(ACTUAL, urls, 2)


def test_successor_full_checker_rebinds_pinned_launch_receipts(native_case, monkeypatch):
    """Exercise the copy's native admission branch with real pinned subprocess readers."""
    from tests.unit.test_native_groundtruth_check import load_checker
    previous, case, run, manifest, records, summary = native_case
    checker = load_checker(TRACE.parents[2] / 'task_memory/task_2026-10-21_native_cold_start/check_groundtruth_run_successor.py')
    monkeypatch.setattr(checker, 'git', previous.git)
    harness = Path(manifest['harness_worktree'])
    original = __import__('subprocess').run
    module = 'tests/comparison/calibration/pd_replay.py'
    content = (TRACE.parents[2] / module).read_bytes()
    (harness / module).write_bytes(content)
    manifest['harness_sha256'][module] = hashlib.sha256(content).hexdigest()
    stamp = (run / 'run_manifest.json').stat()
    (run / 'run_manifest.json').write_text(json.dumps(manifest))
    os.utime(run / 'run_manifest.json', ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    digest = hashlib.sha256((run / 'run_manifest.json').read_bytes()).hexdigest()

    def git_run(args, **kwargs):
        if args[:3] == ['git', '-C', str(harness)] and args[-1].endswith(':' + module):
            return __import__('subprocess').CompletedProcess(args, 0, stdout=content)
        return original(args, **kwargs)

    monkeypatch.setattr(checker.subprocess, 'run', git_run)
    for path in (run / 'nodes').glob('*/result.json'):
        node = json.loads(path.read_text())
        identity = dict(server_pid=101, server_started_at_utc='2026-01-01T00:00:01+00:00',
                        run_manifest_sha256=digest, run_generation=manifest['run_generation'])
        path.write_text(json.dumps(node | identity))
        if node['plan']['lane'] == 0:
            ready = path.with_name('server_ready.json')
            ready.write_text(json.dumps(json.loads(ready.read_text()) | identity | dict(
                health_status=200, ready_at_utc='2026-01-01T00:00:02+00:00')))
    (run / 'launcher_receipt.json').write_text(json.dumps(json.loads((run / 'launcher_receipt.json').read_text()) |
                                                         {'run_manifest_sha256': digest}))
    repetitions = summary['canonical_repetitions']
    first = repetitions['idle_boundaries'][0]
    urls = {role: summary['instances'][role]['http_url'] for role in pd_replay.ROLES}
    first.update(proof='dedicated_services_ready_client_zero_native_counters_zero',
                 client_dispatched_requests=0, fresh_services=pd_replay.dedicated_services(run, urls, 4))
    for role in first['observations'][-1]['roles'].values():
        role.update(idle=False, lanes={str(lane): {} for lane in range(4)}, completion_counters={str(lane):
            {reason: 0 for reason in ('stop', 'length', 'abort', 'error', 'repetition')} for lane in range(4)})
    (run / 'run/canonical_repetitions.json').write_text(json.dumps(repetitions))
    (run / 'run/replay_summary.json').write_text(json.dumps(summary))
    result = checker.check(case, 'native', 'clean')
    (run / 'successor_predicates.json').write_text(json.dumps(result, indent=2))
    assert result['checks']['harness_pinned_and_client_rows'].startswith('PASS')
    assert result['checks']['requests_and_idle_lanes'].startswith('PASS: 7 boundaries;')
    assert result['checks']['native_workers'].startswith('PASS'), result['checks']
    assert result['status'] == 'FAIL' and result['checks']['rank_runtime_graph'].startswith('FAIL')
    assert set(result['checks']) == {'checkout', 'inputs_unchanged', 'harness_pinned_and_client_rows',
        'requests_and_idle_lanes', 'native_workers', 'native_lifecycle', 'rank_runtime_graph', 'terminal_status'}
    assert all(value.startswith('PASS') for key, value in result['checks'].items() if key != 'rank_runtime_graph'), result['checks']
    ready = run / 'nodes/host0/server_ready.json'
    bound = json.loads(ready.read_text())
    faults = dict(role='decode', url=urls['decode'], server_pid=999, server_started_at_utc='2025-01-01T00:00:00+00:00',
                  run_manifest_sha256='0' * 64, run_generation='old', ready_at_utc='2025-01-01T00:00:00+00:00', health_status=503)
    rejected = {}
    for key, value in [(key, None) for key in faults] + list(faults.items()):
        changed = dict(bound)
        if value is None:
            changed.pop(key)
        else:
            changed[key] = value
        ready.write_text(json.dumps(changed))
        result = checker.check(case, 'native', 'clean')
        assert result['checks']['requests_and_idle_lanes'].startswith('FAIL'), (key, value, result)
        assert result['checks']['native_workers'].startswith('FAIL'), (key, value, result)
        rejected[f'{key}_{"missing" if value is None else "wrong"}'] = result['checks']
    for path in (run / 'nodes').glob('*/server_ready.json'):
        receipt = json.loads(path.read_text())
        path.write_text(json.dumps(dict(role=receipt['role'], url=receipt['url'])))
    first.pop('proof')
    for observed in first['observations'][-1]['roles'].values():
        observed.update(idle=True, lanes={str(lane): dict(num_requests_running=0, num_requests_waiting=0,
                          kv_cache_usage_perc=0) for lane in range(4)})
    (run / 'run/canonical_repetitions.json').write_text(json.dumps(repetitions))
    (run / 'run/replay_summary.json').write_text(json.dumps(summary))
    for extra in ({}, {'server_pid': 101}):
        ready.write_text(json.dumps(dict(role='prefill', url=urls['prefill']) | extra))
        current, legacy = checker.check(case, 'native', 'clean'), previous.check(case, 'native', 'clean')
        assert current['checks']['native_workers'] == legacy['checks']['native_workers']
        assert current['checks']['native_workers'].startswith('FAIL' if extra else 'PASS')
        rejected[f'legacy_{"extra_pid" if extra else "strict_role_url"}'] = current['checks']
    (run / 'successor_readiness_negatives.json').write_text(json.dumps(rejected, indent=2))
