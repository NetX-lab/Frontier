"""CPU HTTP integration; no fixture result qualifies a native GPU run."""

import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
import yaml

from tests.comparison.calibration import native_pd_worker, pd_proxy, pd_replay
from tests.comparison.calibration.client_metrics import native_client_rows
from tests.unit.test_native_pd_manifest import REPO, TRACE, git, native_case, sha


CHECKER = Path('/data/ycfeng/Frontier/task_memory/task_2026-09-27_pr_stack_vllm_e2e_accuracy_gate/calibration/check_groundtruth_run.py')
TRACE_FILES = ('trace.csv', 'request_ids.json', 'warmup_pool.csv', 'manifest.json', 'SHA256SUMS')
# The unchanged generator excludes 100 IDs at either end. This fixture-only
# argument yields IDs 100..127, inside the separately recorded model vocab 128.
FIXTURE_PROMPT_GENERATOR_VOCAB = 228
PARAMS = dict(do_remote_prefill=True, do_remote_decode=False, transfer_mode='pull',
              remote_engine_id='cpu-mock-prefill', remote_request_id='cpu-mock-transfer',
              remote_host='127.0.0.1', remote_port=5557, remote_block_ids=[[1]])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def write_records(path, records):
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))


def event(value):
    return ('data: ' + (value if isinstance(value, str) else json.dumps(value)) + '\n\n').encode()


class FixtureClock:
    """Advance only harness timestamps; keep asyncio's actual loop clock intact."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        self.now += .000001
        return self.now

    def time(self):
        return 1700000000.0 + self.now


class FixtureAsyncio:
    """Yield for real HTTP I/O, then advance each scheduled arrival without waiting."""

    def __init__(self, clock):
        self.clock = clock

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def sleep(self, delay):
        # Heartbeat does not represent a new arrival and must not move fixture time.
        if asyncio.current_task().get_coro().__name__ == 'heartbeat':
            await asyncio.sleep(3600)
            return
        due = self.clock.now + delay
        await asyncio.sleep(0)
        self.clock.now = max(self.clock.now, due)


def test_emitted_canonical_http_records_reach_actual_external_checker(tmp_path, monkeypatch):
    if not CHECKER.is_file():
        pytest.skip('The existing task-local external checker is not installed on this host')
    started = time.perf_counter()
    checker_hash = sha(CHECKER)
    config, source, harness = native_case(tmp_path)
    case_dir = Path(config['case_dir'])
    case = yaml.safe_load((case_dir / 'manifest.yaml').read_text())
    case['groundtruth_branch'] = git(source, 'branch', '--show-current')
    (case_dir / 'manifest.yaml').write_text(yaml.safe_dump(case))
    # These are explicit CPU fixture inputs, not installed vLLM/GPU claims.
    config.update(expected_source_hashes={'source.py': sha(source / 'source.py')},
                  data_parallel_size_local=1)
    config['engine']['native_reference']['image'] = config['image']
    write_json(case_dir / 'inputs/native_config.json', config)
    write_json(case_dir / 'inputs/engine.json', config['engine'])
    frozen = {name: sha(TRACE / name) for name in TRACE_FILES}
    for name in TRACE_FILES:
        committed = subprocess.check_output([
            'git', '-C', str(REPO), 'show', f'68f891b:{TRACE.relative_to(REPO)}/{name}'])
        assert committed == (TRACE / name).read_bytes()
        assert sha(case_dir / 'inputs/trace' / name) == frozen[name]
    manifest_path = native_pd_worker.write_run_manifest(
        config, 'CPU localhost HTTP fixture authorization only; no native job or GPU qualification')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['groundtruth_commit'] == git(source, 'rev-parse', 'HEAD')
    assert manifest['harness_commit'] == git(harness, 'rev-parse', 'HEAD')
    assert len(manifest['harness_sha256']) == 8
    run_dir = manifest_path.parent
    run = run_dir / 'run'
    run.mkdir()
    (run_dir / 'nodes').mkdir()  # Intentionally empty: no fabricated native workers.
    workload = pd_replay.canonical_workload(case_dir / 'inputs/trace')
    clock = FixtureClock()
    replay_module = sys.modules[pd_replay.replay.__module__]
    for module in (pd_replay, replay_module, pd_proxy):
        monkeypatch.setattr(module, 'time', clock)
    for module in (pd_replay, replay_module):
        monkeypatch.setattr(module, 'asyncio', FixtureAsyncio(clock))
    completed = {role: [0] * 4 for role in ('prefill', 'decode')}
    submitted = {role: 0 for role in completed}
    seen, scrapes, prompts = [], [], {}

    def endpoint(role):
        async def complete(request):
            body = await request.json()
            identity = request.headers['X-Request-Id']
            template, repeat = identity.rsplit('-repeat', 1)
            assert repeat in ('01', '02', '03')
            assert len(body['prompt']) == 6144
            assert all(type(token) is int and 0 <= token < 128 for token in body['prompt'])
            assert prompts.setdefault(template, body['prompt']) == body['prompt']
            assert body['return_token_ids'] is True
            assert body['max_tokens'] == (1 if role == 'prefill' else 1024)
            assert body['stream'] is (role == 'decode')
            if role == 'prefill':
                assert 'min_tokens' not in body and 'stream_options' not in body
                assert body['kv_transfer_params']['do_remote_decode'] is True
            else:
                assert body['min_tokens'] == 1024
                assert body['stream_options'] == {'include_usage': True}
                assert body['kv_transfer_params'] == PARAMS
            seen.append(dict(role=role, request_id=identity, prompt_tokens=len(body['prompt']),
                             max_tokens=body['max_tokens'], stream=body['stream'],
                             return_token_ids=body['return_token_ids'],
                             kv_transfer_params=body['kv_transfer_params']))
            lane = submitted[role] % 4
            submitted[role] += 1
            if role == 'prefill':
                completed[role][lane] += 1
                return web.json_response({'kv_transfer_params': PARAMS})
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            await response.write(event({'choices': [{'index': 0, 'text': '', 'token_ids': [1]}]}))
            await asyncio.sleep(0)
            await response.write(event({'choices': [{'index': 0, 'text': '', 'token_ids': [2] * 1023,
                                                     'finish_reason': 'length'}]}))
            await response.write(event({'choices': [], 'usage': dict(prompt_tokens=6144,
                completion_tokens=1024, total_tokens=7168)}) + event('[DONE]'))
            await response.write_eof()
            completed[role][lane] += 1
            return response
        return complete

    def metrics(role):
        async def scrape(request):
            scrapes.append(dict(role=role, completed_requests=sum(completed[role])))
            text = '\n'.join(
                f'vllm:{name}{{model_name="dp_pp_case",engine="{lane}"}} {value}'
                for lane, count in enumerate(completed[role])
                for name, value in (('num_requests_running', 0), ('num_requests_waiting', 0),
                                    ('kv_cache_usage_perc', 0), ('request_success_total', count)))
            return web.Response(text=text)
        return scrape

    async def exercise():
        prefill, decode = web.Application(), web.Application()
        for role, app in (('prefill', prefill), ('decode', decode)):
            app.router.add_post('/v1/completions', endpoint(role))
            app.router.add_get('/metrics', metrics(role))
        async with TestServer(prefill) as ps, TestServer(decode) as ds:
            urls = dict(prefill=str(ps.make_url('')).rstrip('/'), decode=str(ds.make_url('')).rstrip('/'))
            proxy = pd_proxy.build_app(urls['prefill'], urls['decode'], '', '',
                                       run / 'proxy_requests.jsonl', vllm_version='0.28.0')
            async with TestServer(proxy) as front:
                def checkpoint(result, records):
                    write_records(run / 'client_requests.jsonl', records)
                    write_json(run / 'canonical_repetitions.json', result)

                result, records = await pd_replay.replay_repetitions(
                    workload, front.port, FIXTURE_PROMPT_GENERATOR_VOCAB, 0., urls, 4,
                    5., .001, checkpoint=checkpoint)
        return result, records, urls

    result, records, urls = asyncio.run(exercise())
    assert result['client_workload_verified'] and result['reference_admission'] == 'UNVERIFIED'
    assert len(records) == len({row['request_id'] for row in records}) == 432
    assert all(row['status'] == 'completed' for row in records)
    assert completed == {'prefill': [108] * 4, 'decode': [108] * 4}
    assert len(prompts) == 144 and len(seen) == 864
    assert {r['request_id'] for r in seen if r['role'] == 'prefill'} == {
        r['request_id'] for r in seen if r['role'] == 'decode'}
    for repeat in (1, 2, 3):
        for phase, templates in (('warmup', workload[2]), ('formal', workload[1])):
            selected = [row for row in records if row['repeat'] == repeat and row['segment'] == phase]
            assert len(selected) == len(templates)
            assert [row['arrived_at'] for row in selected] == list(range(len(templates)))
        assert result['repetitions'][repeat - 1]['metrics']['total_tokens'] == 917504
    assert len(result['idle_boundaries']) == 7 and all(b['verified'] for b in result['idle_boundaries'])
    assert len(scrapes) == 14
    for index, count in enumerate((0, 16, 144, 160, 288, 304, 432)):
        assert sorted((r['role'], r['completed_requests']) for r in scrapes[index * 2:index * 2 + 2]) == [
            ('decode', count), ('prefill', count)]
    proxy_records = [json.loads(line) for line in (run / 'proxy_requests.jsonl').read_text().splitlines()]
    assert len(proxy_records) == 432
    assert {r['request_id'] for r in proxy_records} == {r['request_id'] for r in records}
    assert all(r['prefill_status'] == r['decode_status'] == 200 and r['kv_transfer_params'] == PARAMS
               and r['handoff_protocol'] == 'nixl_pull' for r in proxy_records)
    summary = dict(mode='clean', vllm_version='0.28.0', status=0, canonical_repetitions=result,
                   engine_config=config['engine'], num_requests=len(records), failed_request_ids=[],
                   qualification_verified=False,
                   instances={role: dict(http_url=url, lifecycle_owner='CPU mock') for role, url in urls.items()})
    write_json(run / 'replay_summary.json', summary)
    write_json(run_dir / 'replay_finished.json', {'returncode': 0})
    write_records(tmp_path / 'http_observations.jsonl', seen)
    write_json(tmp_path / 'metrics_scrapes.json', scrapes)
    request_ids = json.loads((case_dir / 'inputs/trace/request_ids.json').read_text())
    frames = native_client_rows(run_dir, workload, request_ids)
    assert {name: len(frame) for name, frame in frames.items()} == dict(repeat01=128, repeat02=128, repeat03=128)
    for name, frame in frames.items():
        assert set(frame['request_id']) == {row['request_id'] for row in workload[1]}
        assert set(frame['request_num_prefill_tokens']) == {6144}
        assert set(frame['request_num_decode_tokens']) == {1024}
        frame.to_csv(tmp_path / f'{name}_native_client_rows.csv', index=False)
    spec = importlib.util.spec_from_file_location('canonical_http_external_checker', CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)

    def check(name):
        outcome = checker.check(case_dir, config['run_tag'], 'clean')
        write_json(tmp_path / f'checker_{name}.json', outcome)
        assert outcome['status'] == 'FAIL'  # Missing runtime evidence must never become a synthetic PASS.
        assert outcome['checks']['rank_runtime_graph'].startswith('FAIL: UNVERIFIED:')
        return outcome['checks']

    baseline = check('cpu_http_baseline')
    assert baseline['checkout'].startswith('PASS:')
    assert baseline['inputs_unchanged'].startswith('PASS:')
    assert baseline['requests_and_idle_lanes'].startswith('PASS: 7 boundaries;')
    assert "{'repeat01': 128, 'repeat02': 128, 'repeat03': 128}" in baseline['harness_pinned_and_client_rows']
    assert "native server commands/environments: {}" in baseline['harness_pinned_and_client_rows']
    assert 'Expected 8 nodes, found 0' in baseline['native_workers']
    assert 'Incomplete native node evidence' in baseline['native_lifecycle']
    assert 'FileNotFoundError' in baseline['terminal_status']
    original_manifest = manifest_path.read_bytes()
    manifest_stat = manifest_path.stat()
    original_records = (run / 'client_requests.jsonl').read_bytes()
    errors = {}
    try:
        stale = copy.deepcopy(manifest)
        stale['groundtruth_commit'] = '0' * 40
        write_json(manifest_path, stale)
        os.utime(manifest_path, ns=(manifest_stat.st_atime_ns, manifest_stat.st_mtime_ns))
        provenance_failure = check('stale_source_provenance')
        assert provenance_failure['checkout'].startswith('FAIL:')
        assert provenance_failure['harness_pinned_and_client_rows'] == baseline['harness_pinned_and_client_rows']
        manifest_path.write_bytes(original_manifest)
        os.utime(manifest_path, ns=(manifest_stat.st_atime_ns, manifest_stat.st_mtime_ns))
        bad_count = copy.deepcopy(records)
        bad_count[-1]['output_token_count'] -= 1
        write_records(run / 'client_requests.jsonl', bad_count)
        with pytest.raises(ValueError, match='Invalid native client identity, scheduled arrival or tokens') as error:
            native_client_rows(run_dir, workload, request_ids)
        errors['wrong_output_count'] = str(error.value)
        count_failure = check('wrong_output_count')
        assert count_failure['checkout'].startswith('PASS:')
        assert 'CalledProcessError' in count_failure['harness_pinned_and_client_rows']
        assert count_failure['harness_pinned_and_client_rows'] != baseline['harness_pinned_and_client_rows']
        (run / 'client_requests.jsonl').write_bytes(original_records)
        os.utime(run / 'client_requests.jsonl', ns=(1, 1))
        with pytest.raises(SystemExit, match='older than') as error:
            native_client_rows(run_dir, workload, request_ids)
        errors['stale_output'] = str(error.value)
        stale_failure = check('stale_client_output')
        assert stale_failure['checkout'].startswith('PASS:')
        assert 'CalledProcessError' in stale_failure['harness_pinned_and_client_rows']
    finally:
        manifest_path.write_bytes(original_manifest)
        os.utime(manifest_path, ns=(manifest_stat.st_atime_ns, manifest_stat.st_mtime_ns))
        (run / 'client_requests.jsonl').write_bytes(original_records)
    assert len(native_client_rows(run_dir, workload, request_ids)) == 3
    assert {name: sha(TRACE / name) for name in TRACE_FILES} == frozen
    assert all(sha(case_dir / 'inputs/trace' / name) == digest for name, digest in frozen.items())
    assert sha(CHECKER) == checker_hash
    write_json(tmp_path / 'negative_reader_errors.json', errors)
    evidence = dict(task_id='W128-GLM-CANONICAL-INTEGRATION-HTTP', fixture_only=True,
        timing='Harness-only virtual arrival clock; real asyncio loop clock and localhost HTTP I/O; not GPU timing',
        model_vocab_size=128, prompt_generator_vocab_argument=FIXTURE_PROMPT_GENERATOR_VOCAB,
        records=len(records), unique_transport_ids=len({r['request_id'] for r in records}),
        http_requests_per_role={role: sum(counts) for role, counts in completed.items()},
        formal_dataframes={name: len(frame) for name, frame in frames.items()}, idle_boundaries=7,
        frozen_trace_sha256=frozen, checker_sha256=checker_hash,
        checker_baseline_status='FAIL', checker_baseline_checks=baseline, negative_reader_errors=errors,
        actual_wall_s=time.perf_counter() - started)
    write_json(tmp_path / 'integration_evidence.json', evidence)
    print('CANONICAL_HTTP_INTEGRATION_EVIDENCE', json.dumps(evidence))
