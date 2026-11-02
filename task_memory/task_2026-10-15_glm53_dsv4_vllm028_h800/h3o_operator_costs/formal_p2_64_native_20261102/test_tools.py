"""CPU tests of failure isolation, coverage evidence and exact package scope."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import coverage

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1] / 'h200_mainline/build_package.py'
HARNESS = Path('/data/ycfeng/Frontier/.worktrees/p2-64-pair-warmup')


def fixture_run(root):
    base = datetime(2026, 11, 2, tzinfo=timezone.utc).timestamp()
    offset = base - 1000.
    root.mkdir()
    (root / 'run').mkdir()
    manifest = dict(run_tag='cpu-fixture', mode='clean', written_before_launch_utc='2026-11-02T00:00:00+00:00',
                    groundtruth_commit='cpu-fixture', resources=dict(replica=4))
    (root / 'run_manifest.json').write_text(json.dumps(manifest))
    client, proxy, phases = [], [], {}
    for phase, count, origin in [('warmup', 4, 1010.), ('formal', 1, 1024.)]:
        phases[phase] = dict(origin_wall=origin+offset, origin_monotonic=origin)
        for i in range(count):
            rid = f'{phase}-r{i:03d}-repeat01'
            t = origin + 3 * i
            pair = dict(prefill=i // 2 if phase == 'warmup' else 0, decode=i % 2 if phase == 'warmup' else 0)
            row = dict(request_id=rid, template_request_id=rid.rsplit('-repeat', 1)[0], repeat=1, segment=phase,
                       dispatch_monotonic=t, last_token_monotonic=t+2.)
            pr = dict(request_id=rid, kv_transfer_params=dict(remote_engine_id=f'cpu_prefill_dp{pair["prefill"]}'))
            for field, delta in [('proxy_arrival', 0.), ('prefill_response', .1), ('decode_first_byte', 1.5), ('decode_response', 2.)]:
                pr[field+'_monotonic'] = t + delta
                pr[field+'_time'] = t + delta + offset
            if phase == 'warmup':
                row['intended_pair'] = pair
                pr['intended_pair'] = pair
            client.append(row)
            proxy.append(pr)
    def boundary(name, t, completed):
        lanes = {str(i): dict(num_requests_running=0, num_requests_waiting=0, kv_cache_usage_perc=0) for i in range(2)}
        return dict(boundary=name, verified=True, observations=[dict(monotonic=t, errors=[],
            roles={role: dict(lanes=lanes, completed_requests=completed, idle=True) for role in ('prefill', 'decode')})])
    repeat = dict(repetitions=[dict(repeat=1, phases=phases)], idle_boundaries=[
        boundary('before_repeat01_warmup', 1001., 0), boundary('after_repeat01_warmup', 1023., 4),
        boundary('after_repeat01_formal', 1027., 5)])
    (root / 'run/canonical_repetitions.json').write_text(json.dumps(repeat))
    for name, records in [('client_requests', client), ('proxy_requests', proxy)]:
        (root / f'run/{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    for role in ('prefill', 'decode'):
        for lane in range(2):
            node = root / f'nodes/{role}{lane}'
            (node / 'serving').mkdir(parents=True)
            ids = [f'cmpl-{r["request_id"]}-0-suffix' for r in client if (
                r.get('intended_pair', dict(prefill=0, decode=0))[role] == lane)]
            (node / 'serving/rank0.json').write_text(json.dumps(dict(role=role,
                inventory=dict(request_owner=lane), counters=dict(dropped_rows=0, request_ids=[dict(request_ids=ids)]))))
            log = ['INFO 11-02 00:00:01 Starting vLLM server on http://cpu']
            if role == 'decode':
                for i in range(4):
                    if i % 2 == lane:
                        stamp = datetime.fromtimestamp(base + 11 + 3*i, timezone.utc).strftime('%m-%d %H:%M:%S')
                        for tp in range(8):
                            log.append(f'Worker_DP{lane}_TP{tp} INFO {stamp} NIXL compatibility check passed')
            (node / 'server.log').write_text('\n'.join(log)+'\n')
            if lane == 0:
                (node / 'server_ready.json').write_text(json.dumps(dict(ready_at_utc='2026-11-02T00:00:01.100000+00:00')))
    return root


def test_complete_pair_evidence(tmp_path):
    result = coverage.safe_analyze(fixture_run(tmp_path / 'run'))
    assert result['evidence_verified'], result.get('error')
    assert result['status'] == 'COVERED'
    assert result['openers'] == {'vllm/cpu-fixture-repeat01': []}
    assert len(result['repetitions']['repeat01']['pairs_reached_in_this_warmup']) == 4


@pytest.mark.parametrize('fault', ['intended', 'idle', 'cohort', 'absent_cohorts', 'configured_nodes', 'missing'])
def test_coverage_is_soft_and_missing_evidence_is_not_an_empty_exclusion(tmp_path, fault):
    root = fixture_run(tmp_path / 'run')
    if fault == 'intended':
        path = root / 'run/client_requests.jsonl'
        records = [json.loads(r) for r in path.read_text().splitlines()]
        records[0]['intended_pair']['decode'] = 1
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    elif fault == 'idle':
        path = root / 'run/canonical_repetitions.json'
        record = json.loads(path.read_text())
        record['idle_boundaries'][1]['observations'][0]['roles']['decode']['lanes']['0']['kv_cache_usage_perc'] = .1
        path.write_text(json.dumps(record))
    elif fault == 'cohort':
        path = root / 'nodes/decode0/server.log'
        path.write_text(path.read_text().replace('Worker_DP0_TP7', 'Worker_DP0_TP99'))
    elif fault == 'absent_cohorts':
        path = root / 'nodes/decode0/server.log'
        path.write_text(path.read_text().splitlines()[0]+'\n')
    elif fault == 'configured_nodes':
        path = root / 'run_manifest.json'
        document = json.loads(path.read_text())
        document['resources']['replica'] = 6
        path.write_text(json.dumps(document))
    else:
        root = tmp_path / 'absent'
    result = coverage.safe_analyze(root)
    assert result['status'] == 'NOT_COVERED'
    if fault in ('cohort', 'absent_cohorts', 'configured_nodes', 'missing'):
        assert result['evidence_verified'] is False and result['openers'] is None
    # CLI always returns zero, even malformed/missing serving evidence.
    process = subprocess.run([sys.executable, str(HERE/'coverage.py'), '--run', str(root), '--output',
        str(tmp_path/'result.json'), '--openers', str(tmp_path/'openers.json')], capture_output=True, text=True)
    assert process.returncode == 0
    assert json.loads((tmp_path/'result.json').read_text())['status'] == 'NOT_COVERED'


@pytest.fixture
def builder_case(tmp_path):
    module = coverage.load_module('test_builder', BUILDER)
    case = module.Case('p2_64_formal', 'package_01')
    case.harness_commit = module.git(HARNESS, 'rev-parse', 'HEAD').strip()
    case.target = tmp_path / 'package'
    config, changed = module.build(case)
    module.compare(case, config, changed)
    return module, case, config, changed


@pytest.mark.parametrize('name,field', [('engine.json', 'seed'), ('config.json', 'timeout_seconds'),
    ('case/inputs/frontier.json', 'replica_config_attn_dp'), ('trace/trace.csv', None),
    ('submit_native_diagnostic.py', None), ('native_dummy_bootstrap.py', None)])
def test_builder_rejects_other_differences(builder_case, name, field):
    module, case, config, changed = builder_case
    path = case.target / name
    if field:
        document = json.loads(path.read_text())
        document[field] += 1
        path.write_text(json.dumps(document))
    else:
        path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='Unexpected formal package bytes'):
        module.compare(case, config, changed)


def test_builder_preemptible_and_trace_are_exact(builder_case):
    module, case, config, changed = builder_case
    assert config['preemptible'] is True and case.h200_limit == 32
    assert config['engine'] == json.loads((case.source/'config.json').read_text())['engine']
    for name in ('engine.json', 'case/inputs/engine.json', 'case/inputs/frontier.json',
                 'trace/trace.csv', 'trace/manifest.json', 'trace/warmup_pool.csv', 'trace/SHA256SUMS'):
        assert (case.target/name).read_bytes() == (case.source/name).read_bytes()
    assert 'preemptible=config.get("preemptible", False)' in (case.target/'submit_native_diagnostic.py').read_text()
    import yaml
    manifest = yaml.safe_load((case.target/'case/manifest.yaml').read_text())
    assert manifest['policy']['preemptible'] is True
    assert manifest['policy']['H200_nonpreemptible_running_plus_queued_limit'] == 32


@pytest.mark.parametrize('fault', ['missing', 'hash', 'equal', 'later'])
def test_build_checks_receipt_before_creating_package(tmp_path, monkeypatch, fault):
    module = coverage.load_module('test_builder_receipt', BUILDER)
    case = module.Case('p2_64_formal', 'package_01')
    case.target = tmp_path / 'not_created'
    case.harness_commit = 'cpu-fixture'
    receipt = tmp_path / 'receipt.json'
    now = datetime.now(timezone.utc)
    issued = now + timedelta(days=1) if fault == 'later' else now
    content = json.dumps(dict(issued_utc=issued.isoformat(), predicate=dict(result=True))).encode()
    if fault != 'missing':
        receipt.write_bytes(content)
    monkeypatch.setattr(module, 'PROMOTION_PATH', receipt)
    monkeypatch.setattr(module, 'PROMOTION_SHA256', 'wrong' if fault == 'hash' else hashlib.sha256(content).hexdigest())
    if fault == 'equal':
        class Clock:
            @staticmethod
            def now(tz):
                return now
        monkeypatch.setattr(module, 'datetime', Clock)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.build(case)
    assert not case.target.exists()


@pytest.mark.parametrize('fault', ['valid', 'missing', 'hash', 'equal', 'later'])
def test_actual_staged_launch_checks_receipt_before_check_only(builder_case, tmp_path, fault):
    module, case, config, changed = builder_case
    receipt = tmp_path / 'bound_receipt.json'
    built = datetime.now(timezone.utc) - timedelta(seconds=2)
    issued = built - timedelta(seconds=2)
    if fault == 'equal':
        issued = built
    elif fault == 'later':
        issued = built + timedelta(seconds=1)
    content = json.dumps(dict(issued_utc=issued.isoformat(), predicate=dict(result=True))).encode()
    if fault != 'missing':
        receipt.write_bytes(content)
    config['promotion_receipt'] = dict(path=str(receipt), build_utc=built.isoformat(),
        sha256='wrong' if fault == 'hash' else hashlib.sha256(content).hexdigest())
    # This independent fixture deliberately reseals its test config. The real
    # candidate was compared above, and is never modified or submitted here.
    (case.target/'config.json').write_text(json.dumps(config))
    subprocess.run([sys.executable, str(case.target/'seal.py')], check=True, capture_output=True, text=True)
    process = subprocess.run([sys.executable, str(case.target/'launch.py'), '--check-only'],
        capture_output=True, text=True)
    if fault == 'valid':
        assert process.returncode == 0, process.stderr
        assert process.stdout.strip() == 'INPUT_HASHES_PASS_NO_SUBMISSION'
    else:
        assert process.returncode != 0
        assert 'INPUT_HASHES_PASS_NO_SUBMISSION' not in process.stdout
        assert 'Promotion receipt' in process.stderr or 'bound_receipt.json' in process.stderr
    assert not (case.target/'launcher_started.json').exists()


def test_staged_resource_constructor_forwards_wholly_preemptible_request(builder_case):
    import ast
    module, case, config, changed = builder_case
    tree = ast.parse((case.target/'submit_native_diagnostic.py').read_text())
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'ResourceConfig')
    observed = eval(compile(ast.Expression(call), '<cpu ResourceConfig stand-in>', 'eval'),
                    dict(ResourceConfig=lambda **fields: fields, config=config))
    assert observed['preemptible'] is True
    assert (observed['replica'], observed['gpu'], observed['positive_tags']) == (8, 8, ['H200'])
    original = ast.parse((case.source/'submit_native_diagnostic.py').read_text())
    old_call = next(node for node in ast.walk(original) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == 'ResourceConfig')
    previous = eval(compile(ast.Expression(old_call), '<retained resource expression>', 'eval'),
                    dict(ResourceConfig=lambda **fields: fields, config=config))
    assert {k:v for k,v in observed.items() if k != 'preemptible'} == previous


def test_compare_missing_manifest_is_durable_and_soft(tmp_path):
    import compare
    download = tmp_path/'download'
    artifacts = download/'artifacts'
    artifacts.mkdir(parents=True)
    (artifacts/'payload.txt').write_text('explicit incomplete CPU fixture')
    inventory = dict(files=[dict(path='payload.txt', bytes=(artifacts/'payload.txt').stat().st_size,
                                 sha256=compare.sha(artifacts/'payload.txt'))])
    (download/'cloud_inventory.json').write_text(json.dumps(inventory))
    (download/'download_receipt.json').write_text(json.dumps(dict(status='HASH_IDENTICAL_DOWNLOAD',
        mismatched=[], inventory_sha256=compare.sha(download/'cloud_inventory.json'))))
    result = compare.run_pipeline(tmp_path/'case', download, tmp_path/'out')
    assert result['status'] == 'UNVERIFIED' and result['errors'] == ['read_run_manifest']
    assert json.loads((tmp_path/'out/result.json').read_text()) == result
