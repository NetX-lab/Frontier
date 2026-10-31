"""CPU acceptance tests for B30b link, NIXL and launch-depth diagnostics."""
import copy
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from frontier.profiling.collectives.cuda_launch_depth import blocking_depth, probe_cases
from frontier.profiling.collectives.nixl_kv_read_benchmark import KVRegionLayout, read_cases
from frontier.profiling.collectives.rank_placement import select_physical_devices, select_visible_devices
from frontier.profiling.collectives.vllm_collective_benchmark import plan_from_config
from frontier.profiling.collectives.vllm_forward_collectives import ServingParallelism

REPO = Path(__file__).resolve().parents[2]
B30B = REPO / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b30b'


def load(name, path=None):
    spec = importlib.util.spec_from_file_location('b30b_test_' + name, path or B30B / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner, fits, analysis, runner = [load(n) for n in ('planner', 'fit_tools', 'launch_analysis', 'phase_runner')]
LINKS = json.loads((B30B / 'stage_b30b_links_kv.json').read_text())
DEPTH = json.loads((B30B / 'stage_b30b_launch_depth.json').read_text())


@pytest.mark.parametrize('phase,count', [(0, 132), (1, 142), (2, 142), (3, 80)])
def test_case_grids_and_existing_admission(phase, count):
    p = LINKS['phases'][phase]
    cases = planner.phase_cases(p)
    assert len(cases) == count
    if phase < 3:
        cfg = p['config']
        for key, values in cfg['grid'].items():
            if key != 'calls':
                assert all(type(v) is int and v > 0 and v & (v - 1) == 0 for v in values)
        # Single-rank TP groups cross no link, so B measures only its EP calls.
        if p['name'].startswith('B_'):
            assert {(c['group'], c['group_size']) for c in cases} == {('ep', 2)}
        parallel = ServingParallelism.from_engine(cfg['engine'])
        model = json.loads(Path(cfg['model_config']).read_text())
        with pytest.raises(ValueError, match='Sequence-parallel'):
            plan_from_config(model, parallel, cfg['grid'])
        for invalid in ({'enable_expert_parallel': False}, {'pipeline_parallel_size': 2},
                        {'all2all_backend': 'unsupported'}, {'dtype': 'unsupported'}):
            bad = ServingParallelism.from_engine({**cfg['engine'], **invalid})
            with pytest.raises(ValueError):
                plan_from_config(model, bad, cfg['grid'], allow_link_probe_parallelism=True)
    else:
        assert max(c['blocks_per_request'] * c['requests_in_flight'] for c in cases) == 864
        assert {c['active_pairs'] for c in cases} == {1, 8}


def test_native_layout_established_and_group_counts():
    layout = planner.establish_layout(LINKS)
    assert layout.num_regions == 1 and layout.block_bytes == 1002240
    assert layout.request_blocks(6144) == 46
    assert layout.request_blocks(8192) == 54
    assert layout.request_blocks(64) == 13
    legacy = KVRegionLayout((16,), 64, 128)
    assert legacy.request_blocks(65) == 2
    with pytest.raises(ValueError, match='exceed'):
        read_cases([{'prompt_tokens': [8192], 'active_pairs': [8], 'requests_in_flight': [32]}], layout, 8)
    with pytest.raises(ValueError, match='positive'):
        read_cases([{'prompt_tokens': [64], 'active_pairs': [1], 'requests_in_flight': [0]}], layout, 8)


@pytest.mark.parametrize('failure', ['missing', 'changed_line', 'stride', 'groups'])
def test_layout_fails_closed(tmp_path, failure):
    config = copy.deepcopy(LINKS)
    evidence = json.loads((REPO / config['kv_layout_evidence']).read_text())
    if failure == 'missing':
        evidence['logs'] = []
    elif failure == 'changed_line':
        evidence['logs'][0]['text'] = 'not the retained native log'
    elif failure == 'stride':
        cache = config['phases'][3]['config']['kv_cache']
        cache['regions'][0]['block_bytes'] += 1
        cache['resident_bytes_per_block'] += 1
    else:
        config['phases'][3]['config']['kv_cache']['logical_groups'] = []
    path = tmp_path / 'evidence.json'
    path.write_text(json.dumps(evidence))
    config['kv_layout_evidence'] = str(path)
    with pytest.raises(ValueError):
        planner.establish_layout(config)


@pytest.mark.parametrize('order', [list(range(8)), list(reversed(range(8))), [7, 0, 1, 2, 3, 4, 5, 6]])
@pytest.mark.parametrize('uuid_prefix', ['GPU-', ''])
def test_physical_placement_is_independent_of_cuda_order(order, uuid_prefix):
    inventory = [f'{i}, GPU-{i}, 0000:{i:02x}:00.0, NVIDIA H200' for i in range(8)]
    allocated = [f'{uuid_prefix}{i}' for i in order]
    assert select_physical_devices(allocated, [7, 0], inventory, 8) == 'GPU-7,GPU-0'
    assert select_visible_devices(','.join(allocated), [0], 8) == allocated[0]
    with pytest.raises(ValueError, match='unavailable'):
        select_physical_devices(allocated[:-1], [order[-1]], inventory, 7)
    with pytest.raises(ValueError):
        select_physical_devices(allocated, [0, 0], inventory, 8)
    with pytest.raises(ValueError):
        select_visible_devices('0,1', [7], 8)


def test_phase_torchrun_rank_placement():
    peers = [dict(host='n0', ip='10.0.0.1', port=12000), dict(host='n1', ip='10.0.0.2', port=12001)]
    for node in (0, 1):
        command = runner.phase_command(LINKS, LINKS['phases'][0], peers, node, Path('/leaf'))
        assert command[command.index('--nnodes') + 1] == '1'
        assert command[command.index('--nproc-per-node') + 1] == '8'
        assert command[command.index('--master-addr') + 1] == peers[node]['ip']
        command = runner.phase_command(LINKS, LINKS['phases'][2], peers, node, Path('/leaf'))
        assert command[command.index('--nnodes') + 1] == '2'
        assert command[command.index('--nproc-per-node') + 1] == '1'
        assert command[command.index('--node-rank') + 1] == str(node)
        assert command[command.index('--master-addr') + 1] == peers[0]['ip']
    assert LINKS['phases'][2]['gpu_indices'] == [[7], [0]]


def test_phase_failure_continues_and_independent_node_survives(tmp_path, monkeypatch):
    config = copy.deepcopy(DEPTH)
    first = config['phases'][0]
    second = copy.deepcopy(first)
    second['name'] = 'next'
    first['independent_nodes'] = True
    config['phases'] = [first, second]
    node = tmp_path / 'nodes/n0'
    node.mkdir(parents=True)
    def write(path, value):
        path.write_text(json.dumps(value))
    helpers = SimpleNamespace(write_json=write, interface_of=lambda ip: 'eth0')
    barriers = []
    def barrier(output, name, peers, seconds):
        barriers.append(name)
        return [{'ok': True}, {'ok': False}] if name.endswith('_prepared') else [{}]
    monkeypatch.setattr(runner, 'await_phase', barrier)
    calls = []
    def child(*args):
        calls.append(args[0])
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr(runner, 'run_child', child)
    rows = runner.run_phases(config, tmp_path, node, [dict(host='n0', ip='127.0.0.1', port=1)], 0, helpers)
    assert len(rows) == 2 and len(calls) == 1
    assert 'TimeoutExpired' in rows[0]['exception']
    assert 'skipped' in rows[1]
    assert barriers == ['F2_launch_depth_prepared', 'F2_launch_depth_result', 'next_prepared', 'next_result']


def test_timeout_kills_process_group_even_when_leader_exits(monkeypatch):
    class Child:
        pid = 4321
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self, timeout):
            if timeout == 1: raise subprocess.TimeoutExpired('child', 1)
            return 0
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: Child())
    killed = []
    monkeypatch.setattr(runner.os, 'killpg', lambda *args: killed.append(args))
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_child(['child'], {}, None, 1)
    assert killed == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]


def trial(k, depth, sleep=500, censored=False):
    launches = [dict(host_us=10, start_us=i * 10, gate_pending=True) for i in range(depth)]
    if not censored:
        launches.append(dict(host_us=sleep * 900, start_us=depth * 10, gate_pending=True))
    return dict(mode='graph', kernels_per_graph=k, sleep_ms=sleep, trial=0,
                baseline_launch_us=10, actual_sleep_us=sleep * 1000, launches=launches)


def test_depth_counts_accepted_launches_and_censors_finished_sleep():
    row = trial(32, 19)
    assert blocking_depth(row)['depth'] == 19
    assert not blocking_depth(row)['censored']
    assert blocking_depth(trial(32, 19, censored=True))['censored']
    row['launches'][-1]['gate_pending'] = False
    assert blocking_depth(row)['censored']
    row['launches'][-1]['gate_pending'] = True
    row['launches'][-1]['start_us'] = 450000
    assert blocking_depth(row)['censored']
    row['launches'] = []
    assert blocking_depth(row)['depth'] == 0 and blocking_depth(row)['censored']


@pytest.mark.parametrize('capacity_kind', ['graph_launches', 'kernel_nodes'])
def test_launch_analysis_distinguishes_queue_units(capacity_kind):
    rows = [trial(k, 128 if capacity_kind == 'graph_launches' else 1024 // k) for k in (1, 2, 4, 8, 16)]
    result = analysis.analyze(rows, 64)
    assert result['status'] == 'PREDICTED'
    assert result['preferred_model'] == capacity_kind
    assert result['predicted_K'] == pytest.approx(128 if capacity_kind == 'graph_launches' else 16)
    assert result['predicted_minus_19'] == pytest.approx(109 if capacity_kind == 'graph_launches' else -3)
    assert analysis.analyze(rows[:2], 64)['status'] == 'INSUFFICIENT_EVIDENCE'
    assert analysis.analyze([trial(1, 10, censored=True)], 64)['status'] == 'INSUFFICIENT_EVIDENCE'


def synthetic_astra(n, latency, bandwidth):
    rows = []
    for op in ('all_reduce', 'sp_all_gather', 'sp_reduce_scatter', 'all_gatherv', 'reduce_scatterv'):
        for size in (1024, 8192, 1048576):
            steps = 2 * (n - 1) if op == 'all_reduce' else n - 1
            chunk = size if op in ('sp_all_gather', 'all_gatherv') else size / n
            observed = steps * (latency + chunk * 8 / 1000 / bandwidth)
            rows.append(dict(scenario=str(size), call=op, op=op, group='ep' if op.endswith('v') else 'tp',
                             group_size=n, cuda_graph=False, max_input_bytes=size, median_us=observed))
    return rows


@pytest.mark.parametrize('link,n,latency,bandwidth', [('intra', 8, 2.5, 320), ('inter', 2, 11.5, 125)])
def test_astra_fit_recovers_known_link_parameters(link, n, latency, bandwidth):
    rows = synthetic_astra(n, latency, bandwidth)
    fitted = fits.fit_astra(rows, link)
    assert fitted['latency_us'] == pytest.approx(latency)
    assert fitted['bandwidth_gbps'] == pytest.approx(bandwidth)
    assert all(abs(r['residual_us']) < 1e-8 for r in fitted['residuals'])
    assert len(fitted['curves']) == 5
    assert fitted['operation_correction'] is False
    # Operation-specific native latency differences must remain in residuals.
    rows[0]['median_us'] += 5
    assert any(abs(r['residual_us']) > 0.1 for r in fits.fit_astra(rows, link)['residuals'])


def synthetic_nixl(latency=75, bandwidth=110):
    return [dict(active_pairs=active, requests_in_flight=requests, bytes_per_reader=size * requests,
                 complete_median_us=(latency + size * 8 / 1000 / bandwidth) * requests,
                 data_verified=True, notifications_received=8, notifications_expected=8)
            for active in (1, 8) for size in (1002240, 14031360, 54120960) for requests in (1, 4, 16)]


def test_nixl_fit_recovers_known_parameters_and_refuses_bad_data():
    rows = synthetic_nixl()
    fitted = fits.fit_nixl(rows)
    assert fitted['latency_us'] == pytest.approx(75)
    assert fitted['bandwidth_gbps'] == pytest.approx(110)
    assert len(fitted['residuals']) == 9
    assert fitted['h800_baseline'] == dict(latency_us=64.7, bandwidth_gbps=126.2)
    for key, value in [('data_verified', False), ('notifications_received', 0)]:
        bad = copy.deepcopy(rows)
        bad[-1][key] = value
        with pytest.raises(ValueError): fits.fit_nixl(bad)
    with pytest.raises(ValueError, match='Unidentifiable'):
        fits.least_squares([[1, 1], [1, 1]], [5, 6])


def test_dry_run_counts_budget_and_caps():
    g5h, k5 = planner.read_timings(planner.DEFAULT_G5H), planner.read_timings([planner.DEFAULT_K5])
    for cfg, cases, gpus in [(LINKS, 628, 16), (DEPTH, 84, 1)]:
        report = planner.dry_run(cfg, g5h, k5)
        assert report['total_cases'] == cases and report['allocated_gpus'] == gpus
        assert report['gpu_hours'] == pytest.approx(report['wall_seconds'] * gpus / 3600)
        assert report['wall_seconds'] < cfg['timeout_seconds']
        for p, ph in zip(report['phases'], cfg['phases']):
            assert p['estimate']['wall_seconds'] * cfg['estimate']['safety_factor'] < ph['timeout_seconds']
    assert len(probe_cases(DEPTH['phases'][0]['config'])) == 84


def test_flat_assignments_are_supported_config_fields():
    from dataclasses import fields
    from frontier.config.config import SimulationConfig
    from frontier.config.flat_dataclass import create_flat_dataclass
    known = {f.name for f in fields(create_flat_dataclass(SimulationConfig))}
    assignments = fits.astra_fields(dict(latency_us=2, bandwidth_gbps=300),
                                    dict(latency_us=10, bandwidth_gbps=125))
    assignments.update(analytical_kv_cache_transfer_config_network_latency_ms=0.075,
                       analytical_kv_cache_transfer_config_network_bandwidth_gbps=110)
    assert set(assignments) <= known


def test_shared_runtime_checker_extension_keeps_default_command_and_locks_extra_sources(tmp_path, monkeypatch):
    runtime = load('runtime_check', B30B.parent / 'g5h/launch/runtime_check.py')
    cfg = copy.deepcopy(DEPTH)
    cfg['workspace'] = str(tmp_path)
    stage = tmp_path / 'b30b/stage.json'
    stage.parent.mkdir()
    stage.write_text(json.dumps(cfg))
    installed = tmp_path / 'installed'
    cfg['native_checkout'] = str(tmp_path / 'native')
    overlay = tmp_path / 'overlay.json'
    overlay.write_text(json.dumps(dict(installed_package=str(installed),
                                      binary_sha256={runtime.NCCL_LIBRARY: 'pinned'})))
    cfg['p2_overlay'] = str(overlay)
    paths = [tmp_path / 'frontier/profiling/collectives' / n for n in runtime.BENCHMARK_MODULES]
    paths += [tmp_path / runtime.K5_NODE_ENTRY, tmp_path / cfg['runtime_check']]
    paths += [tmp_path / n for n in cfg['lock_sources']]
    paths += [Path(cfg['native_checkout']) / n for n in runtime.VLLM_SOURCES]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('CPU hash fixture\n')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='fixture-head' if command[-1] == 'HEAD' else '')
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    lock = runtime.build_lock(cfg, stage)
    assert str(tmp_path / cfg['runtime_check']) in lock['benchmark_sources']
    assert str(tmp_path / cfg['node_entry']) in lock['benchmark_sources']
    assert all(str(tmp_path / p) in lock['benchmark_sources'] for p in cfg['lock_sources'])
    assert len(lock['benchmark_sources']) == len(set(lock['benchmark_sources']))
    assert calls[0][:3] == ['git', '-C', str(tmp_path)]
    command = runtime.worker_command(cfg, stage, tmp_path / 'lock.json')
    assert str(tmp_path / cfg['runtime_check']) in command
    assert str(tmp_path / cfg['node_entry']) in command
    plain = {k: v for k, v in cfg.items() if k not in ('runtime_check', 'node_entry')}
    assert runtime.checker_path(plain, stage) == stage.parent / 'runtime_check.py'
    assert str(tmp_path / runtime.K5_NODE_ENTRY) in runtime.worker_command(plain, stage, tmp_path / 'lock.json')


def test_fit_and_launch_analysis_clis(tmp_path):
    import csv
    def write_csv(path, rows):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    a, b, n = [tmp_path / (s + '.csv') for s in ('intra', 'inter', 'nixl')]
    write_csv(a, synthetic_astra(8, 2.5, 320))
    write_csv(b, synthetic_astra(2, 11.5, 125))
    write_csv(n, synthetic_nixl())
    for kind, args, latency in [('astra', ['--intra', str(a), '--inter', str(b)], 2.5),
                                ('nixl', ['--input', str(n)], 75)]:
        output = tmp_path / kind
        result = subprocess.run([sys.executable, str(B30B / 'fit_tools.py'), kind, *args,
                                 '--output', str(output)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        fitted = json.loads((output / 'fit.json').read_text())
        assert (fitted['intra'] if kind == 'astra' else fitted)['latency_us'] == pytest.approx(latency)
        assert all('=' in line for line in (output / 'fields.txt').read_text().splitlines())
    raw = tmp_path / 'launches.jsonl'
    raw.write_text(''.join(json.dumps(trial(k, 128)) + '\n' for k in (1, 2, 4, 8)))
    output = tmp_path / 'launch_analysis.json'
    result = subprocess.run([sys.executable, str(B30B / 'launch_analysis.py'), '--input', str(raw),
                             '--native-count', str(B30B / 'native_dummy_kernels.json'),
                             '--output', str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    fitted = json.loads(output.read_text())
    assert fitted['native_kernels'] == 2240 and fitted['predicted_K'] == pytest.approx(128)


def test_fit_can_recover_completed_raw_collective_rows(tmp_path):
    path = tmp_path / 'measurements.jsonl'
    record = dict(pass_=0, scenario='s', call='c', op='all_gatherv', group='ep', group_size=2,
                  cuda_graph=False, calls_per_forward=43, tokens_per_dp_rank=[1, 2],
                  max_input_bytes=16384, max_output_bytes=24576,
                  per_rank_call_us=[[11, 12], [13, 14]],
                  per_rank_launch_call_us=[[11, 12], [13, 14]],
                  per_rank_queue_held=[[True, True], [True, True]])
    record['pass'] = record.pop('pass_')
    path.write_text(json.dumps(record) + '\n')
    rows = fits.csv_rows([path])
    assert rows[0]['median_us'] == 13.5
    assert rows[0]['max_input_bytes'] == 16384


def test_sleep_dependent_depth_does_not_predict_k():
    rows = [trial(k, 32, sleep=500) for k in (1, 2, 4)]
    rows += [trial(k, 64, sleep=1000) for k in (1, 2, 4)]
    result = analysis.analyze(rows, 2240)
    assert result['status'] == 'SLEEP_DEPENDENT'
    assert 'predicted_K' not in result


def test_native_dummy_count_excludes_real_and_piecewise_forwards(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / 'tests/analysis'))
    module = load('v4_dummy_kernel_count', REPO / 'tests/analysis/v4_dummy_kernel_count.py')
    forward = lambda mode, real, n: SimpleNamespace(record=dict(mode=mode, real=real),
                                                   kernels_by_segment=dict(dispatch=tuple(range(n))))
    counts = module.count_dummies([forward('FULL', False, 2240), forward('FULL', False, 2240),
                                   forward('FULL', True, 100), forward('PIECEWISE', False, 55)])
    assert dict(counts) == {2240: 2}
    with pytest.raises(ValueError):
        module.count_dummies([forward('FULL', False, 0)])


def test_launch_depth_interpolation_preserves_model_ambiguity_and_measured_bracket():
    rows = [trial(k, depth) for k, depth in [(1, 1020), (2, 1020), (32, 1020),
                                            (64, 610), (1024, 40), (2048, 20), (4096, 10)]]
    result = analysis.analyze(rows, 2240)
    assert result["status"] == "AMBIGUOUS"
    assert result["predicted_K"] is None
    local = result["local_interpolation"]
    assert local["native"]["depth"] == pytest.approx(40960 / 2240)
    assert local["native"]["complete_forwards"] == 18
    assert [p["kernels_per_graph"] for p in local["native"]["bracket"]] == [2048, 4096]
    heldout = next(p for p in local["heldout"] if p["kernels_per_graph"] == 2048)
    assert heldout["relative_error"] == pytest.approx(0)
    assert analysis.interpolate_depth(result["points"], 8192) is None
    assert analysis.interpolate_depth(result["points"], 2048)["complete_forwards"] == 20
