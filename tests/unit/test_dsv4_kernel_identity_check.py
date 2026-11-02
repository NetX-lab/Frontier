"""CPU shape coverage, launch containment, native segmentation and CLI gates."""

from collections import Counter
from dataclasses import asdict, replace
import gzip
import json

import pytest

import dsv4_kernel_identity_check as identity
import v4_runtime_census as census
from dsv4_graph_bridge_check import BridgeConfig, BridgeRun, graph_mode_flags
from frontier.profiling.attention import dsv4_attention_main as cli
from frontier.profiling.kernel_gap.dsv4_replay_gaps import ReplayRecords
from test_dsv4_attention_profiling import PLAN_ARGS

K = ('attention', (1, 1, 1), (32, 1, 1))


def _row_kernel(rows):
    return '_fused_q_kv_rmsnorm_kernel', (rows, 1, 1), (32, 1, 1)


def _candidate(kind='decode', mode='FULL', padded_rows=1, eager_rows=None, kernels=None, count=1):
    return identity.NativeCandidate(kind, mode, padded_rows, eager_rows,
                                    Counter({K: 43, _row_kernel(padded_rows): 43}) if kernels is None else kernels,
                                    count, ('decode',))


def _native_trace(root, role, kind, mode, kernels, repeat=1, step_tokens=None, with_step_range=True):
    dispatch = {'FULL': 'model_dispatch_run_fullgraph', 'PIECEWISE': 'model_dispatch_run_pw_graph',
                'NONE': 'model_forward'}[mode]
    events = []
    for index in range(repeat):
        base = index * 1000

        def event(name, cat, ts, dur, args=None, tid=1):
            return dict(ph='X', name=name, cat=cat, ts=base + ts, dur=dur, pid=10, tid=tid, args=args or {})

        events.append(event(f'frontier_native_metadata|{index}|attention_metadata', 'user_annotation', 20, 10))
        events.append(event(dispatch, 'user_annotation', 40, 200))
        events.append(event('gloo:all_reduce', 'user_annotation', 5, 10, tid=99))
        if kind != 'dummy':
            events.append(event(f'frontier_native_ownership|{index}|logits', 'user_annotation', 250, 10))
            if with_step_range:
                c, g = {'decode': (0, 1), 'prefill': (1, 0), 'first_tail_alone': (1, 0),
                        'first_tail_with_decodes': (1, 3)}[kind]
                ct, gt = (c, g) if step_tokens is None else step_tokens[index]
                events.append(event(f'execute_context_{c}({ct})_generation_{g}({gt})', 'user_annotation', 0, 400))
        for ordinal, ((name, grid, block), count) in enumerate(kernels.items()):
            correlation = index * 100 + ordinal
            events.append(event('cudaLaunchKernel', 'cuda_runtime', 50 + ordinal, 1, {'correlation': correlation}))
            for _ in range(count):
                events.append(event(name, 'kernel', 300 + ordinal, 2,
                                    {'correlation': correlation, 'stream': 7, 'grid': list(grid),
                                     'block': list(block)}, tid=7))
            events.append(event('copy', 'gpu_memcpy', 350, 1, {'correlation': correlation, 'stream': 7}, tid=7))
    path = root / role / 'dp0' / 'dp0_pp0_tp0_synthetic.pt.trace.json.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as handle:
        json.dump({'traceEvents': events}, handle)
    return path


def test_native_dispatch_parsing_deduplicates_forward_counts_and_ignores_copies(tmp_path):
    kernels = Counter({K: 4, _row_kernel(2): 43})
    path = _native_trace(tmp_path, 'decode', 'decode', 'FULL', kernels, repeat=3)
    forwards = census.dispatch_kernel_multisets('decode', path)
    assert [forward.scheduled_tokens for forward in forwards] == [1, 1, 1]
    result = identity.native_candidates([tmp_path])
    assert len(result) == 1
    assert result[0].kernels == kernels
    assert (result[0].forward_count, result[0].shape_key, result[0].roles) == (
        3, ('decode', 'FULL', 2, None), ('decode',))


def test_native_scheduled_tokens_sum_context_and_generation_without_changing_census(tmp_path):
    kernels = Counter({K: 1, _row_kernel(16): 43})
    path = _native_trace(tmp_path, 'decode', 'first_tail_with_decodes', 'PIECEWISE', kernels,
                         step_tokens=[(7, 3)])
    forward, = census.dispatch_kernel_multisets('decode', path)
    assert forward.scheduled_tokens == 10
    candidate, = identity.native_candidates([tmp_path])
    assert candidate.shape_key == ('first_tail_with_decodes', 'PIECEWISE', 16, 10)
    record, = census.forward_census('decode', path)['forwards']
    assert record['step'] == {'context_requests': 1, 'context_tokens': 7,
                              'generation_requests': 3, 'generation_tokens': 3}
    assert 'scheduled_tokens' not in record


def test_native_forward_without_step_range_has_no_scheduled_token_count(tmp_path):
    path = _native_trace(tmp_path, 'decode', 'decode', 'NONE', Counter({_row_kernel(1): 43}),
                         with_step_range=False)
    forward, = census.dispatch_kernel_multisets('decode', path)
    assert forward.scheduled_tokens is None
    assert forward.step_class == 'real_without_step_range'


@pytest.mark.parametrize('mode', ['FULL', 'PIECEWISE', 'NONE'])
def test_native_deduplication_uses_eager_rows_only_outside_full_graph(tmp_path, mode):
    _native_trace(tmp_path, 'decode', 'decode', mode, Counter({K: 1, _row_kernel(16): 43}),
                  repeat=3, step_tokens=[(0, 8), (0, 9), (0, 8)])
    result = identity.native_candidates([tmp_path])
    if mode == 'FULL':
        assert [(row.shape_key, row.forward_count) for row in result] == [(('decode', mode, 16, None), 3)]
    else:
        assert [(row.shape_key, row.forward_count) for row in result] == [
            (('decode', mode, 16, 8), 2), (('decode', mode, 16, 9), 1)]


@pytest.mark.parametrize('kernels', [Counter({K: 1}), Counter({_row_kernel(1): 1, _row_kernel(2): 1}),
    Counter({_row_kernel(1): 1, ('_fused_q_kv_rmsnorm_kernel', (1, 2, 1), K[2]): 1})])
def test_native_missing_or_nonunique_row_count_grid_raises_with_trace_and_forward(tmp_path, kernels):
    path = _native_trace(tmp_path, 'decode', 'decode', 'FULL', kernels)
    with pytest.raises(ValueError, match='forward 0: row-count kernel.*expected exactly one') as error:
        identity.native_candidates([tmp_path])
    assert str(path) in str(error.value)


@pytest.mark.parametrize('phase,batch,kind', [('prefill', 1, 'prefill'), ('decode', 8, 'decode'),
    ('first_tail', 1, 'first_tail_alone'), ('first_tail', 4, 'first_tail_with_decodes'), ('dummy', 1, 'dummy')])
def test_native_class_mapping_and_mode_match(phase, batch, kind):
    assert identity.native_step_class(phase, batch) == kind
    candidate = _candidate(kind, kernels=Counter({K: 2}))
    assert identity.match_kernels(Counter({K: 1}), kind, 'FULL', 1, None, [candidate])['status'] == 'PASS'
    for other_kind, mode, rows, eager in [(kind, 'PIECEWISE', 1, 1), ('other', 'FULL', 1, None),
                                        (kind, 'FULL', 2, None)]:
        result = identity.match_kernels(Counter({K: 1}), other_kind, mode, rows, eager, [candidate])
        assert result['status'] == 'NO_REFERENCE'
        assert result['candidate_count'] == 0
        assert result['missing_kernel_count'] is None and result['missing_kernels'] is None


def test_missing_and_grid_mismatch_report_closest_candidate_of_the_same_shape():
    other = K[0], (2, 1, 1), K[2]
    expected = Counter({K: 2, other: 1})
    candidates = [_candidate(kernels=Counter({other: 2}), count=5),
                  _candidate(kernels=Counter(), count=1),
                  _candidate(padded_rows=2, kernels=expected, count=7)]
    result = identity.match_kernels(expected, 'decode', 'FULL', 1, None, candidates)
    assert result['status'] == 'FAIL' and result['missing_kernel_count'] == 2
    assert result['candidate_count'] == 2 and result['closest_candidate_forward_count'] == 5
    assert result['missing_kernels'] == [{'name': K[0], 'grid': list(K[1]), 'block': list(K[2]), 'count': 2}]
    assert result['grid_mismatches'] == [{'name': K[0], 'block': list(K[2]), 'expected_grid': list(K[1]),
                                         'native_grids': [[2, 1, 1]], 'missing_count': 2}]


def _role_expansion_inputs():
    from test_dsv4_attention_profiling import _build_plan, CONFIG_PATH
    from frontier.profiling.attention.dsa_plan import load_hf_config
    plan = _build_plan(load_hf_config(CONFIG_PATH))
    variants = {layer.global_layer_id: layer.attention_variant for layer in plan.shape.layers}
    roles = {layer.global_layer_id: layer.plans_tile_schedule for layer in plan.shape.layers}
    counts = {('uncompressed', True): 2, ('csa', True): 1, ('csa', False): 20,
              ('hca', True): 1, ('hca', False): 19}
    # Distinct multisets make every full-model expansion factor observable.
    signatures = {role: (role[0] + str(role[1]), K[1], K[2]) for role in counts}
    kernels = {layer: Counter({signatures[(variants[layer], roles[layer])]: 1}) for layer in variants}
    return kernels, variants, roles, counts, signatures


def test_planner_reuse_full_model_expansion_with_several_layers_per_role():
    kernels, variants, roles, counts, signatures = _role_expansion_inputs()
    assert identity.expand_attention_kernels(kernels, variants, roles, counts) == {
        signatures[role]: count for role, count in counts.items()}


@pytest.mark.parametrize('layer', [1, 6, 7])
def test_unequal_multisets_within_role_raise(layer):
    kernels, variants, roles, counts, _ = _role_expansion_inputs()
    kernels[layer][K] += 1
    with pytest.raises(ValueError, match='role.*unequal kernel multisets.*layers'):
        identity.expand_attention_kernels(kernels, variants, roles, counts)


def test_missing_full_model_role_raises():
    kernels, variants, roles, counts, _ = _role_expansion_inputs()
    del variants[2]
    with pytest.raises(ValueError, match="no reduced layers.*csa.*True"):
        identity.expand_attention_kernels(kernels, variants, roles, counts)


def test_reduced_role_absent_from_full_model_raises():
    kernels, variants, roles, counts, _ = _role_expansion_inputs()
    roles[0] = False
    with pytest.raises(ValueError, match="roles absent from the full model.*uncompressed.*False"):
        identity.expand_attention_kernels(kernels, variants, roles, counts)


def test_missing_kernels_in_any_reduced_layer_raise():
    kernels, variants, roles, counts, _ = _role_expansion_inputs()
    del kernels[10]
    with pytest.raises(ValueError, match="role.*csa.*False.*no attention kernels.*layer 10"):
        identity.expand_attention_kernels(kernels, variants, roles, counts)


def _pilot(tmp_path, phase='decode', mode='FULL', padded_rows=1, requests=None):
    graph_mode = 'FULL_AND_PIECEWISE'
    plan_args = tuple(PLAN_ARGS[:-1])
    plan = cli.build_plan(cli.build_parser().parse_args([
        'plan', *plan_args, *graph_mode_flags(graph_mode, (1, 2, 4, 8))]))
    case = next(case for case in plan.cases if case.phase == phase and case.engine_index == 0)
    engine = {'kind': 'engine', 'engine_policy': 'graph_replay', 'engine_index': 0, 'device_name': 'fake',
        'driver_version': 'fake', 'target_tensor_parallel_size': 8,
        'runtime_versions': {'torch': 'fake', 'torch_cuda': 'fake'}, 'step_hold_us': 0.0,
        'hold_cycles_per_us': None, 'multi_stream_gemm_token_threshold': 1024,
        'cudagraph_capture': {'cudagraph_mode': graph_mode}, 'kv_cache_groups': plan.summary()['kv_cache_groups'],
        'layers': [dict(global_layer_id=layer.global_layer_id, attention_variant=layer.attention_variant,
                        compress_ratio=max(1, layer.compress_ratio)) for layer in plan.shape.layers]}
    if requests is None:
        requests = [] if phase == 'dummy' else [{'num_scheduled_tokens': 1}]
    step = dict(kind='step', engine_index=0, case_id=case.case_id, step_index=0, phase=phase,
                requests=requests, cudagraph_dispatch={'cg_mode': mode, 'num_tokens': padded_rows})
    intervals = tuple(dict(kind='interval', engine_index=0, case_id=case.case_id, step_index=0,
        global_layer_id=layer.global_layer_id, scope='dsv4_sparse_attention', activity='kernel',
        name=name, grid=list(grid), block=list(block))
        for layer in plan.shape.layers for name, grid, block in (K, _row_kernel(padded_rows)))
    record = ReplayRecords('synthetic', engine, (step,), (), intervals)
    config = BridgeConfig({}, plan_args, (), (BridgeRun(graph_mode, 0, (case.case_id,)),),
                          (1, 2, 4, 8), tmp_path / 'bridge')
    return config, record, case


def test_saved_pilot_full_expansion_and_shape_coverage(tmp_path):
    config, record, case = _pilot(tmp_path)
    assert identity.layer_role_counts(config) == {('uncompressed', True): 2, ('csa', True): 1,
        ('csa', False): 20, ('hca', True): 1, ('hca', False): 19}
    candidates = [_candidate(count=2), _candidate(kernels=Counter({K: 44, _row_kernel(1): 43}), count=3)]
    report = identity.check_identity(config, [record], candidates)
    assert report['verdict'] == 'PASS' and report['steps'][0]['expected_kernel_count'] == 86
    assert report['layer_role_counts'] == [
        {'variant': variant, 'plans_tile_schedule': planner, 'count': count}
        for (variant, planner), count in sorted(identity.layer_role_counts(config).items())]
    assert 'variant_layer_counts' not in report
    assert report['shape_coverage'] == [{
        'shape_key': ['decode', 'FULL', 1, None], 'pilot_case_ids': [case.case_id],
        'step_status_counts': {'PASS': 1, 'FAIL': 0, 'NO_REFERENCE': 0}, 'native_forward_count': 5}]
    assert report['native_candidates'][0]['shape_key'] == ['decode', 'FULL', 1, None]


def test_pilot_different_padded_rows_has_no_reference(tmp_path):
    config, record, _ = _pilot(tmp_path, padded_rows=2)
    report = identity.check_identity(config, [record], [_candidate()])
    assert report['verdict'] == report['cases'][0]['status'] == report['steps'][0]['status'] == 'NO_REFERENCE'
    assert report['steps'][0]['shape_key'] == ['decode', 'FULL', 2, None]
    assert report['shape_coverage'][0]['native_forward_count'] == 0


@pytest.mark.parametrize('mode', ['PIECEWISE', 'NONE'])
def test_pilot_different_eager_rows_has_no_reference(tmp_path, mode):
    config, record, _ = _pilot(tmp_path, mode=mode, padded_rows=16,
                              requests=[{'num_scheduled_tokens': 3}, {'num_scheduled_tokens': 5}])
    report = identity.check_identity(config, [record], [_candidate(mode=mode, padded_rows=16, eager_rows=9)])
    assert report['steps'][0]['shape_key'] == ['decode', mode, 16, 8]
    assert report['steps'][0]['status'] == 'NO_REFERENCE'
    assert report['steps'][0]['grid_mismatches'] == []


def test_pilot_full_matching_ignores_scheduled_rows(tmp_path):
    config, record, _ = _pilot(tmp_path, padded_rows=16, requests=[{'num_scheduled_tokens': 8}])
    report = identity.check_identity(config, [record], [_candidate(padded_rows=16)])
    assert report['steps'][0]['shape_key'] == ['decode', 'FULL', 16, None]
    assert report['verdict'] == 'PASS'


def test_pilot_row_count_grid_must_equal_dispatch(tmp_path):
    config, record, _ = _pilot(tmp_path)
    record = replace(record, steps=(dict(record.steps[0], cudagraph_dispatch={'cg_mode': 'FULL', 'num_tokens': 2}),))
    with pytest.raises(ValueError, match='Pilot.*case.*step 0: row-count kernel grid x 1.*num_tokens 2'):
        identity.check_identity(config, [record], [])


@pytest.mark.parametrize('ambiguous', [False, True])
def test_pilot_requires_unique_row_count_grid_even_without_a_reference(tmp_path, ambiguous):
    config, record, _ = _pilot(tmp_path)
    intervals = tuple(row for row in record.intervals if row['name'] == K[0])
    if ambiguous:
        intervals = record.intervals + tuple(dict(row, grid=[1, 2, 1]) for row in record.intervals
                                             if row['name'] == _row_kernel(1)[0])
    with pytest.raises(ValueError, match='Pilot.*step 0: row-count kernel.*expected exactly one'):
        identity.check_identity(config, [replace(record, intervals=intervals)], [])


@pytest.mark.parametrize('mode', ['FULL', 'PIECEWISE', 'NONE'])
def test_dummy_eager_rows_equal_padded_rows_for_pilot_and_native(tmp_path, monkeypatch, mode):
    # Reuse the real reduced-layer plan; dummy metadata needs no request rows.
    config, record, case = _pilot(tmp_path, mode=mode, padded_rows=4, requests=[])
    plans = identity._plans(config)
    plan = plans[config.runs[0].mode]
    dummy_case = replace(case, phase='dummy')
    monkeypatch.setattr(identity, '_plans', lambda _: {
        config.runs[0].mode: replace(plan, cases=tuple(dummy_case if row.case_id == case.case_id else row
                                                     for row in plan.cases))})
    record = replace(record, steps=(dict(record.steps[0], phase='dummy'),))
    path = _native_trace(tmp_path / 'native', 'decode', 'dummy', mode,
                         Counter({K: 43, _row_kernel(4): 43}))
    forward, = census.dispatch_kernel_multisets('decode', path)
    assert forward.scheduled_tokens is None
    candidate, = identity.native_candidates([tmp_path / 'native'])
    eager_rows = None if mode == 'FULL' else 4
    assert candidate.shape_key == ('dummy', mode, 4, eager_rows)
    report = identity.check_identity(config, [record], [candidate])
    assert report['steps'][0]['shape_key'] == ['dummy', mode, 4, eager_rows]
    assert report['verdict'] == 'PASS'


def _steps_at_rows(record, rows):
    return replace(record,
        steps=tuple(dict(record.steps[0], step_index=index,
                         cudagraph_dispatch={'cg_mode': 'FULL', 'num_tokens': count})
                    for index, count in enumerate(rows)),
        intervals=tuple(dict(row, step_index=index, grid=[count, 1, 1]) if row['name'] == _row_kernel(1)[0]
                        else dict(row, step_index=index)
                        for index, count in enumerate(rows) for row in record.intervals))


@pytest.mark.parametrize('rows,native_kernels,case_status,verdict', [
    ((1, 2), Counter({K: 43, _row_kernel(1): 43}), 'NO_REFERENCE', 'PASS'),
    ((1, 2), Counter({K: 42, _row_kernel(1): 43}), 'FAIL', 'FAIL'),
    ((2, 2), Counter({K: 43, _row_kernel(1): 43}), 'NO_REFERENCE', 'NO_REFERENCE'),
    ((), Counter({K: 43, _row_kernel(1): 43}), 'PASS', 'NO_REFERENCE'),
])
def test_case_status_and_verdict_priorities(tmp_path, rows, native_kernels, case_status, verdict):
    config, record, _ = _pilot(tmp_path)
    report = identity.check_identity(config, [_steps_at_rows(record, rows)], [_candidate(kernels=native_kernels)])
    assert report['cases'][0]['status'] == case_status
    assert report['verdict'] == verdict


def test_coverage_groups_all_case_ids_and_step_status_counts(tmp_path):
    config, record, case = _pilot(tmp_path)
    plan = identity._plans(config)[config.runs[0].mode]
    # A second real decode case keeps coverage grouped across cases at the same shape.
    second_case = next(row for row in plan.cases_for_engine(0) if row.phase == 'decode' and row.case_id != case.case_id)
    second_id = second_case.case_id
    record = _steps_at_rows(record, (1, 2))
    record = replace(record, steps=record.steps + tuple(dict(row, case_id=second_id) for row in record.steps),
                     intervals=record.intervals + tuple(dict(row, case_id=second_id) for row in record.intervals))
    config = replace(config, runs=(replace(config.runs[0], case_ids=(case.case_id, second_id)),))
    report = identity.check_identity(config, [record], [_candidate(count=4)])
    assert report['step_status_counts'] == {'PASS': 2, 'FAIL': 0, 'NO_REFERENCE': 2}
    assert report['shape_coverage'] == [
        {'shape_key': ['decode', 'FULL', 1, None], 'pilot_case_ids': sorted([case.case_id, second_id]),
         'step_status_counts': {'PASS': 2, 'FAIL': 0, 'NO_REFERENCE': 0}, 'native_forward_count': 4},
        {'shape_key': ['decode', 'FULL', 2, None], 'pilot_case_ids': sorted([case.case_id, second_id]),
         'step_status_counts': {'PASS': 0, 'FAIL': 0, 'NO_REFERENCE': 2}, 'native_forward_count': 0}]


@pytest.mark.parametrize('count,rows,status,exit_code', [(43, 1, 'PASS', 0), (42, 1, 'FAIL', 1),
                                                        (43, 2, 'NO_REFERENCE', 1)])
def test_identity_cli_reads_raw_and_native_traces_and_returns_gate(tmp_path, capsys, count, rows, status, exit_code):
    config, record, _ = _pilot(tmp_path)
    raw = config.output_root / 'FULL_AND_PIECEWISE/raw/dsv4_raw_e0.jsonl'
    raw.parent.mkdir(parents=True)
    raw.write_text(''.join(json.dumps(row) + '\n' for row in (record.engine, *record.steps, *record.intervals)))
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(dict(asdict(config), output_root=str(config.output_root))))
    native = tmp_path / 'native'
    _native_trace(native, 'decode', 'decode', 'FULL', Counter({K: count, _row_kernel(rows): 43}))
    output = tmp_path / 'report.json'
    args = ['--config', str(config_path), '--native_trace_dirs', str(native), '--output', str(output)]
    assert identity.main(args) == exit_code
    report = json.loads(output.read_text())
    assert report['verdict'] == report['steps'][0]['status'] == status
    assert report['steps'][0]['missing_kernel_count'] == (None if status == 'NO_REFERENCE' else max(0, 43 - count))
    summary = capsys.readouterr().out
    assert f'[dsv4_identity] {status}: 1 steps' in summary
    assert 'referenced shapes' in summary and 'unreferenced shapes' in summary


def test_identity_cli_invalid_native_grid_returns_failure_report(tmp_path):
    config, record, _ = _pilot(tmp_path)
    raw = config.output_root / 'FULL_AND_PIECEWISE/raw/dsv4_raw_e0.jsonl'
    raw.parent.mkdir(parents=True)
    raw.write_text(''.join(json.dumps(row) + '\n' for row in (record.engine, *record.steps, *record.intervals)))
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(dict(asdict(config), output_root=str(config.output_root))))
    _native_trace(tmp_path / 'native', 'decode', 'decode', 'FULL', Counter({K: 43}))
    output = tmp_path / 'report.json'
    assert identity.main(['--config', str(config_path), '--native_trace_dirs', str(tmp_path / 'native'),
                          '--output', str(output)]) == 1
    report = json.loads(output.read_text())
    assert report['verdict'] == 'FAIL' and report['shape_coverage'] == []
    assert 'forward 0' in report['reasons'][0] and 'row-count kernel' in report['reasons'][0]
