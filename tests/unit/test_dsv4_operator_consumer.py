"""B16b: committed H200 rows, trained stage pricing, gaps and cache parity on CPU."""

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.config import (
    ClusterConfig, MetricsConfig, RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig, VllmV1SchedulerConfig, global_vars,
)
from frontier.entities import Batch, Request
from frontier.entities.batch import DecodeCudaGraphMetadata
from frontier.execution_time_predictor.deepseek_v4 import (
    component_ledger, serving_context, source_sp_token_widths, timing_operator_times,
)
from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase
from frontier.execution_time_predictor.dsv4_operator_predictor import (
    GATE_OPERATORS, GROUPED_GEMM, load_dsv4_operator_dataframe, load_dsv4_moe_dataframe,
)
from frontier.execution_time_predictor.dsv4_runtime import (
    V4_SEGMENT_OWNERS, add_dsv4_operator_gaps, compose_trained_attention, operator_segment_class,
)
from frontier.execution_time_predictor.kernel_gap import load_kernel_gap_ms
from frontier.execution_time_predictor.random_forrest_execution_time_predictor import RandomForrestExecutionTimePredictor
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import SklearnMoEExecutionTimePredictor
from frontier.moe_ep_workload import materialize_balanced_ep_workload
from frontier.operators.families import COMM_FAMILY
from frontier.profiling.linear_op.deepseek_v4_plan import build_operator_plans
from frontier.types import ClusterType, MeasurementType
from test_dsv4_source_execution import RecordingCCBackend

TABLES = Path(__file__).resolve().parents[2] / 'data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash'
HOST_FIELDS = ('schedule_time', 'sampler_e2e_time', 'prepare_inputs_e2e_time',
               'process_model_outputs_time', 'ray_comm_time')


def replica():
    r = ReplicaConfig(
        model_name='deepseek-ai/DeepSeek-V4-Flash', device='h200', network_device='h200_dgx',
        attn_tensor_parallel_size=8, attn_dp=2, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=16, total_expert_num=256, router_topk=6,
        attention_backend='FLASHMLA_SPARSE_DSV4', kv_cache_dtype='fp8',
        indexer_kv_cache_dtype='fp8', moe_routing_distribution_type='balanced',
    )
    r.cluster_num_replicas = 1
    return r


def config():
    return RandomForrestExecutionTimePredictorConfig(
        enable_dummy_mode=False, skip_cpu_overhead_modeling=True,
        k_fold_cv_splits=2, num_training_job_threads=1, num_estimators=[1],
        max_depth=[2], min_samples_split=[2],
    )


def construct(root, *, manager=None, predictor_config=None):
    return RandomForrestExecutionTimePredictor(
        predictor_config or config(), replica(), VllmV1SchedulerConfig(block_size=256),
        MetricsConfig(cache_dir=str(root / 'cache'), output_dir=str(root / 'metrics')),
        cluster_type=ClusterType.MONOLITHIC, model_manager=manager,
        cc_backend=RecordingCCBackend(),
    )


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp('v4_models')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(global_vars, 'KV_CONNECTOR', 'none')
        patch.setattr(global_vars, 'DECODE_CUDA_GRAPH_MODE', 'full_and_piecewise')

        def forbidden(_self):
            raise AssertionError('non-dummy V4 read _dummy_execution_time')

        patch.setattr(SklearnMoEExecutionTimePredictor, '_dummy_execution_time',
                      property(forbidden, lambda _self, _value: None), raising=False)
        yield construct(root), root


def batch(mode):
    if mode == 'NONE':
        b = Batch(0, [Request(0, 6144, 4)], [6144], is_moe=True,
                  num_context_tokens=[0], request_is_decoding=[False])
    else:
        b = Batch(0, [Request(0, 6144, 4) for _ in range(16)], [1] * 16,
                  is_moe=True, num_context_tokens=[6144] * 16, request_is_decoding=[True] * 16)
        b.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            'full_and_piecewise', mode, True, False, 16, 16, 16, 16,
        )
    b._cluster_type = ClusterType.MONOLITHIC
    return b


def test_every_committed_operator_row_admitted_and_plan_unchanged(trained):
    p, _ = trained
    linear = load_dsv4_operator_dataframe(TABLES / 'linear_op_kernel_only.csv', p._model_config, p._replica_config)
    moe = load_dsv4_moe_dataframe(TABLES / 'moe_kernel_only.csv', p._model_config, p._replica_config)
    assert len(pd.read_csv(TABLES / 'linear_op_kernel_only.csv')) == 41
    assert len(linear) == 80 and len(moe) == 33
    grid = dict(attention_tp_size=8, sp_local_rows=[1, 2, 3, 4, 768],
                descriptor_rows=[1, 2, 4, 8, 16, 24, 6144], logits_rows=list(range(1, 25)))
    plans = build_operator_plans(p._model_config, grid)
    # Golden serialized producer plans recorded before the TP selector move.
    assert hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest() == '184bdcfc08f720b25b60a35a14aaa875691a0156329824513dfff3f36f919833'


@pytest.mark.parametrize('column,value', [('selected_tensor_parallel_size', 2), ('tensor_parallel_mode', 'attention_tp')])
def test_non_layer_tp_contract_mismatch_names_operator(trained, tmp_path, column, value):
    p, _ = trained
    rows = pd.read_csv(TABLES / 'linear_op_kernel_only.csv', float_precision='round_trip')
    contracts = json.loads(rows.loc[0, 'typed_operator_contracts'])
    name = 'dsv4_mhc_first_pre'
    contracts[name][column] = value
    rows.loc[0, 'typed_operator_contracts'] = json.dumps(contracts)
    path = tmp_path / 'invalid.csv'
    rows.to_csv(path, index=False)
    with pytest.raises(DeepSeekV4CoverageError, match=name):
        load_dsv4_operator_dataframe(path, p._model_config, p._replica_config)


@pytest.mark.parametrize('loader,filename', [(load_dsv4_operator_dataframe, 'linear_op_kernel_only.csv'),
                                           (load_dsv4_moe_dataframe, 'moe_kernel_only.csv')])
def test_loader_missing_file_preserves_cause(trained, tmp_path, loader, filename):
    p, _ = trained
    with pytest.raises(DeepSeekV4CoverageError) as error:
        loader(tmp_path / filename, p._model_config, p._replica_config)
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_all_exact_operator_and_moe_coordinates_and_duplicate_gate_mean(trained):
    p, _ = trained
    runtime = p._predictions['dsv4_operators']
    linear = load_dsv4_operator_dataframe(TABLES / 'linear_op_kernel_only.csv', p._model_config, p._replica_config)
    for _, row in linear.iterrows():
        result = runtime.operator(row.operator, int(row.num_tokens), row.selected_tensor_parallel_size)
        assert result.time_ms == row[f'time_stats.{row.operator}.median']
        assert result.kernel_count == row[f'time_stats.{row.operator}.kernel_count']
    moe = load_dsv4_moe_dataframe(TABLES / 'moe_kernel_only.csv', p._model_config, p._replica_config)
    for local, rows in moe.groupby('gating_num_tokens'):
        for name in GATE_OPERATORS:
            result = runtime.gate(name, int(local))
            assert result.time_ms == pytest.approx(rows[f'time_stats.{name}.median'].mean(), rel=1e-14)
            assert result.kernel_count == rows[f'time_stats.{name}.kernel_count'].mean()
    for (ep, population), rows in moe.groupby(['expert_parallel_size', 'num_tokens']):
        work = materialize_balanced_ep_workload(routing_token_count=int(population), router_topk=6,
                                               total_expert_num=256, moe_expert_parallel_size=int(ep))
        for rank in range(int(ep)):
            result = runtime.grouped_gemm(work.lane(rank))
            matching = rows[rows.local_routed_rows == work.per_ep_routed_tokens[rank]]
            assert not matching.empty
            assert result.time_ms == pytest.approx(matching[f'time_stats.{GROUPED_GEMM}.median'].mean(), rel=1e-14)
            assert result.kernel_count == matching[f'time_stats.{GROUPED_GEMM}.kernel_count'].mean()
    with pytest.raises(DeepSeekV4CoverageError, match='dsv4_mhc_first_pre.*5'):
        runtime.operator('dsv4_mhc_first_pre', 5, 1)
    work = materialize_balanced_ep_workload(routing_token_count=80, router_topk=6,
                                           total_expert_num=256, moe_expert_parallel_size=16)
    with pytest.raises(DeepSeekV4CoverageError, match='moe_grouped_gemm.*80'):
        runtime.grouped_gemm(work.lane(0))
    # Same total is insufficient: a changed expert vector must not alias a measured class.
    lane = materialize_balanced_ep_workload(routing_token_count=32, router_topk=6,
                                           total_expert_num=256, moe_expert_parallel_size=16).lane(0)
    changed = list(lane.global_token_counts)
    changed[0] += 1
    changed[1] -= 1
    with pytest.raises(DeepSeekV4CoverageError, match='moe_grouped_gemm.*balanced'):
        runtime.grouped_gemm(replace(lane, global_token_counts=tuple(changed), local_token_counts=tuple(changed[:16])))


@pytest.mark.parametrize('mode', ['FULL', 'NONE'])
def test_a1_every_layer_and_endpoints_through_stage_and_ep_lanes(trained, mode):
    p, root = trained
    b = batch(mode)
    attention = p.predict_stage_execution_time(b, 0, ClusterType.MONOLITHIC,
                                               num_layers=43, include_ffn=False)
    widths = source_sp_token_widths(p, {0: b, 1: b})
    work = materialize_balanced_ep_workload(routing_token_count=widths.gathered_tokens,
                                           router_topk=6, total_expert_num=256, moe_expert_parallel_size=16)
    lane_layers = []
    for layer_id in range(43):
        lane_layers.append(p._predict_layer_with_kernel_gap(
            p._predict_dsv4_structural_execution, b, layer_id, 0,
            lane_workload=work.lane(0), sp_token_widths=widths,
            include_attention=False, include_ffn=True, **serving_context(p, b, layer_id),
        ))
    ffn = p._assemble_stage(lane_layers, first_layer_id=0)
    assert p._dsv4_step_costs is None
    assert len(attention.layer_execution_times) == len(ffn.layer_execution_times) == 43
    assert all(getattr(stage, field) == 0 for stage in (attention, ffn) for field in HOST_FIELDS)
    names = set().union(*(layer.op_times for stage in (attention, ffn) for layer in stage.layer_execution_times))
    assert {'dsv4_embedding', 'dsv4_mhc_first_pre', 'dsv4_mhc_final_post', 'dsv4_mhc_head',
            'dsv4_final_norm', 'dsv4_lm_head', 'dsv4_logits_gather', GROUPED_GEMM} <= names
    assert 'moe_shuffling' not in names
    assert not any(x.has_kernel_gap for stage in (attention, ffn) for x in stage.layer_execution_times)
    records = {'mode': mode, 'attention': [component_ledger(x) for x in attention.layer_execution_times],
               'ffn': [component_ledger(x) for x in ffn.layer_execution_times],
               'attention_elapsed_ms': sum(x.get_single_layer_attention_scope_time() for x in attention.layer_execution_times),
               'ffn_phase_ms': {phase: sum(x._moe_phase_time(phase) for x in ffn.layer_execution_times)
                                for phase in ('pre_dispatch', 'dispatch', 'routed_compute', 'combine', 'post_combine')}}
    (root / f'a1_{mode}.json').write_text(json.dumps(records, indent=2))


@pytest.mark.parametrize('mode', ['FULL', 'PIECEWISE', 'NONE'])
@pytest.mark.parametrize('rows', [2, 5])
def test_sentinel_composition_and_every_operator_segment_gap(mode, rows):
    names = ('dsv4_fused_qkv_projection', 'dsv4_compressor_projection', 'dsv4_index_weights_projection',
             'dsv4_index_compressor_projection', 'dsv4_q_projection', 'dsv4_compressor_save_partial_states',
             'dsv4_sparse_attention', 'dsv4_output_projection')
    costs = {name: float(i + 1) for i, name in enumerate(names)}
    counts = {name: 3 for name in V4_SEGMENT_OWNERS}
    all_costs = {name: 2.0 for name in V4_SEGMENT_OWNERS if name != 'dsv4_index_short_context_fill'}
    all_costs.update(costs)
    inputs = dict(variant='csa', graph_runtime_mode=mode, rows=rows,
                  multi_stream_gemm_token_threshold=3, indexer_short_context=False)
    layer = SimpleNamespace(op_times=all_costs, _dsv4_scope_gap_ms={name: -.1 for name in costs},
                            _dsv4_kernel_counts=counts, _dsv4_composition_inputs=inputs)
    p = SimpleNamespace(_kernel_gap_ms=dict(cuda_graph=.01, cuda_graph_piecewise=.02,
                                            eager=.03, eager_preparation=.04))
    b = SimpleNamespace(get_decode_cuda_graph_runtime_mode=lambda: mode)
    compose_names = {name for _, name, _ in timing_operator_times(layer, 'attention')}
    expected_costs = {name: all_costs[name] for name in compose_names}
    comm = {op.name: op.kernel_count for op in COMM_FAMILY.operators}
    expected_gaps = {name: (-.1 if name in costs else (comm.get(name, 3) * p._kernel_gap_ms[operator_segment_class(mode, name)]))
                     for name in compose_names}
    compose_trained_attention(p, layer, b)
    assert layer._dsv4_attention_schedule == compose_attention_phase(expected_costs, expected_gaps, **inputs)
    layer.op_times['dsv4_index_short_context_fill'] = 2.0
    add_dsv4_operator_gaps(p, layer, b)
    for name, gap in layer._dsv4_operator_gap_ms.items():
        assert gap == comm.get(name, 3) * p._kernel_gap_ms[operator_segment_class(mode, name)]
    assert layer._dsv4_operator_gap_ms['dsv4_lm_head'] == 3 * .04
    assert layer._dsv4_operator_gap_ms['dsv4_logits_gather'] == comm['dsv4_logits_gather'] * .04


def test_host_and_missing_gap_errors(trained, tmp_path):
    p, _ = trained
    with pytest.raises(DeepSeekV4CoverageError, match='skip_cpu_overhead_modeling=True'):
        construct(tmp_path, predictor_config=replace(config(), skip_cpu_overhead_modeling=False))
    with pytest.raises(DeepSeekV4CoverageError, match='kernel_gap.csv'):
        load_kernel_gap_ms(config(), str(tmp_path / 'missing.csv'), p._replica_config, sys_arch='co-location')


def test_runner_zeroing_is_typed_coverage(trained):
    p, _ = trained
    b = batch('NONE')
    b.cache_block_zeroing = SimpleNamespace(block_ids=tuple(range(1, 34)), physical_block_size_bytes=1, physical_reservation_extent_bytes=33)
    with pytest.raises(DeepSeekV4CoverageError, match='kv_cache_block_zero'):
        p.predict_stage_execution_time(b, 0, ClusterType.MONOLITHIC, include_ffn=False)


def test_both_training_paths_and_warm_cache_exact_identity(trained, tmp_path):
    p, root = trained
    before = {path.name for path in (root / 'cache').iterdir()}
    warm = construct(root)
    assert before == {path.name for path in (root / 'cache').iterdir()}
    cluster = ClusterConfig(replica_config=replica(), execution_time_predictor_config=config(),
                            replica_scheduler_config=VllmV1SchedulerConfig(block_size=256))
    manager = ExecutionTimePredictionModelManager({ClusterType.MONOLITHIC: cluster},
                                                  MetricsConfig(cache_dir=str(tmp_path / 'cache')))
    shared = construct(tmp_path, manager=manager)
    assert p._models.keys() == warm._models.keys() == shared._models.keys()
    assert all(name.startswith(('dsv4_attention__', 'dsv4_operator__')) for name in shared._models)
    for name, model in p._models.items():
        for other in (warm._models[name], shared._models[name]):
            assert model._frontier_exact_lookup == other._frontier_exact_lookup
            counts = getattr(model, '_frontier_kernel_count_model', None)
            if counts is not None:
                assert counts._frontier_exact_lookup == other._frontier_kernel_count_model._frontier_exact_lookup
    for mode in ('FULL', 'NONE'):
        b = batch(mode)
        for owner in (warm, shared):
            a = p.predict_stage_execution_time(b, 0, ClusterType.MONOLITHIC, num_layers=43, include_ffn=False)
            result = owner.predict_stage_execution_time(b, 0, ClusterType.MONOLITHIC, num_layers=43, include_ffn=False)
            assert [component_ledger(x) for x in a.layer_execution_times] == [component_ledger(x) for x in result.layer_execution_times]
    (root / 'model_names.json').write_text(json.dumps(sorted(p._models), indent=2))
    assert all(manager._get_measurement_types_for_cluster(role, replica()) == [MeasurementType.KERNEL_ONLY]
               for role in ClusterType)


def test_ep_wave_uses_the_segment_gap_wrapper_for_every_lane(trained, monkeypatch):
    from frontier.scheduler.utils.expert_parallel import (
        get_ep_phase_times_ms, predict_ep_wave_phase_times,
    )

    p, _ = trained
    b = batch('FULL')
    sources = {0: b, 1: b}
    widths = source_sp_token_widths(p, sources)
    work = materialize_balanced_ep_workload(routing_token_count=widths.gathered_tokens,
                                           router_topk=6, total_expert_num=256, moe_expert_parallel_size=16)
    calls = []
    original = p._predict_layer_with_kernel_gap

    def wrapped(predict_layer, *args, **kwargs):
        layer = original(predict_layer, *args, **kwargs)
        calls.append((kwargs['lane_workload'].ep_id, layer))
        return layer

    monkeypatch.setattr(p, '_predict_layer_with_kernel_gap', wrapped)
    result = predict_ep_wave_phase_times(
        layer_workload=work, source_batch=b, stage_id=0, layer_id=3,
        cluster_type=ClusterType.MONOLITHIC, predictor=p,
        lane_builder=lambda **args: SimpleNamespace(lane_workload=work.lane(args['ep_id'])),
        phase_getter=get_ep_phase_times_ms, workload_logger=lambda **args: None,
        trace_identity=None, batch_id=b.id, source_batches=sources, sp_token_widths=widths,
    )
    assert [rank for rank, _ in calls] == list(range(16))
    for rank, layer in calls:
        assert layer._dsv4_operator_gap_ms[GROUPED_GEMM] == layer._dsv4_kernel_counts[GROUPED_GEMM] * p._kernel_gap_ms['cuda_graph']
        assert not layer.has_kernel_gap
        assert result.routed_compute_times_ms[rank] == layer._moe_phase_time('routed_compute')


def test_measurement_selection_is_kernel_only_for_every_role_and_mode(trained, monkeypatch):
    p, _ = trained
    for role in ClusterType:
        monkeypatch.setattr(p, '_cluster_type', role)
        for mode in ('FULL', 'PIECEWISE', 'NONE'):
            assert p._get_default_measurement_type_for_cluster() is MeasurementType.KERNEL_ONLY
            assert p._select_measurement_type_for_batch(batch(mode)) is MeasurementType.KERNEL_ONLY
            assert p._should_enable_measurement_family(MeasurementType.KERNEL_ONLY)
            assert not p._should_enable_measurement_family(MeasurementType.CUDA_EVENT)
