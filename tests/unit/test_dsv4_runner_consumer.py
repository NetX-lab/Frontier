"""B17: runner tables, exact costs, activity gaps and PD coverage on CPU."""

import json
from pathlib import Path
from dataclasses import replace

import pandas as pd
import pytest

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.entities import Batch, Request, DummyForwardBatch
from frontier.entities.batch import DecodeCudaGraphMetadata, KVBlockTableWrites, KVCacheBlockZeroing
from frontier.execution_time_predictor.deepseek_v4 import serving_context, source_sp_token_widths, timing_operator_times
from frontier.execution_time_predictor.dsv4_operator_predictor import load_dsv4_runner_dataframe, runner_model_name
from frontier.execution_time_predictor.dsv4_runner import (
    ZERO_SCOPE, WRITE_SCOPE, INPUT_SCOPE, SAMPLING_SCOPE, BLOCK_TABLE, REPLAY_TABLE,
    PREPARATION_COLUMNS, preparation_coordinate,
)
from frontier.moe_ep_workload import materialize_balanced_ep_workload
from frontier.types import ClusterType
from test_dsv4_operator_consumer import trained, config, replica, TABLES, construct
from frontier.execution_time_predictor.random_forrest_execution_time_predictor import RandomForrestExecutionTimePredictor
from frontier.config import MetricsConfig, VllmV1SchedulerConfig, global_vars
from test_dsv4_source_execution import RecordingCCBackend


def real_forward(role, *, first_tail=False):
    prefill = role is ClusterType.PREFILL
    b = 1 if prefill or first_tail else 8
    query = 6144 if prefill else 1
    batch = Batch(0, [Request(0, 6144, 16) for _ in range(b)], [query] * b,
                  is_moe=True, num_context_tokens=[0 if prefill else 6143 if first_tail else 6144] * b,
                  request_is_decoding=[not (prefill or first_tail)] * b)
    batch._cluster_type = role
    if not prefill:
        mode = "PIECEWISE" if first_tail else "FULL"
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            "full_and_piecewise", mode, True, first_tail, b, b, b, b)
    return batch


def layer(p, batch, index, *, include_ffn=True):
    widths = source_sp_token_widths(p, {0: batch, 1: batch})
    work = materialize_balanced_ep_workload(
        routing_token_count=widths.gathered_tokens, router_topk=6,
        total_expert_num=256, moe_expert_parallel_size=16)
    return p._predict_layer_with_kernel_gap(
        p._predict_dsv4_structural_execution, batch, index, 0,
        lane_workload=work.lane(0), sp_token_widths=widths,
        include_attention=True, include_ffn=include_ffn, **serving_context(p, batch, index))


def test_committed_block_coordinates_and_counts():
    rows = load_dsv4_runner_dataframe(TABLES / BLOCK_TABLE)
    zero, writes = rows[rows.operator == ZERO_SCOPE], rows[rows.operator == WRITE_SCOPE]
    assert set(map(int, zero.n)) == set(range(1, 33)) | {2520}
    assert len(zero) == 33 and len(writes) == 164
    assert set(zip(map(int, writes.w_c), map(int, writes.r))) == {
        (w, r) for w in range(33) for r in range(5) if (w, r) != (0, 0)}
    assert set(rows.kernel_count) == {1} and set(rows.memcpy_count) == {1}
    assert set(rows.memset_count) == {0} and set(rows.activity_count) == {2}
    assert {tuple(json.loads(value)) for value in writes.new_row_entries} == {(24, 96, 96, 1536, 768)}


def test_all_runner_coordinates_exact_and_duplicate_mean(trained):
    p, _ = trained
    runtime = p._predictions['dsv4_operators']
    directory = Path(p._attention_input_file).parent
    for table in (BLOCK_TABLE, REPLAY_TABLE):
        rows = load_dsv4_runner_dataframe(directory / table)
        for name, group in rows.groupby('operator'):
            columns = (PREPARATION_COLUMNS if name == INPUT_SCOPE else
                       ('n',) if name == ZERO_SCOPE else ('w_c', 'r') if name == WRITE_SCOPE else ('b',))
            for key, measured in group.groupby(list(columns), dropna=False):
                key = key if isinstance(key, tuple) else (key,)
                coordinate = dict(zip(columns, key))
                result = runtime.runner(name, coordinate)
                expected = (measured[f'kernel_stats.{name}.median'].astype(float) +
                            measured[f'memcpy_stats.{name}.median'].astype(float)).mean()
                assert result.time_ms == pytest.approx(expected, rel=1e-14)
                assert result.kernel_count == measured.activity_count.mean()
    assert runtime.runner(ZERO_SCOPE, {'n': 0}).time_ms == 0
    assert runtime.runner(WRITE_SCOPE, {'w_c': 0, 'r': 0}).kernel_count == 0


def test_runner_coverage_is_typed_and_includes_coordinate(trained, tmp_path):
    runtime = trained[0]._predictions['dsv4_operators']
    for name, coordinate in (
        (ZERO_SCOPE, {'n': 33}), (WRITE_SCOPE, {'w_c': 33, 'r': 0}),
        (SAMPLING_SCOPE, {'b': 25}),
        (INPUT_SCOPE, preparation_coordinate(real_forward(ClusterType.PREFILL)) | {'padded_rows': 6145}),
    ):
        with pytest.raises(DeepSeekV4CoverageError, match=name):
            runtime.runner(name, coordinate)
    with pytest.raises(DeepSeekV4CoverageError, match='continuing pair.*2 new entries'):
        runtime.block_table_writes(KVBlockTableWrites(((1, 'compressed', 2),), ()))
    with pytest.raises(DeepSeekV4CoverageError, match='new row.*differ'):
        runtime.block_table_writes(KVBlockTableWrites((), ((1, (('compressed', 1),)),)))
    for filename in (BLOCK_TABLE, REPLAY_TABLE):
        with pytest.raises(DeepSeekV4CoverageError, match=filename):
            load_dsv4_runner_dataframe(tmp_path / filename)


def test_runner_activity_mismatch_names_coordinate(tmp_path):
    rows = pd.read_csv(TABLES / BLOCK_TABLE, keep_default_na=False)
    duplicate = rows.iloc[[0]].copy()
    duplicate['kernel_count'], duplicate['activity_count'] = 2, 3
    target = tmp_path / BLOCK_TABLE
    pd.concat([rows, duplicate]).to_csv(target, index=False)
    with pytest.raises(DeepSeekV4CoverageError, match='activity count varies.*kv_cache_block_zero'):
        load_dsv4_runner_dataframe(target)


def test_runner_cache_identity_covers_changed_table(trained):
    p = trained[0]
    for filename in (BLOCK_TABLE, REPLAY_TABLE):
        rows = load_dsv4_runner_dataframe(Path(p._attention_input_file).parent / filename)
        row = rows.iloc[0]
        key = (runner_model_name(row.operator, row.phase, row.graph_runtime_mode)
               if row.operator == INPUT_SCOPE else runner_model_name(row.operator))
        altered = rows.copy()
        altered.loc[0, f'kernel_stats.{row.operator}.median'] = float(altered.loc[0, f'kernel_stats.{row.operator}.median']) * 1.01
        assert p._get_model_hash(key, rows) != p._get_model_hash(key, altered)


@pytest.mark.parametrize('role', [ClusterType.PREFILL, ClusterType.DECODE])
def test_pd_predictors_construct_and_price_supported_phases(trained, tmp_path, monkeypatch, role):
    monkeypatch.setattr(global_vars, 'KV_CONNECTOR', 'nixl')
    p = RandomForrestExecutionTimePredictor(
        config(), replica(), VllmV1SchedulerConfig(block_size=256),
        MetricsConfig(cache_dir=str(trained[1] / 'cache'), output_dir=str(tmp_path / 'metrics')),
        cluster_type=role, cc_backend=RecordingCCBackend())
    b = real_forward(role, first_tail=role is ClusterType.DECODE)
    first = layer(p, b, 0, include_ffn=False)
    assert first.op_times[INPUT_SCOPE] > 0
    assert 'phase' in first._dsv4_workloads[INPUT_SCOPE]
    assert first._dsv4_workloads[INPUT_SCOPE]['phase'] == ('prefill' if role is ClusterType.PREFILL else 'first_tail')
    if role is ClusterType.PREFILL:
        with pytest.raises(DeepSeekV4CoverageError, match='PREFILL sampling is unmeasured'):
            layer(p, b, 42)
    else:
        assert layer(p, b, 42).op_times[SAMPLING_SCOPE] > 0
    dummy = DummyForwardBatch(0, 0, 0)
    dummy._cluster_type = role
    dummy_first = layer(p, dummy, 0)
    dummy_last = layer(p, dummy, 42)
    assert dummy_first.op_times[INPUT_SCOPE] > 0
    assert not {ZERO_SCOPE, WRITE_SCOPE, 'dsv4_lm_head', 'dsv4_logits_gather', SAMPLING_SCOPE} & (
        dummy_first.op_times.keys() | dummy_last.op_times.keys())


def test_pdaf_roles_still_rejected(trained, monkeypatch):
    p = trained[0]
    for role in (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN):
        monkeypatch.setattr(p, '_cluster_type', role)
        with pytest.raises((ValueError, NotImplementedError), match='MONOLITHIC, PREFILL (?:or|or unified) DECODE'):
            layer(p, real_forward(ClusterType.DECODE), 0, include_ffn=False)


def test_runner_cost_and_own_gap_at_phase_boundaries(trained):
    p = trained[0]
    b = real_forward(ClusterType.MONOLITHIC)
    b.cache_block_zeroing = KVCacheBlockZeroing(tuple(range(1, 9)), 1002240)
    b.kv_block_table_writes = KVBlockTableWrites(tuple((i, 'compressed', 1) for i in range(8)), ())
    first, last = layer(p, b, 0), layer(p, b, 42)
    assert [name for _, name, _ in timing_operator_times(first, 'pre_dp')] == [ZERO_SCOPE, WRITE_SCOPE]
    assert timing_operator_times(first, 'attention')[0][1] == INPUT_SCOPE
    assert timing_operator_times(last, 'post_combine')[-2:][0][1] == 'dsv4_logits_gather'
    assert timing_operator_times(last, 'post_combine')[-1][1] == SAMPLING_SCOPE
    for record, name in ((first, ZERO_SCOPE), (first, WRITE_SCOPE), (first, INPUT_SCOPE), (last, SAMPLING_SCOPE)):
        prediction = p._predictions['dsv4_operators'].runner(name, record._dsv4_workloads[name])
        assert record.op_times[name] == prediction.time_ms
        assert record._dsv4_operator_gap_ms[name] == prediction.kernel_count * p._kernel_gap_ms['eager_preparation']
    assert first._dsv4_attention_schedule.start_ms[INPUT_SCOPE] == 0
    assert first._dsv4_attention_schedule.start_ms['dsv4_embedding'] == pytest.approx(
        first.op_times[INPUT_SCOPE] + first._dsv4_operator_gap_ms[INPUT_SCOPE])


def test_runtime_limitations_describe_operator_path():
    limitations = replica().model_config.get_model_architecture_profile().runtime_limitations
    text = ' '.join(limitations)
    assert 'trained operator execution' in text and 'PREFILL' in text and 'unified DECODE' in text
    assert 'diagnostic' in text and 'min_tokens' in text
    assert 'CPU-overhead' in text and 'PD-AF' in text


def test_committed_tables_only_constructor_names_missing_replay(tmp_path, monkeypatch):
    if (TABLES / REPLAY_TABLE).exists():
        pytest.skip("accepted replay table has been promoted")
    monkeypatch.delenv('FRONTIER_DSV4_REPLAY_TABLE', raising=False)
    monkeypatch.setattr(global_vars, 'KV_CONNECTOR', 'none')
    monkeypatch.setattr(global_vars, 'DECODE_CUDA_GRAPH_MODE', 'full_and_piecewise')
    with pytest.raises(DeepSeekV4CoverageError, match=REPLAY_TABLE):
        construct(tmp_path)


@pytest.mark.parametrize('kind', ['decode', 'first_tail', 'prefill', 'dummy_FULL', 'dummy_PIECEWISE', 'dummy_NONE'])
def test_a1_pd_all_layers_use_tables_and_explicit_prefill_sampling_limit(trained, monkeypatch, kind):
    p = trained[0]
    role = ClusterType.PREFILL if kind == 'prefill' else ClusterType.DECODE
    monkeypatch.setattr(p, '_cluster_type', role)
    if kind.startswith('dummy_'):
        b = DummyForwardBatch(0, 0, 0)
        b._cluster_type = role
        mode = kind.removeprefix('dummy_')
        if mode != 'NONE':
            b.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
                'full_and_piecewise', mode, True, False, 1, 8, 1, 8)
    else:
        b = real_forward(role, first_tail=kind == 'first_tail')
    records = []
    for index in range(p._model_config.num_layers):
        if kind == 'prefill' and index == p._model_config.num_layers - 1:
            with pytest.raises(DeepSeekV4CoverageError, match='PREFILL sampling is unmeasured'):
                layer(p, b, index)
        else:
            records.append(layer(p, b, index))
    names = [name for record in records for name in record.op_times]
    assert names.count(INPUT_SCOPE) == 1
    assert names.count(SAMPLING_SCOPE) == (0 if kind.startswith('dummy_') or kind == 'prefill' else 1)
    assert all(record._dsv4_kernel_counts for record in records)
