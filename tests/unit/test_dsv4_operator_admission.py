"""Moved isolated-reader contract, backend and target validation at load time."""

import json

import pandas as pd
import pytest

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.execution_time_predictor.dsv4_operator_predictor import (
    load_dsv4_moe_dataframe, load_dsv4_operator_dataframe,
)
from test_dsv4_operator_consumer import TABLES, replica


@pytest.mark.parametrize('column,value,match', [
    ('typed_operator_contracts', '{}', 'typed_operator_contracts'),
    ('num_tensor_parallel_workers', None, 'num_tensor_parallel_workers'),
    ('time_stats.dsv4_mhc_first_pre.median', 0, 'dsv4_mhc_first_pre'),
    ('time_stats.dsv4_mhc_first_pre.median', float('inf'), 'dsv4_mhc_first_pre'),
    ('time_stats.dsv4_mhc_first_pre.kernel_count', 0, 'dsv4_mhc_first_pre'),
    ('time_stats.dsv4_mhc_first_pre.kernel_count', 1.5, 'dsv4_mhc_first_pre'),
    ('runtime_fingerprint', 'invalid', 'runtime_fingerprint'),
])
def test_linear_loader_rejects_uncovered_identity_or_targets(tmp_path, column, value, match):
    r = replica()
    rows = pd.read_csv(TABLES / 'linear_op_kernel_only.csv', float_precision='round_trip')
    if isinstance(value, float) and not value.is_integer():
        rows[column] = rows[column].astype(float)
    rows.loc[0, column] = value
    path = tmp_path / 'linear.csv'
    rows.to_csv(path, index=False)
    with pytest.raises(DeepSeekV4CoverageError, match=match):
        load_dsv4_operator_dataframe(path, r.model_config, r)


@pytest.mark.parametrize('column,value,match', [
    ('expert_parallel_size', None, 'expert_parallel_size'),
    ('local_routed_rows', -1, 'rank-class'),
    ('moe_grouped_gemm_backend', 'unqualified', 'moe_grouped_gemm'),
    ('moe_native_backend', 'unqualified', 'moe_grouped_gemm'),
    ('moe_quantization_mode', 'bf16', 'moe_grouped_gemm'),
    ('time_stats.moe_grouped_gemm.kernel_count', 1.5, 'moe_grouped_gemm'),
])
def test_moe_loader_rejects_uncovered_rank_backend_or_targets(tmp_path, column, value, match):
    r = replica()
    rows = pd.read_csv(TABLES / 'moe_kernel_only.csv', float_precision='round_trip')
    if isinstance(value, float) and not value.is_integer():
        rows[column] = rows[column].astype(float)
    rows.loc[0, column] = value
    path = tmp_path / 'moe.csv'
    rows.to_csv(path, index=False)
    with pytest.raises(DeepSeekV4CoverageError, match=match):
        load_dsv4_moe_dataframe(path, r.model_config, r)


def test_shared_expert_keeps_layer_contract_and_qualified_linear_backend(tmp_path):
    r = replica()
    rows = pd.read_csv(TABLES / 'linear_op_kernel_only.csv', float_precision='round_trip')
    index = rows[rows.linear_backends.notna()].index[0]
    backends = json.loads(rows.loc[index, 'linear_backends'])
    backends['share_expert_up_proj'] = ['unqualified']
    rows.loc[index, 'linear_backends'] = json.dumps(backends)
    path = tmp_path / 'shared.csv'
    rows.to_csv(path, index=False)
    with pytest.raises(DeepSeekV4CoverageError, match='share_expert_up_proj.*backend'):
        load_dsv4_operator_dataframe(path, r.model_config, r)
    rows = pd.read_csv(TABLES / 'linear_op_kernel_only.csv', float_precision='round_trip')
    contracts = json.loads(rows.loc[index, 'typed_operator_contracts'])
    contracts['share_expert_up_proj']['selected_tensor_parallel_size'] = 2
    rows.loc[index, 'typed_operator_contracts'] = json.dumps(contracts)
    rows.to_csv(path, index=False)
    with pytest.raises(DeepSeekV4CoverageError, match='share_expert_up_proj'):
        load_dsv4_operator_dataframe(path, r.model_config, r)
