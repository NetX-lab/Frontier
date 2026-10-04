"""KV extract of vLLM's P2P NCCL connector in PDD prefill steps (ticket 33 decision T33-G14)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import frontier.config  # noqa: F401  (imports the operator registry in dependency order)
from frontier.attention.families import DENSE_ATTENTION_FAMILY, DENSE_ATTENTION_KV_CACHE_EXTRACT
from frontier.config import AnalyticalKVCacheTransferConfig, SimulationConfig, global_vars
from frontier.entities import StageExecutionTime
from frontier.entities.time_components import AttentionTime
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from tests.unit.test_kernel_gap_pricing import _CountModel, _FamilyTrainers, _Pricing, _priced_model
from tests.unit.test_stage_finalized_contract import _layer
from tests.unit.test_stage_reporting_contract import _isolated_simulation_globals, _store  # noqa: F401

EXTRACT = DENSE_ATTENTION_KV_CACHE_EXTRACT.name


def test_kv_connector_accepts_none_and_p2p_nccl_only():
    assert AnalyticalKVCacheTransferConfig().kv_connector == "none"
    assert AnalyticalKVCacheTransferConfig(kv_connector="p2p_nccl").kv_connector == "p2p_nccl"
    with pytest.raises(ValueError, match="kv_connector must be one of"):
        AnalyticalKVCacheTransferConfig(kv_connector="mooncake")


def test_a_kv_connector_outside_pdd_fails_fast():
    with pytest.raises(ValueError, match="prices the KV extract of PDD prefill steps; sys_arch='co-location'"):
        SimulationConfig(
            sys_arch="co-location",
            kv_cache_transfer_config=AnalyticalKVCacheTransferConfig(kv_connector="p2p_nccl"),
        )


def test_the_extract_is_a_dense_attention_operator_outside_every_trace_and_table():
    assert DENSE_ATTENTION_KV_CACHE_EXTRACT in DENSE_ATTENTION_FAMILY.operators
    for ops in (DENSE_ATTENTION_FAMILY.profiling_ops(), DENSE_ATTENTION_FAMILY.predictor_ops(),
                DENSE_ATTENTION_FAMILY.e2e_trace_ops()):
        assert DENSE_ATTENTION_KV_CACHE_EXTRACT not in ops


class _ExtractPricing(_Pricing):
    _get_attention_kv_cache_extract_execution_time = (
        SklearnExecutionTimePredictor._get_attention_kv_cache_extract_execution_time
    )
    _block_size = 16
    _attention_input_file = "attention_kernel_only.csv"

    def __init__(self, predictions=None):
        super().__init__()
        self._predictions = {EXTRACT: {"_on_demand_prediction": True}} if predictions is None else predictions
        self.model = _priced_model(EXTRACT, _CountModel(predicted=1.0))
        self.lookups = []

    def _get_on_demand_prediction(self, model_name, features):
        key = (float(features["num_blocks"]),)
        self.lookups.append(key)
        self._record_kernel_count(model_name, self.model, key, ["num_blocks"])
        return 0.01 * features["num_blocks"]


def _request(num_prefill_tokens, num_processed_tokens=0):
    return SimpleNamespace(num_prefill_tokens=num_prefill_tokens, num_processed_tokens=num_processed_tokens)


def test_each_prompt_the_step_completes_pays_one_extract_over_all_its_blocks():
    pricing = _ExtractPricing()
    batch = SimpleNamespace(
        requests=[_request(2048), _request(4096), _request(1000, num_processed_tokens=512)],
        num_tokens=[2048, 1024, 488],
    )

    with pricing._counting_kernels() as counts:
        extract_time = pricing._get_attention_kv_cache_extract_execution_time(batch)

    # The 4096-token prompt is still mid-prefill; the 1000-token prompt's last
    # chunk gathers all 63 of its blocks.
    assert pricing.lookups == [(128.0,), (63.0,)]
    assert extract_time == pytest.approx(0.01 * (128 + 63))
    assert counts.operators == {EXTRACT: 2.0}
    assert pricing._kernel_count_scopes == ()


def test_an_attention_table_without_extract_rows_fails_fast():
    with pytest.raises(ValueError, match="rows of attention_kernel_only.csv, which has none"):
        _ExtractPricing(predictions={})._get_attention_kv_cache_extract_execution_time(
            SimpleNamespace(requests=[_request(16)], num_tokens=[16])
        )


class _ExtractTrainers(_FamilyTrainers):
    _measurement_family_name = staticmethod(SklearnExecutionTimePredictor._measurement_family_name)

    def _fit_single_model(self, model_name, df, feature_cols, target_col, **kwargs):
        self.rows = getattr(self, "rows", {})
        self.rows[model_name] = df[[*feature_cols, target_col]].values.tolist()
        return super()._fit_single_model(model_name, df, feature_cols, target_col, **kwargs)


def test_the_extract_trains_on_one_request_rows_by_block_count():
    target = f"time_stats.{EXTRACT}.median"
    rows = pd.DataFrame({
        "batch_size": [1, 1, 2, 1],
        "kv_cache_size": [0, 0, 0, 0],
        "prefill_chunk_size": [2048, 40, 64, 96],
        target: [0.02, 0.002, 0.004, np.nan],
        f"time_stats.{EXTRACT}.kernel_count": [1, 1, 1, np.nan],
    })
    trainers = _ExtractTrainers()

    models = trainers._train_kv_cache_extract_model(rows, 16, None, {})

    assert list(models) == [EXTRACT]
    assert trainers.rows[EXTRACT] == [[128.0, 0.02], [3.0, 0.002]]
    assert trainers.fitted == [EXTRACT, f"{EXTRACT}_kernel_count"]
    assert trainers._train_kv_cache_extract_model(rows.drop(columns=[target]), 16, None, {}) == {}


def test_the_extract_joins_the_layer_attention_time():
    attention = AttentionTime(attention_prefill_execution_time=2.0, attention_kv_cache_extract_execution_time=0.5)
    layer = _layer(7, attention_prefill_execution_time=2.0, attention_kv_cache_extract_execution_time=0.5)

    assert attention.total_time() == 2.5
    assert layer.attention_kv_cache_extract_execution_time == 0.5
    assert layer.get_single_layer_attention_time() == 2.5


def test_runs_with_a_kv_connector_report_the_extract_in_the_stage_ledger(tmp_path):
    store, _ = _store(tmp_path, enable_op_level_tracing=False, store_frontier_stage_batch_ledger=True)
    layers = [_layer(layer_id, attention_kv_cache_extract_execution_time=0.25) for layer_id in (7, 8)]
    extract_key = DENSE_ATTENTION_KV_CACHE_EXTRACT.execution_time_attr

    assert extract_key not in store._build_frontier_stage_batch_component_ledger(StageExecutionTime(layers))
    global_vars.set_kv_connector("p2p_nccl")
    assert store._build_frontier_stage_batch_component_ledger(StageExecutionTime(layers))[extract_key] == 0.5
