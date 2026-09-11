from types import MethodType, SimpleNamespace

import pandas as pd
import pytest

from frontier.entities.time_components import AttentionOperatorTimes, AttentionTime
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.gdn.predictor import GDNBatchFeatures, ProfiledGDNPredictor
from frontier.gdn.profiling_schema import get_required_gdn_profiling_columns
from frontier.profiling.common.model_config import ModelConfig
from frontier.types import ClusterType, MeasurementType


MODEL_NAME = "Qwen3.8-2.4T-A95B-Quark-MXFP4"


def _profile_row(
    *,
    batch_size: int,
    prefill_tokens: int,
    decode_tokens: int,
    max_query_len: int,
    has_initial_state: bool,
    core_name: str,
    core_time: float,
) -> dict:
    config = ModelConfig.from_model_name(MODEL_NAME)
    gdn_config = config.get_gdn_config()
    row = {column: 0 for column in get_required_gdn_profiling_columns()}
    row.update(
        {
            "measurement_type": "CUDA_EVENT",
            "model_architecture_profile": "qwen3_5_moe",
            "quant_signature": config.get_quant_signature(),
            "device": "mi355x",
            "runtime_stack_signature": "vllm=test;torch=test;hip=test;triton=test",
            "gdn_runtime_backend": "vllm_rocm_standard_dispatch",
            "gdn_rank_aggregation": "single_rank",
            "gdn_prefill_backend": "triton",
            "gdn_decode_backend": "packed_recurrent_triton",
            "gqa_interleaved_layout": False,
            "packed_recurrent_decode": True,
            "model_dtype": "torch.bfloat16",
            "conv_state_dtype": "bfloat16",
            "recurrent_state_dtype": "float32",
            "weight_source": "synthetic_bf16",
            "num_tensor_parallel_workers": 1,
            "hidden_size": config.embedding_dim,
            "conv_kernel_size": gdn_config.conv_kernel_size,
            "key_head_dim": gdn_config.key_head_dim,
            "value_head_dim": gdn_config.value_head_dim,
            "num_key_heads": gdn_config.num_key_heads,
            "num_value_heads": gdn_config.num_value_heads,
            "batch_size": batch_size,
            "batch_num_tokens": prefill_tokens + decode_tokens,
            "batch_num_prefill_tokens": prefill_tokens,
            "batch_num_decode_tokens": decode_tokens,
            "max_query_len": max_query_len,
            "has_initial_state": has_initial_state,
            "time_stats.gdn_input_projections.median": 0.1,
            f"time_stats.{core_name}.median": core_time,
            "time_stats.gdn_output_projection.median": 0.3,
        }
    )
    return row


@pytest.fixture
def profile_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _profile_row(
                batch_size=1,
                prefill_tokens=16,
                decode_tokens=0,
                max_query_len=16,
                has_initial_state=False,
                core_name="gdn_core_prefill",
                core_time=0.2,
            ),
            _profile_row(
                batch_size=1,
                prefill_tokens=0,
                decode_tokens=1,
                max_query_len=1,
                has_initial_state=True,
                core_name="gdn_core_decode",
                core_time=0.4,
            ),
            _profile_row(
                batch_size=2,
                prefill_tokens=16,
                decode_tokens=1,
                max_query_len=16,
                has_initial_state=True,
                core_name="gdn_core_mixed",
                core_time=0.5,
            ),
        ]
    )


def _build_predictor(dataframe: pd.DataFrame) -> ProfiledGDNPredictor:
    return ProfiledGDNPredictor(
        dataframe,
        model_config=ModelConfig.from_model_name(MODEL_NAME),
        device="mi355x",
        tensor_parallel_size=1,
        measurement_type=MeasurementType.CUDA_EVENT,
    )


def test_profiled_gdn_predictor_uses_exact_phase_specific_rows(
    profile_dataframe: pd.DataFrame,
) -> None:
    predictor = _build_predictor(profile_dataframe)

    prefill = predictor.predict_operator_times(
        GDNBatchFeatures(1, 16, 16, 0, 16, False)
    )
    decode = predictor.predict_operator_times(
        GDNBatchFeatures(1, 1, 0, 1, 1, True)
    )
    mixed = predictor.predict_operator_times(
        GDNBatchFeatures(2, 17, 16, 1, 16, True)
    )

    assert prefill == {
        "gdn_input_projections": 0.1,
        "gdn_core_prefill": 0.2,
        "gdn_output_projection": 0.3,
    }
    assert decode["gdn_core_decode"] == 0.4
    assert mixed["gdn_core_mixed"] == 0.5


def test_profiled_gdn_predictor_builds_legacy_and_structured_times_once(
    profile_dataframe: pd.DataFrame,
) -> None:
    predictor = _build_predictor(profile_dataframe)
    request = SimpleNamespace(num_processed_tokens=16, is_prefill_complete=True)
    batch = SimpleNamespace(
        num_tokens=[1],
        requests=[request],
        total_num_tokens=1,
        num_prefill_tokens=0,
        num_decode_tokens=1,
    )

    result = predictor.predict_attention_time(batch, norm_time_ms=0.05)

    assert result.attention_layer_pre_proj_execution_time == 0.1
    assert result.attention_decode_execution_time == 0.4
    assert result.attention_layer_post_proj_execution_time == 0.3
    assert result.total_time() == pytest.approx(0.85)
    assert result.operator_times.op_times["gdn_core_decode"] == 0.4


def test_profiled_gdn_predictor_rejects_mixed_runtime_stacks(
    profile_dataframe: pd.DataFrame,
) -> None:
    profile_dataframe.loc[1, "runtime_stack_signature"] = "different"

    with pytest.raises(ValueError, match="incompatible runtime contracts"):
        _build_predictor(profile_dataframe)


def test_unseen_decode_projections_cannot_learn_prefill_latency_scale():
    rows = []
    for size in (16, 32):
        for prefill in (True, False):
            row = _profile_row(
                batch_size=size, prefill_tokens=size * 1024 if prefill else 0,
                decode_tokens=0 if prefill else size, max_query_len=1024 if prefill else 1,
                has_initial_state=not prefill,
                core_name="gdn_core_prefill" if prefill else "gdn_core_decode",
                core_time=2.0 if prefill else 0.04,
            )
            row["time_stats.gdn_input_projections.median"] = 10.0 if prefill else 0.03
            row["time_stats.gdn_output_projection.median"] = 5.0 if prefill else 0.01
            rows.append(row)
    predictor = _build_predictor(pd.DataFrame(rows).fillna(0))
    result = predictor.predict_operator_times(GDNBatchFeatures(24, 24, 0, 24, 1, True))
    assert result == pytest.approx({"gdn_input_projections": 0.03,
                                    "gdn_core_decode": 0.04,
                                    "gdn_output_projection": 0.01})


def test_qwen_hybrid_stage_attention_is_exact_weighted_average() -> None:
    class ConcretePredictor(SklearnExecutionTimePredictor):
        def _get_grid_search_params(self):
            return {}

        def _get_estimator(self):
            raise AssertionError("estimator is not used by this contract test")

    predictor = object.__new__(ConcretePredictor)
    predictor._model_config = ModelConfig.from_model_name(MODEL_NAME)
    predictor._num_layers_per_pipeline_stage = 92

    def predict_layer(self, batch, layer_id, cluster_type):
        del batch, cluster_type
        if self._model_config.is_gdn_layer(layer_id):
            return AttentionTime(
                attention_layer_pre_proj_execution_time=1.0,
                attention_prefill_execution_time=2.0,
                attention_layer_post_proj_execution_time=3.0,
                operator_times=AttentionOperatorTimes(
                    {
                        "gdn_input_projections": 1.0,
                        "gdn_core_prefill": 2.0,
                        "gdn_output_projection": 3.0,
                    }
                ),
            )
        return AttentionTime(
            attention_layer_pre_proj_execution_time=5.0,
            attention_prefill_execution_time=7.0,
            attention_layer_post_proj_execution_time=11.0,
        )

    predictor.predict_attention_layer_time = MethodType(predict_layer, predictor)
    averaged = predictor._predict_stage_attention_time(
        batch=object(),
        stage_id=0,
        num_layers=92,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert averaged.operator_times is None
    assert averaged.attention_layer_pre_proj_execution_time == pytest.approx(
        (69 * 1.0 + 23 * 5.0) / 92
    )
    assert averaged.total_time() * 92 == pytest.approx(69 * 6.0 + 23 * 23.0)


def test_matching_batch_sweep_interpolates_between_measured_endpoints():
    rows = []
    for size, time in ((16, 1.0), (32, 2.0)):
        rows.append(_profile_row(batch_size=size, prefill_tokens=size * 128,
                                  decode_tokens=0, max_query_len=128,
                                  has_initial_state=False, core_name="gdn_core_prefill",
                                  core_time=time))
    predictor = _build_predictor(pd.DataFrame(rows))
    midpoint = predictor.predict_operator_times(GDNBatchFeatures(24, 24 * 128, 24 * 128, 0, 128, False))
    assert midpoint["gdn_core_prefill"] == pytest.approx(1.5)
    endpoint = predictor.predict_operator_times(GDNBatchFeatures(16, 16 * 128, 16 * 128, 0, 128, False))
    assert endpoint["gdn_core_prefill"] == 1.0
    # Outside the calibrated interval retain RF fallback, never extrapolate.
    outside = predictor.predict_operator_times(GDNBatchFeatures(64, 64 * 128, 64 * 128, 0, 128, False))
    assert 1.0 <= outside["gdn_core_prefill"] <= 2.0
