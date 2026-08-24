from types import SimpleNamespace

from frontier.entities.batch import AFDStageMetadata, DecodeCudaGraphMetadata
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ClusterType, MeasurementType


def test_decode_cuda_graph_activation_record_uses_runtime_metadata() -> None:
    predictor = SimpleNamespace()
    batch = SimpleNamespace(
        id=17,
        decode_cuda_graph_metadata=DecodeCudaGraphMetadata(
            config_mode="piecewise",
            runtime_mode="PIECEWISE",
            capture_hit=True,
            is_mixed_batch=True,
            original_total_tokens=5,
            padded_total_tokens=8,
            original_decode_batch_size=3,
            padded_decode_batch_size=3,
        ),
        afd_stage_metadata=None,
    )

    records = SklearnExecutionTimePredictor._build_cuda_graph_activation_records(
        predictor,
        batch,
        MeasurementType.KERNEL_ONLY,
        ClusterType.MONOLITHIC,
    )

    assert records == [
        {
            "batch_id": 17,
            "cluster_role": "MONOLITHIC",
            "config_mode": "piecewise",
            "runtime_mode": "PIECEWISE",
            "capture_hit": True,
            "capture_sizes": [8],
            "original_tokens": [5],
            "padded_tokens": [8],
            "original_decode_batch_size": 3,
            "padded_decode_batch_size": 3,
            "measurement_family": "kernel_only",
        }
    ]


def test_pdaf_cuda_graph_activation_records_role_specific_captures() -> None:
    predictor = SimpleNamespace()
    metadata = AFDStageMetadata(
        num_stages=2,
        original_total_tokens=5,
        padded_total_tokens=6,
        ffn_compute_total_tokens=8,
        original_stage_token_lens=[2, 3],
        padded_stage_token_lens=[2, 4],
        ffn_compute_stage_token_lens=[2, 4],
        attention_use_cuda_graph=True,
        attention_cudagraph_capture_sizes=[1, 2, 4, 8],
        ffn_use_cuda_graph=True,
        ffn_cudagraph_capture_sizes=[1, 2, 4, 8],
    )
    batch = SimpleNamespace(
        id=23,
        decode_cuda_graph_metadata=None,
        afd_stage_metadata=metadata,
        afd_stage_idx=0,
        afd_stage_represents_all_stages=True,
    )

    attention_records = (
        SklearnExecutionTimePredictor._build_cuda_graph_activation_records(
            predictor,
            batch,
            MeasurementType.KERNEL_ONLY,
            ClusterType.DECODE_ATTN,
        )
    )
    ffn_records = SklearnExecutionTimePredictor._build_cuda_graph_activation_records(
        predictor,
        batch,
        MeasurementType.KERNEL_ONLY,
        ClusterType.DECODE_FFN,
    )

    assert attention_records[0]["cluster_role"] == "DECODE_ATTN"
    assert attention_records[0]["capture_hit"] is True
    assert attention_records[0]["capture_sizes"] == [2, 4]
    assert attention_records[0]["original_tokens"] == [2, 3]
    assert attention_records[0]["padded_tokens"] == [2, 4]
    assert attention_records[0]["measurement_family"] == "kernel_only"

    assert ffn_records[0]["cluster_role"] == "DECODE_FFN"
    assert ffn_records[0]["capture_hit"] is True
    assert ffn_records[0]["capture_sizes"] == [8]
    assert ffn_records[0]["original_tokens"] == [6]
    assert ffn_records[0]["padded_tokens"] == [8]
    assert ffn_records[0]["measurement_family"] == "kernel_only"
