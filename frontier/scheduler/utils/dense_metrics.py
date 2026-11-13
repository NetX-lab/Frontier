"""Dense-layer completion and physical-layer timing for mixed-MoE metrics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from frontier.config.model_config import BaseModelConfig
from frontier.entities.execution_time import ExecutionTime
from frontier.entities.stage_execution_time import StageExecutionTime
from frontier.entities.time_components import CommunicationOperatorTimes
from frontier.operators.families import get_comm_operator

from frontier.scheduler.utils.collective_timing import advance_decode_layer
from frontier.scheduler.utils.execution_time_metrics import (
    build_metrics_execution_time,
    build_single_layer_metrics_execution_time,
)
from frontier.scheduler.utils.request_selection import collect_active_requests


_DENSE_FFN_COMMUNICATION_ATTRS = frozenset(
    {
        "moe_tensor_parallel_allreduce_time",
        "tensor_parallel_allgather_time",
    }
)


def _restore_structural_dsa_attention(
    layer: ExecutionTime,
    predictor: Any,
) -> ExecutionTime:
    """Use the existing dummy DSA source scope on the PDD reporting path."""
    if layer.attention_family_id != "dsa_attention":
        return layer
    if not getattr(predictor, "_enable_dummy_mode", False):
        raise NotImplementedError(
            "DSA stage metrics require the supported dummy structural path"
        )
    predict_dsa = getattr(predictor, "_predict_dsa_dummy_attention_time", None)
    if not callable(predict_dsa):
        raise ValueError("DSA predictor is missing its structural attention source")
    restored_attention = predict_dsa(layer.global_layer_id)
    restored = layer.as_single_layer(
        global_layer_id=layer.global_layer_id,
        attention_family_id=layer.attention_family_id,
        attention_variant_id=layer.attention_variant_id,
    )
    old_attention_ops = restored.attention_operator_times
    restored._attention_time = deepcopy(restored_attention)
    restored_op_times = dict(restored._op_times)
    if old_attention_ops is not None:
        for op_name in old_attention_ops.op_times:
            restored_op_times.pop(op_name, None)
    if restored_attention.operator_times is not None:
        restored_op_times.update(restored_attention.operator_times.op_times)
    restored._op_times = restored_op_times
    restored._refresh_op_time_attr_values()
    return restored.finalized_copy()


def _merge_dense_ffn_metrics_layer(
    attention_layer: ExecutionTime,
    dense_layer: ExecutionTime,
) -> ExecutionTime:
    """Add one dense FFN probe without replacing the executed attention scope.

    The attention source scope is predicted without its FFN, so it carries no
    residual adds. A routed layer's EP waves own its adds; no EP wave runs for
    a dense layer, so the probe supplies both of its residual adds.
    """
    if attention_layer.global_layer_id != dense_layer.global_layer_id:
        raise ValueError(
            "Dense FFN metrics require matching physical layer identities: "
            f"{attention_layer.global_layer_id} != {dense_layer.global_layer_id}"
        )
    if dense_layer._is_moe:
        raise ValueError("Dense FFN metrics require a dense prediction")

    merged = attention_layer.as_single_layer(
        global_layer_id=attention_layer.global_layer_id,
        attention_family_id=attention_layer.attention_family_id,
        attention_variant_id=attention_layer.attention_variant_id,
    )
    merged._moe_or_mlp_time = deepcopy(dense_layer._moe_or_mlp_time)
    merged._residual_time = dense_layer.residual_time_component

    dense_communication = dense_layer.communication_time_component
    merged_communication = merged._communication_time
    for attr_name in _DENSE_FFN_COMMUNICATION_ATTRS:
        setattr(
            merged_communication,
            attr_name,
            getattr(dense_communication, attr_name),
        )

    original_comm_ops = merged_communication.operator_times
    dense_comm_ops = dense_communication.operator_times
    communication_op_times = (
        {} if original_comm_ops is None else dict(original_comm_ops.op_times)
    )
    if dense_comm_ops is not None:
        for op_name, time_ms in dense_comm_ops.op_times.items():
            if get_comm_operator(op_name).execution_time_attr in (
                _DENSE_FFN_COMMUNICATION_ATTRS
            ):
                communication_op_times[op_name] = time_ms
    merged_communication.operator_times = (
        CommunicationOperatorTimes(communication_op_times)
        if dense_comm_ops is not None or original_comm_ops is not None
        else None
    )

    merged_op_times = dict(merged._op_times)
    original_mlp_ops = merged.mlp_operator_times
    if original_mlp_ops is not None:
        for op_name in original_mlp_ops.op_times:
            merged_op_times.pop(op_name, None)
    dense_mlp_ops = dense_layer.mlp_operator_times
    if dense_mlp_ops is not None:
        merged_op_times.update(dense_mlp_ops.op_times)
    for op_name, time_ms in communication_op_times.items():
        merged_op_times[op_name] = time_ms
    merged._op_times = merged_op_times
    merged._refresh_op_time_attr_values()
    if dense_layer.has_kernel_gap:
        merged._kernel_gap_times["mlp"] = dense_layer._kernel_gap_times.get(
            "mlp", 0.0
        )
    return merged.finalized_copy()


def complete_dense_layer(
    scheduler: Any,
    *,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Any,
    layer_id: int,
    phase: str,
    metrics_store: Any,
) -> list:
    """Advance a dense layer through the scheduler's existing phase handler.

    A dense layer completes per source, outside the shared forward completion
    that credits a routed layer to every decoding request in the cohort. The
    decode handler credits its own batch, but the prefill handler credits
    nothing, so a decoding request carried in a prefill-mode batch (chunked
    prefill mixes them) would miss this layer. Credit it here, once per
    executed layer; a request still prefilling or recomputing has no decode
    layer to credit.
    """
    if phase == "prefill":
        advance_decode_layer(
            (
                request
                for request in collect_active_requests([batch])
                if request.is_decoding
            ),
            scheduler._config.replica_config.model_config.num_layers,
        )
        return scheduler.on_prefill_sync_collective(
            time,
            replica_id,
            stage_id,
            int(batch.global_id),
            "post_moe",
            layer_id,
            metrics_store,
            direct_batch=batch,
        )
    if phase == "decode":
        return scheduler.on_decode_sync_collective(
            time,
            replica_id,
            stage_id,
            scheduler._get_decode_sync_wait_key(batch),
            "post_moe",
            layer_id,
            metrics_store,
            direct_batch=batch,
        )
    raise ValueError(f"Unsupported dense layer completion phase: {phase!r}")


def build_prefill_metrics_execution_time(
    *,
    original_execution_time: ExecutionTime | StageExecutionTime,
    sample_batch: Any,
    predictor: Any,
    stage_id: int,
    cluster_type: Any,
    model_config: BaseModelConfig,
) -> ExecutionTime | StageExecutionTime:
    """Preserve stage scope and include each executed dense FFN's own timing.

    EP wave records own routed MoE timing. Stage records own attention and
    dense FFNs, whose timings must be predicted at their actual layer IDs.
    """
    if not isinstance(original_execution_time, StageExecutionTime) or not (
        0 < model_config.get_num_moe_layers() < model_config.num_layers
    ):
        return build_metrics_execution_time(original_execution_time)
    layers = []
    for layer in original_execution_time.layer_execution_times:
        layer = _restore_structural_dsa_attention(layer, predictor)
        if model_config.is_moe_layer(layer.global_layer_id):
            layers.append(layer)
            continue
        dense_stage = predictor.predict_stage_execution_time(
            sample_batch, stage_id, cluster_type=cluster_type,
            num_layers=1, layer_id=layer.global_layer_id,
        )
        dense_layer = build_single_layer_metrics_execution_time(dense_stage)
        layers.append(_merge_dense_ffn_metrics_layer(layer, dense_layer))
    return StageExecutionTime(
        layers, stage_execution_time=original_execution_time.stage_execution_time,
    )
