"""Step attention cost lookup using the shared measured-model contract."""

from __future__ import annotations

from dataclasses import asdict
import json

from frontier.attention.step5 import describe_step5_layer
from frontier.entities.time_components import AttentionOperatorTimes, AttentionTime, OperatorCost
from frontier.execution_time_predictor.step5 import (
    STEP5_FEATURE_COLUMNS,
    STEP5_DURATION_STATISTICS,
    STEP5_OPERATOR_NAMES,
    step5_measurement_type,
    step5_row_features,
)
from frontier.types import ClusterType, MeasurementType


def step5_measurements_for_role(role, decode_graph_mode):
    """Attention's measured modality is independent of the step's FFN family."""
    decode = step5_measurement_type(decode_graph_mode)
    return {
        ClusterType.PREFILL: (MeasurementType.KERNEL_ONLY,),
        ClusterType.DECODE: (decode,),
        ClusterType.DECODE_ATTN: (decode,),
        ClusterType.DECODE_FFN: (),
        ClusterType.MONOLITHIC: tuple(dict.fromkeys((MeasurementType.KERNEL_ONLY, decode))),
    }[role]


def step5_predictions(models, measurement_type):
    """Expose trained root/count estimators at the existing on-demand seam."""
    predictions = {}
    for root in STEP5_OPERATOR_NAMES:
        for name in (root, f"{root}_kernel_count"):
            if name not in models:
                continue
            model = models[name]
            if tuple(model._frontier_feature_names) != STEP5_FEATURE_COLUMNS:
                raise ValueError(f"Step feature schema mismatch for {name}")
            identity = json.loads(model._frontier_measurement_identity)
            if identity["measurement_type"] != measurement_type.value:
                raise ValueError(f"Step source measurement identity mismatch for {name}")
            predictions[name] = {
                "_on_demand_prediction": True, "_model": model,
                "_feature_names": list(STEP5_FEATURE_COLUMNS),
                "_exact_lookup": model._frontier_exact_lookup,
                "_feature_bounds": model._frontier_feature_bounds,
            }
    return predictions


def step5_batch_features(predictor, batch, layer_id):
    """Bind actual batch extents to the same source geometry used by the producer."""
    from frontier.entities.batch import DummyForwardBatch

    if isinstance(batch, DummyForwardBatch):
        raise ValueError("Step measured request-free attention coverage is unavailable")
    config = predictor._model_config
    metadata = describe_step5_layer(
        config, layer_id,
        attention_tp_size=predictor._replica_config.attn_tensor_parallel_size,
        moe_tp_size=predictor._replica_config.moe_tensor_parallel_size,
        expert_parallel_size=predictor._replica_config.moe_expert_parallel_size,
    )
    workload = predictor._step5_attention_workload(batch)
    geometry = metadata.geometry
    row = {
        **asdict(geometry),
        "global_layer_id": layer_id,
        "num_tensor_parallel_workers": predictor._replica_config.attn_tensor_parallel_size,
        "n_q_head": geometry.num_q_heads, "n_kv_head": geometry.num_kv_heads,
        "head_size": geometry.head_dim,
        "region_block_size": workload.region_tokens, "topk_regions": workload.topk_regions,
        "sliding_window": config.sliding_window, "rope_theta": metadata.rope_theta,
        "rotary_dim": metadata.rotary_dim, "block_size": workload.block_size_tokens,
        "batch_size": len(workload.query_lengths), "batch_num_tokens": workload.valid_tokens,
        "batch_num_prefill_tokens": batch.num_prefill_tokens,
        "batch_num_decode_tokens": batch.num_decode_tokens,
        "valid_tokens": workload.valid_tokens, "padded_tokens": workload.padded_tokens,
        "max_model_len": workload.max_model_len_tokens,
        "block_table_columns": workload.block_table_columns, "score_regions": workload.score_regions,
        "attention_variant": metadata.attention_spec.variant_id,
        "phase": workload.phase, "graph_mode": workload.graph_mode,
        "query_lens": workload.query_lengths,
        "context_lens": tuple(length - query for length, query in
                              zip(workload.sequence_lengths, workload.query_lengths)),
    }
    return step5_row_features(row), workload


def predict_step5_attention_time(predictor, batch, layer_id):
    """Charge the inclusive physical root once; child scopes are diagnostics only."""
    features, workload = step5_batch_features(predictor, batch, layer_id)
    measurement = step5_measurement_type(workload.graph_mode)
    if measurement is MeasurementType.KERNEL_ONLY and predictor._kernel_gap_ms is None:
        raise ValueError("Step GPU-active attention needs measured kernel_gap.csv coverage")
    key = tuple(features[name] for name in STEP5_FEATURE_COLUMNS)
    root = STEP5_OPERATOR_NAMES[0]
    with predictor._temporary_measurement_type(measurement):
        names = (root, f"{root}_kernel_count")
        missing = [name for name in names if name not in predictor._predictions]
        if missing:
            raise ValueError(f"Step measured attention model coverage is unavailable: {missing}")
        models = [predictor._predictions[name]["_model"] for name in names]
        for name, model in zip(names, models):
            if key not in model._frontier_step5_coverage:
                raise ValueError(
                    f"Step measured attention coverage is unavailable for {name}: {features}"
                )
        if models[0]._frontier_measurement_identity != models[1]._frontier_measurement_identity:
            raise ValueError("Step duration and kernel-count measurement identities differ")
        identity = json.loads(models[0]._frontier_measurement_identity)
        expected = {
            "model_architecture_profile": predictor._model_config.get_model_architecture_profile().profile_id,
            "quant_signature": predictor._model_config.get_quant_signature(),
            "device": predictor._replica_config.device,
            "num_tensor_parallel_workers": predictor._replica_config.attn_tensor_parallel_size,
            "block_size": predictor._block_size,
            "physical_scope": "native_attention_layer_without_tp_allreduce",
            "duration_statistic": STEP5_DURATION_STATISTICS[measurement],
        }
        if any(identity.get(name) != value for name, value in expected.items()):
            raise ValueError("Step runtime query conflicts with measured physical identity")
        targets = (f"time_stats.{root}.{expected['duration_statistic']}", f"time_stats.{root}.kernel_count")
        if tuple(model._frontier_target_col for model in models) != targets:
            raise ValueError("Step measurement target policy conflicts with fitted model")
        duration = predictor._get_on_demand_prediction(root, features)
        count = predictor._get_on_demand_prediction(names[1], features)
        proof = OperatorCost(
            duration, count, measurement, models[0]._frontier_model_hash,
            models[1]._frontier_model_hash, identity.get("evidence_kind"),
            all(key in model._frontier_exact_lookup for model in models),
        )
    return AttentionTime(
        operator_times=AttentionOperatorTimes({root: duration}, costs={root: proof}),
    )
