"""Step fixed layer bindings and cost queries through the measured-model contract."""

from __future__ import annotations

from dataclasses import asdict
import json
from frontier.execution_time_predictor.prediction_model_registry import PredictionModelRegistry

from frontier.attention.step5 import Step5AttentionWorkload, describe_step5_layer
from frontier.attention.model_binding import bind_layer_attention
from frontier.attention.step5_inputs import (
    MRV2_DUMMY_ROW_IDENTITY, step5_input_kind, step5_workload_kind, step5_workload_label,
)
from frontier.entities.time_components import AttentionOperatorTimes, AttentionTime, OperatorCost
from frontier.execution_time_predictor.step5 import (
    STEP5_FEATURE_COLUMNS,
    STEP5_DURATION_STATISTICS,
    STEP5_OPERATOR_NAMES,
    step5_measurement_type,
    step5_row_features,
)
from frontier.execution_time_predictor.prediction_model_identity import _serialize_selected_layer_cache_identity
from frontier.operators.binding import bind_operator_query
from frontier.types import ClusterType, MeasurementType


def step5_layer_metadata(predictor, layer_id):
    """Reuse source geometry for a predictor's fixed model and parallel groups."""
    bind_layer_attention(predictor._model_config, layer_id)
    metadata_by_layer = getattr(predictor, "_step5_metadata_by_layer", None)
    if metadata_by_layer is None:
        metadata_by_layer = predictor._step5_metadata_by_layer = {}
    if layer_id not in metadata_by_layer:
        metadata_by_layer[layer_id] = describe_step5_layer(
            predictor._model_config, layer_id,
            attention_tp_size=predictor._replica_config.attn_tensor_parallel_size,
            moe_tp_size=predictor._replica_config.moe_tensor_parallel_size,
            expert_parallel_size=predictor._replica_config.moe_expert_parallel_size,
        )
    return metadata_by_layer[layer_id]


def step5_operator_binding(predictor, metadata, model_name):
    """Reuse fixed precision and contract identity, never a selected estimator."""
    bindings = getattr(predictor, "_step5_operator_bindings", None)
    if bindings is None:
        bindings = predictor._step5_operator_bindings = {}
    key = (metadata.attention_spec.global_layer_id, model_name)
    if key not in bindings:
        binding = bind_operator_query(model_name)
        projections = [p for p in metadata.projections if p.operator_name == model_name]
        precision = metadata.model_precision
        if projections:
            precisions = {p.weight_precision for p in projections}
            if len(precisions) != 1:
                raise ValueError(f"Step {model_name} has heterogeneous source precision")
            precision = precisions.pop()
        identity = None
        if binding.family_id in {"ffn", "share_expert", "moe"}:
            contract = predictor._get_model_architecture_profile().resolve_layer_contract(
                predictor._model_config, layer_id=key[0], operator_name=model_name,
                attention_tp_size=predictor._replica_config.attn_tensor_parallel_size,
                ffn_tp_size=predictor._get_ffn_tp_key_for_linear_op(),
                moe_tp_size=predictor._replica_config.moe_tensor_parallel_size,
                expert_parallel_size=predictor._replica_config.moe_expert_parallel_size,
            )
            identity = _serialize_selected_layer_cache_identity(contract)
        bindings[key] = (precision, identity)
    return bindings[key]


def step5_measurements_for_role(role, decode_graph_mode):
    """Attention's measured modality is independent of the step's FFN family."""
    from frontier.config.graph_policy import graph_policy

    policy = graph_policy(decode_graph_mode)
    decode = step5_measurement_type(policy.uniform_decode.lower())
    # D's initial prompt tail and graph misses still execute without a graph.
    decode_measurements = tuple(dict.fromkeys((MeasurementType.KERNEL_ONLY, decode)))
    return {
        ClusterType.PREFILL: decode_measurements,
        ClusterType.DECODE: decode_measurements,
        ClusterType.DECODE_ATTN: (decode,),
        ClusterType.DECODE_FFN: (),
        ClusterType.MONOLITHIC: decode_measurements,
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


def step5_batch_workload(predictor, batch):
    """Bind real local history or MRv2's agreed metadata-bearing dummy input."""
    from frontier.entities.batch import DummyForwardBatch

    config = predictor._model_config
    descriptor = getattr(batch, "forward_start_descriptor", None)
    if getattr(predictor, "requires_forward_start_agreement", False):
        from frontier.scheduler.utils.forward_start_agreement import ForwardStartDescriptor

        if not isinstance(descriptor, ForwardStartDescriptor):
            raise ValueError("Measured Step first-layer prediction requires a forward-start descriptor")
    if isinstance(batch, DummyForwardBatch):
        dummy = getattr(batch, "mrv2_dummy_descriptor", None)
        if descriptor is not None:
            from frontier.attention.step5_inputs import MRv2DummyDescriptor

            if not isinstance(dummy, MRv2DummyDescriptor) or (
                dummy.num_tokens != descriptor.num_tokens
                or dummy.num_reqs != descriptor.num_reqs
                or dummy.graph_mode.upper() != descriptor.graph_mode
                or dummy.max_query_len != descriptor.max_query_len
                or dummy.uniform_token_count != descriptor.uniform_token_count
            ):
                raise ValueError("Step dummy forward requires its agreed MRv2 InputBatch descriptor")
            queries = dummy.query_lens
            return Step5AttentionWorkload.from_lengths(
                config, phase="decode" if max(queries) == 1 else "prefill",
                query_lengths=queries, sequence_lengths=queries,
                physical_query_rows=dummy.num_tokens,
                block_size_tokens=predictor._block_size,
                block_table_columns=(predictor._step5_max_model_len + predictor._block_size - 1) // predictor._block_size,
                max_model_len_tokens=predictor._step5_max_model_len, graph_mode=dummy.graph_mode,
                has_attention_metadata=True,
            )
        # This isolated diagnostic is not MRv2's metadata-bearing DP dummy path.
        return Step5AttentionWorkload.from_lengths(
            config, phase="decode", query_lengths=tuple(batch.num_tokens),
            sequence_lengths=tuple(batch.num_tokens),
            physical_query_rows=batch.get_effective_total_tokens_for_compute(predictor._cluster_type),
            block_size_tokens=predictor._block_size, block_table_columns=0,
            max_model_len_tokens=predictor._step5_max_model_len,
            graph_mode=batch.get_decode_cuda_graph_runtime_mode().lower(),
            has_attention_metadata=False,
        )
    phase = "mixed" if batch.num_prefill_tokens and batch.num_decode_tokens else (
        "prefill" if batch.num_prefill_tokens else "decode"
    )
    queries = tuple(batch.num_tokens)
    # MRv2 also runs a one-token prompt extension over existing context through
    # its decode graph. Ownership and DES prefill credit remain on the batch.
    if descriptor is not None and descriptor.graph_mode == "FULL":
        phase = "decode"
    max_model_len = predictor._step5_max_model_len
    return Step5AttentionWorkload.from_lengths(
        config, phase=phase, query_lengths=queries,
        sequence_lengths=tuple(context + query for context, query in zip(batch.num_context_tokens, queries)),
        physical_query_rows=batch.get_effective_total_tokens_for_compute(predictor._cluster_type),
        block_size_tokens=predictor._block_size,
        block_table_columns=(max_model_len + predictor._block_size - 1) // predictor._block_size,
        max_model_len_tokens=max_model_len,
        graph_mode=batch.get_decode_cuda_graph_runtime_mode().lower(),
    )


def step5_batch_features(predictor, batch, layer_id):
    """Bind actual batch extents to the same source geometry used by the producer."""
    config = predictor._model_config
    metadata = step5_layer_metadata(predictor, layer_id)
    workload = predictor._step5_attention_workload(batch)
    dummy_descriptor = getattr(batch, "mrv2_dummy_descriptor", None)
    kind = step5_input_kind(has_attention_metadata=workload.has_attention_metadata,
                           dummy_descriptor=dummy_descriptor)
    uses_physical_phase = dummy_descriptor is not None or (
        getattr(batch, "forward_start_descriptor", None) is not None and workload.graph_mode == "full"
    )
    prefill_tokens = workload.valid_tokens if workload.phase == "prefill" else 0
    decode_tokens = workload.valid_tokens if workload.phase == "decode" else 0
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
        "batch_num_prefill_tokens": prefill_tokens if uses_physical_phase else batch.num_prefill_tokens,
        "batch_num_decode_tokens": decode_tokens if uses_physical_phase else batch.num_decode_tokens,
        "valid_tokens": workload.valid_tokens, "padded_tokens": workload.padded_tokens,
        "max_model_len": workload.max_model_len_tokens,
        "block_table_columns": workload.block_table_columns, "score_regions": workload.score_regions,
        "has_attention_metadata": int(workload.has_attention_metadata),
        "native_workload": kind, "physical_scope": step5_workload_kind(kind).physical_scope,
        "attention_variant": metadata.attention_spec.variant_id,
        "phase": workload.phase, "graph_mode": workload.graph_mode,
        "query_lens": workload.query_lengths,
        "context_lens": tuple(length - query for length, query in
                               zip(workload.sequence_lengths, workload.query_lengths)),
    }
    if dummy_descriptor is not None:
        row.update(MRV2_DUMMY_ROW_IDENTITY, dummy_descriptor=asdict(dummy_descriptor))
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
        identity = PredictionModelRegistry._measurement_identity(models[0])
        expected = {
            "model_architecture_profile": predictor._model_config.get_model_architecture_profile().profile_id,
            "quant_signature": predictor._model_config.get_quant_signature(),
            "device": predictor._replica_config.device,
            "num_tensor_parallel_workers": predictor._replica_config.attn_tensor_parallel_size,
            "block_size": predictor._block_size,
            "duration_statistic": STEP5_DURATION_STATISTICS[measurement],
        }
        if any(identity.get(name) != value for name, value in expected.items()):
            raise ValueError("Step runtime query conflicts with measured physical identity")
        kind = step5_workload_label(features["native_workload_code"])
        if identity.get("physical_scope", {}).get(kind) != step5_workload_kind(kind).physical_scope:
            raise ValueError("Step runtime query conflicts with measured kind-bound physical identity")
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
