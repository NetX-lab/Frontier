"""Source-scoped V4 workloads and qualified isolated measured components.

Source: vLLM 2cf0a6915ce544dc493a0990f2ea38d81601128a,
models/deepseek_v4/nvidia/model.py and common/ops/sequence_parallel.py.
"""

from frontier.attention.deepseek_v4 import (
    DeepSeekV4CoverageError, _integer, _validate_source_batch, compute_rows,
    validate_deepseek_v4_dummy_forward,
)
from frontier.entities.batch import DummyForwardBatch
from frontier.entities.time_components import MoEOperatorTimes, MoETime
from frontier.operators.binding import build_operator_manifest
from frontier.operators.families import get_comm_operator


def timing_operator_times(execution_time, phase):
    """Project physical V4 scopes into the existing serial structural phases.

    This ordering is a synthetic execution contract, not an observed native
    launch schedule or a claim about auxiliary-stream overlap.
    """
    from frontier.operators.families import COMM_FAMILY

    names_by_phase = {
        "attention": (
            "kv_cache_block_zero",
            "dsv4_embedding", "dsv4_embedding_allreduce",
            "dsv4_mhc_first_pre", "dsv4_mhc_ffn_post_attn_pre",
            "dsv4_sp_attention_allgather",
            "dsv4_fused_qkv_projection", "dsv4_compressor_projection",
            "dsv4_index_weights_projection", "dsv4_index_compressor_projection",
            "dsv4_fused_qkv_rmsnorm", "dsv4_q_projection",
            "dsv4_fused_q_kv_prepare", "dsv4_query_padding",
            "dsv4_compressor_save_partial_states", "dsv4_compressor_fused_compress_store",
            "dsv4_index_query_projection", "dsv4_index_fused_query_prepare",
            "dsv4_index_compressor_save_partial_states",
            "dsv4_index_compressor_fused_compress_store", "dsv4_index_scores",
            "dsv4_index_topk", "dsv4_index_short_context_fill", "dsv4_sparse_attention",
            "dsv4_attention_output_zero", "dsv4_output_inv_rope_quant",
            "dsv4_grouped_output_projection", "dsv4_output_projection",
            "dsv4_sp_attention_reduce_scatter",
        ),
        "pre_dispatch": (
            "dsv4_mhc_attn_post_ffn_pre", "share_expert_up_proj",
            "share_expert_act", "share_expert_down_proj", "moe_gating_linear",
            "moe_gating_routing_topk",
        ),
        "dispatch": (
            "dsv4_ep_activation_allgather", "dsv4_ep_weights_allgather",
            "dsv4_ep_ids_allgather",
        ),
        "routed_compute": ("moe_shuffling", "moe_grouped_gemm"),
        "combine": ("dsv4_ep_output_reduce_scatter",),
        "post_combine": (
            "dsv4_mhc_final_post", "dsv4_sp_final_mhc_allgather",
            "dsv4_mhc_head", "dsv4_final_norm", "dsv4_lm_head", "dsv4_logits_gather",
        ),
    }
    supported = {name for names in names_by_phase.values() for name in names}
    unsupported = set(execution_time.op_times) - supported
    if unsupported:
        raise ValueError(f"V4 physical timing scopes have no phase owner: {sorted(unsupported)}")
    comm_names = {op.name for op in COMM_FAMILY.operators}
    return tuple(
        ("COMM" if name in comm_names else "COMPUTE", name, execution_time.op_times[name])
        for name in names_by_phase[phase] if name in execution_time.op_times
    )


def predict_source_execution(
    predictor, batch, layer_id, tp_rank, *, execution_mode,
    has_attention_metadata, prefix_caching, decode_context_parallel_size,
    prefill_context_parallel_size, enable_expert_parallel, moe_backend,
    all2all_backend, skip_padding, input_padding_mask, selected_logits_rows,
    include_attention=True, include_ffn=True,
):
    """Return selected synthetic components; routed EP and stage work are absent.

    ``source_order`` orders only the selected wrapper components. It does not
    establish the ordinary FusedMoE backend's routed/shared stream overlap.
    ``selected_logits_rows`` is an explicit row count after native row selection.
    No physical vocabulary padding or logits-collection tensor is inferred.
    """
    model = predictor._model_config
    replica = predictor._replica_config
    profile = model.get_model_architecture_profile()
    if profile.profile_id != "deepseek_v4":
        raise ValueError("V4 source components require the DeepSeek-V4 profile")
    if enable_expert_parallel is not True:
        raise NotImplementedError("V4 SP source scope requires expert parallel enabled")
    if moe_backend != "auto":
        raise NotImplementedError("V4 source scope requires the pinned ordinary auto backend")
    if all2all_backend != "allgather_reducescatter":
        raise NotImplementedError("V4 source scope requires all2all allgather_reducescatter")
    if input_padding_mask is not None:
        raise NotImplementedError("V4 source scope does not represent an existing input padding mask")
    if isinstance(batch, DummyForwardBatch):
        validate_deepseek_v4_dummy_forward(batch)
        if has_attention_metadata is not False:
            raise ValueError("V4 idle source must declare absent attention metadata")
    else:
        _validate_source_batch(batch)
        if has_attention_metadata is not True:
            raise NotImplementedError("V4 real Batch requires real metadata eager execution")
    tp = _integer(replica.attn_tensor_parallel_size, "attention TP")
    rank = _integer(tp_rank, "source TP rank", minimum=0)
    if rank >= tp:
        raise ValueError("source TP rank must be less than attention TP")
    if type(skip_padding) is not bool:
        raise ValueError("skip_padding must be an explicit bool")
    manifest = build_operator_manifest(model, global_layer_id=layer_id)
    canonical = {operator.name for operator in manifest.operators()}
    actual_tokens = sum(batch.num_tokens)
    tokens = compute_rows(batch)
    last = layer_id == model.num_layers - 1
    if last:
        selected_logits_rows = _integer(
            selected_logits_rows, "selected_logits_rows", minimum=0,
        )
        if selected_logits_rows > actual_tokens:
            raise ValueError("selected_logits_rows cannot exceed source rows")
        if isinstance(batch, DummyForwardBatch) and selected_logits_rows:
            raise ValueError("V4 DummyForwardBatch cannot select LM rows")
    elif selected_logits_rows is not None:
        raise ValueError("selected_logits_rows is only accepted at the last layer")
    from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths

    widths = sequence_parallel_moe_token_widths([tokens], tp, rank)
    local = widths.local_tokens[0]
    valid = max(0, min(local, actual_tokens - rank * local))
    base = predictor._dummy_execution_time
    hidden = _integer(model.embedding_dim, "hidden width")
    streams = _integer(model.hc_mult, "mHC streams")
    intermediate = _integer(model.share_expert_dim, "shared intermediate width")
    shared_contract = profile.resolve_layer_contract(
        model, layer_id=layer_id, operator_name="share_expert_up_proj",
        attention_tp_size=tp, attention_dp_size=replica.attn_dp,
        num_pipeline_stages=replica.num_pipeline_stages,
        moe_tp_size=replica.moe_tensor_parallel_size,
        expert_parallel_size=replica.moe_expert_parallel_size,
    )
    shared_tp = shared_contract.tensor_parallel_size
    if shared_tp != 1:
        raise ValueError("V4 SP shared expert weights must be replicated")
    shared_names = ("share_expert_up_proj", "share_expert_act", "share_expert_down_proj")
    shared_values = {name: base for name in shared_names}
    shared = MoETime(operator_times=MoEOperatorTimes(shared_values))
    first_pre = "dsv4_mhc_first_pre" if layer_id == 0 else "dsv4_mhc_ffn_post_attn_pre"
    component_times = {}
    order = []
    workloads = {}

    def local_scope(name, input_shape, output_shape, **features):
        if not predictor._enable_dummy_mode:
            raise DeepSeekV4CoverageError(
                f"V4 serving scope {name} lacks phase/TP-qualified measurements; "
                "isolated eager rows do not admit a native serving forward"
            )
        component_times[name] = base
        workloads[name] = {
            "input_shape": input_shape, "output_shape": output_shape, **features,
        }
        order.append({"scope": name, "time_ms": base})

    def communication(name):
        value = predictor._predict_comm_operator(get_comm_operator(name), batch)
        component_times[name] = value
        order.append({"scope": name, "time_ms": value})

    if include_attention and layer_id == 0:
        local_scope(
            "dsv4_embedding", (tokens,), (tokens, hidden),
            cost_scope="local vocab embedding; reduction is a separate physical scope",
        )
    attention = None
    if include_attention:
        local_scope(
            first_pre, (local, hidden), (local, hidden),
            residual_shape=(local, streams, hidden),
            post_mix_shape=(local, streams, 1), res_mix_shape=(local, streams, streams),
            state_dtype="float32", activation_dtype="bfloat16",
        )
        communication("dsv4_sp_attention_allgather")
        attention = predictor._predict_dsv4_dummy_attention_time(
            batch, layer_id, predictor._cluster_type, execution_mode=execution_mode,
            has_attention_metadata=has_attention_metadata, prefix_caching=prefix_caching,
            decode_context_parallel_size=decode_context_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
        )
        component_times.update(attention.op_times)
        workloads.update(attention._dsv4_workloads)
        # The private component map is not a native launch schedule.
        order.append({"scope": "attention", "time_ms": attention.total_time()})
        communication("dsv4_sp_attention_reduce_scatter")
    if include_ffn:
        local_scope(
            "dsv4_mhc_attn_post_ffn_pre", (local, hidden), (local, hidden),
            residual_shape=(local, streams, hidden),
            post_mix_shape=(local, streams, 1), res_mix_shape=(local, streams, streams),
            state_dtype="float32", activation_dtype="bfloat16",
        )
        local_scope(shared_names[0], (local, hidden), (local, 2 * intermediate), weight_tp=shared_tp)
        local_scope(shared_names[1], (local, 2 * intermediate), (local, intermediate))
        local_scope(shared_names[2], (local, intermediate), (local, hidden), weight_tp=shared_tp)
    if last and include_ffn:
        local_scope(
            "dsv4_mhc_final_post", (local, hidden), (local, streams, hidden),
            residual_shape=(local, streams, hidden), state_dtype="float32",
        )
        communication("dsv4_sp_final_mhc_allgather")
        local_scope("dsv4_mhc_head", (tokens, streams, hidden), (tokens, hidden))
        local_scope("dsv4_final_norm", (tokens, hidden), (tokens, hidden))
        if selected_logits_rows:
            local_scope(
                "dsv4_lm_head", (selected_logits_rows, hidden), None,
                logical_unpartitioned_output_shape=(selected_logits_rows, model.vocab_size),
                weight_tp=tp,
            )
    comm_names = {
        "dsv4_sp_attention_allgather", "dsv4_sp_attention_reduce_scatter",
        "dsv4_sp_final_mhc_allgather",
    }
    if not set(component_times).difference(comm_names).issubset(canonical):
        raise ValueError("V4 source components are absent from the layer manifest")
    return {
        "status": "PARTIAL", "source_rows": actual_tokens, "compute_rows": tokens,
        "physical_local_rows": local, "padded_rows": local * tp,
        "valid_local_rows": valid,
        "padding_mask_rows": local - valid if skip_padding else 0,
        # The score fast path need not consume the mask; geometry is not routing.
        "routed_assignments": None,
        "shared_ffn_time": shared, "attention_time": attention,
        "component_times": component_times, "workloads": workloads,
        "source_order": tuple(order),
        "selected_component_time_ms": sum(component_times.values()),
        "omitted_scopes": (
            "routing implementation and assignments", "routed EP dispatch/compute/combine",
            "embedding internal vocab reduction", "logits gathering and sampling",
            "phase and stage timing publication", "routed/shared overlap",
        ),
    }


def predict_structural_execution(
    predictor, batch, layer_id, tp_rank, *, lane_workload=None,
    include_attention=True, include_ffn=True, include_stage_owned=False,
    **source_context,
):
    """Connect one source rank and an explicitly synthetic destination lane.

    The ordinary modular W4A16 scenario transports BF16 activations, FP32
    top-k weights and int32 IDs. Native auto-backend selection, padding-mask
    consumption, routing outcomes and overlap are not inferred from this test.
    """
    from frontier.entities import ExecutionTime
    from frontier.moe_ep_workload import EPLaneWorkload

    if source_context["skip_padding"] is not False:
        raise NotImplementedError("V4 structural routing requires an explicit unmasked scenario")
    if include_ffn and not isinstance(lane_workload, EPLaneWorkload):
        raise TypeError("V4 structural execution requires an EPLaneWorkload")
    replica = predictor._replica_config
    if include_ffn and (
        lane_workload.moe_expert_parallel_size != replica.moe_expert_parallel_size
        or lane_workload.total_expert_num != replica.total_expert_num
        or lane_workload.router_topk != replica.router_topk
    ):
        raise ValueError("V4 destination workload does not match the source topology")
    source = predict_source_execution(
        predictor, batch, layer_id, tp_rank, include_attention=include_attention,
        include_ffn=include_ffn, **source_context,
    )
    times = dict(source["component_times"])
    workloads = dict(source["workloads"])
    base = predictor._dummy_execution_time
    zeroing = batch.cache_block_zeroing
    if include_attention and layer_id == 0 and zeroing is not None:
        times["kv_cache_block_zero"] = base
        workloads["kv_cache_block_zero"] = {
            "block_ids": zeroing.block_ids,
            "block_assignment_count": len(zeroing.block_ids),
            "unique_block_count": len(set(zeroing.block_ids)),
            "physical_block_size_bytes": zeroing.physical_block_size_bytes,
            "physical_reservation_extent_bytes": zeroing.physical_reservation_extent_bytes,
            "extent_scope": "distinct physical slab reservations; not native zeroed segments or store traffic",
            "execution_scope": "one before-forward invocation per physical worker and scheduler step",
            "native_store_traffic_bytes": "UNKNOWN", "native_h2d_cost": "UNKNOWN",
            "cost_source": "one synthetic dummy duration; no byte or layer scaling",
        }
    if include_attention and layer_id == 0:
        # The embedding scope now excludes its separately owned reduction.
        times["dsv4_embedding_allreduce"] = base
    if include_ffn:
        times.update({
            "moe_gating_linear": base, "moe_gating_routing_topk": base,
            # The fused expert composite owns its local alignment/reduction work.
            "moe_grouped_gemm": base if lane_workload.routed_token_count else 0.0,
        })
        local = source["physical_local_rows"]
        hidden = predictor._model_config.embedding_dim
        workloads.update({
            "moe_gating_linear": {
                "input_shape": (local, hidden),
                "output_shape": (local, predictor._model_config.num_experts),
            },
            "moe_gating_routing_topk": {
                "input_shape": (local, predictor._model_config.num_experts),
                "topk_weights_shape": (local, replica.router_topk),
                "topk_ids_shape": (local, replica.router_topk),
                "topk_weights_dtype": "float32", "topk_ids_dtype": "int32",
                "routing_variant": "hash" if layer_id < predictor._model_config.num_hash_layers else "score",
            },
            "moe_grouped_gemm": {
                "input_shape": (lane_workload.routed_token_count, hidden),
                "output_shape": (lane_workload.routed_token_count, hidden),
                "cost_scope": "inclusive fused expert FFN including local alignment/reduction",
            },
        })
        for name in (
            "dsv4_ep_activation_allgather", "dsv4_ep_weights_allgather",
            "dsv4_ep_ids_allgather", "dsv4_ep_output_reduce_scatter",
        ):
            times[name] = predictor._predict_comm_operator(
                get_comm_operator(name), batch,
                lane_workload=(lane_workload if name == "dsv4_ep_output_reduce_scatter" else None),
            )
        if layer_id == predictor._model_config.num_layers - 1 and source_context["selected_logits_rows"]:
            times["dsv4_logits_gather"] = base
            workloads["dsv4_logits_gather"] = {
                "selected_logits_rows": source_context["selected_logits_rows"],
                "logical_vocabulary_size": predictor._model_config.vocab_size,
                "physical_vocabulary_padding": "UNKNOWN",
            }
    from frontier.config import get_quantization_manager
    from frontier.operators.families import COMM_FAMILY
    from frontier.operators.spec import CommOperatorSpec, CommPayloadContext

    for operator in COMM_FAMILY.operators:
        if operator.name not in times or not isinstance(operator, CommOperatorSpec):
            continue
        payload_context = CommPayloadContext(
            batch, predictor._model_config, replica, predictor._cluster_type,
            get_quantization_manager(),
            lane_workload=lane_workload if operator.name == "dsv4_ep_output_reduce_scatter" else None,
        )
        workloads[operator.name] = {
            "data_size_bytes": operator.build_payload_bytes(payload_context),
            "collective": operator.collective_alias,
            "comm_domain": operator.resolve_comm_domain(payload_context),
            "num_devices": operator.num_devices(payload_context),
            "payload_scope": "CC collective input tensor; not native wire traffic",
        }
    if "dsv4_embedding_allreduce" in times:
        workloads["dsv4_embedding_allreduce"] = {
            "input_shape": (source["compute_rows"], predictor._model_config.embedding_dim),
            "activation_dtype": "bfloat16", "collective": "allreduce",
            "cost_scope": "unmeasured synthetic dummy duration",
        }
    spec = predictor._model_config.get_layer_attention_spec(layer_id)
    cpu = base if include_stage_owned else 0.0
    timing = ExecutionTime(
        num_layers_per_pipeline_stage=1,
        attention_rope_execution_time=0.0,
        attention_kv_cache_save_execution_time=0.0,
        attention_decode_execution_time=0.0, attention_prefill_execution_time=0.0,
        attention_layer_pre_proj_execution_time=0.0,
        attention_layer_post_proj_execution_time=0.0,
        attn_norm_time=0.0, mlp_norm_time=0.0, add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0, moe_shuffling_time=0.0,
        schedule_time=cpu, sampler_e2e_time=cpu, prepare_inputs_e2e_time=cpu,
        process_model_outputs_time=cpu, ray_comm_time=cpu, is_moe=True,
        op_times=times, global_layer_id=layer_id,
        attention_family_id=spec.family_id, attention_variant_id=spec.variant_id,
    )
    from types import MappingProxyType

    timing._dsv4_source_rows = MappingProxyType({
        "source_rows": source["source_rows"],
        "compute_rows": source["compute_rows"],
        "physical_local_rows": source["physical_local_rows"],
        "valid_local_rows": source["valid_local_rows"],
        "request_valid_local_rows": (0 if isinstance(batch, DummyForwardBatch)
                                     else source["valid_local_rows"]),
        "source_forward_kind": ("request-free dummy" if isinstance(batch, DummyForwardBatch)
                                else "scheduled requests"),
        "reference_workflow_scope": (
            f"{batch.get_decode_cuda_graph_runtime_mode()}; no DBO/ubatching; "
            "model-internal SP enabled; native timing unqualified"
        ),
        "source_tp_rank": tp_rank,
        "source_batch_id": batch.id,
        "source_request_ids": [str(request.id) for request in batch.requests],
        "routing_policy": "explicit synthetic unmasked SP rows",
        "destination_provenance": "configured synthetic expert ratios",
        "native_routed_assignments": "UNKNOWN",
        "routing_operator_variant": "hash" if layer_id < predictor._model_config.num_hash_layers else "score",
        "execution_order": "synthetic serial; native shared/routed overlap UNKNOWN",
        "ep_transport": "structural modular W4A16 BF16/topk FP32/int32; native backend UNKNOWN",
        "ep_collective_cost": "uniform CC calls; ragged native latency unqualified",
        "endpoint_collective_cost": "unmeasured synthetic dummy duration",
        "synthetic_destination_assignments": lane_workload.routed_token_count if include_ffn else None,
    })
    timing._dsv4_workloads = MappingProxyType(workloads)
    return timing


def operator_trace_metadata(op_name, op_type, context):
    """Publish source tensor contracts without converting weight precision to IO dtype."""
    from frontier.config import get_quantization_manager
    from frontier.operators.families import COMM_FAMILY

    if op_type == "COMM":
        operators = {operator.name: operator for operator in COMM_FAMILY.operators}
        if op_name not in operators:
            raise ValueError(f"Unsupported V4 communication trace scope: {op_name}")
        operator = operators[op_name]
        return {
            "precision_op": operator.precision_name(),
            "source_scope": "vllm 2cf0a6915ce544dc493a0990f2ea38d81601128a",
            "collective": getattr(operator, "collective_alias", "source endpoint composite"),
            "cost_source": "synthetic structural communication, not native measurements",
        }
    if op_type != "COMPUTE":
        raise ValueError(f"Unsupported V4 trace kind: {op_type}")
    if op_name == "kv_cache_block_zero":
        return {
            "source_scope": "vllm 2cf0a6915ce544dc493a0990f2ea38d81601128a: KVBlockZeroer",
            "execution_boundary": "before forward; structural serial ordering",
            "cache_storage_precision": "mixed cache views; byte initialization has no compute dtype",
            "cost_source": "one synthetic dummy duration; native segment, store and H2D costs unqualified",
        }
    inventory = {row["operation"]: row for row in
                 get_quantization_manager().get_operation_precision_metadata()}
    if op_name not in inventory:
        raise ValueError(f"V4 compute scope has no source precision contract: {op_name}")
    return {"precision_op": op_name, **inventory[op_name],
            "cost_source": "synthetic dummy duration, not native measurements"}


def operator_workload_metadata(execution_time, op_name):
    """Attach the existing component builder's logical calls to one physical scope."""
    workload = getattr(execution_time, "_dsv4_workloads", {}).get(op_name)
    if workload is None:
        return {}
    calls = workload if isinstance(workload, tuple) else (workload,)
    return {"source_tensor_calls": calls}


def serving_context(predictor, batch, layer_id):
    """Use actual graph state, not the process-wide compilation configuration."""
    from frontier.config import global_vars

    if global_vars.KV_CONNECTOR != "none":
        raise NotImplementedError("V4 predictor PD phase/graph equivalence remains unqualified")
    compute_rows(batch)
    return dict(
        execution_mode=("eager" if batch.get_decode_cuda_graph_runtime_mode() == "NONE"
                        else "cuda_graph"),
        has_attention_metadata=not isinstance(batch, DummyForwardBatch),
        # The component cache independently rejects prefix caching/MTP/PD.
        prefix_caching=False, decode_context_parallel_size=1,
        prefill_context_parallel_size=1, enable_expert_parallel=True,
        moe_backend="auto", all2all_backend="allgather_reducescatter",
        skip_padding=False, input_padding_mask=None,
        # Ordinary runner _prepare_inputs selects the last scheduled row of
        # each request, including partial prefill (gpu_model_runner.py:2304).
        selected_logits_rows=(len(batch.requests)
                              if layer_id == predictor._model_config.num_layers - 1 else None),
    )


def component_ledger(execution_time):
    """Keep canonical device operators and existing once-only CPU fields disjoint."""
    components = dict(execution_time.op_times)
    for name in (
        "schedule_time", "sampler_e2e_time", "prepare_inputs_e2e_time",
        "process_model_outputs_time", "ray_comm_time",
    ):
        components[name] = float(getattr(execution_time, name))
    return components


def source_physical_rows(predictor, source_batches):
    """Resolve all DP source rows, including explicit one-row idle forwards."""
    replica = predictor._replica_config
    if tuple(sorted(source_batches)) != tuple(range(replica.attn_dp)):
        raise ValueError("V4 source cohort must contain every attention DP rank")
    for batch in source_batches.values():
        if isinstance(batch, DummyForwardBatch):
            validate_deepseek_v4_dummy_forward(batch)
        else:
            _validate_source_batch(batch)
    tp = replica.attn_tensor_parallel_size
    return sum(((compute_rows(batch) + tp - 1) // tp) * tp
               for batch in source_batches.values())


def load_attention_dataframe(file_path, model, tp, block_size, measurement_type, *, float_precision=None):
    """Read the existing wide V4 schema without passing it to the dense loader."""
    import pandas as pd
    from frontier.profiling.attention.dsv4_attention_rows import DSV4_ROW_SCHEMA_VERSION

    try:
        rows = pd.read_csv(file_path, float_precision=float_precision)
    except FileNotFoundError as exc:
        raise DeepSeekV4CoverageError(f"V4 attention table missing: {file_path}") from exc
    expected = {
        "schema_version": DSV4_ROW_SCHEMA_VERSION,
        "measurement_type": measurement_type.value,
        "attention_family": "deepseek_v4_attention",
        "model_architecture_profile": model.get_model_architecture_profile().profile_id,
        "quant_signature": model.get_quant_signature(),
        "profiling_precision": model.get_default_precision().name,
        "attention_backend": "FLASHMLA_SPARSE_DSV4",
        "n_embd": model.embedding_dim, "n_q_head": model.num_q_heads,
        "n_q_head_per_rank": model.num_q_heads // tp,
        "num_tensor_parallel_workers": tp, "o_groups": model.o_groups,
        "o_groups_per_rank": model.o_groups // tp, "block_size": block_size,
        **{name: getattr(model, name) for name in (
            "q_lora_rank", "o_lora_rank", "head_dim", "qk_rope_head_dim",
            "sliding_window", "index_topk", "index_n_heads", "index_head_dim",
        )},
    }
    return _select_rows(rows, expected, "attention schema/model/TP")


def _select_rows(rows, expected, scope):
    for column, value in expected.items():
        if column not in rows:
            raise DeepSeekV4CoverageError(f"V4 {scope} missing column {column}")
        rows = rows[rows[column].isna() if value is None else rows[column] == value]
    if rows.empty:
        raise DeepSeekV4CoverageError(f"V4 {scope} has no exact measured rows: {expected}")
    return rows


def _read_isolated_component(reader, scope, *args, **kwargs):
    """Translate the existing readers' known data-admission errors, not bugs."""
    import pandas as pd
    from frontier.operators.typed_contracts import TYPED_OPERATOR_CONTRACTS_COLUMN

    try:
        rows = reader(*args, **kwargs)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DeepSeekV4CoverageError(f"V4 {scope} measured table cannot be read: {exc}") from exc
    except KeyError as exc:
        if exc.args[0] not in {
            "num_experts", "router_topk", "hidden_dim", "num_tensor_parallel_workers",
            "expert_hidden_dim", "expert_parallel_size",
        }:
            raise
        raise DeepSeekV4CoverageError(f"V4 {scope} measured table missing {exc}") from exc
    except ValueError as exc:
        # These legacy readers use ValueError for their explicit coverage
        # checks. Unrelated ValueErrors retain their original type/traceback.
        if not str(exc).startswith((
            "Column 'num_tensor_parallel_workers' not found ",
            "No data matches the filtering criteria in ",
            "No linear-op rows match the selected typed layer contract ",
            "Legacy linear-op profiling data is missing ",
            "Required attention linear op columns are missing or all-NaN in ",
            "MoE profiling data is missing 'expert_parallel_size' ",
            "Missing load imbalance features", "typed_operator_contracts ",
            "typed metadata ", "typed contract ", "model_architecture_profile ",
            "Multiple model_architecture_profile values found ",
            "Unknown typed operator ",
        )):
            raise
        raise DeepSeekV4CoverageError(f"V4 {scope} measured coverage unavailable: {exc}") from exc
    if TYPED_OPERATOR_CONTRACTS_COLUMN not in rows:
        raise DeepSeekV4CoverageError(f"V4 {scope} missing {TYPED_OPERATOR_CONTRACTS_COLUMN}")
    return rows


def build_isolated_prefill_layouts(plan, records, *, expected_identity, input_file_sha256):
    """Declare flat per-layer contracts from accepted non-timing engine/step facts."""
    import pandas as pd
    from frontier.profiling.attention.dsa_rows import index_records
    from frontier.profiling.attention.dsv4_attention_rows import (
        _ENGINE_COLUMNS, _model_columns, _variant_metadata,
    )

    engines, steps, _ = index_records(r for r in records if r["kind"] != "call")
    layouts = {}
    for step in steps.values():
        if step["phase"] != "prefill":
            raise DeepSeekV4CoverageError("V4 isolated prefill contract requires prefill raw metadata")
        engine = engines[step["engine_index"]]
        columns = _model_columns(plan, engine)
        for layer in plan.shape.layers:
            try:
                metadata = pd.json_normalize(_variant_metadata(step, layer.attention_variant),
                                             sep="_metadata.").iloc[0].to_dict()
            except ValueError as exc:
                raise DeepSeekV4CoverageError("V4 isolated prefill contract lacks native metadata") from exc
            layout = {
                **{name: columns[name] for name in (*_ENGINE_COLUMNS, "block_size",
                    "kv_cache_dtype", "max_model_len", "max_num_batched_tokens", "max_num_seqs")},
                **{name: columns[name] for name in ("routing_runtime_path",
                    "routing_strategy_sha256", "moe_routing") if name in columns},
                **metadata, "compress_ratio": max(1, layer.compress_ratio),
                "attention_variant": layer.attention_variant,
                "plans_tile_schedule": layer.plans_tile_schedule,
                **expected_identity, "input_file_sha256": input_file_sha256,
            }
            if layer.global_layer_id in layouts and layouts[layer.global_layer_id] != layout:
                raise DeepSeekV4CoverageError("V4 isolated prefill raw layouts disagree between steps")
            layouts[layer.global_layer_id] = layout
    metadata_columns = {name for layout in layouts.values() for name in layout if "_metadata." in name}
    return {layer: {**dict.fromkeys(metadata_columns), **layout} for layer, layout in layouts.items()}


def read_isolated_scope(predictor, batch, layer_id, scope, *, lane_workload=None,
                        attention_layout=None):
    """Consume one isolated eager measurement, never publish a serving duration.

    Attention scopes may overlap and composites cannot be split into their
    constituent operators. Physical TP1 with target per-rank TP8 dimensions is
    retained as such; this receipt does not assert TP8 or phase equivalence.
    Linear/MoE selection uses the existing typed readers and feature classifier.
    Attention requires an independently qualified complete flattened layout;
    accepted prefill and decode bind input bytes and runtime identity in that
    same explicit or externally admitted sidecar contract, never inferred from candidate rows. Reduced metadata
    in that contract never asserts serving capacity.
    """
    import json
    import math
    import pandas as pd
    from frontier.execution_time_predictor.profiling_dataframe_loaders import ProfilingDataFrameLoaders
    from frontier.profiling.attention.dsv4_attention_plan import DSV4_SCOPE_NAMES, scope_operators
    from frontier.types import ClusterType, MeasurementType

    _validate_source_batch(batch)
    if predictor._cluster_type is not ClusterType.MONOLITHIC:
        raise DeepSeekV4CoverageError("V4 isolated measured scopes do not qualify PD batches")
    if batch.get_decode_cuda_graph_runtime_mode() != "NONE":
        raise DeepSeekV4CoverageError("V4 native graph/compiled scope requires qualified replay measurements")
    phase = "prefill"
    if any(batch.request_is_decoding) or any(q == 1 for q in batch.num_tokens):
        # Native SWA classifies short prompt extends as decode kernels too.
        # The scheduled semantic snapshot and prompt extent exclude first-tail.
        if (scope not in DSV4_SCOPE_NAMES or len(batch.requests) != 1
                or batch.request_is_decoding != [True] or batch.num_tokens != [1]
                or batch.num_context_tokens[0] < batch.requests[0].num_prefill_tokens):
            raise DeepSeekV4CoverageError("V4 isolated decode requires one ongoing decode, not first-tail or mixed phases")
        phase = "decode"
    model, replica = predictor._model_config, predictor._replica_config
    if model.get_model_architecture_profile().profile_id != "deepseek_v4":
        raise ValueError("V4 measured scopes require the DeepSeek-V4 profile")
    measurement = predictor._active_measurement_type
    if measurement not in (MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY):
        raise DeepSeekV4CoverageError(f"V4 measurement type is unsupported: {measurement}")
    loaders = ProfilingDataFrameLoaders()
    loaders._model_config = model
    loaders._replica_config = replica
    tp = replica.attn_tensor_parallel_size
    layer = model.get_layer_attention_spec(layer_id)
    if scope in DSV4_SCOPE_NAMES:
        from frontier.attention.deepseek_v4 import build_deepseek_v4_attention_workload
        from frontier.profiling.attention.dsv4_attention_rows import _ENGINE_COLUMNS

        path = predictor._attention_input_file
        if attention_layout is None:
            from frontier.execution_time_predictor.dsv4_attention_input import load_isolated_prefill_layout

            attention_layout = load_isolated_prefill_layout(path, layer_id, measurement, phase)
        accepted_prefill = phase == "prefill" and (getattr(replica,
            "moe_gating_routing_runtime_path", None) == "balanced_topk" or attention_layout is not None
            and "input_file_sha256" in attention_layout)
        rows = (load_attention_dataframe(path, model, tp, predictor._block_size, measurement,
                                        float_precision="round_trip")
                if phase == "decode" or accepted_prefill else predictor._load_attention_df(path))
        layout_columns = set(_ENGINE_COLUMNS) | {
            "block_size", "kv_cache_dtype", "compress_ratio", "max_model_len",
            "max_num_batched_tokens", "max_num_seqs",
        } | {name for name in rows if "_metadata." in name}
        provenance_columns = {"source_revision", "runtime_fingerprint", "model_config_sha256",
                               "input_file_sha256"} if phase == "decode" or accepted_prefill else set()
        layout_columns |= provenance_columns
        if accepted_prefill:
            layout_columns |= {"attention_variant", "plans_tile_schedule"}
        if getattr(replica, "moe_gating_routing_runtime_path", None) == "balanced_topk":
            layout_columns |= {"routing_runtime_path", "routing_strategy_sha256", "moe_routing"}
        if attention_layout is None or set(attention_layout) != layout_columns:
            raise DeepSeekV4CoverageError("V4 attention requires a complete isolated layout contract")
        if provenance_columns:
            from frontier.profiling.common.runtime_identity import file_sha256

            if (any(not isinstance(attention_layout[name], str) or not attention_layout[name]
                    for name in provenance_columns)
                    or file_sha256(path) != attention_layout["input_file_sha256"]):
                raise DeepSeekV4CoverageError("V4 accepted input provenance does not match the measured table")
            rows = _select_rows(rows, {name: attention_layout[name] for name in provenance_columns
                                      if name != "input_file_sha256"}, "accepted input provenance")
        if getattr(replica, "moe_gating_routing_runtime_path", None) == "balanced_topk":
            from frontier.profiling.attention.dsv4_attention_rows import BALANCED_ROUTING_SHA256, routing_columns

            for name in ("routing_runtime_path", "routing_strategy_sha256", "moe_routing"):
                if name not in rows or rows[name].isna().any() or len(rows[name].unique()) != 1:
                    raise DeepSeekV4CoverageError(f"V4 balanced isolated attention missing/ambiguous {name}")
            rows = _select_rows(rows, {"routing_runtime_path": "balanced_topk",
                "routing_strategy_sha256": BALANCED_ROUTING_SHA256}, "balanced isolated attention")
            try:
                routing_columns(json.loads(rows.moe_routing.iloc[0]))
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise DeepSeekV4CoverageError("V4 balanced isolated attention routing identity mismatch") from exc
        rows = _select_rows(rows, {name: value for name, value in attention_layout.items()
                                  if name not in provenance_columns}, "isolated attention layout")
        source = build_deepseek_v4_attention_workload(model, replica, batch, layer_id,
            cluster_type=predictor._cluster_type, execution_mode="eager",
            has_attention_metadata=True, prefix_caching=False,
            decode_context_parallel_size=1, prefill_context_parallel_size=1)
        rows = _select_rows(rows, {
            "padded_heads": source["num_padded_q_heads"],
            "kv_cache_dtype": replica.kv_cache_dtype,
            "attention_kv_cache_dtype": replica.kv_cache_dtype,
        }, "attention source/cache layout")
        rows = _select_rows(rows, {
            "phase": phase, "attention_variant": layer.variant_id,
            "compress_ratio": max(1, model.compress_ratios[layer_id]),
            "plans_tile_schedule": not any(
                model.get_layer_attention_spec(i).variant_id == layer.variant_id
                for i in range(layer_id)
            ),
            "total_tokens": batch.total_num_tokens,
            "batch_size": len(batch.requests),
        }, scope)
        if phase == "decode":
            key_len = batch.num_context_tokens[0] + 1
            native = {
                "is_prefill": False, "is_mixed_batch": False,
                "num_prefill_seqs": 0, "num_decode_seqs": 1,
                "total_prefill_tokens": 0, "total_decode_tokens": 1,
                "max_key_len": key_len,
                "swa_metadata.num_decodes": 1, "swa_metadata.num_decode_tokens": 1,
                "swa_metadata.num_prefills": 0, "swa_metadata.num_prefill_tokens": 0,
                "swa_metadata.max_decode_query_len": 1,
                "global_layer_ids": json.dumps([layer_id]),
                "num_layers_sampled": 1, "num_steps_sampled": 1,
            }
            if layer.variant_id != "uncompressed":
                native.update({
                    "attention_metadata.max_query_len": 1,
                    "attention_metadata.max_seq_len": key_len,
                    "attention_metadata.num_actual_tokens": 1,
                    "attention_metadata.num_reqs": 1,
                })
            if layer.variant_id == "csa":
                native.update({
                    "indexer_metadata.max_seq_len": key_len,
                    "indexer_metadata.num_decodes": 1, "indexer_metadata.num_decode_tokens": 1,
                    "indexer_metadata.num_prefills": 0, "indexer_metadata.num_prefill_tokens": 0,
                })
            rows = _select_rows(rows, native, "ongoing decode native metadata")
        for column in ("prefill_seq_lens", "prefill_kv_cache_sizes", "decode_kv_cache_sizes", "operator_scopes"):
            if column not in rows:
                raise DeepSeekV4CoverageError(f"V4 {scope} missing {column}")
        pairs = sorted(zip(batch.num_tokens, batch.num_context_tokens))
        def matches_geometry(row):
            queries = json.loads(row["prefill_seq_lens"])
            contexts = json.loads(row["prefill_kv_cache_sizes"])
            if phase == "decode":
                return (queries == contexts == []
                        and json.loads(row["decode_kv_cache_sizes"]) == batch.num_context_tokens)
            return (len(queries) == len(contexts) == len(pairs)
                    and sorted(zip(queries, contexts)) == pairs
                    and json.loads(row["decode_kv_cache_sizes"]) == [])

        rows = rows[rows.apply(matches_geometry, axis=1)]
        rows = _select_rows(rows, {}, f"{scope} actual query/context")
        if phase == "decode" and len(rows) != 1:
            raise DeepSeekV4CoverageError("V4 isolated decode requires one exact measured observation")
        expected_operators = list(scope_operators(scope, layer.variant_id))
        if any(json.loads(value).get(scope) != expected_operators for value in rows.operator_scopes):
            raise DeepSeekV4CoverageError(f"V4 {scope} composite ownership is missing or mismatched")
    else:
        contract = model.get_model_architecture_profile().resolve_layer_contract(
            model, layer_id=layer_id, operator_name=scope,
            attention_tp_size=tp, attention_dp_size=replica.attn_dp,
            num_pipeline_stages=replica.num_pipeline_stages,
            moe_tp_size=replica.moe_tensor_parallel_size,
            expert_parallel_size=replica.moe_expert_parallel_size,
        ) if scope in ("share_expert_up_proj", "share_expert_act", "share_expert_down_proj", "moe_grouped_gemm") else None
        if contract is None:
            raise DeepSeekV4CoverageError(f"V4 {scope} has no qualified native measurement; generic routing columns do not qualify")
        if scope == "moe_grouped_gemm":
            from frontier.execution_time_predictor.moe_dataset_training import select_moe_operator_features
            from frontier.moe_ep_workload import EPLaneWorkload
            from frontier.moe_load_imbalance import MOE_LOAD_IMBALANCE_FEATURES, MoELoadImbalanceInput

            if not isinstance(lane_workload, EPLaneWorkload):
                raise DeepSeekV4CoverageError("V4 grouped GEMM requires an explicit expert lane, not inferred routing")
            if (lane_workload.moe_expert_parallel_size != contract.expert_parallel_size
                    or lane_workload.total_expert_num != model.num_experts
                    or lane_workload.router_topk != model.num_experts_per_tok):
                raise DeepSeekV4CoverageError("V4 grouped GEMM lane does not match the typed model/EP contract")
            path = predictor._moe_input_file
            rows = _read_isolated_component(loaders._load_moe_df, scope, path, replica,
                tensor_parallel_size=contract.tensor_parallel_size,
                expert_parallel_size=contract.expert_parallel_size,
                layer_contract=contract, operator_name=scope,
                # Preserve serialized float features for exact (not fitted)
                # lookup. The default parser rounds the accepted entropy row.
                float_precision="round_trip")
            rows = _select_rows(rows, {"moe_grouped_gemm_backend": "vllm_marlin_mxfp4",
                "moe_quantization_mode": "mxfp4", "moe_native_backend": "MARLIN"}, scope)
            rows, columns = select_moe_operator_features(scope, rows)
            if tuple(columns) != MOE_LOAD_IMBALANCE_FEATURES:
                raise DeepSeekV4CoverageError("V4 MARLIN isolated scope requires exact expert-load feature coverage")
            features = MoELoadImbalanceInput(
                num_tokens=sum(lane_workload.global_token_counts) // lane_workload.router_topk,
                num_experts_per_device=lane_workload.local_expert_width,
                hidden_dim=model.embedding_dim, expert_hidden_dim=contract.effective_ffn_width,
                router_topk=lane_workload.router_topk,
                expert_token_counts=list(lane_workload.local_token_counts),
            ).to_features_dict()
            rows = _select_rows(rows, {
                "num_tokens": sum(lane_workload.global_token_counts) // lane_workload.router_topk,
                **{name: features[name] for name in columns},
            }, scope)
        else:
            path = predictor._compute_input_file
            rows = _read_isolated_component(loaders._load_linear_op_df, scope,
                path, contract.tensor_parallel_size,
                required_columns=[f"time_stats.{scope}.median"],
                layer_contract=contract, operator_name=scope)
            local = (compute_rows(batch) + tp - 1) // tp
            rows = _select_rows(rows, {"num_tokens": local}, scope)
            if scope != "share_expert_act" and "linear_backends" not in rows:
                raise DeepSeekV4CoverageError(f"V4 {scope} missing projection backend identity")
            if scope != "share_expert_act" and any(
                not all(name == "FlashInferFp8DeepGEMMDynamicBlockScaledKernel"
                        for name in json.loads(value).get(scope, []))
                or not json.loads(value).get(scope) for value in rows.linear_backends
            ):
                raise DeepSeekV4CoverageError(f"V4 {scope} requires the qualified FP8 DeepGEMM projection backend")
    _validate_isolated_identity(rows, predictor, measurement)
    target = f"time_stats.{scope}.median"
    if target not in rows or any(not math.isfinite(float(v)) or float(v) <= 0 for v in rows[target]):
        raise DeepSeekV4CoverageError(f"V4 {scope} requires finite positive measured {target}")
    kernel_column = f"time_stats.{scope}.kernel_count"
    if measurement is MeasurementType.KERNEL_ONLY and (
        kernel_column not in rows or any(
            not math.isfinite(float(v)) or float(v) <= 0 or int(v) != v
            for v in rows[kernel_column]
        )
    ):
        raise DeepSeekV4CoverageError(f"V4 {scope} requires measured kernel counts, not relabelled CUDA_EVENT rows")
    row = rows.iloc[0]
    return {
        "scope": scope, "time_ms": float(rows[target].median()),
        "measurement_type": measurement.value, "graph_mode": "eager",
        "physical_tensor_parallel_size": 1, "target_tensor_parallel_size": tp,
        "serving_admission": False,
        "qualification": ("isolated eager scope only; reduced-model layer observation, not full serving"
                          if phase == "decode" else "isolated eager scope only"),
        "source_revision": rows.source_revision.iloc[0], "input_file": str(path),
        "runtime_fingerprint": rows.runtime_fingerprint.iloc[0],
        "model_config_sha256": rows.model_config_sha256.iloc[0],
        "parallel_groups": json.loads(row.parallel_groups),
        "attention_layout": dict(attention_layout) if scope in DSV4_SCOPE_NAMES else None,
        "row_metadata": {name: value.item() if hasattr(value, "item") else value
                         for name, value in row.items()
                         if (name.endswith(".block_size") or name in (
                              "phase", "padded_heads", "attention_variant", "plans_tile_schedule",
                              "attention_kv_cache_dtype", "max_model_len",
                              "max_num_batched_tokens", "max_num_seqs",
                              "global_layer_ids", "num_layers_sampled", "num_steps_sampled",
                              "prefill_seq_lens", "prefill_kv_cache_sizes", "decode_kv_cache_sizes",
                              "weight_source",
                              "routing_runtime_path", "routing_strategy_sha256", "moe_routing",
                         )) and not pd.isna(value)},
    }


def _validate_isolated_identity(rows, predictor, measurement):
    import json
    from frontier.profiling.common.runtime_identity import IDENTITY_COLUMNS, RuntimeIdentity

    for name in (*IDENTITY_COLUMNS, "model_architecture_profile", "measurement_type", "profiling_precision"):
        if name not in rows or rows[name].isna().any() or len(rows[name].unique()) != 1:
            raise DeepSeekV4CoverageError(f"V4 isolated scope has missing/ambiguous {name}")
    row = rows.iloc[0]
    expected = {
        "graph_mode": "eager", "measurement_type": measurement.value,
        "model_architecture_profile": "deepseek_v4",
        "quant_signature": predictor._model_config.get_quant_signature(),
        "profiling_precision": predictor._model_config.get_default_precision().name,
    }
    _select_rows(rows, expected, "isolated identity")
    groups = json.loads(row.parallel_groups)
    if groups.get("engine_tensor_parallel") != 1:
        raise DeepSeekV4CoverageError("V4 isolated scope requires its measured physical TP1 identity")
    if str(predictor._replica_config.device).lower() not in row.device_name.lower():
        raise DeepSeekV4CoverageError("V4 measured device does not match the selected replica")
    identity = RuntimeIdentity(**{name: (json.loads(row[name]) if name in (
        "runtime_versions", "parallel_groups") else row[name]) for name in (
        "runtime_versions", "source_revision", "model_config_revision", "model_config_sha256",
        "device_name", "device_capability", "schema_version", "quant_signature",
        "parallel_groups", "graph_mode",
    )})
    if identity.fingerprint != row.runtime_fingerprint:
        raise DeepSeekV4CoverageError("V4 measured runtime fingerprint does not match its provenance")
