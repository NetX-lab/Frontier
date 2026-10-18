"""Source-scoped eager V4 dummy execution, with native execution gates closed.

Source: vLLM 2cf0a6915ce544dc493a0990f2ea38d81601128a,
models/deepseek_v4/nvidia/model.py and common/ops/sequence_parallel.py.
"""

from frontier.attention.deepseek_v4 import (
    _integer, _validate_source_batch, validate_deepseek_v4_dummy_forward,
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
    tokens = sum(batch.num_tokens)
    last = layer_id == model.num_layers - 1
    if last:
        selected_logits_rows = _integer(
            selected_logits_rows, "selected_logits_rows", minimum=0,
        )
        if selected_logits_rows > tokens:
            raise ValueError("selected_logits_rows cannot exceed source rows")
        if isinstance(batch, DummyForwardBatch) and selected_logits_rows:
            raise ValueError("V4 DummyForwardBatch cannot select LM rows")
    elif selected_logits_rows is not None:
        raise ValueError("selected_logits_rows is only accepted at the last layer")
    local = (tokens + tp - 1) // tp
    valid = max(0, min(local, tokens - rank * local))
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
        "status": "PARTIAL", "source_rows": tokens,
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
            "input_shape": (source["source_rows"], predictor._model_config.embedding_dim),
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
        "physical_local_rows": source["physical_local_rows"],
        "valid_local_rows": source["valid_local_rows"],
        "request_valid_local_rows": (0 if isinstance(batch, DummyForwardBatch)
                                     else source["valid_local_rows"]),
        "source_forward_kind": ("request-free dummy" if isinstance(batch, DummyForwardBatch)
                                else "scheduled requests"),
        "reference_workflow_scope": "eager NONE; no DBO/ubatching; compiler-pass SP disabled; model-internal SP enabled",
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
    """Resolve the existing ordinary eager dummy workload, with no new config."""
    from frontier.config import global_vars

    if not predictor._enable_dummy_mode:
        raise NotImplementedError("V4 measured/native execution remains unqualified")
    if (global_vars.USE_CUDA_GRAPH or global_vars.DECODE_CUDA_GRAPH_MODE != "none"
            or global_vars.KV_CONNECTOR != "none"):
        raise NotImplementedError("V4 structural serving requires eager co-location without PD")
    return dict(
        execution_mode="eager", has_attention_metadata=not isinstance(batch, DummyForwardBatch),
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
    return sum(((sum(batch.num_tokens) + tp - 1) // tp) * tp
               for batch in source_batches.values())
