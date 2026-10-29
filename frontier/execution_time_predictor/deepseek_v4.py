"""Source-scoped V4 workloads and qualified isolated measured components.

Source: vLLM 2cf0a6915ce544dc493a0990f2ea38d81601128a,
models/deepseek_v4/nvidia/model.py and common/ops/sequence_parallel.py.
"""

from frontier.attention.deepseek_v4 import (
    DeepSeekV4CoverageError, _integer, _validate_source_batch, compute_rows,
    validate_deepseek_v4_dummy_forward, build_deepseek_v4_attention_workload,
)
from frontier.entities.batch import DummyForwardBatch
from frontier.execution_time_predictor.dsv4_composition import serial_attention_phase
from frontier.entities.time_components import AttentionOperatorTimes, MoEOperatorTimes, MoETime
from frontier.attention.dsv4_features import build_dsv4_features, build_dsv4_dummy_features, dsv4_request_is_decoding
from frontier.operators.binding import build_operator_manifest
from frontier.operators.families import get_comm_operator


def timing_operator_times(execution_time, phase):
    """Project physical V4 scopes into the existing serial structural phases.

    This ordering is a synthetic execution contract, not an observed native
    launch schedule or a claim about auxiliary-stream overlap.
    """
    from frontier.operators.families import COMM_FAMILY
    from frontier.execution_time_predictor.dsv4_runner import ZERO_SCOPE, WRITE_SCOPE, INPUT_SCOPE, SAMPLING_SCOPE

    names_by_phase = {
        # Work before the forward's DP rendezvous: the runner zeroes newly
        # allocated cache blocks before it synchronizes the DP descriptor.
        "pre_dp": (ZERO_SCOPE, WRITE_SCOPE),
        "attention": (
            INPUT_SCOPE, "dsv4_embedding", "dsv4_embedding_allreduce",
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
            "dsv4_ep_dispatch_allgather",
        ),
        "routed_compute": ("moe_grouped_gemm",),
        "combine": ("dsv4_ep_output_reduce_scatter",),
        "post_combine": (
            "dsv4_mhc_final_post", "dsv4_sp_final_mhc_allgather",
            "dsv4_mhc_head", "dsv4_final_norm", "dsv4_lm_head", "dsv4_logits_gather",
            SAMPLING_SCOPE,
        ),
    }
    supported = {name for names in names_by_phase.values() for name in names}
    unsupported = set(execution_time.op_times) - supported
    if unsupported:
        raise ValueError(f"V4 physical timing scopes have no phase owner: {sorted(unsupported)}")
    comm_names = {op.name for op in COMM_FAMILY.operators}
    return tuple(
        ("COMM" if name in comm_names else "COMPUTE", name,
         execution_time.op_times[name] + getattr(execution_time, "_dsv4_operator_gap_ms", {}).get(name, 0.0))
        for name in names_by_phase[phase] if name in execution_time.op_times
    )


def predict_attention_operator_times(
    predictor, batch, layer_id, cluster_type, *, execution_mode,
    prefix_caching, decode_context_parallel_size, prefill_context_parallel_size,
) -> AttentionOperatorTimes:
    """Price V2 attention scopes with synthetic calls or replay costs and gaps."""
    model = predictor._model_config
    workload = build_deepseek_v4_attention_workload(
        model, predictor._replica_config, batch, layer_id,
        cluster_type=cluster_type, execution_mode=execution_mode,
        prefix_caching=prefix_caching,
        decode_context_parallel_size=decode_context_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
    )
    if predictor._enable_dummy_mode:
        timing = AttentionOperatorTimes({
            scope: predictor._dummy_execution_time * len(calls)
            for scope, calls in workload["operators"].items()
        })
    else:
        rows = workload["num_tokens"]
        variant = workload["variant"]
        features = (build_dsv4_dummy_features(rows) if isinstance(batch, DummyForwardBatch)
                    else build_dsv4_features(
                        batch.num_tokens, batch.num_context_tokens, dsv4_request_is_decoding(batch),
                        variant=variant, compress_ratio=max(1, model.compress_ratios[layer_id]),
                        sliding_window=model.sliding_window, index_topk=model.index_topk,
                        padded_rows=rows,
                    ))
        attention_predictor = predictor._predictions["dsv4_attention"]
        prediction = attention_predictor.scope_times(
            features, variant=variant, graph_runtime_mode=workload["graph_runtime_mode"],
            plans_tile_schedule=workload["plans_tile_schedule"],
            scopes=tuple(workload["operators"]),
        )
        timing = AttentionOperatorTimes(prediction.cost_ms)
        timing._dsv4_scope_gap_ms = prediction.gap_ms
        timing._dsv4_composition_inputs = dict(
            variant=variant, graph_runtime_mode=workload["graph_runtime_mode"], rows=rows,
            multi_stream_gemm_token_threshold=attention_predictor.multi_stream_gemm_token_threshold,
            indexer_short_context=workload["indexer_short_context"],
        )
    timing._dsv4_workloads = workload["operators"]
    return timing


def predict_source_execution(
    predictor, batch, layer_id, tp_rank, *, execution_mode,
    prefix_caching, decode_context_parallel_size,
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
    else:
        _validate_source_batch(batch)
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
    base = predictor._dummy_execution_time if predictor._enable_dummy_mode else 0.0
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
    first_pre = "dsv4_mhc_first_pre" if layer_id == 0 else "dsv4_mhc_ffn_post_attn_pre"
    component_times = {}
    order = []
    workloads = {}
    kernel_counts = {}

    def local_scope(name, input_shape, output_shape, **features):
        if predictor._enable_dummy_mode:
            value = base
        else:
            from frontier.execution_time_predictor.dsv4_operator_predictor import (
                operator_rows, operator_tensor_parallel_size,
            )
            rows = operator_rows(name, sp_local_rows=local, descriptor_rows=tokens,
                                 logits_rows=selected_logits_rows)
            selected_tp = operator_tensor_parallel_size(model, replica, name)
            prediction = predictor._predictions["dsv4_operators"].operator(name, rows, selected_tp)
            value = prediction.time_ms
            kernel_counts[name] = prediction.kernel_count
        component_times[name] = value
        workloads[name] = {
            "input_shape": input_shape, "output_shape": output_shape, **features,
        }
        order.append({"scope": name, "time_ms": value})

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
        attention = predict_attention_operator_times(
            predictor, batch, layer_id, predictor._cluster_type, execution_mode=execution_mode,
            prefix_caching=prefix_caching,
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
    shared = MoETime(operator_times=MoEOperatorTimes(
        {name: component_times[name] for name in shared_names if name in component_times}
    ))
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
        "kernel_counts": kernel_counts,
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
    predictor, batch, layer_id, tp_rank, *, lane_workload=None, sp_token_widths=None,
    include_attention=True, include_ffn=True, include_stage_owned=False,
    **source_context,
):
    """Connect one source rank and an explicitly synthetic destination lane.

    The ordinary modular W4A16 scenario transports BF16 activations, FP32
    top-k weights and profile-owned int64 IDs. Native auto-backend selection, padding-mask
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
    from frontier.config import get_quantization_manager
    from frontier.operators.spec import CommOperatorSpec, CommPayloadContext

    payload_context = CommPayloadContext(
        batch, predictor._model_config, replica, predictor._cluster_type,
        get_quantization_manager(), sp_token_widths=sp_token_widths,
        selected_logits_rows=source_context["selected_logits_rows"],
    )

    def communication(name):
        return predictor._predict_comm_operator_with_context(get_comm_operator(name), payload_context)

    times = dict(source["component_times"])
    workloads = dict(source["workloads"])
    base = predictor._dummy_execution_time if predictor._enable_dummy_mode else 0.0
    kernel_counts = dict(source["kernel_counts"])
    from frontier.execution_time_predictor.dsv4_runtime import add_runner_scopes

    add_runner_scopes(
        predictor, batch, layer_id, include_attention=include_attention, include_ffn=include_ffn,
        selected_logits_rows=source_context["selected_logits_rows"],
        times=times, counts=kernel_counts, workloads=workloads,
    )
    if include_attention and layer_id == 0:
        # The embedding scope now excludes its separately owned reduction.
        times["dsv4_embedding_allreduce"] = communication("dsv4_embedding_allreduce")
    if include_ffn:
        times.update({
            "moe_gating_linear": base, "moe_gating_routing_topk": base,
            # The fused expert composite owns its local alignment/reduction work.
            "moe_grouped_gemm": base if lane_workload.routed_token_count else 0.0,
        })
        local = source["physical_local_rows"]
        if not predictor._enable_dummy_mode:
            operator_predictor = predictor._predictions["dsv4_operators"]
            for name in ("moe_gating_linear", "moe_gating_routing_topk"):
                prediction = operator_predictor.gate(name, local)
                times[name] = prediction.time_ms
                kernel_counts[name] = prediction.kernel_count
            prediction = operator_predictor.grouped_gemm(lane_workload)
            times["moe_grouped_gemm"] = prediction.time_ms
            kernel_counts["moe_grouped_gemm"] = prediction.kernel_count
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
                "topk_weights_dtype": "float32",
                "topk_ids_dtype": predictor._model_config.get_model_architecture_profile().require_gathered_topk_ids_dtype(),
                "routing_variant": "hash" if layer_id < predictor._model_config.num_hash_layers else "score",
            },
            "moe_grouped_gemm": {
                "input_shape": (lane_workload.routed_token_count, hidden),
                "output_shape": (lane_workload.routed_token_count, hidden),
                "cost_scope": "inclusive fused expert FFN including local alignment/reduction",
            },
        })
        for name in (
            "dsv4_ep_dispatch_allgather", "dsv4_ep_output_reduce_scatter",
        ):
            times[name] = communication(name)
        if layer_id == predictor._model_config.num_layers - 1 and source_context["selected_logits_rows"]:
            times["dsv4_logits_gather"] = communication("dsv4_logits_gather")
            workloads["dsv4_logits_gather"] = {
                "selected_logits_rows": source_context["selected_logits_rows"],
                "logical_vocabulary_size": predictor._model_config.vocab_size,
                "physical_vocabulary_padding": "UNKNOWN",
            }
    from frontier.operators.families import COMM_FAMILY

    for operator in COMM_FAMILY.operators:
        if operator.name not in times or not isinstance(operator, CommOperatorSpec):
            continue
        workloads.setdefault(operator.name, {}).update({
            "data_size_bytes": operator.build_payload_bytes(payload_context),
            "collective": operator.collective_alias,
            "comm_domain": operator.resolve_comm_domain(payload_context),
            "num_devices": operator.num_devices(payload_context),
            "payload_scope": "CC collective input tensor; not native wire traffic",
        })
    if "dsv4_embedding_allreduce" in times:
        workloads["dsv4_embedding_allreduce"].update({
            "input_shape": (source["compute_rows"], predictor._model_config.embedding_dim),
            "activation_dtype": "bfloat16", "collective": "allreduce",
            "cost_scope": "communication backend device-time prediction",
        })
    spec = predictor._model_config.get_layer_attention_spec(layer_id)
    cpu = base if predictor._enable_dummy_mode and include_stage_owned else 0.0
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
        "ep_transport": "grouped BF16 hidden/FP32 weights/profile-owned int64 ids",
        "ep_collective_cost": "communication backend; measured unequal combine requires exact ordered SP rows",
        "endpoint_collective_cost": "communication backend device-time prediction",
        "synthetic_destination_assignments": lane_workload.routed_token_count if include_ffn else None,
    })
    timing._dsv4_workloads = MappingProxyType(workloads)
    if predictor._enable_dummy_mode:
        timing._dsv4_attention_schedule = serial_attention_phase(
            (name, value) for _, name, value in timing_operator_times(timing, "attention")
        )
    else:
        from frontier.execution_time_predictor.dsv4_runtime import compose_trained_attention

        timing._dsv4_kernel_counts = MappingProxyType(kernel_counts)
        attention = source["attention_time"]
        timing._dsv4_scope_gap_ms = attention._dsv4_scope_gap_ms if include_attention else {}
        if include_attention:
            timing._dsv4_composition_inputs = attention._dsv4_composition_inputs
            compose_trained_attention(predictor, timing, batch)
        else:
            timing._dsv4_attention_schedule = serial_attention_phase(())
    return timing


def predict_step_cost_execution(
    predictor, batch, layer_id, tp_rank, *, lane_workload=None,
    include_attention=True, include_ffn=True, include_stage_owned=False,
    **source_context,
):
    """Spread one measured lane step evenly over the layers of its forward.

    Each layer's attention call carries the step's per-layer share, so the
    per-layer EP synchronization prices the attention-DP cohort at its slowest
    lane. FFN, EP-wave, idle-forward and CPU work add nothing: the measured
    step already contains them.
    """
    from types import MappingProxyType

    from frontier.entities import ExecutionTime

    times, workloads = {}, {}
    if include_attention and not isinstance(batch, DummyForwardBatch):
        table = predictor._dsv4_step_costs
        step_ms = table.step_time_ms(predictor._cluster_type.name, batch)
        times["dsv4_sparse_attention"] = step_ms / predictor._model_config.num_layers
        workloads["dsv4_sparse_attention"] = {
            "cost_scope": "measured native lane step divided evenly over the layers; "
                          "covers the whole layer, not sparse attention alone",
            "step_cost_table": table.path,
        }
    spec = predictor._model_config.get_layer_attention_spec(layer_id)
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
        schedule_time=0.0, sampler_e2e_time=0.0, prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0, ray_comm_time=0.0, is_moe=True,
        op_times=times, global_layer_id=layer_id,
        attention_family_id=spec.family_id, attention_variant_id=spec.variant_id,
    )
    timing._dsv4_workloads = MappingProxyType(workloads)
    timing._dsv4_attention_schedule = serial_attention_phase(
        (name, value) for _, name, value in timing_operator_times(timing, "attention")
    )
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
    from frontier.execution_time_predictor.dsv4_runner import ZERO_SCOPE, WRITE_SCOPE, INPUT_SCOPE, SAMPLING_SCOPE

    if op_name in (ZERO_SCOPE, WRITE_SCOPE, INPUT_SCOPE, SAMPLING_SCOPE):
        return {
            "source_scope": "vllm 2cf0a6915ce544dc493a0990f2ea38d81601128a: V2 model runner",
            "cost_source": "runner scope kernel and memcpy medians; dummy mode uses its configured duration",
            "gap_scope": "eager_preparation per device activity; host time zero",
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
    compute_rows(batch)
    return dict(
        execution_mode=("eager" if batch.get_decode_cuda_graph_runtime_mode() == "NONE"
                        else "cuda_graph"),
        # The component cache independently rejects prefix caching/MTP.
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


def source_sp_token_widths(predictor, source_batches):
    """Resolve ordered V2 descriptor SP widths, including idle forwards."""
    from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths
    replica = predictor._replica_config
    if tuple(sorted(source_batches)) != tuple(range(replica.attn_dp)):
        raise ValueError("V4 source cohort must contain every attention DP rank")
    for batch in source_batches.values():
        if isinstance(batch, DummyForwardBatch):
            validate_deepseek_v4_dummy_forward(batch)
        else:
            _validate_source_batch(batch)
    tp = replica.attn_tensor_parallel_size
    return sequence_parallel_moe_token_widths(
        [compute_rows(source_batches[d]) for d in range(replica.attn_dp)], tp,
    )


def _select_rows(rows, expected, scope):
    for column, value in expected.items():
        if column not in rows:
            raise DeepSeekV4CoverageError(f"V4 {scope} missing column {column}")
        rows = rows[rows[column].isna() if value is None else rows[column] == value]
    if rows.empty:
        raise DeepSeekV4CoverageError(f"V4 {scope} has no exact measured rows: {expected}")
    return rows


def validate_measured_identity(rows, model_config, replica_config, measurement):
    """Check measured provenance for both eager scopes and graph replay tables."""
    import json
    from frontier.profiling.common.runtime_identity import IDENTITY_COLUMNS, RuntimeIdentity

    for name in (*IDENTITY_COLUMNS, "model_architecture_profile", "measurement_type", "profiling_precision"):
        if name not in rows or rows[name].isna().any() or len(rows[name].unique()) != 1:
            raise DeepSeekV4CoverageError(f"V4 isolated scope has missing/ambiguous {name}")
    row = rows.iloc[0]
    expected = {
        "measurement_type": measurement.value,
        "model_architecture_profile": "deepseek_v4",
        "quant_signature": model_config.get_quant_signature(),
        "profiling_precision": model_config.get_default_precision().name,
    }
    _select_rows(rows, expected, "isolated identity")
    groups = json.loads(row.parallel_groups)
    if groups.get("engine_tensor_parallel") != 1:
        raise DeepSeekV4CoverageError("V4 isolated scope requires its measured physical TP1 identity (parallel_groups)")
    if str(replica_config.device).lower() not in row.device_name.lower():
        raise DeepSeekV4CoverageError("V4 measured device does not match the selected replica (device_name)")
    identity = RuntimeIdentity(**{name: (json.loads(row[name]) if name in (
        "runtime_versions", "parallel_groups") else row[name]) for name in (
        "runtime_versions", "source_revision", "model_config_revision", "model_config_sha256",
        "device_name", "device_capability", "schema_version", "quant_signature",
        "parallel_groups", "graph_mode",
    )})
    if identity.fingerprint != row.runtime_fingerprint:
        raise DeepSeekV4CoverageError("V4 measured runtime fingerprint does not match its provenance (runtime_fingerprint)")
