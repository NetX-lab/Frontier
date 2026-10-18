"""Eager source-rank V4 component queries, without full execution admission.

Source: vLLM 2cf0a6915ce544dc493a0990f2ea38d81601128a,
models/deepseek_v4/nvidia/model.py and common/ops/sequence_parallel.py.
"""

from frontier.attention.deepseek_v4 import _integer, _validate_source_batch
from frontier.entities.time_components import MoEOperatorTimes, MoETime
from frontier.operators.binding import build_operator_manifest
from frontier.operators.families import get_comm_operator


def predict_source_execution(
    predictor, batch, layer_id, tp_rank, *, execution_mode,
    has_attention_metadata, prefix_caching, decode_context_parallel_size,
    prefill_context_parallel_size, enable_expert_parallel, moe_backend,
    all2all_backend, skip_padding, input_padding_mask, selected_logits_rows,
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
    _validate_source_batch(batch)
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

    if layer_id == 0:
        local_scope(
            "dsv4_embedding", (tokens,), (tokens, hidden),
            cost_scope="inclusive synthetic embedding wrapper",
        )
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
    # Attention is one opaque component query: its private map is not a launch schedule.
    order.append({"scope": "attention", "time_ms": attention.total_time()})
    communication("dsv4_sp_attention_reduce_scatter")
    local_scope(
        "dsv4_mhc_attn_post_ffn_pre", (local, hidden), (local, hidden),
        residual_shape=(local, streams, hidden),
        post_mix_shape=(local, streams, 1), res_mix_shape=(local, streams, streams),
        state_dtype="float32", activation_dtype="bfloat16",
    )
    local_scope(shared_names[0], (local, hidden), (local, 2 * intermediate), weight_tp=shared_tp)
    local_scope(shared_names[1], (local, 2 * intermediate), (local, intermediate))
    local_scope(shared_names[2], (local, intermediate), (local, hidden), weight_tp=shared_tp)
    if last:
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
