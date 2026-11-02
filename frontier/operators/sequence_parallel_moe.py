"""vLLM sequence-parallel collective geometry and source-owned timing."""

# The frontier.config package imports the operator families, which import this
# module while they initialize, so functions import frontier.config names when
# they run.


SP_TP_OPERATORS = (
    "sp_moe_attention_reduce_scatter",
    "sp_moe_layer_input_allgather",
    "sp_moe_final_allgather",
)
EMBEDDING_LOGITS_OPERATORS = ("embedding_allreduce", "logits_allgather")


def uses_sequence_parallel_moe(model, replica):
    return model.get_model_architecture_profile().uses_sequence_parallel_moe(
        replica.attn_tensor_parallel_size, replica.attn_dp,
        replica.num_pipeline_stages, replica.moe_expert_parallel_size,
    )


def source_tp_operator_names(model, replica, layer_id):
    """Select source-owned calls, including the dense-to-MoE boundary."""
    if replica.attn_tensor_parallel_size <= 1:
        return ()
    names = []
    if uses_sequence_parallel_moe(model, replica) and model.is_moe_layer(layer_id):
        if layer_id > 0 and model.is_moe_layer(layer_id - 1):
            names.append("sp_moe_layer_input_allgather")
        names.append("sp_moe_attention_reduce_scatter")
        if layer_id == model.num_layers - 1:
            names.append("sp_moe_final_allgather")
    if model.get_model_architecture_profile().embedding_logits_collectives:
        if layer_id == 0:
            names.append("embedding_allreduce")
        if layer_id == model.num_layers - 1:
            names.append("logits_allgather")
    return tuple(names)


def tp_workload(name, model, tp, tokens, logits_rows=None):
    """Largest rank input; activations stay BF16 even with FP8 weights."""
    from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths
    from frontier.config.precision_type import PrecisionType

    hidden = model.embedding_dim
    bf16_bytes = int(PrecisionType.BF16.bytes_per_element)
    rows = sequence_parallel_moe_token_widths([tokens], tp).local_tokens[0]
    if name == "sp_moe_attention_reduce_scatter":
        shape, collective = [tp * rows, hidden], "reduce_scatter"
    elif name == "sp_moe_layer_input_allgather":
        shape, collective = [rows, hidden], "all_gather"
    elif name == "sp_moe_final_allgather":
        # deepseek_v2.py:1498-1501 concatenates hidden and residual before gather.
        shape, collective = [rows, 2 * hidden], "all_gather"
    elif name == "dsv4_sp_attention_allgather":
        shape, collective = [rows, hidden], "sp_all_gather"
    elif name == "dsv4_sp_attention_reduce_scatter":
        shape, collective = [tokens, hidden], "sp_reduce_scatter"
    elif name == "dsv4_sp_final_mhc_allgather":
        shape, collective = [rows, model.hc_mult * hidden], "sp_all_gather"
    elif name in ("embedding_allreduce", "dsv4_embedding_allreduce"):
        shape, collective = [tokens, hidden], "all_reduce"
    elif name in ("logits_allgather", "dsv4_logits_gather"):
        if model.vocab_size % tp:
            raise ValueError("Logits all-gather requires vocabulary divisible by attention TP")
        if logits_rows is None:
            raise ValueError("Logits all-gather requires selected_logits_rows")
        shape, collective = [logits_rows, model.vocab_size // tp], "all_gather"
    else:
        raise ValueError(f"Unknown source collective {name!r}")
    size = shape[0] * shape[1] * bf16_bytes
    return {
        "collective": collective, "tensor_shape": {"input": shape},
        "tensor_dtype": {"input": "BF16"}, "data_size_bytes": size,
        "base_size_bytes": size, "dtype": "BF16", "dtype_bytes": bf16_bytes,
        "byte_scope": "largest_rank_input",
    }


def tp_payload(name, ctx):
    return tp_workload(
        name, ctx.model_config, ctx.replica_config.attn_tensor_parallel_size,
        ctx.batch.get_effective_total_tokens_rounded(ctx.cluster_type),
        # gpu_model_runner.py:2304 selects one row per scheduled request,
        # including partial prefill; dummy forwards contain no requests.
        logits_rows=len(ctx.batch.requests),
    )["data_size_bytes"]


def source_tp_operator_times(predictor, batch, model, replica, role, layer_id):
    """Price the same physical source calls in live and structural PD execution."""
    from frontier.config import get_quantization_manager
    from frontier.operators.families import get_comm_operator
    from frontier.operators.spec import CommPayloadContext

    profile = predictor._get_cluster_model_architecture_profile(role)
    if not (profile.replicate_shared_expert_in_sequence_parallel
            or profile.embedding_logits_collectives):
        return {}
    names = source_tp_operator_names(model, replica, layer_id)
    if predictor._enable_dummy_mode:
        return {name: predictor._dummy_execution_time for name in names
                if name != "logits_allgather" or batch.requests}
    ctx = CommPayloadContext(batch, model, replica, role, get_quantization_manager())
    return {name: predictor._predict_comm_operator_with_context(get_comm_operator(name), ctx)
            for name in names}


def ep_widths(ctx):
    widths = ctx.sp_token_widths
    if widths is None:
        widths = ctx.batch.moe_sequence_parallel_token_widths
    if widths is None:
        raise ValueError("SP EP collective requires source DP-lane token widths")
    return widths


def ep_workload(name, model, topk, widths):
    from frontier.config.precision_type import PrecisionType

    bf16_bytes = int(PrecisionType.BF16.bytes_per_element)
    if name in ("expert_parallel_alltoall_dispatch", "dsv4_ep_dispatch_allgather"):
        import numpy as np

        ids_dtype = model.get_model_architecture_profile().require_gathered_topk_ids_dtype()
        ids_bytes = np.dtype(ids_dtype).itemsize
        rows = max(widths.local_tokens)
        shapes = {"hidden": [rows, model.embedding_dim],
                  "topk_weights": [rows, topk], "topk_ids": [rows, topk]}
        dtypes = {"hidden": "BF16", "topk_weights": "FP32", "topk_ids": ids_dtype.upper()}
        size = rows * (model.embedding_dim * bf16_bytes
                       + topk * int(PrecisionType.FP32.bytes_per_element) + topk * ids_bytes)
        collective = "all_gatherv"
    elif name in ("expert_parallel_alltoall_combine", "dsv4_ep_output_reduce_scatter"):
        shapes = {"hidden": [widths.gathered_tokens, model.embedding_dim]}
        dtypes = {"hidden": "BF16"}
        size = widths.gathered_tokens * model.embedding_dim * bf16_bytes
        collective = "reduce_scatterv"
    else:
        raise ValueError(f"Unknown EP phase {name!r}")
    return {
        "collective": collective, "tensor_shape": shapes, "tensor_dtype": dtypes,
        "data_size_bytes": size, "base_size_bytes": size,
        "dtype": "BF16", "dtype_bytes": bf16_bytes,
        "byte_scope": "largest_rank_input", "source_sp_rows": widths.local_tokens,
    }


def combine_sp_rows(ctx):
    """Supply ordered rows for SP combine specs; ordinary all-to-all has none."""
    if uses_sequence_parallel_moe(ctx.model_config, ctx.replica_config):
        return ep_widths(ctx).local_tokens
    return None


def measured_collective_time(operator, ctx, backend, size):
    """Price every measured communication spec through its implementation."""
    from frontier.cc_backend.backends.measured_cc_backend import MeasuredCCBackend

    if not isinstance(backend, MeasuredCCBackend):
        return None
    rows = operator.sp_rows_builder(ctx) if operator.sp_rows_builder is not None else None
    return backend.predict_operator_collective(
        operator.resolve_collective_alias(ctx), operator.collective_implementation,
        size, operator.num_devices(ctx), operator.resolve_comm_domain(ctx), rows,
    )
