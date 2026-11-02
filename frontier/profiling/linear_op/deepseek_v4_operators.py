"""Native V4 standalone calls and timing, imported without torch or vLLM.

The worker constructs synthetic parameters with native shapes/dtypes. No engine,
TP collectives, routed experts, or checkpoint loading belongs to these scopes.
"""

from __future__ import annotations

from pathlib import Path

from frontier.profiling.linear_op.deepseek_v4_plan import row_contracts


def _mhc_call(model, name, rows):
    import torch
    from vllm.model_executor.kernels.mhc.tilelang import (
        hc_head_fused_kernel_tilelang, mhc_fused_post_pre_tilelang,
        mhc_post_tilelang, mhc_pre_broadcast_tilelang,
    )

    hidden, streams = model.embedding_dim, model.hc_mult

    def activation(shape):
        return torch.randn(shape, dtype=model.dtype, device="cuda")

    def state(shape):
        return torch.randn(shape, dtype=torch.float32, device="cuda")

    if name == "dsv4_mhc_head":
        x = activation((rows, streams, hidden))
        fn, scale, base = state((streams, streams * hidden)), state((1,)), state((streams,))
        return lambda: hc_head_fused_kernel_tilelang(
            x, fn, scale, base, model.rms_norm_eps, model.hc_eps,
        )
    x = activation((rows, hidden))
    residual = activation((rows, streams, hidden))
    post_mix, res_mix = state((rows, streams, 1)), state((rows, streams, streams))
    if name == "dsv4_mhc_final_post":
        return lambda: mhc_post_tilelang(x, residual, post_mix, res_mix)
    mix_width = (2 + streams) * streams
    fn, scale, base = state((mix_width, streams * hidden)), state((3,)), state((mix_width,))
    norm_weight = torch.ones(hidden, dtype=model.dtype, device="cuda")
    # Native DeepseekV4DecoderLayer.hc_post_alpha is 2.0 (model.py:853).
    args = (fn, scale, base, model.rms_norm_eps, model.hc_eps, model.hc_eps,
            2.0, model.hc_sinkhorn_iters)
    if name == "dsv4_mhc_first_pre":
        # Same checkpoint-weight fold as finalize_mhc_broadcast_weights.
        broadcast = fn.view(-1, streams, hidden).sum(dim=1)
        return lambda: mhc_pre_broadcast_tilelang(
            x, *args, norm_weight=norm_weight, norm_eps=model.rms_norm_eps,
            fn_broadcast=broadcast,
        )
    # Separate calls/closures allocate distinct attention-side and FFN-side parameters.
    return lambda: mhc_fused_post_pre_tilelang(
        x, residual, post_mix, res_mix, *args, n_splits=1, tile_n=1,
        norm_weight=norm_weight, norm_eps=model.rms_norm_eps,
    )


def _vocab_shard(model, tp):
    import torch
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod, VocabParallelEmbedding, pad_vocab_size,
    )

    padded = pad_vocab_size(model.vocab_size)
    indices = VocabParallelEmbedding._get_indices(
        padded, padded, model.vocab_size, model.vocab_size, 0, tp,
    )
    layer = torch.nn.Module()
    method = UnquantizedEmbeddingMethod()
    # VocabParallelEmbedding's constructor reads the live TP group; its local
    # forward parts only need these indices and the method's native weight.
    method.create_weights(
        layer, input_size_per_partition=model.embedding_dim,
        output_partition_sizes=[indices.num_elements_padded],
        input_size=model.embedding_dim, output_size=padded,
        params_dtype=model.dtype, weight_loader=None,
    )
    with torch.no_grad():
        layer.weight.normal_()
    return layer, method, indices


def _embedding_call(model, rows, tp):
    import torch
    from vllm.model_executor.layers.vocab_parallel_embedding import get_masked_input_and_mask

    layer, method, indices = _vocab_shard(model, tp)
    input_ids = torch.randint(model.vocab_size, (rows,), dtype=torch.long, device="cuda")

    def call():
        if tp > 1:
            masked_input, input_mask = get_masked_input_and_mask(
                input_ids, indices.org_vocab_start_index, indices.org_vocab_end_index,
                indices.num_org_vocab_padding, indices.added_vocab_start_index,
                indices.added_vocab_end_index,
            )
        else:
            masked_input = input_ids
        output = method.embedding(layer, masked_input.long())
        if tp > 1:
            output.masked_fill_(input_mask.unsqueeze(-1), 0)
        return output

    return call


def _lm_head_call(model, rows, tp):
    import torch

    layer, method, _ = _vocab_shard(model, tp)
    x = torch.randn((rows, model.embedding_dim), dtype=model.dtype, device="cuda")
    # ParallelLMHead has no quantization config and uses this same method.
    return lambda: method.apply(layer, x, bias=None)


def _final_norm_call(model, rows):
    import torch
    from vllm import ir
    from vllm.model_executor.layers.layernorm import RMSNorm

    norm = RMSNorm(model.embedding_dim, model.rms_norm_eps, dtype=model.dtype).cuda()
    x = torch.randn((rows, model.embedding_dim), dtype=model.dtype, device="cuda")

    def call():
        # v0.28 RMSNorm.forward_cuda dispatches via IR. This supported provider
        # invokes torch.ops._C.rms_norm, rather than compiling a torch fallback.
        with ir.ops.rms_norm.set_priority(["vllm_c"]):
            return norm.forward_cuda(x)

    return call


def build_operator_call(model, name, rows, tp):
    if name == "dsv4_embedding":
        return _embedding_call(model, rows, tp)
    if name == "dsv4_lm_head":
        return _lm_head_call(model, rows, tp)
    if name == "dsv4_final_norm":
        return _final_norm_call(model, rows)
    return _mhc_call(model, name, rows)


class DeepseekV4OperatorWrapper:
    """Profile each planned scope using the producer's store and tracer."""

    def __init__(self, model_config, num_tensor_parallel_workers, profile_method,
                 rank, output_dir, profiling_plan):
        from frontier.profiling.common.timer_stats_store import TimerStatsStore
        from frontier.profiling.utils import normalize_profile_method

        self.model_config = model_config
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.profile_method = normalize_profile_method(profile_method)
        self.rank = rank
        self.output_dir = output_dir
        self.profiling_plan = profiling_plan
        self.timer_stats_store = TimerStatsStore(profile_method=self.profile_method)
        Path(output_dir, "profiler_traces").mkdir(parents=True, exist_ok=True)

    def profile(self, num_tokens):
        import torch
        from frontier.profiling.common.cuda_timer import CudaTimer
        from frontier.profiling.common.vllm_compat import vllm_config_context
        from frontier.profiling.linear_op.fp8_quantization import checkpoint_hf_config
        from frontier.profiling.linear_op.linear_op_wrapper import (
            ACTIVE_STEPS, WARMUP_STEPS, LinearOpWrapper,
        )
        from frontier.profiling.utils import ProfileMethod
        from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer

        stats = {}
        contracts = row_contracts(
            self.model_config, self.profiling_plan["typed_operator_contracts"], num_tokens,
        )
        with torch.inference_mode(), torch.device("cuda"), vllm_config_context(
            model_config=checkpoint_hf_config(self.model_config),
        ):
            for name in self.profiling_plan["enabled_ops"]:
                call = build_operator_call(
                    self.model_config, name, num_tokens,
                    contracts[name]["selected_tensor_parallel_size"],
                )
                for _ in range(WARMUP_STEPS):
                    output = call()
                torch.cuda.synchronize()
                expected = contracts[name].get("local_output_shape", contracts[name]["output_shape"])
                measured = output[-1] if isinstance(output, tuple) else output
                if tuple(measured.shape) != expected:
                    raise RuntimeError(f"{name}: native output {tuple(measured.shape)} != {expected}")
                self.timer_stats_store.clear_stats()
                timer = CudaTimer(name)
                if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
                    # Traces stay under profiler_traces; operator and row are named
                    # in the child directory so kernel/grid review is unambiguous.
                    trace_dir = Path(self.output_dir, "profiler_traces", f"{name}_{num_tokens}")
                    (trace_dir / "profiler_traces").mkdir(parents=True, exist_ok=True)
                    tracer = RecordFunctionTracer(str(trace_dir))
                    with tracer:
                        for _ in range(ACTIVE_STEPS):
                            with timer:
                                output = call()
                    timed = tracer.get_operation_time_stats(debug=True)
                else:
                    for _ in range(ACTIVE_STEPS):
                        with timer:
                            output = call()
                    torch.cuda.synchronize()
                    timed = self.timer_stats_store.get_stats()
                if name not in timed:
                    raise RuntimeError(f"Missing native timing for {name}, rows={num_tokens}")
                stats[name] = timed[name]
        result = LinearOpWrapper._build_profile_result(self, stats, num_tokens=num_tokens)
        result["typed_operator_contracts"] = contracts
        result["row_axis"] = self.profiling_plan["row_axis"]
        self.timer_stats_store.clear_stats()
        return result
