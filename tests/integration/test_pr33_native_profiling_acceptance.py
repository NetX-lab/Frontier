"""Real PR33 profiling acceptance; GPU/runtime absence yields explicit skips.

Run on an already authorized worker with its native profiling environment.
This module never provisions workers or substitutes simulated native kernels.
"""

import json
import math
import os
from types import SimpleNamespace

import pytest


@pytest.fixture
def native_torch():
    torch = pytest.importorskip("torch", reason="Native profiling requires PyTorch")
    if not torch.cuda.is_available():
        pytest.skip("Native profiling requires an available CUDA or ROCm GPU")
    return torch


@pytest.fixture
def mi355x_torch(native_torch):
    torch = native_torch
    if torch.version.hip is None:
        pytest.skip("This lane requires ROCm; NVIDIA execution is separate evidence")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if not properties.gcnArchName.startswith("gfx950"):
        pytest.skip("Pinned AMD profiling acceptance requires gfx950/MI355X")
    return torch


@pytest.fixture
def attention_runtime(mi355x_torch):
    pytest.importorskip("vllm.v1.attention.backends.rocm_attn",
                        reason="Native vLLM ROCm attention backend unavailable")
    from frontier.profiling.common.timer_stats_store import TimerStatsStore
    from frontier.profiling.utils.singleton import Singleton

    previous = Singleton._instances.pop(TimerStatsStore, None)
    try:
        store = TimerStatsStore(profile_method="device_event")
        yield mi355x_torch, store
    finally:
        Singleton._instances.pop(TimerStatsStore, None)
        if previous is not None:
            Singleton._instances[TimerStatsStore] = previous


@pytest.mark.parametrize("model_name", [
    "meta-llama/Llama-2-7b-hf", "Qwen3.8-2.4T-A95B-Quark-MXFP4",
])
@pytest.mark.parametrize("phase,prefix,length", [("prefill", 0, 16), ("decode", 16, 1)])
def test_native_rocm_attention_matches_causal_reference(attention_runtime, phase, prefix, length, model_name):
    torch, store = attention_runtime
    from frontier.profiling.attention.backends.vllm_rocm_attention_wrapper import VllmRocmAttentionWrapper
    from frontier.profiling.attention.sequence_proxy import SequenceMetadataProxy
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.common.parallel_config import ParallelConfig
    from frontier.profiling.common.vllm_compat import vllm_config_context

    model = ModelConfig.from_model_name(model_name)
    wrapper = VllmRocmAttentionWrapper()
    with vllm_config_context():
        wrapper.init(model, ParallelConfig(1, 1), 16, torch.device("cuda"))
        total = prefix + length
        query = torch.randn(total, wrapper.num_q_heads, wrapper.head_dim,
                            device="cuda", dtype=model.dtype) * 0.1
        key = torch.randn(total, wrapper.num_kv_heads, wrapper.head_dim,
                          device="cuda", dtype=model.dtype) * 0.1
        value = torch.randn_like(key)
        cache = wrapper.get_cache_block(2, device="cuda", dtype=model.dtype)
        scale = wrapper.head_dim ** -0.5
        try:
            if prefix:
                wrapper.begin_forward([SequenceMetadataProxy(True, prefix, 0, [0, 1])])
                wrapper.forward(query[:prefix], key[:prefix], value[:prefix], cache, scale)
                wrapper.end_forward()
            store.clear_stats()
            wrapper.begin_forward([SequenceMetadataProxy(phase == "prefill", total, prefix, [0, 1])])
            output = wrapper.forward(query[prefix:], key[prefix:], value[prefix:], cache, scale)
            torch.cuda.synchronize()
        finally:
            wrapper.end_forward()

    repeat = wrapper.num_q_heads // wrapper.num_kv_heads
    full_keys = key.repeat_interleave(repeat, dim=1).float()
    full_values = value.repeat_interleave(repeat, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", query[prefix:].float(), full_keys) * scale
    causal = torch.arange(total, device="cuda")[None, :] <= torch.arange(prefix, total, device="cuda")[:, None]
    scores.masked_fill_(~causal.unsqueeze(0), float("-inf"))
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), full_values).reshape_as(output)
    torch.testing.assert_close(output.float(), reference, atol=0.02, rtol=0.02)
    assert torch.isfinite(output).all()
    samples = store.get_times()[f"attn_{phase}"]
    assert len(samples) == 1 and math.isfinite(samples[0]) and samples[0] >= 0


@pytest.fixture
def mxfp4_runtime(mi355x_torch):
    pytest.importorskip("vllm", reason="Native online MXFP4 requires current vLLM")
    pytest.importorskip("aiter", reason="Native online MXFP4 requires AITER")
    for name in ("VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"):
        if os.environ.get(name) != "1":
            pytest.skip(f"Select the supported native backend with {name}=1 before starting Python")
    from frontier.profiling.moe import moe_vllm_kernel

    if moe_vllm_kernel.VLLM_API_VERSION != "functional_fused_experts":
        pytest.skip("Native online MXFP4 requires vLLM's functional fused-experts API")
    pytest.importorskip("vllm.model_executor.layers.quantization.online.base",
                        reason="vLLM online quantization API unavailable")
    return mi355x_torch, moe_vllm_kernel


def test_native_mxfp4_materializes_packed_weights_and_measures_events(mxfp4_runtime):
    torch, kernel = mxfp4_runtime
    ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32, device="cuda")
    weights = torch.ones((4, 1), dtype=torch.float32, device="cuda")
    result = kernel.profile_fused_moe_kernel(
        num_tokens=4, num_experts=2, hidden_dim=256, expert_hidden_dim=256,
        top_k=1, topk_weights=weights, topk_ids=ids, dtype=torch.bfloat16,
        use_mxfp4=True, model_type="qwen3_5_moe_text", profile_method="device_event",
        warmup_steps=1, active_steps=2,
    )
    assert result["native_backend"].rsplit(".", 1)[-1].startswith("AITER")
    assert result["mean"] > 0
    assert all(math.isfinite(result[name]) and result[name] >= 0
               for name in ("min", "max", "mean", "median", "std"))
    state = kernel._get_functional_mxfp4_state(
        num_experts=2, hidden_dim=256, expert_hidden_dim=256, top_k=1,
        dtype=torch.bfloat16, model_type="qwen3_5_moe_text",
    )
    kernel.validate_mxfp4_materialized_layout(state["layer"], state["layout"])


@pytest.mark.parametrize("platform", ["cuda", "rocm"], ids=["NCCL", "RCCL"])
@pytest.mark.parametrize("precision,bytes_per_element", [("BF16", 2), ("FP32", 4)])
def test_native_collective_runner_reports_actual_dtype_bytes(native_torch, platform, precision, bytes_per_element):
    torch = native_torch
    actual = "rocm" if torch.version.hip is not None else "cuda"
    if actual != platform:
        pytest.skip(f"The {platform} collective lane requires its own native runtime")
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Native {platform} all-reduce acceptance requires two visible GPUs")
    if not torch.distributed.is_nccl_available():
        pytest.skip("Native NCCL/RCCL process-group backend unavailable")
    from frontier.profiling.collectives.collectives_input import CollectivesInput
    from frontier.profiling.collectives.main import _run_local_collective

    result = _run_local_collective(CollectivesInput(2, 2, 128, "all_reduce", precision), 2)
    assert result["rank"] == 0 and result["num_workers"] == 2
    assert result["size"] == 128 * bytes_per_element
    stats = result["time_stats"]["all_reduce"]
    assert stats["median"] > 0
    assert all(math.isfinite(stats[name]) and stats[name] >= 0
               for name in ("min", "max", "mean", "median", "std"))


@pytest.fixture
def sglang_runtime(mi355x_torch, tmp_path, request):
    torch = mi355x_torch
    pytest.importorskip("aiter", reason="Native SGLang routed primitives require AITER")
    state = pytest.importorskip("sglang.srt.distributed.parallel_state",
                                reason="Native SGLang distributed runtime unavailable")
    primitive = request.node.callspec.params["primitive"]
    if primitive == "moe_routing_topk":
        topk = pytest.importorskip("sglang.srt.layers.moe.topk", reason="Native SGLang top-k API unavailable")
        if not topk._use_aiter:
            pytest.skip("Native SGLang top-k requires an AITER-enabled SGLang environment")
    if primitive == "moe_experts_quant_gemm_combine" and not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("Native SGLang MXFP4 requires the packed float4_e2m1fn_x2 dtype")
    if torch.distributed.is_initialized():
        pytest.skip("SGLang acceptance requires an isolated process without an existing process group")
    if not torch.distributed.is_nccl_available():
        pytest.skip("Native SGLang graph replay requires the RCCL process-group backend")
    try:
        state.init_distributed_environment(
            world_size=1, rank=0, local_rank=torch.cuda.current_device(),
            distributed_init_method=(tmp_path / "sglang-rendezvous").as_uri(),
            backend="nccl", timeout=120,
        )
        state.initialize_model_parallel(tensor_model_parallel_size=1)
        yield torch, state.get_tp_group()
    finally:
        try:
            state.destroy_model_parallel()
        finally:
            state.destroy_distributed_environment()


@pytest.mark.parametrize("primitive", ["moe_routing_topk", "moe_sorting", "moe_experts_quant_gemm_combine"])
def test_native_sglang_replay_checks_outputs_and_emits_kernel_trace(sglang_runtime, primitive, tmp_path):
    torch, group = sglang_runtime
    from frontier.profiling.experimental.sglang.graph_replay import attach_kernel_evidence, profile_graph
    from frontier.profiling.experimental.sglang.routed_moe_replay import profile_routed_graph

    model = SimpleNamespace(embedding_dim=256, routed_mlp_hidden_dim=256,
                            is_moe=True, num_experts=2, num_experts_per_tok=1)
    if primitive == "moe_routing_topk":
        model.num_experts, model.num_experts_per_tok = 4, 2
        row, replay = profile_graph(primitive, 64, 2, 5, model, group, 0, trace=True)
    else:
        query = dict(component=primitive, physical_size=64, physical_expert_counts=(33, 31))
        row, replay = profile_routed_graph(query, 2, 5, model, group, trace=True)
    assert row["measurement_type"] == "HIP_GRAPH_REPLAY" and row["experimental"] is True
    assert row["correctness_checked"] is True
    assert len(row["samples_ms"]) == 5
    assert all(math.isfinite(sample) and sample > 0 for sample in row["samples_ms"])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        replay()
    trace = tmp_path / f"{primitive}.json"
    profiler.export_chrome_trace(str(trace))
    attach_kernel_evidence([row], json.loads(trace.read_text())["traceEvents"])
    assert row["kernel_names"]
