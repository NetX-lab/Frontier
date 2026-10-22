"""Step-5-Preview routed-expert profiling: layer groups, router and expert kernels.

Step-5 routed layers differ from the single model-wide MoE setting: layers 3-87
run block-FP8 experts, layers 88-90 run BF16 experts, all of them use the
clamped ``swiglustep`` activation, and the router is an FP32 gate with biased
sigmoid top-k selection. These CPU tests check that the profiler measures each
group with that native configuration; none of them measures timing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from frontier.config import PrecisionType
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.moe import main as moe_main
from frontier.profiling.moe.routed_layer_groups import (
    SWIGLUSTEP_ACTIVATION,
    RoutedLayerRouter,
    resolve_routed_layer_groups,
)
from tests.unit.test_step5_fixed_probe import installed_profile_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP5 = "stepfun-ai/Step-5-Preview"
STEP5_ROUTER = RoutedLayerRouter(
    scoring_func="sigmoid",
    uses_selection_bias=True,
    renormalize=True,
    routed_scaling_factor=3.0,
    gate_precision=PrecisionType.FP32,
)


def test_step5_routed_layers_split_by_expert_precision():
    groups = resolve_routed_layer_groups(ModelConfig.from_model_name(STEP5))

    assert [group.layer_ids for group in groups] == [
        tuple(range(3, 88)),
        (88, 89, 90),
    ]
    assert [group.precision for group in groups] == [PrecisionType.FP8, PrecisionType.BF16]
    assert {group.activation for group in groups} == {SWIGLUSTEP_ACTIVATION}
    assert {group.router for group in groups} == {STEP5_ROUTER}
    assert groups[1].row_metadata() == {
        "moe_layer_ids": "88-90",
        "moe_routed_precision": "bf16",
        "moe_activation": "swiglustep",
    }


def test_models_without_layer_groups_keep_one_model_wide_setting():
    assert resolve_routed_layer_groups(ModelConfig.from_model_name("zai-org/GLM-5.3")) == ()


def test_dry_run_counts_each_step5_layer_group(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "moe.main", "--dry_run", "--device", "h800", "--models", STEP5,
            "--num_tensor_parallel_workers", "1", "--expert_parallel_sizes", "32",
            "--num_tokens_list", "1", "16", "256",
        ],
    )
    moe_main.main()
    out = capsys.readouterr().out
    plan = json.loads(out[out.index("{\n"):])
    model = plan["models"][STEP5]

    assert model["local_experts_by_expert_parallel_size"] == {"32": 11}
    assert [group["moe_layer_ids"] for group in model["routed_layer_groups"]] == [
        "3-87",
        "88-90",
    ]
    # Three token counts, one EP size and two layer groups.
    assert model["profiled_steps"] == 3 * 1 * 2 * plan["cases_per_parallel_point"]


@pytest.mark.parametrize(
    ("quantization_mode", "block_shape", "weight_dtype_name"),
    [("fp8", (128, 128), "float8_e4m3fn"), ("bf16", None, "bfloat16")],
)
def test_builder_builds_swiglustep_triton_experts_on_cpu(
    monkeypatch, cpu_vllm_platform, quantization_mode, block_shape, weight_dtype_name
):
    torch = pytest.importorskip("torch")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle
    from vllm.model_executor.layers.fused_moe.oracle import unquantized as unquantized_oracle

    from frontier.profiling.moe.fp8_block_experts import _build_native_routed_experts

    selected = {}

    # Native Step-5 serving on H800 selects Triton experts for both precisions.
    def select_fp8_on_hopper(*, config, weight_key, activation_key, allow_vllm_cutlass):
        selected["config"] = config
        return fp8_oracle.Fp8MoeBackend.TRITON, TritonExperts

    def select_unquantized_on_hopper(moe_config):
        selected["config"] = moe_config
        return unquantized_oracle.UnquantizedMoeBackend.TRITON, TritonExperts

    monkeypatch.setattr(fp8_oracle, "select_fp8_moe_backend", select_fp8_on_hopper)
    monkeypatch.setattr(
        unquantized_oracle, "select_unquantized_moe_backend", select_unquantized_on_hopper
    )
    routed = _build_native_routed_experts(
        quantization_mode=quantization_mode, activation=SWIGLUSTEP_ACTIVATION,
        num_local_experts=11, global_num_experts=352, hidden_dim=256,
        intermediate_dim=128, top_k=8, block_shape=block_shape,
        expert_parallel_size=32, expert_parallel_rank=0, dtype=torch.bfloat16,
        model_type="step5", device=torch.device("cpu"),
    )

    config = selected["config"]
    assert config.activation is MoEActivation.SWIGLUSTEP
    assert (config.num_experts, config.num_local_experts) == (352, 11)
    assert (routed.backend, routed.experts_class) == ("TRITON", "TritonExperts")
    assert type(routed.kernel.fused_experts) is TritonExperts
    assert type(routed.kernel.prepare_finalize).__name__ == (
        "MoEPrepareAndFinalizeNoDPEPModular"
    )
    assert routed.activation is MoEActivation.SWIGLUSTEP
    # No tuned file exists for 11 local experts of width 128.
    assert routed.triton_tuning_config == "default"
    assert routed.w13.dtype == routed.w2.dtype == getattr(torch, weight_dtype_name)
    assert tuple(routed.w13.shape) == (11, 256, 256)
    assert tuple(routed.w2.shape) == (11, 256, 128)


ROUTER_FRESH_PROCESS_SCRIPT = """
import contextlib

import torch

# CPU stand-in for the worker's CUDA device.
torch.cuda.current_device = lambda: "cpu"

from frontier.config import PrecisionType
from frontier.profiling.moe.moe_impl import SigmoidBiasGatingNetwork
from frontier.profiling.moe.routed_layer_groups import RoutedLayerRouter

router = RoutedLayerRouter(
    scoring_func="sigmoid", uses_selection_bias=True, renormalize=True,
    routed_scaling_factor=3.0, gate_precision=PrecisionType.FP32,
)
gating = SigmoidBiasGatingNetwork(
    hidden_dim=64, num_experts=352, router_topk=8, router=router
)
# vLLM's weight loading step; on CPU it binds the CPU GEMM of the gate.
gating.gate.quant_method.process_weights_after_loading(gating.gate)
gating.gating_linear_timer = contextlib.nullcontext()
gating.gating_routing_topk_timer = contextlib.nullcontext()
calls = []


def record_routing(**kwargs):
    calls.append(kwargs)
    num_tokens = kwargs["hidden_states"].shape[0]
    return torch.zeros(num_tokens, 8), torch.zeros(num_tokens, 8, dtype=torch.int32)


gating._fused_topk_bias = record_routing
gating(torch.randn(4, 64, dtype=torch.bfloat16))
call = calls[0]
print("gate", type(gating.gate).__name__, gating.gate.out_dtype)
print("logits", call["gating_output"].dtype, *call["gating_output"].shape)
print("bias", call["e_score_correction_bias"].dtype, *call["e_score_correction_bias"].shape)
print(
    "routing", call["scoring_func"], call["topk"], call["renormalize"],
    call["routed_scaling_factor"],
)
print("weight_policy", gating.routing_weight_policy)
"""


def test_sigmoid_bias_router_runs_fp32_gate_and_biased_topk():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU check; GPU profiling builds the router on CUDA")
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("requires vLLM")
    env = {
        **os.environ,
        "VLLM_TARGET_DEVICE": "cpu",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])
        ),
    }
    # vLLM's parallel state is process-global, so the router runs in a fresh process.
    result = subprocess.run(
        [sys.executable, "-c", ROUTER_FRESH_PROCESS_SCRIPT],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    lines = result.stdout.splitlines()
    assert "gate GateLinear torch.float32" in lines
    assert "logits torch.float32 4 352" in lines
    assert "bias torch.float32 352" in lines
    assert "routing sigmoid 8 True 3.0" in lines
    assert "weight_policy sigmoid_renorm_scaled" in lines


def test_sigmoid_bias_router_rejects_other_routers():
    pytest.importorskip("torch")
    from dataclasses import replace

    from frontier.profiling.moe.moe_impl import SigmoidBiasGatingNetwork

    for router in (
        replace(STEP5_ROUTER, scoring_func="softmax"),
        replace(STEP5_ROUTER, uses_selection_bias=False),
        replace(STEP5_ROUTER, gate_precision=PrecisionType.BF16),
    ):
        with pytest.raises(ValueError, match="sigmoid routers"):
            SigmoidBiasGatingNetwork(
                hidden_dim=64, num_experts=8, router_topk=2, router=router
            )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"use_vllm_kernel": False}, "use_vllm_kernel=True"),
        ({"routing_runtime_path": "uniform_topk"}, "time their own router"),
    ],
)
def test_wrapper_rejects_layer_groups_on_paths_that_ignore_them(updates, message):
    pytest.importorskip("torch")
    from frontier.profiling.moe.moe_wrapper import MoEWrapper

    model_config = ModelConfig.from_model_name(STEP5)
    kwargs = dict(
        model_config=model_config, num_tensor_parallel_workers=1,
        expert_parallel_size=32, profile_method="cuda_event", rank=0,
        output_dir="unused", routed_layer_group=resolve_routed_layer_groups(model_config)[0],
    )
    kwargs.update(updates)
    with pytest.raises(ValueError, match=message):
        MoEWrapper(**kwargs)


def test_kernel_times_bf16_swiglustep_experts_of_the_last_layers(cuda_runtime):
    torch = cuda_runtime.torch
    cuda_runtime.expert_map.numel.return_value = 352
    cuda_runtime.layout.return_value = (32, 3)
    cuda_runtime.routed_experts.backend = "TRITON"
    cuda_runtime.routed_experts.experts_class = "TritonExperts"
    cuda_runtime.routed_experts.triton_tuning_config = "default"
    cuda_runtime.args.update(
        num_experts=11, global_num_experts=352, hidden_dim=4096,
        expert_hidden_dim=1536, use_fp8=False, block_shape=None,
        dtype=torch.bfloat16, model_type="step5", activation=SWIGLUSTEP_ACTIVATION,
    )

    stats = cuda_runtime.kernel.profile_fused_moe_kernel(**cuda_runtime.args)

    assert stats == {
        "mean": 2.5,
        "native_backend": "TRITON",
        "native_experts_class": "TritonExperts",
        "triton_tuning_config": "default",
    }
    assert cuda_runtime.build.call_args.kwargs == dict(
        quantization_mode="bf16", activation="swiglustep",
        num_local_experts=11, global_num_experts=352, hidden_dim=4096,
        intermediate_dim=1536, top_k=8, block_shape=None,
        expert_parallel_size=32, expert_parallel_rank=3,
        dtype=torch.bfloat16, model_type="step5",
    )


def test_profile_row_records_the_measured_layer_group():
    pytest.importorskip("torch")
    from frontier.profiling.moe.moe_wrapper import MoEWrapper

    model_config = ModelConfig.from_model_name(STEP5)
    group = resolve_routed_layer_groups(model_config)[1]
    wrapper = object.__new__(MoEWrapper)
    for name, value in dict(
        model_config=model_config, routed_layer_group=group,
        num_experts=352, num_experts_per_device=11, expert_parallel_size=32,
        routing_runtime_metadata={"routing_assignment_policy": "biased_score_topk"},
        gating_runtime_context_metadata={}, router_topk=8, hidden_dim=4096,
        expert_hidden_dim=1536, use_gated=True, num_tensor_parallel_workers=1,
    ).items():
        setattr(wrapper, name, value)

    row = wrapper._build_profile_result(
        {"moe_grouped_gemm": {"mean": 2.5}}, num_tokens=16
    )

    assert row["moe_layer_ids"] == "88-90"
    assert row["moe_routed_precision"] == "bf16"
    assert row["moe_activation"] == "swiglustep"
    assert row["routing_assignment_policy"] == "biased_score_topk"
    assert row["expert_hidden_dim"] == 1536


def test_position_adapter_binds_one_source_tensor_pair_and_real_factory_contract(monkeypatch):
    torch = pytest.importorskip("torch")
    from contextlib import contextmanager, nullcontext
    from types import ModuleType, SimpleNamespace
    from frontier.profiling.moe import moe_impl
    from frontier.profiling.moe.inputs import MoESourceFrame
    calls, state = {}, SimpleNamespace(routing_positions=None, is_padding=None)
    gate_module = ModuleType("vllm.model_executor.layers.fused_moe.router.gate_linear")
    class GateLinear(torch.nn.Module):
        def __init__(self, hidden, experts, **kwargs):
            super().__init__()
            calls["gate"] = kwargs
            self.experts = experts
        def forward(self, x):
            return torch.zeros((len(x), self.experts), dtype=torch.float32), None
    gate_module.GateLinear = GateLinear
    bias_module = ModuleType("vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router")
    bias_module.fused_topk_bias = lambda **kwargs: pytest.fail("position path reached legacy top-k")
    context = ModuleType("vllm.forward_context")
    @contextmanager
    def set_forward_context(metadata, native, **kwargs):
        old = state.is_padding
        state.is_padding = kwargs["is_padding"]
        try:
            yield
        finally:
            state.is_padding = old
    @contextmanager
    def set_routing_positions(positions):
        old = state.routing_positions
        state.routing_positions = positions
        try:
            yield
        finally:
            state.routing_positions = old
    context.set_forward_context, context.set_routing_positions = set_forward_context, set_routing_positions
    config = ModuleType("vllm.config")
    config.get_current_vllm_config_or_none = lambda: object()
    config.VllmConfig = lambda: pytest.fail("an active config must be retained")
    factory = ModuleType("vllm.model_executor.layers.fused_moe.router.router_factory")
    class Router:
        routing_method_type = SimpleNamespace(name="Simulated")
        def select_experts(self, x, logits):
            calls["inputs"] = (state.routing_positions, state.is_padding, logits.dtype)
            if calls.get("fail"):
                raise RuntimeError("native routing failure")
            return "native-weights", "native-ids"
    def create_fused_moe_router(**kwargs):
        calls["factory"] = kwargs
        return Router()
    factory.create_fused_moe_router = create_fused_moe_router
    for module in (gate_module, bias_module, context, config, factory):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(moe_impl, "ensure_vllm_single_rank_parallel_state", lambda: None)
    monkeypatch.setenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "position_cyclic")
    monkeypatch.setenv("FRONTIER_STEP5_PROFILE_MANIFEST", "CPU_FACTORY_CONTRACT_ONLY")
    monkeypatch.setattr("frontier.profiling.common.runtime_identity.verify_step5_profile_runtime",
                        lambda path, **kwargs: calls.setdefault("provenance", kwargs))
    gate = moe_impl.SigmoidBiasGatingNetwork(hidden_dim=4, num_experts=352, router_topk=8,
                                           router=STEP5_ROUTER, routing_runtime_path="position_cyclic")
    gate.gating_linear_timer = gate.gating_routing_topk_timer = nullcontext()
    gate.bind_source_frame(MoESourceFrame(0, (2,), (6128,), 3, "standalone_legacy"), device="cpu")
    positions, mask = gate.routing_positions, gate.routing_is_padding
    assert gate(torch.zeros((3, 4))) == ("native-weights", "native-ids")
    assert calls["inputs"][0] is positions and calls["inputs"][1] is mask
    assert positions.tolist() == [6128, 6129, 0] and mask.tolist() == [False, False, True]
    assert calls["inputs"][2] is torch.float32
    assert calls["factory"]["top_k"] == 8 and calls["factory"]["global_num_experts"] == 352
    assert calls["factory"]["scoring_func"] == "sigmoid" and calls["factory"]["routed_scaling_factor"] == 3.
    assert gate.gate.forward in calls["provenance"]["callables"]
    calls["fail"] = True
    with pytest.raises(RuntimeError, match="native routing failure"):
        gate(torch.zeros((3, 4)))
    assert state.routing_positions is None and state.is_padding is None


@pytest.mark.parametrize("failure", [None, "call", "parser"])
@pytest.mark.parametrize("record_function", [False, True])
def test_real_device_timer_annotations_parser_replay_and_event_preservation(tmp_path, monkeypatch, failure, record_function):
    torch = pytest.importorskip("torch")
    from frontier.profiling.common.device_timer import DeviceTimer
    from frontier.profiling.common.timer_stats_store import TimerStatsStore
    from frontier.profiling.moe.moe_wrapper import MoEWrapper, ACTIVE_STEPS
    from frontier.profiling.utils import ProfileMethod
    from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer
    store = TimerStatsStore.get_existing() or TimerStatsStore("cuda_event")
    previous_method = store.profile_method
    previous_disabled = store.disabled
    method = ProfileMethod.RECORD_FUNCTION if record_function else ProfileMethod.CUDA_EVENT
    store.profile_method, store.disabled = method, False
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            pass
        def elapsed_time(self, end):
            return 2.5
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    captured = []
    tracing = []
    def transport_enter(tracer):
        tracing.append(True)
        tracer.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        tracer.profiler.__enter__()
        return tracer
    def transport_exit(tracer, *exc):
        tracer.profiler.__exit__(*exc)
        annotations = [event for event in tracer.profiler.events() if event.name.startswith("vidur_")]
        captured.extend(event.name for event in annotations)
        events = []
        for index, annotation in enumerate(annotations):
            name = annotation.name
            events.append(dict(cat="user_annotation", name=name, ts=100*index, dur=80))
            for kernel in range(3):
                correlation = 10*index + kernel
                events.append(dict(cat="cuda_runtime", name="cudaLaunchKernel", ts=100*index+2+kernel, dur=1,
                                   args={"correlation": correlation}))
                if failure != "parser":
                    events.append(dict(cat="kernel", name="cpu_transport_only", dur=100,
                                       args={"correlation": correlation}))
        Path(tracer.trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tracer.trace_path).write_text(json.dumps({"traceEvents": events}))
    monkeypatch.setattr(RecordFunctionTracer, "__enter__", transport_enter)
    monkeypatch.setattr(RecordFunctionTracer, "__exit__", transport_exit)
    wrapper = object.__new__(MoEWrapper)
    wrapper.timer_stats_store, wrapper.output_dir = store, str(tmp_path)
    timer = DeviceTimer("moe_grouped_gemm")
    methods = []
    def run():
        methods.append(store.profile_method)
        with timer:
            torch.ones(1).add_(1)
            if failure == "call" and tracing:
                raise RuntimeError("replay call failure")
    try:
        if failure:
            with pytest.raises((ValueError, RuntimeError)):
                wrapper._profile_cell_callable(run, ("moe_grouped_gemm",))
        else:
            row = wrapper._profile_cell_callable(run, ("moe_grouped_gemm",))["moe_grouped_gemm"]
            assert row["median"] == (pytest.approx(.3) if record_function else 2.5)
            assert row["count"] == (1 if record_function else ACTIVE_STEPS)
            assert row["std"] == 0 and row["kernel_count"] == 3
        assert captured == ["vidur_moe_grouped_gemm"]
        assert methods[-1] is ProfileMethod.RECORD_FUNCTION
        assert set(methods[:-1]) == {method}
        assert store.profile_method is method and store.TIMING_STATS == {}
        assert TimerStatsStore.get_existing() is store
    finally:
        store.profile_method = previous_method
        store.disabled = previous_disabled
        store.clear_stats()


def test_masked_expert_sentinel_is_filtered_before_local_map_and_bincount():
    torch = pytest.importorskip("torch")
    from frontier.profiling.moe.load_distribution import compute_expert_token_counts
    from frontier.profiling.moe.moe_wrapper import MoEWrapper
    wrapper = object.__new__(MoEWrapper)
    wrapper.num_experts, wrapper.num_experts_per_device, wrapper.expert_parallel_size = 4, 2, 2
    wrapper._get_profiling_ep_rank = lambda: 1
    ids = torch.tensor([[-1, -1], [2, 3]], dtype=torch.int32)
    assert wrapper._compute_local_expert_token_counts(ids) == [1, 1]
    assert compute_expert_token_counts(ids, 4) == [0, 0, 1, 1]


@pytest.mark.parametrize("module_name", ["vllm.models.step5.nvidia.layernorm",
    "vllm.model_executor.layers.fused_moe.oracle.fp8", "vllm.model_executor.layers.fused_moe.oracle.unquantized"])
@pytest.mark.parametrize("change", [None, "spec", "hash"])
def test_norm_and_both_oracle_callable_receipts_use_full_installed_sources(installed_profile_receipt, tmp_path, module_name, change):
    from frontier.profiling.common.runtime_identity import verify_step5_profile_runtime
    path, modules, manifest = installed_profile_receipt
    module = modules[module_name]
    if change == "spec":
        module.__spec__.origin = str(tmp_path / "byte_identical_shadow.py")
    elif change == "hash":
        Path(module.__file__).write_text("# same version, different oracle bytes\n")
    if change:
        with pytest.raises((ValueError, FileNotFoundError)):
            verify_step5_profile_runtime(path, callables=(module.make_step5_rms_norm, module.GateLinear().forward))
    else:
        facts = verify_step5_profile_runtime(path, callables=(module.make_step5_rms_norm, module.GateLinear().forward))
        assert module_name + ".make_step5_rms_norm" in facts["callables"]


def test_lazy_norm_verifies_factory_instance_and_bound_method_before_caching(installed_profile_receipt, monkeypatch):
    torch = pytest.importorskip("torch")
    from frontier.profiling.common.layers.layernorm import Step5RMSNorm
    path, modules, _ = installed_profile_receipt
    monkeypatch.setenv("FRONTIER_STEP5_PROFILE_MANIFEST", str(path))
    norm = Step5RMSNorm(4, eps=1e-6, zero_centered=True, dtype=torch.float32, fp32_residual_connection=True)
    impl = norm._native_impl()
    assert impl is norm._native_impl() and impl.weight is norm.weight
    assert len(norm.native_facts["callables"]) == 3
    modules["vllm.models.step5.nvidia.layernorm"].__spec__.origin = str(path.parent / "foreign.py")
    other = Step5RMSNorm(4, eps=1e-6, zero_centered=True, dtype=torch.float32, fp32_residual_connection=True)
    with pytest.raises(FileNotFoundError):
        other._native_impl()
    assert other._impl is None
