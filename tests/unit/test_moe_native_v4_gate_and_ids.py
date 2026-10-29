"""CPU regression checks for the native V4 gate, routing dtypes and row schema."""

from collections.abc import Mapping
from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from frontier.profiling.moe import moe_impl, moe_vllm_kernel as kernel
from frontier.profiling.moe.moe_wrapper import MOE_FAMILY, MoEWrapper

ROUTING_PATHS = (
    ("balanced_topk", torch.int64),
    ("grouped_topk_sigmoid", torch.int32),
    ("standard_fused_topk", torch.int32),
    ("uniform_topk", torch.int32),
)


@pytest.mark.parametrize("local_rows", [1, 2, 3, 4, 768])
def test_balanced_gate_uses_model_parameters_fp32_output_and_no_bias(monkeypatch, local_rows):
    seen = {}

    class NativeGate(torch.nn.Module):
        # GateLinear's pinned signature defaults bias=False.
        def __init__(self, input_size, output_size, bias=False, out_dtype=None, params_dtype=None):
            super().__init__()
            seen.update(input_size=input_size, output_size=output_size, bias=bias,
                        out_dtype=out_dtype, params_dtype=params_dtype)
            self.weight = torch.nn.Parameter(torch.empty(output_size, input_size, dtype=params_dtype))
            self.out_dtype = out_dtype

        def forward(self, hidden_states):
            seen["input"] = hidden_states
            return torch.empty(hidden_states.shape[0], self.weight.shape[0], dtype=self.out_dtype), None

    native = ModuleType("vllm.model_executor.layers.fused_moe")
    native.GateLinear = Mock(side_effect=NativeGate)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(moe_impl, "HAS_VLLM", True)
    monkeypatch.setattr(moe_impl, "ensure_vllm_single_rank_parallel_state", lambda: None)
    monkeypatch.setattr(moe_impl, "CudaTimer", lambda *_: nullcontext())
    monkeypatch.setattr(moe_impl, "raise_if_fp8_requested", lambda *_: None)
    config = SimpleNamespace(dtype=torch.bfloat16)
    gate = moe_impl.MoEGatingNetwork(4096, 256, 6, routing_runtime_path="balanced_topk", router_config=config)
    hidden_states = torch.empty(local_rows, 4096, dtype=config.dtype)
    weights, ids = gate(hidden_states)
    assert {name: seen[name] for name in ("input_size", "output_size", "bias", "out_dtype", "params_dtype")} == {
        "input_size": 4096, "output_size": 256, "bias": False,
        "out_dtype": torch.float32, "params_dtype": config.dtype,
    }
    native.GateLinear.assert_called_once_with(4096, 256, out_dtype=torch.float32, params_dtype=config.dtype)
    assert gate.gate.weight.dtype == config.dtype
    assert seen["input"] is hidden_states
    assert gate.linear_backend == "vllm_gate_linear"
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int64
    assert ids.shape == (local_rows, 6)


@pytest.fixture
def cpu_functional_runtime(monkeypatch):
    # Replace CUDA allocation/transfer/timing only; the tensors and dtype casts
    # are real CPU operations, and the native adapter is the observation point.
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    runtime = SimpleNamespace(
        version=SimpleNamespace(hip=None, cuda="12.8"),
        cuda=SimpleNamespace(synchronize=Mock()),
        randn=lambda *shape, **kwargs: torch.zeros(*shape, dtype=kwargs["dtype"]),
        empty_like=torch.empty_like, bfloat16=torch.bfloat16,
        float16=torch.float16, float32=torch.float32, int32=torch.int32,
    )
    monkeypatch.setattr(kernel, "torch", runtime)
    monkeypatch.setattr(kernel, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(kernel, "VLLM_API_VERSION", "functional_fused_experts")
    monkeypatch.setattr("frontier.profiling.common.vllm_compat.vllm_config_context", lambda **kwargs: nullcontext())
    def collect(step_fn, **kwargs):
        step_fn()
        return {"median": 1.0}
    monkeypatch.setattr(kernel, "_collect_cuda_event_stats", collect)
    return runtime


@pytest.mark.parametrize("routing_path,expected_dtype", ROUTING_PATHS)
@pytest.mark.parametrize("weight_source", [None, "checkpoint"])
def test_functional_kernel_receives_the_routing_paths_native_id_dtype(
    monkeypatch, cpu_functional_runtime, routing_path, expected_dtype, weight_source
):
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32 if expected_dtype == torch.int64 else torch.int64)
    weights = torch.full(ids.shape, 0.5, dtype=torch.float32)
    functional = Mock()
    adapter = SimpleNamespace(backend="MXFP4_MARLIN", workspaces=Mock(return_value=(None, None)), apply=Mock())
    monkeypatch.setattr(kernel, "_functional_fused_experts", functional)
    monkeypatch.setattr(kernel, "validate_mxfp4_runtime", lambda **kwargs: None)
    monkeypatch.setattr(kernel, "get_mxfp4_marlin_routed_experts", Mock(return_value=adapter))
    result = kernel.profile_fused_moe_kernel(
        num_tokens=2, num_experts=4, hidden_dim=32, expert_hidden_dim=32, top_k=2,
        topk_weights=weights, topk_ids=ids, warmup_steps=0, active_steps=1,
        routing_runtime_path=routing_path, use_mxfp4=weight_source is not None,
        mxfp4_weight_source=weight_source, model_type="deepseek_v4",
    )
    if weight_source is None:
        received_ids = functional.call_args.kwargs["topk_ids"]
        functional.assert_called_once()
    else:
        received_ids = adapter.apply.call_args.args[3]
        assert adapter.workspaces.call_args.args[1] is received_ids
        assert result["native_backend"] == "MXFP4_MARLIN"
        adapter.apply.assert_called_once()
    assert received_ids.dtype == expected_dtype
    assert received_ids.is_contiguous()
    assert received_ids.device.type == "cpu"
    assert received_ids.tolist() == ids.tolist()


@pytest.mark.parametrize("routing_path,expected_dtype", ROUTING_PATHS)
def test_shuffling_alignment_receives_the_routing_paths_native_id_dtype(monkeypatch, routing_path, expected_dtype):
    monkeypatch.setattr(moe_impl, "HAS_VLLM", True)
    monkeypatch.setattr(moe_impl, "CudaTimer", lambda *_: nullcontext())
    shuffler = moe_impl.MoETokenShuffler(256, 6, 32, 32, torch.bfloat16, True,
        num_local_experts=16, routing_runtime_path=routing_path)
    monkeypatch.setattr(shuffler, "_get_block_size_m", lambda rows: 8)
    align = Mock(return_value=(torch.tensor([0, 1], dtype=torch.int32), None, None))
    monkeypatch.setattr(moe_impl, "moe_align_block_size", align, raising=False)
    shuffler(torch.empty(2, 32), torch.zeros(2, 6, dtype=torch.int64))
    assert align.call_args.kwargs["topk_ids"].dtype == expected_dtype


@pytest.mark.parametrize("routing_path,expected_dtype", ROUTING_PATHS)
def test_prepared_balanced_sample_keeps_each_runtime_paths_id_dtype(monkeypatch, routing_path, expected_dtype):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    wrapper = MoEWrapper.__new__(MoEWrapper)
    wrapper.rank = 0
    wrapper.expert_parallel_size = 16
    wrapper.num_experts = 256
    wrapper.num_experts_per_device = 16
    wrapper.router_topk = 6
    wrapper.routing_runtime_path = routing_path
    sample = wrapper._prepare_routing_inputs(32, "balanced")
    assert sample["topk_ids"].dtype == expected_dtype


# Frozen _build_profile_result from HEAD eaee355046ea809b716f2e9abf8b00d18fa72cea;
# only the function name/indentation changed. This preserves an independent row
# oracle, including typed metadata and routed-width substitution.
def head_row_builder(
    self,
    time_stats: Mapping[str, object],
    *,
    num_tokens: int,
) -> dict:
    """Build one MoE row and attach metadata for measured routed operators."""

    if not isinstance(time_stats, Mapping):
        raise TypeError(
            f"MoE profiling time_stats must be a mapping, got {type(time_stats).__name__}"
        )

    stats = {
        "time_stats": dict(time_stats),
        "num_tokens": num_tokens,
        "num_experts": self.num_experts,
        "num_experts_per_device": self.num_experts_per_device,
        "expert_parallel_size": self.expert_parallel_size,
        **self.routing_runtime_metadata,
        **self.gating_runtime_context_metadata,
        "router_topk": self.router_topk,
        "hidden_dim": self.hidden_dim,
        "expert_hidden_dim": self.expert_hidden_dim,
        "use_gated": self.use_gated,
        "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
    }

    measured_names = set(time_stats)
    moe_operator_names = tuple(
        operator.profiling_name() for operator in MOE_FAMILY.profiling_ops()
    )
    measured_moe_names = [
        operator_name
        for operator_name in moe_operator_names
        if operator_name in measured_names
    ]
    if measured_moe_names:
        typed_contracts = {}
        routed_width = None
        for operator_name in measured_moe_names:
            routed_contract = self._resolve_routed_layer_contract(
                operator_name=operator_name
            )
            metadata = routed_contract.typed_metadata_identity()
            typed_contracts[operator_name] = dict(metadata)
            if routed_width is None:
                routed_width = routed_contract.effective_ffn_width
            elif routed_width != routed_contract.effective_ffn_width:
                raise ValueError(
                    "MoE operators resolved to conflicting routed widths: "
                    f"{routed_width} and {routed_contract.effective_ffn_width}"
                )
        stats["typed_operator_contracts"] = typed_contracts
        if routed_width is not None:
            stats["expert_hidden_dim"] = routed_width

    return stats


@pytest.mark.parametrize("routing_path", ["grouped_topk_sigmoid", "standard_fused_topk", "uniform_topk"])
@pytest.mark.parametrize("distribution", ["uniform", "skewed", "extremely_skewed"])
def test_non_balanced_row_keys_and_values_equal_head(routing_path, distribution):
    wrapper = MoEWrapper.__new__(MoEWrapper)
    wrapper.num_experts = 8
    wrapper.num_experts_per_device = 4
    wrapper.expert_parallel_size = 2
    wrapper.routing_runtime_metadata = moe_impl.get_routing_runtime_metadata(routing_path)
    wrapper.gating_runtime_context_metadata = {"gating_runtime_context": "isolated"}
    wrapper.router_topk = 2
    wrapper.hidden_dim = 32
    wrapper.expert_hidden_dim = 64
    wrapper.use_gated = True
    wrapper.num_tensor_parallel_workers = 1
    wrapper._resolve_routed_layer_contract = lambda **kwargs: SimpleNamespace(
        typed_metadata_identity=lambda: {"layout": "routed", "width": 48}, effective_ffn_width=48)
    stats = {name: {"median": 1.0} for name in (
        "moe_gating_linear", "moe_gating_routing_topk", "moe_shuffling", "moe_grouped_gemm")}
    expected = head_row_builder(wrapper, stats, num_tokens=32)
    actual = wrapper._build_profile_result(stats, num_tokens=32, load_distribution=distribution)
    assert actual.keys() == expected.keys()
    assert actual == expected
    assert actual["expert_hidden_dim"] == 48
    assert set(actual["typed_operator_contracts"]) == set(stats)


@pytest.mark.parametrize("routing_path,expected_dtype", ROUTING_PATHS)
def test_grouped_scope_passes_the_wrappers_routing_path_to_the_kernel(monkeypatch, routing_path, expected_dtype):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    wrapper = MoEWrapper.__new__(MoEWrapper)
    wrapper.rank = 0
    wrapper.expert_parallel_size = 16
    wrapper.num_experts = 256
    wrapper.num_experts_per_device = 16
    wrapper.router_topk = 6
    wrapper.routing_runtime_path = routing_path
    wrapper.hidden_dim = 32
    wrapper.expert_hidden_dim = 32
    wrapper.use_gated = True
    wrapper.num_tensor_parallel_workers = 1
    wrapper.use_vllm_kernel = True
    wrapper.use_fp8 = False
    wrapper.use_mxfp4 = False
    wrapper.mxfp4_weight_source = None
    wrapper.quantization_mode = "bf16"
    wrapper._dtype = torch.bfloat16
    wrapper.per_channel_quant = False
    wrapper.block_shape = None
    wrapper.profile_method = "cuda_event"
    wrapper.output_dir = None
    wrapper.model_config = SimpleNamespace(model_type="deepseek_v4", swiglu_limit=None)
    sample = wrapper._prepare_routing_inputs(32, "balanced")
    profile = Mock(return_value={"median": 1.0})
    monkeypatch.setattr(kernel, "profile_fused_moe_kernel", profile)
    wrapper.profile_grouped_gemm(32, "balanced", routing_inputs=sample)
    assert profile.call_args.kwargs["routing_runtime_path"] == routing_path
    assert profile.call_args.kwargs["topk_ids"].dtype == expected_dtype
