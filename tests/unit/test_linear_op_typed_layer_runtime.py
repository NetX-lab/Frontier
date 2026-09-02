"""Runtime wiring regressions for profile-owned typed FFN contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.non_kv_cache_overhead import runtime_estimator
from frontier.profiling.linear_op import linear_op_impl
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter


class _CaptureLinear(torch.nn.Module):
    """Small CPU-safe stand-in that records parallel linear dimensions."""

    instances: list["_CaptureLinear"] = []

    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def forward(self, hidden_states):
        return hidden_states, None


class _NoopTimer:
    """CPU-safe timer context for constructor-only wiring tests."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args
        return False


def _tiny_step3_config() -> ModelConfig:
    """Return a small Step3-shaped config with deliberately distinct widths."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    config.embedding_dim = 8
    config.num_q_heads = 2
    config.num_kv_heads = 1
    config.mlp_hidden_dim = 7
    config.dense_mlp_hidden_dim = 12
    config.routed_mlp_hidden_dim = 6
    config.share_expert_dim = 6
    return config


def _tiny_mixed_step3_config() -> ModelConfig:
    """Return a four-layer mixed config with distinct dense/routed domains."""

    config = _tiny_step3_config()
    config.num_layers = 4
    config.moe_layers_enum = "2,3"
    config.num_experts = 2
    config.norm = "layer_norm"
    config.post_attn_norm = False
    return config


def _ffn_only_plan() -> dict:
    """Return a constructor-only plan with sharded FFN materialization enabled."""

    return {
        "attn_enabled": False,
        "ffn_enabled": True,
        "ffn_sharded_enabled": True,
        "enabled_ops": [],
        "padded_n_embd": 8,
        "padded_n_expanded_embd": 12,
    }


def test_mlp_uses_profile_owned_dense_width(monkeypatch) -> None:
    """The standard FFN module must resolve Step3 dense width independently."""

    _CaptureLinear.instances = []
    monkeypatch.setattr(linear_op_impl, "ColumnParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "RowParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "SiluAndMul", torch.nn.Identity)
    monkeypatch.setattr(linear_op_impl, "CudaTimer", _NoopTimer)

    mlp = linear_op_impl.MLP(_tiny_step3_config(), world_size=2)

    assert mlp.mlp_hidden_dim == 12
    assert [layer.output_size for layer in _CaptureLinear.instances] == [24, 8]


def test_mlp_rejects_explicit_width_below_profile_width(monkeypatch) -> None:
    """A legacy override cannot silently narrow a typed FFN contract."""

    monkeypatch.setattr(linear_op_impl, "ColumnParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "RowParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "SiluAndMul", torch.nn.Identity)
    monkeypatch.setattr(linear_op_impl, "CudaTimer", _NoopTimer)

    with pytest.raises(ValueError, match="below.*profile|profile.*width"):
        linear_op_impl.MLP(
            _tiny_step3_config(),
            world_size=2,
            mlp_hidden_dim=10,
        )


def test_shared_expert_mlp_uses_profile_owned_shared_width(monkeypatch) -> None:
    """The shared-expert module must resolve its own typed width."""

    _CaptureLinear.instances = []
    monkeypatch.setattr(linear_op_impl, "ColumnParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "RowParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "SiluAndMul", torch.nn.Identity)
    monkeypatch.setattr(linear_op_impl, "CudaTimer", _NoopTimer)

    shared = linear_op_impl.ShareExpertMLP(_tiny_step3_config(), world_size=2)

    assert shared.mlp_hidden_dim == 6
    assert [layer.output_size for layer in _CaptureLinear.instances] == [12, 8]


def test_shared_expert_mlp_accepts_profile_declared_width_alias(monkeypatch) -> None:
    """A profile-owned shared width must not require a legacy config field."""

    _CaptureLinear.instances = []
    monkeypatch.setattr(linear_op_impl, "ColumnParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "RowParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "SiluAndMul", torch.nn.Identity)
    monkeypatch.setattr(linear_op_impl, "CudaTimer", _NoopTimer)

    config = SimpleNamespace(
        name="profile-shared-alias",
        is_moe=True,
        num_layers=1,
        moe_layers_enum=None,
        model_type="generic",
        model_arch="generic",
        model_architecture_profile=None,
        embedding_dim=8,
        share_expert_dim=None,
        shared_expert_intermediate_size=6,
        use_gated_mlp=True,
        quantization_config=None,
        supports_share_expert=lambda: True,
    )

    shared = linear_op_impl.ShareExpertMLP(config, world_size=2)

    assert shared.mlp_hidden_dim == 6
    assert [layer.output_size for layer in _CaptureLinear.instances] == [12, 8]


def test_pure_routed_moe_does_not_construct_dense_mlp(monkeypatch) -> None:
    """A pure routed-MoE profile must not materialize a dense FFN module."""

    config = ModelConfig.from_model_name("mixtral_8x7b_moe")
    config.embedding_dim = 8
    config.num_q_heads = 2
    config.num_kv_heads = 1
    config.mlp_hidden_dim = 6
    config.routed_mlp_hidden_dim = 6
    config.norm = "layer_norm"
    config.post_attn_norm = False

    monkeypatch.setattr(linear_op_impl, "ColumnParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "RowParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "SiluAndMul", torch.nn.Identity)
    monkeypatch.setattr(linear_op_impl, "CudaTimer", _NoopTimer)

    block = linear_op_impl.GPTBlock(
        config,
        world_size=2,
        profiling_plan={
            "attn_enabled": False,
            "ffn_enabled": True,
            "ffn_sharded_enabled": True,
            "enabled_ops": [],
            "padded_n_embd": 8,
            "padded_n_expanded_embd": 6,
        },
    )

    assert isinstance(block.mlp, linear_op_impl.DummyMLP)


def test_gpt_block_rejects_typed_metadata_with_wrong_parallel_domain(monkeypatch) -> None:
    """Runtime construction must reject metadata that changes the TP domain."""

    monkeypatch.setattr(linear_op_impl, "ColumnParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "RowParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "SiluAndMul", torch.nn.Identity)
    monkeypatch.setattr(linear_op_impl, "CudaTimer", _NoopTimer)

    config = _tiny_step3_config()
    # Keep this constructor-only contract test independent from the timed RMSNorm
    # implementation; the typed metadata check is the behavior under test.
    config.norm = "layer_norm"
    config.post_attn_norm = False

    with pytest.raises(ValueError, match="tensor_parallel_mode"):
        linear_op_impl.GPTBlock(
            config,
            world_size=2,
            profiling_plan={
                "attn_enabled": False,
                "ffn_enabled": True,
                "ffn_sharded_enabled": True,
                "enabled_ops": [],
                "padded_n_embd": 8,
                "padded_n_expanded_embd": 12,
                "typed_operator_contracts": {
                    "mlp_up_proj": {
                        "profile_id": "step3_text",
                        "operator_family_id": "ffn",
                        "operator_family_ids": ["ffn"],
                        "layer_kind": "dense",
                        "dimension_source": "dense_mlp_hidden_dim",
                        "effective_ffn_width": 12,
                        "tensor_parallel_mode": "moe_tp",
                        "expert_parallel_mode": "off",
                        "selected_expert_parallel_size": None,
                        "tensor_parallel_sizes": [2],
                        "selected_tensor_parallel_size": 2,
                        "selected_padded_ffn_width": 12,
                    },
                    "share_expert_up_proj": {
                        "profile_id": "step3_text",
                        "operator_family_id": "share_expert",
                        "operator_family_ids": ["share_expert"],
                        "layer_kind": "shared",
                        "dimension_source": "share_expert_dim",
                        "effective_ffn_width": 6,
                        "tensor_parallel_mode": "attention_tp",
                        "expert_parallel_mode": "off",
                        "selected_expert_parallel_size": None,
                        "tensor_parallel_sizes": [2],
                        "selected_tensor_parallel_size": 2,
                        "selected_padded_ffn_width": 6,
                    },
                },
            },
        )


def test_runtime_typed_metadata_rejects_float_for_integer_width() -> None:
    """Runtime plan validation must preserve exact typed field representations."""

    config = _tiny_step3_config()
    contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        operator_name="mlp_up_proj",
        tensor_parallel_size=2,
    )
    metadata = contract.typed_metadata_identity()
    metadata["effective_ffn_width"] = float(metadata["effective_ffn_width"])
    metadata["selected_padded_ffn_width"] = 12

    with pytest.raises(ValueError, match="effective_ffn_width"):
        linear_op_impl._validate_typed_linear_metadata(  # pylint: disable=protected-access
            metadata,
            contract,
            operator_name="mlp_up_proj",
            world_size=2,
        )


def test_attention_builder_uses_config_owned_profile(monkeypatch) -> None:
    """Attention construction must use the config's resolved profile snapshot."""

    step3_attention = object()
    generic_attention = object()
    step3_profile = SimpleNamespace(
        linear_attention=SimpleNamespace(
            sharded_impl=linear_op_impl.LinearAttentionImplementation.STEP3_TEXT,
            has_replicated_pre_projection=lambda _enabled: False,
        )
    )
    generic_profile = SimpleNamespace(
        linear_attention=SimpleNamespace(
            sharded_impl=linear_op_impl.LinearAttentionImplementation.GENERIC,
            has_replicated_pre_projection=lambda _enabled: False,
        )
    )
    config = SimpleNamespace()

    monkeypatch.setattr(
        linear_op_impl,
        "_get_architecture_profile",
        lambda _config: step3_profile,
    )
    monkeypatch.setattr(
        linear_op_impl,
        "get_model_architecture_profile",
        lambda _config: generic_profile,
    )
    monkeypatch.setattr(
        linear_op_impl,
        "Step3TextCausalSelfAttention",
        lambda *_args: step3_attention,
    )
    monkeypatch.setattr(
        linear_op_impl,
        "CausalSelfAttention",
        lambda *_args: generic_attention,
    )

    selected = linear_op_impl.build_linear_op_attention_module(
        config,
        world_size=2,
        enabled_ops=set(),
        attn_sharded_enabled=True,
    )

    assert selected is step3_attention


def test_gpt_block_materializes_only_the_identity_selected_domain(monkeypatch) -> None:
    """A layer-aware block must not duplicate dense or shared MoE weights."""

    monkeypatch.setattr(linear_op_impl, "ColumnParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "RowParallelLinear", _CaptureLinear)
    monkeypatch.setattr(linear_op_impl, "SiluAndMul", torch.nn.Identity)
    monkeypatch.setattr(linear_op_impl, "CudaTimer", _NoopTimer)

    dense_block = linear_op_impl.GPTBlock(
        _tiny_mixed_step3_config(),
        world_size=2,
        profiling_plan=_ffn_only_plan(),
        layer_id=0,
    )
    assert isinstance(dense_block.mlp, linear_op_impl.MLP)
    assert dense_block.share_expert is None
    assert dense_block.mlp_layer_contract.layer_id == 0

    routed_block = linear_op_impl.GPTBlock(
        _tiny_mixed_step3_config(),
        world_size=2,
        profiling_plan=_ffn_only_plan(),
        layer_id=2,
    )
    assert isinstance(routed_block.mlp, linear_op_impl.DummyMLP)
    assert routed_block.share_expert is None


def test_full_structure_passes_layer_identity_to_each_block(monkeypatch) -> None:
    """Full-structure runtime loading must preserve each block's global layer ID."""

    class _NoopModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, value, *args, **kwargs):
            del args, kwargs
            return value

    class _CaptureBlock(torch.nn.Module):
        calls: list[tuple[int, dict | None]] = []

        def __init__(self, config, world_size, profiling_plan=None, *, layer_id):
            super().__init__()
            del config, world_size
            self.__class__.calls.append((int(layer_id), profiling_plan))

    _CaptureBlock.calls = []
    monkeypatch.setattr(runtime_estimator, "GPTBlock", _CaptureBlock)
    monkeypatch.setattr(runtime_estimator, "VocabParallelEmbedding", _NoopModule)
    monkeypatch.setattr(runtime_estimator, "RMSNorm", _NoopModule)

    plan = _ffn_only_plan()
    runtime_estimator._FullStructureGPTModel(
        _tiny_mixed_step3_config(),
        world_size=2,
        profiling_plan=plan,
    )

    assert [layer_id for layer_id, _ in _CaptureBlock.calls] == [0, 1, 2, 3]
    assert _CaptureBlock.calls[0][1] is plan
    assert _CaptureBlock.calls[1][1] is plan
    assert _CaptureBlock.calls[2][1]["ffn_sharded_enabled"] is False
    assert _CaptureBlock.calls[3][1]["ffn_sharded_enabled"] is False


def test_full_structure_uses_typed_widths_for_moe_weight_containers(monkeypatch) -> None:
    """Full-structure MoE containers must use routed and shared widths independently."""

    class _NoopModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _NoopBlock(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(runtime_estimator, "GPTBlock", _NoopBlock)
    monkeypatch.setattr(runtime_estimator, "VocabParallelEmbedding", _NoopModule)
    monkeypatch.setattr(runtime_estimator, "RMSNorm", _NoopModule)

    model = runtime_estimator._FullStructureGPTModel(
        _tiny_mixed_step3_config(),
        world_size=2,
        profiling_plan=_ffn_only_plan(),
        ep_size=1,
        moe_tp_size=1,
    )

    assert model.moe_expert_weights is not None
    routed_expert = model.moe_expert_weights[0].experts[0]
    assert routed_expert["up_proj"].out_features == 12
    assert routed_expert["down_proj"].in_features == 6

    assert model.moe_shared_expert_weights is not None
    shared_expert = model.moe_shared_expert_weights[0]
    assert shared_expert.up_proj.out_features == 6
    assert shared_expert.down_proj.in_features == 3


def test_full_structure_attention_only_skips_zero_moe_domains(monkeypatch) -> None:
    """DECODE_ATTN must keep zero MoE domains and construct attention only."""

    class _NoopModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _CaptureBlock(torch.nn.Module):
        calls: list[dict | None] = []

        def __init__(self, config, world_size, profiling_plan=None, *, layer_id):
            super().__init__()
            del config, world_size, layer_id
            self.__class__.calls.append(profiling_plan)

    monkeypatch.setattr(runtime_estimator, "GPTBlock", _CaptureBlock)
    monkeypatch.setattr(runtime_estimator, "VocabParallelEmbedding", _NoopModule)
    monkeypatch.setattr(runtime_estimator, "RMSNorm", _NoopModule)

    plan = {
        "attn_enabled": True,
        "ffn_enabled": False,
        "ffn_sharded_enabled": False,
        "enabled_ops": ["attn_pre_proj_wq"],
    }
    _CaptureBlock.calls = []
    model = runtime_estimator._FullStructureGPTModel(
        _tiny_mixed_step3_config(),
        world_size=2,
        profiling_plan=plan,
        ep_size=0,
        moe_tp_size=0,
    )

    assert model.moe_expert_weights is None
    assert model.moe_shared_expert_weights is None
    assert _CaptureBlock.calls
    assert all(block_plan["ffn_enabled"] is False for block_plan in _CaptureBlock.calls)


def test_attention_only_forward_does_not_add_dummy_mlp_identity(monkeypatch) -> None:
    """Attention-only blocks must add attention output to residual exactly once."""

    class _Attention(torch.nn.Module):
        def forward(self, *, positions, hidden_states):
            del positions
            return hidden_states * 2

    monkeypatch.setattr(linear_op_impl, "build_linear_op_attention_module", lambda **_kwargs: _Attention())
    monkeypatch.setattr(linear_op_impl, "CudaTimer", _NoopTimer)

    config = _tiny_mixed_step3_config()
    config.post_attn_norm = False
    block = linear_op_impl.GPTBlock(
        config,
        world_size=2,
        profiling_plan={
            "attn_enabled": True,
            "ffn_enabled": False,
            "ffn_sharded_enabled": False,
            "enabled_ops": [],
        },
        layer_id=0,
    )
    block.input_layernorm = torch.nn.Identity()

    hidden_states = torch.tensor([[1.0, 2.0]])
    output, residual = block(
        torch.tensor([0], dtype=torch.long),
        hidden_states,
        None,
    )

    assert torch.equal(residual, hidden_states)
    assert torch.equal(output, hidden_states * 3)


def test_param_counter_matches_runtime_mixed_layer_ownership(monkeypatch) -> None:
    """Parameter counting must follow the modules owned by each layer kind."""

    class _CPUColumnParallelLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, *, world_size=1, bias=False, **kwargs):
            super().__init__()
            del kwargs
            partitioned_output = int(output_size) // int(world_size)
            self.weight = torch.nn.Parameter(
                torch.empty(partitioned_output, int(input_size))
            )
            self.bias = (
                torch.nn.Parameter(torch.empty(partitioned_output)) if bias else None
            )

        def forward(self, hidden_states):
            return hidden_states, None

    class _CPURowParallelLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, *, world_size=1, bias=False, **kwargs):
            super().__init__()
            del kwargs
            partitioned_input = int(input_size) // int(world_size)
            self.weight = torch.nn.Parameter(
                torch.empty(int(output_size), partitioned_input)
            )
            self.bias = (
                torch.nn.Parameter(torch.empty(int(output_size))) if bias else None
            )

        def forward(self, hidden_states):
            return hidden_states, None

    class _CPUEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim, **kwargs):
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(
                torch.empty(int(num_embeddings), int(embedding_dim))
            )

    class _CPUNorm(torch.nn.Module):
        def __init__(self, hidden_dim, *args, **kwargs):
            super().__init__()
            del args, kwargs
            self.weight = torch.nn.Parameter(torch.empty(int(hidden_dim)))

    class _NoopTimer:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args
            return False

    monkeypatch.setattr(linear_op_impl, "ColumnParallelLinear", _CPUColumnParallelLinear)
    monkeypatch.setattr(linear_op_impl, "RowParallelLinear", _CPURowParallelLinear)
    monkeypatch.setattr(linear_op_impl, "SiluAndMul", torch.nn.Identity)
    monkeypatch.setattr(linear_op_impl, "CudaTimer", _NoopTimer)
    monkeypatch.setattr(runtime_estimator, "VocabParallelEmbedding", _CPUEmbedding)
    monkeypatch.setattr(runtime_estimator, "RMSNorm", _CPUNorm)

    config = _tiny_mixed_step3_config()
    config.vocab_size = 16
    runtime_model = runtime_estimator._FullStructureGPTModel(
        config,
        world_size=2,
        profiling_plan=_ffn_only_plan(),
        ep_size=1,
        moe_tp_size=1,
    )

    def _parameter_count(module: torch.nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    runtime_ffn_parameters = sum(
        _parameter_count(layer.mlp) for layer in runtime_model.layers
    )
    runtime_ffn_parameters += sum(
        _parameter_count(container)
        for container in runtime_model.moe_expert_weights or []
    )
    runtime_ffn_parameters += sum(
        _parameter_count(container)
        for container in runtime_model.moe_shared_expert_weights or []
    )

    from types import SimpleNamespace

    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        num_pipeline_stages=1,
    )
    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    assert runtime_ffn_parameters == 1040
    assert counter.get_num_mlp_parameters_per_device() == runtime_ffn_parameters


def test_runtime_model_loader_passes_profiling_plan_to_full_structure(monkeypatch) -> None:
    """The runtime-model-load boundary must preserve the typed profiling plan."""

    captured: dict[str, object] = {}

    class _CaptureFullStructure(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            del config
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_estimator,
        "_FullStructureGPTModel",
        _CaptureFullStructure,
    )
    plan = _ffn_only_plan()

    runtime_estimator._build_runtime_profile_model(
        profiling_model_config=_tiny_mixed_step3_config(),
        tp_size=2,
        pad_vocab_size=False,
        weights_memory_source="runtime_model_load",
        profiling_plan=plan,
    )

    assert captured["profiling_plan"] is plan


def test_runtime_profile_production_path_passes_typed_profiling_plan(monkeypatch) -> None:
    """The production non-KV profile closure must pass a typed plan to the loader."""

    captured: dict[str, object] = {}

    class _FakeModel:
        def to(self, **kwargs):
            del kwargs
            return self

        def cuda(self):
            return self

        def eval(self):
            return self

    class _FakeTimerStore:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def clear_stats(self):
            return None

    class _FakeParamCounter:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def get_num_mtp_parameters_per_device(self):
            return 0

    class _FakeSnapshot:
        total_memory = 123

    class _FakeAllocation:
        def __init__(self, size_bytes):
            del size_bytes

        def allocate(self):
            return None

        def free(self):
            return None

    def _capture_loader(**kwargs):
        captured["profiling_plan"] = kwargs.get("profiling_plan")
        return _FakeModel()

    def _fake_profile(profile_input):
        del profile_input
        return SimpleNamespace(
            breakdown=SimpleNamespace(
                non_kv_cache_memory_bytes=12,
                weights_memory_bytes=10,
                torch_peak_increase_bytes=1,
                non_torch_increase_bytes=1,
            )
        )

    monkeypatch.setattr(runtime_estimator, "_build_runtime_profile_model", _capture_loader)
    monkeypatch.setattr(runtime_estimator, "initialize_dummy_weights", lambda _model: None)
    monkeypatch.setattr(runtime_estimator, "initialize_model_parallel", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_estimator, "destroy_model_parallel", lambda: None)
    monkeypatch.setattr(runtime_estimator, "clear_rope_cache", lambda: None)
    monkeypatch.setattr(
        runtime_estimator,
        "configure_quantization_manager_for_model_name",
        lambda _model_name: None,
    )
    monkeypatch.setattr(runtime_estimator, "TimerStatsStore", _FakeTimerStore)
    monkeypatch.setattr(runtime_estimator, "ParamCounter", _FakeParamCounter)
    monkeypatch.setattr(runtime_estimator, "MemorySnapshot", _FakeSnapshot)
    monkeypatch.setattr(runtime_estimator, "_CudaNonTorchAllocation", _FakeAllocation)
    monkeypatch.setattr(
        runtime_estimator,
        "estimate_vllm_worker_non_torch_bytes",
        lambda **_kwargs: SimpleNamespace(total_bytes=0),
    )
    monkeypatch.setattr(runtime_estimator, "run_single_rank_profile", _fake_profile)
    monkeypatch.setattr(
        runtime_estimator.torch,
        "randint",
        lambda *args, **kwargs: torch.zeros(kwargs["size"], dtype=torch.long),
    )
    monkeypatch.setattr(
        runtime_estimator.torch,
        "arange",
        lambda size, **kwargs: torch.zeros(size, dtype=torch.long),
    )

    replica_model_config = _tiny_mixed_step3_config()
    replica_model_config.get_name = lambda: replica_model_config.name
    # The production path deliberately reloads the canonical model config by
    # model name. Return the tiny fixture here so the plan and loader describe
    # the same model structure without weakening that production boundary.
    monkeypatch.setattr(
        runtime_estimator.ModelConfig,
        "from_model_name",
        lambda _model_name: replica_model_config,
    )
    replica_config = SimpleNamespace(
        model_config=replica_model_config,
        attn_tensor_parallel_size=2,
        attn_dp=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        num_pipeline_stages=1,
    )

    runtime_estimator._profile_non_kv_cache_overhead_bytes_uncached(
        replica_config=replica_config,
        cluster_type=ClusterType.MONOLITHIC,
        max_num_batched_tokens=1,
        weights_memory_bytes=10,
        weights_memory_source="param_counter",
    )

    plan = captured["profiling_plan"]
    assert isinstance(plan, dict)
    assert plan["typed_operator_contracts"]["mlp_up_proj"]["layer_kind"] == "dense"
    assert plan["typed_operator_contracts"]["mlp_up_proj"]["effective_ffn_width"] == 12


def test_decode_attn_preserves_zero_moe_config_but_uses_attention_nccl_domain(
    monkeypatch,
) -> None:
    """DECODE_ATTN keeps zero MoE config values while NCCL sees attention only."""

    captured: dict[str, object] = {}

    class _FakeModel:
        def to(self, **kwargs):
            del kwargs
            return self

        def cuda(self):
            return self

        def eval(self):
            return self

    class _FakeTimerStore:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def clear_stats(self):
            return None

    class _FakeParamCounter:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def get_num_mtp_parameters_per_device(self):
            return 0

    class _FakeSnapshot:
        total_memory = 123

    class _FakeAllocation:
        def __init__(self, size_bytes):
            del size_bytes

        def allocate(self):
            return None

        def free(self):
            return None

    def _capture_loader(**kwargs):
        captured["loader"] = kwargs
        return _FakeModel()

    def _fake_profile(profile_input):
        del profile_input
        return SimpleNamespace(
            breakdown=SimpleNamespace(
                non_kv_cache_memory_bytes=12,
                weights_memory_bytes=10,
                torch_peak_increase_bytes=1,
                non_torch_increase_bytes=1,
            )
        )

    def _capture_nccl(**kwargs):
        captured["nccl"] = kwargs
        return SimpleNamespace(total_bytes=0)

    monkeypatch.setattr(runtime_estimator, "_build_runtime_profile_model", _capture_loader)
    monkeypatch.setattr(runtime_estimator, "initialize_dummy_weights", lambda _model: None)
    monkeypatch.setattr(runtime_estimator, "initialize_model_parallel", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_estimator, "destroy_model_parallel", lambda: None)
    monkeypatch.setattr(runtime_estimator, "clear_rope_cache", lambda: None)
    monkeypatch.setattr(
        runtime_estimator,
        "configure_quantization_manager_for_model_name",
        lambda _model_name: None,
    )
    monkeypatch.setattr(runtime_estimator, "TimerStatsStore", _FakeTimerStore)
    monkeypatch.setattr(runtime_estimator, "ParamCounter", _FakeParamCounter)
    monkeypatch.setattr(runtime_estimator, "MemorySnapshot", _FakeSnapshot)
    monkeypatch.setattr(runtime_estimator, "_CudaNonTorchAllocation", _FakeAllocation)
    monkeypatch.setattr(
        runtime_estimator,
        "estimate_vllm_worker_non_torch_bytes",
        _capture_nccl,
    )
    monkeypatch.setattr(runtime_estimator, "run_single_rank_profile", _fake_profile)
    monkeypatch.setattr(
        runtime_estimator.torch,
        "randint",
        lambda *args, **kwargs: torch.zeros(kwargs["size"], dtype=torch.long),
    )
    monkeypatch.setattr(
        runtime_estimator.torch,
        "arange",
        lambda size, **kwargs: torch.zeros(size, dtype=torch.long),
    )

    replica_model_config = _tiny_mixed_step3_config()
    replica_model_config.get_name = lambda: replica_model_config.name
    monkeypatch.setattr(
        runtime_estimator.ModelConfig,
        "from_model_name",
        lambda _model_name: replica_model_config,
    )
    replica_config = SimpleNamespace(
        model_config=replica_model_config,
        attn_tensor_parallel_size=2,
        attn_dp=1,
        moe_tensor_parallel_size=0,
        moe_expert_parallel_size=0,
        num_pipeline_stages=1,
    )

    runtime_estimator._profile_non_kv_cache_overhead_bytes_uncached(
        replica_config=replica_config,
        cluster_type=ClusterType.DECODE_ATTN,
        max_num_batched_tokens=1,
        weights_memory_bytes=10,
        weights_memory_source="param_counter",
    )

    loader_kwargs = captured["loader"]
    assert loader_kwargs["ep_size"] == 0
    assert loader_kwargs["moe_tp_size"] == 0
    plan = loader_kwargs["profiling_plan"]
    assert plan["ffn_enabled"] is False
    assert plan["ffn_sharded_enabled"] is False

    nccl_kwargs = captured["nccl"]
    assert nccl_kwargs["tp_size"] == 2
    assert nccl_kwargs["ep_size"] == 1
    assert nccl_kwargs["is_moe"] is False
