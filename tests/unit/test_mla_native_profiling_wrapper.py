"""Contracts for Frontier MLA attention profiling wrapper boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY
from frontier.profiling.attention.attention_input import AttentionInput
from frontier.profiling.attention.backends.flashinfer_attention_wrapper import (
    FlashinferAttentionWrapper,
)
from frontier.profiling.attention.attention_wrapper import AttentionWrapper
from frontier.attention.ops import (
    AttentionFamilySpec,
    AttentionMemoryLayout,
    AttentionOperatorRole,
    AttentionOperatorSpec,
    AttentionPhase,
)
from frontier.profiling.attention.mixed_attention_input import MixedAttentionInput
from frontier.profiling.attention.true_mixed_batch_input import TrueMixedBatchInput
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig
from frontier.types import ActivationType, NormType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLASHINFER_WRAPPER_PATH = (
    PROJECT_ROOT
    / "frontier/profiling/attention/backends/flashinfer_attention_wrapper.py"
)


class _FakeBackendWrapper:
    def __init__(self, supported_family_ids: set[str] | None = None) -> None:
        self.init_args = None
        self.cache_kwargs = None
        self.supported_family_ids = supported_family_ids or {"dense_attention"}

    def supports_attention_family(self, attention_family) -> bool:
        return attention_family.family_id in self.supported_family_ids

    def init(self, model_config, parallel_config, block_size, device) -> None:
        self.init_args = (model_config, parallel_config, block_size, device)

    def get_cache_block(self, num_blocks, **kwargs):
        self.cache_kwargs = {"num_blocks": num_blocks, **kwargs}
        return ("cache", num_blocks, kwargs)


def _mla_model_config() -> ModelConfig:
    return ModelConfig(
        name="deepseek-ai/DeepSeek-V2-MLA-Profiling-Unit",
        num_layers=60,
        num_q_heads=128,
        num_kv_heads=128,
        embedding_dim=5120,
        mlp_hidden_dim=12288,
        max_position_embeddings=163840,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=102400,
        dtype="bfloat16",
        model_type="deepseek_v2",
        use_mla=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
    )


def _dense_model_config() -> ModelConfig:
    return ModelConfig(
        name="meta-llama/Llama-3-8B-Dense-Profiling-Unit",
        num_layers=32,
        num_q_heads=32,
        num_kv_heads=8,
        embedding_dim=4096,
        mlp_hidden_dim=14336,
        max_position_embeddings=8192,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=128256,
        dtype="bfloat16",
        model_type="llama",
    )


def _construct_attention_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    model_config: ModelConfig | None = None,
    max_num_blocks: int = 4,
    fake_backend: _FakeBackendWrapper | None = None,
):
    import frontier.profiling.attention.attention_wrapper as attention_wrapper_module

    fake_backend = fake_backend or _FakeBackendWrapper(
        supported_family_ids={"dense_attention", "latent_mla_attention"}
    )
    monkeypatch.setattr(attention_wrapper_module, "set_attention_backend", lambda _backend: None)
    monkeypatch.setattr(
        attention_wrapper_module,
        "get_attention_wrapper",
        lambda: fake_backend,
    )
    monkeypatch.setattr(
        attention_wrapper_module,
        "configure_quantization_manager_for_model_name",
        lambda _name: None,
    )
    monkeypatch.setattr(attention_wrapper_module.torch, "device", lambda name: name)

    wrapper = AttentionWrapper(
        model_config=model_config or _mla_model_config(),
        parallel_config=ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=8),
        max_num_blocks=max_num_blocks,
        max_model_len=4096,
        block_size=64,
        attention_backend="FLASHINFER",
        dtype="bfloat16",
        profile_method="cuda_event",
        output_dir="unused",
    )
    return wrapper, fake_backend


def test_attention_wrapper_rejects_mla_when_backend_is_not_mla_capable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = _FakeBackendWrapper(supported_family_ids={"dense_attention"})

    with pytest.raises(NotImplementedError, match="latent_mla_attention.*FLASHINFER"):
        _construct_attention_wrapper(monkeypatch, fake_backend=fake_backend)

    assert fake_backend.init_args is None
    assert fake_backend.cache_kwargs is None


@pytest.mark.parametrize("attention_backend", ["FLASHINFER", "NO_OP"])
def test_attention_wrapper_rejects_mla_for_current_real_backends(
    monkeypatch: pytest.MonkeyPatch,
    attention_backend: str,
    tmp_path: Path,
) -> None:
    import frontier.profiling.attention.attention_wrapper as attention_wrapper_module

    monkeypatch.setattr(
        attention_wrapper_module,
        "configure_quantization_manager_for_model_name",
        lambda _name: None,
    )

    with pytest.raises(
        NotImplementedError,
        match=rf"latent_mla_attention.*{attention_backend}",
    ):
        AttentionWrapper(
            model_config=_mla_model_config(),
            parallel_config=ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=8,
            ),
            max_num_blocks=4,
            max_model_len=4096,
            block_size=64,
            attention_backend=attention_backend,
            dtype="bfloat16",
            profile_method="cuda_event",
            output_dir=str(tmp_path),
        )


def test_attention_wrapper_allows_mla_only_for_explicit_mla_capable_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = _FakeBackendWrapper(
        supported_family_ids={LATENT_MLA_ATTENTION_FAMILY.family_id}
    )

    wrapper, fake_backend = _construct_attention_wrapper(
        monkeypatch,
        fake_backend=fake_backend,
    )

    assert wrapper._attention_family.family_id == LATENT_MLA_ATTENTION_FAMILY.family_id
    assert wrapper._uses_latent_mla is True
    assert fake_backend.init_args is not None
    assert fake_backend.cache_kwargs == {
        "num_blocks": 4,
        "dtype": "bfloat16",
        "device": "cuda",
    }


def test_attention_wrapper_uses_mla_qk_dim_for_query_and_latent_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, fake_backend = _construct_attention_wrapper(monkeypatch)
    captured_randn_shapes: list[tuple[object, ...]] = []

    def fake_randn(*shape, **_kwargs):
        captured_randn_shapes.append(shape)
        return {"shape": shape}

    monkeypatch.setattr("frontier.profiling.attention.attention_wrapper.torch.randn", fake_randn)

    seq_metadata_list, query, key, value, kv_cache = wrapper._get_input_tensors(
        AttentionInput(
            prefill_chunk_size=16,
            kv_cache_size=64,
            batch_size=2,
            is_prefill=True,
        )
    )

    assert fake_backend.cache_kwargs == {
        "num_blocks": 4,
        "dtype": "bfloat16",
        "device": "cuda",
    }
    assert wrapper._head_dim == 576
    assert wrapper._qk_head_dim == 192
    assert wrapper._v_head_dim == 128
    assert wrapper._softmax_scale == pytest.approx(1.0 / (192**0.5))
    assert captured_randn_shapes == [
        (32, 16 * 192),
        (32, 1 * 512),
        (32, 1 * 64),
    ]
    assert query["shape"] == (32, 16 * 192)
    assert key["shape"] == (32, 1 * 512)
    assert value["shape"] == (32, 1 * 64)
    assert kv_cache == ("cache", 4, {"dtype": "bfloat16", "device": "cuda"})
    assert len(seq_metadata_list) == 2


def test_attention_wrapper_preserves_dense_qkv_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, _fake_backend = _construct_attention_wrapper(
        monkeypatch,
        model_config=_dense_model_config(),
    )
    captured_randn_shapes: list[tuple[object, ...]] = []

    def fake_randn(*shape, **_kwargs):
        captured_randn_shapes.append(shape)
        return {"shape": shape}

    monkeypatch.setattr("frontier.profiling.attention.attention_wrapper.torch.randn", fake_randn)

    _seq_metadata_list, query, key, value, _kv_cache = wrapper._get_input_tensors(
        AttentionInput(
            prefill_chunk_size=16,
            kv_cache_size=64,
            batch_size=2,
            is_prefill=True,
        )
    )

    assert captured_randn_shapes == [
        (32, 4 * 128),
        (32, 1 * 128),
        (32, 1 * 128),
    ]
    assert query["shape"] == (32, 4 * 128)
    assert key["shape"] == (32, 1 * 128)
    assert value["shape"] == (32, 1 * 128)


def test_attention_wrapper_rejects_incomplete_mla_shape_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, _fake_backend = _construct_attention_wrapper(monkeypatch)
    wrapper._kv_lora_rank = None

    with pytest.raises(ValueError, match="kv_lora_rank"):
        wrapper._make_qkv_tensors(total_tokens=1)


def test_attention_wrapper_dense_profiling_controls_follow_family_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import frontier.profiling.attention.attention_wrapper as module

    wrapper, fake_backend = _construct_attention_wrapper(
        monkeypatch,
        model_config=_dense_model_config(),
    )
    renamed_family = AttentionFamilySpec(
        family_id="renamed_dense_attention",
        display_name="Renamed Dense-KV Attention",
        supported_variants=("gqa",),
        operators=(
            AttentionOperatorSpec(
                name="role_cache_profile",
                role=AttentionOperatorRole.CACHE_WRITE,
                phases=(AttentionPhase.PREFILL, AttentionPhase.DECODE),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_prefill_profile",
                role=AttentionOperatorRole.PREFILL_KERNEL,
                phases=(AttentionPhase.PREFILL,),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_decode_profile",
                role=AttentionOperatorRole.DECODE_KERNEL,
                phases=(AttentionPhase.DECODE,),
                e2e_trace_target=False,
            ),
        ),
        memory_layout=AttentionMemoryLayout.DENSE_KV,
        dense_compatible=True,
        requires_runtime_kv_helpers=False,
    )
    checked_ops: list[str] = []

    monkeypatch.setattr(module, "DENSE_ATTENTION_FAMILY", renamed_family)
    monkeypatch.setattr(
        module,
        "raise_if_fp8_requested",
        lambda op_name, _message: checked_ops.append(op_name),
    )
    fake_backend.contains_prefill = False
    fake_backend.contains_decode = False

    wrapper._validate_precision()
    allow_zero_ops = wrapper._get_allow_zero_cuda_ops_for_current_forward()

    assert checked_ops == [
        "role_cache_profile",
        "role_prefill_profile",
        "role_decode_profile",
    ]
    assert {"role_prefill_profile", "role_decode_profile"}.issubset(allow_zero_ops)
    assert "attn_prefill" not in allow_zero_ops
    assert "attn_decode" not in allow_zero_ops


def test_attention_wrapper_uses_mla_shapes_for_mixed_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, _fake_backend = _construct_attention_wrapper(monkeypatch)
    captured_randn_shapes: list[tuple[object, ...]] = []

    def fake_randn(*shape, **_kwargs):
        captured_randn_shapes.append(shape)
        return {"shape": shape}

    monkeypatch.setattr("frontier.profiling.attention.attention_wrapper.torch.randn", fake_randn)

    seq_metadata_list, query, key, value, _kv_cache = wrapper._get_mixed_input_tensors(
        MixedAttentionInput(seq_lens=[8, 16, 24], kv_cache_size=32)
    )

    assert captured_randn_shapes == [
        (48, 16 * 192),
        (48, 1 * 512),
        (48, 1 * 64),
    ]
    assert query["shape"] == (48, 16 * 192)
    assert key["shape"] == (48, 1 * 512)
    assert value["shape"] == (48, 1 * 64)
    assert len(seq_metadata_list) == 3


def test_attention_wrapper_uses_mla_shapes_for_true_mixed_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, _fake_backend = _construct_attention_wrapper(monkeypatch, max_num_blocks=8)
    captured_randn_shapes: list[tuple[object, ...]] = []

    def fake_randn(*shape, **_kwargs):
        captured_randn_shapes.append(shape)
        return {"shape": shape}

    monkeypatch.setattr("frontier.profiling.attention.attention_wrapper.torch.randn", fake_randn)

    seq_metadata_list, query, key, value, _kv_cache = wrapper._get_true_mixed_input_tensors(
        TrueMixedBatchInput(
            prefill_seq_lens=[32, 64],
            prefill_kv_cache_sizes=[0, 16],
            decode_kv_cache_sizes=[0, 16, 32],
        )
    )

    assert captured_randn_shapes == [
        (99, 16 * 192),
        (99, 1 * 512),
        (99, 1 * 64),
    ]
    assert query["shape"] == (99, 16 * 192)
    assert key["shape"] == (99, 1 * 512)
    assert value["shape"] == (99, 1 * 64)
    assert len(seq_metadata_list) == 5


def test_flashinfer_wrapper_rejects_mla_during_init(monkeypatch: pytest.MonkeyPatch) -> None:
    import frontier.profiling.attention.backends.flashinfer_attention_wrapper as module

    monkeypatch.setattr(module, "HAS_FLASHINFER", True)
    monkeypatch.setattr(module, "HAS_VLLM", True)
    monkeypatch.setattr(module, "get_kv_cache_layout", lambda: "NHD")
    monkeypatch.setattr(
        module.BaseAttentionWrapper,
        "init",
        lambda self, model_config, parallel_config, block_size, device: None,
    )

    wrapper = FlashinferAttentionWrapper()

    with pytest.raises(NotImplementedError, match="MLA profiling is not implemented"):
        wrapper.init(
            _mla_model_config(),
            ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=8),
            block_size=64,
            device="cuda",
        )


def test_flashinfer_wrapper_rejects_mla_before_dense_cache_or_forward() -> None:
    wrapper = FlashinferAttentionWrapper()
    wrapper._uses_latent_mla = True
    wrapper.is_metadata_initialized = True

    with pytest.raises(NotImplementedError, match="MLA profiling is not implemented"):
        wrapper.get_cache_block(1)

    with pytest.raises(NotImplementedError, match="dense FlashInfer path cannot be reused"):
        wrapper.forward(
            query=object(),
            key=object(),
            value=object(),
            kv_cache=object(),
            softmax_scale=1.0,
        )
