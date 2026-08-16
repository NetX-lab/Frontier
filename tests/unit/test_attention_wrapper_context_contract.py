"""CPU-torch tests for attention wrapper context validation."""

import pytest

from frontier.profiling.attention.attention_input import AttentionInput
from frontier.profiling.attention.attention_wrapper import AttentionWrapper
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig
from frontier.types import ActivationType, NormType


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="attention-wrapper-contract",
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=4,
        embedding_dim=256,
        mlp_hidden_dim=512,
        max_position_embeddings=8192,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=False,
        vocab_size=1024,
        head_dim=64,
    )


def _patch_backend(monkeypatch: pytest.MonkeyPatch, fake_backend: object) -> None:
    import frontier.profiling.attention.attention_wrapper as wrapper_module

    monkeypatch.setattr(wrapper_module, "set_attention_backend", lambda _backend: None)
    monkeypatch.setattr(
        wrapper_module,
        "get_attention_wrapper",
        lambda: fake_backend,
    )
    monkeypatch.setattr(
        wrapper_module,
        "configure_quantization_manager_for_model_name",
        lambda _name: None,
    )
    monkeypatch.setattr(wrapper_module.torch, "device", lambda name: name)


def test_attention_wrapper_profile_rejects_invalid_shape_with_value_error() -> None:
    wrapper = AttentionWrapper.__new__(AttentionWrapper)
    wrapper._profile_max_seq_len = 16
    invalid_decode = AttentionInput(
        prefill_chunk_size=0,
        kv_cache_size=16,
        batch_size=1,
        is_prefill=False,
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid standard attention profiling input.*profile_max_seq_len=16",
    ):
        wrapper.profile(invalid_decode)


def test_attention_wrapper_validates_capacity_before_backend_init(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeBackend:
        init_called = False

        def supports_attention_family(self, _family) -> bool:
            return True

        def init(self, *_args) -> None:
            self.init_called = True
            pytest.fail("backend initialized before physical capacity validation")

        def get_cache_block(self, *_args, **_kwargs):
            pytest.fail("KV cache allocated before physical capacity validation")

    fake_backend = FakeBackend()
    _patch_backend(monkeypatch, fake_backend)

    with pytest.raises(
        ValueError,
        match=r"Physical KV-cache capacity.*required_blocks_per_sequence=128",
    ):
        AttentionWrapper(
            model_config=_model_config(),
            parallel_config=ParallelConfig(1, 1),
            max_num_blocks=127,
            max_model_len=4096,
            profile_max_seq_len=8192,
            block_size=64,
            attention_backend="FLASHINFER",
            dtype="bfloat16",
            profile_method="cuda_event",
            output_dir=str(tmp_path),
        )

    assert fake_backend.init_called is False


def test_attention_wrapper_allocates_disjoint_standard_block_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeBackend:
        def supports_attention_family(self, _family) -> bool:
            return True

        def init(self, *_args) -> None:
            return None

        def get_cache_block(self, num_blocks, **kwargs):
            return num_blocks, kwargs

    _patch_backend(monkeypatch, FakeBackend())
    wrapper = AttentionWrapper(
        model_config=_model_config(),
        parallel_config=ParallelConfig(1, 1),
        max_num_blocks=16,
        max_model_len=512,
        profile_max_seq_len=512,
        block_size=64,
        attention_backend="FLASHINFER",
        dtype="bfloat16",
        profile_method="cuda_event",
        output_dir=str(tmp_path),
    )
    wrapper._make_qkv_tensors = lambda _total_tokens: (None, None, None)

    metadata, *_ = wrapper._get_input_tensors(
        AttentionInput(
            prefill_chunk_size=64,
            kv_cache_size=64,
            batch_size=2,
            is_prefill=True,
        )
    )

    assert metadata[0].block_table == [0, 1]
    assert metadata[1].block_table == [2, 3]
