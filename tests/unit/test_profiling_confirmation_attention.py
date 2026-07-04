from types import SimpleNamespace

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.utils.confirmation import build_attention_config_sections
from frontier.types import ActivationType, NormType


def test_attention_confirmation_reports_replicated_kv_heads_for_mqa() -> None:
    args = SimpleNamespace(
        disable_ray=True,
        num_gpus=8,
        output_dir="out",
        profile_method="cuda_event",
        attention_backend="FLASHINFER",
        num_tensor_parallel_workers=[1, 2, 4, 8],
        use_fp8=False,
        block_shape=None,
        max_model_len=128,
        max_seq_len=128,
        min_batch_size=1,
        max_batch_size=8,
        block_size=16,
        profile_only_prefill=False,
        profile_only_decode=False,
        enable_mixed_prefill=False,
    )
    model_config = ModelConfig(
        name="mqa-confirmation-test",
        num_layers=1,
        num_q_heads=64,
        num_kv_heads=1,
        embedding_dim=16384,
        mlp_hidden_dim=1024,
        max_position_embeddings=128,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=False,
        vocab_size=1024,
        head_dim=256,
    )

    sections = build_attention_config_sections(
        args,
        model_config,
        input_combinations_count=35,
        mixed_combinations_count=0,
        true_mixed_combinations_count=96,
        precision_str="BF16",
        torch_dtype="torch.bfloat16",
    )

    attention_section = dict(sections)["Attention Parameters by TP"]
    per_tp = dict(attention_section)["Per-TP Configuration"]

    assert "TP=1: Q_heads=64, KV_heads=1, head_dim=256" in per_tp
    assert "TP=2: Q_heads=32, KV_heads=1, head_dim=256" in per_tp
    assert "TP=4: Q_heads=16, KV_heads=1, head_dim=256" in per_tp
    assert "TP=8: Q_heads=8, KV_heads=1, head_dim=256" in per_tp
    assert "KV_heads=0" not in per_tp
