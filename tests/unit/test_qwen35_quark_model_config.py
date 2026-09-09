from frontier.config.model_config import BaseModelConfig, QuantizationConfig
from frontier.config.precision_type import PrecisionType
from frontier.config.quantization_manager import QuantizationManager
from frontier.gdn import SequenceMixerType
from frontier.profiling.common.model_config import ModelConfig


def test_quark_mxfp4_config_maps_to_selective_fp4_operations() -> None:
    quantization = QuantizationConfig.from_dict(
        {
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {"dtype": "fp4", "group_size": 32},
                "input_tensors": {
                    "dtype": "fp4",
                    "group_size": 32,
                    "is_dynamic": True,
                },
            },
            "frontier_quantized_operations": ["moe_grouped_gemm"],
        }
    )

    assert quantization.quant_method == "fp4"
    assert quantization.activation_scheme == "dynamic"
    assert quantization.quantized_operations == ["moe_grouped_gemm"]


def test_qwen38_profile_config_preserves_hybrid_moe_shape_and_precision() -> None:
    model_name = "Qwen3.8-2.4T-A95B-Quark-MXFP4"
    base_config = BaseModelConfig.create_from_name(model_name)
    profile_config = ModelConfig.from_model_name(model_name)

    assert base_config.num_layers == 92
    assert base_config.num_experts == 512
    assert base_config.num_experts_per_tok == 10
    assert base_config.embedding_dim == 8192
    assert base_config.mlp_hidden_dim == 2048
    assert base_config.quantization_config is not None
    assert base_config.quantization_config.quant_method == "fp4"
    assert base_config.quantization_config.quantized_operations == [
        "moe_grouped_gemm"
    ]
    assert base_config.partial_rotary_factor == 0.25
    assert profile_config.use_qk_norm is True
    assert profile_config.attn_output_gate is True
    assert profile_config.partial_rotary_factor == 0.25
    assert profile_config.share_expert_dim == 2048
    assert base_config.get_model_architecture_profile().profile_id == "qwen3_5_moe"
    assert profile_config.get_model_architecture_profile().profile_id == "qwen3_5_moe"
    assert base_config.get_num_gdn_layers() == 69
    assert base_config.get_num_full_attention_layers() == 23
    assert profile_config.get_num_gdn_layers() == 69
    assert profile_config.get_num_full_attention_layers() == 23
    assert base_config.get_sequence_mixer_type(0) is SequenceMixerType.GATED_DELTA_NET
    assert base_config.get_sequence_mixer_type(3) is SequenceMixerType.FULL_ATTENTION
    assert base_config.get_sequence_mixer_layer_ids(
        SequenceMixerType.FULL_ATTENTION
    ) == list(range(3, 92, 4))

    gdn_config = base_config.get_gdn_config()
    assert gdn_config is not None
    assert gdn_config.conv_dim == 20480
    assert gdn_config.key_dim == 2048
    assert gdn_config.value_dim == 16384


def test_qwen38_gdn_state_layout_matches_vllm_shapes() -> None:
    config = BaseModelConfig.create_from_name(
        "Qwen3.8-2.4T-A95B-Quark-MXFP4"
    )
    gdn_config = config.get_gdn_config()
    assert gdn_config is not None

    expected_bytes_by_tp = {
        1: 8_511_488,
        2: 4_255_744,
        4: 2_127_872,
        8: 1_063_936,
    }
    for tp_size, expected_bytes in expected_bytes_by_tp.items():
        state_layout = gdn_config.get_state_layout(tensor_parallel_size=tp_size)
        assert state_layout.conv_state_shape == (3, 20480 // tp_size)
        assert state_layout.recurrent_state_shape == (
            128 // tp_size,
            128,
            128,
        )
        assert state_layout.total_bytes == expected_bytes

    assert (
        gdn_config.get_state_layout(tensor_parallel_size=1).total_bytes
        * config.get_num_gdn_layers()
        == 587_292_672
    )


def test_qwen38_quantization_manager_keeps_non_routed_ops_bf16() -> None:
    manager = QuantizationManager()
    config = BaseModelConfig.create_from_name(
        "Qwen3.8-2.4T-A95B-Quark-MXFP4"
    )
    try:
        manager.configure_from_model_config(config)
        assert manager.get_precision("moe_grouped_gemm") is PrecisionType.FP4
        assert manager.get_precision("attn_pre_proj") is PrecisionType.BF16
        assert manager.get_precision("share_expert_up_proj") is PrecisionType.BF16
    finally:
        manager.load_config()
