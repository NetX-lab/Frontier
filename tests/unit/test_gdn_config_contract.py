import pytest

from frontier.gdn import (
    GatedDeltaNetConfig,
    SequenceMixerType,
    build_sequence_mixer_schedule,
)


def test_interval_schedule_matches_qwen_three_gdn_then_attention() -> None:
    schedule = build_sequence_mixer_schedule(
        num_layers=8,
        layer_types=None,
        full_attention_interval=4,
        has_gated_delta_net=True,
    )

    assert schedule == (
        SequenceMixerType.GATED_DELTA_NET,
        SequenceMixerType.GATED_DELTA_NET,
        SequenceMixerType.GATED_DELTA_NET,
        SequenceMixerType.FULL_ATTENTION,
        SequenceMixerType.GATED_DELTA_NET,
        SequenceMixerType.GATED_DELTA_NET,
        SequenceMixerType.GATED_DELTA_NET,
        SequenceMixerType.FULL_ATTENTION,
    )


def test_explicit_hf_linear_attention_alias_normalizes_to_gdn() -> None:
    schedule = build_sequence_mixer_schedule(
        num_layers=2,
        layer_types=["linear_attention", "full_attention"],
        full_attention_interval=None,
        has_gated_delta_net=True,
    )

    assert schedule == (
        SequenceMixerType.GATED_DELTA_NET,
        SequenceMixerType.FULL_ATTENTION,
    )


def test_schedule_rejects_gdn_without_shape_contract() -> None:
    with pytest.raises(ValueError, match="shape configuration is incomplete"):
        build_sequence_mixer_schedule(
            num_layers=4,
            layer_types=None,
            full_attention_interval=4,
            has_gated_delta_net=False,
        )


def test_gdn_state_layout_rejects_non_divisible_tensor_parallelism() -> None:
    config = GatedDeltaNetConfig(
        conv_kernel_size=4,
        key_head_dim=128,
        value_head_dim=128,
        num_key_heads=16,
        num_value_heads=128,
    )

    with pytest.raises(ValueError, match="key heads must be divisible"):
        config.get_state_layout(tensor_parallel_size=3)


def test_speculative_tokens_expand_only_conv_state_width() -> None:
    config = GatedDeltaNetConfig(
        conv_kernel_size=4,
        key_head_dim=128,
        value_head_dim=128,
        num_key_heads=16,
        num_value_heads=128,
    )

    layout = config.get_state_layout(
        tensor_parallel_size=8,
        num_speculative_tokens=4,
    )

    assert layout.conv_state_shape == (7, 2560)
    assert layout.recurrent_state_shape == (16, 128, 128)
