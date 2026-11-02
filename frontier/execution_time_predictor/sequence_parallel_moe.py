"""Sequence-parallel widths for local normalization outside the EP wave."""

from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths


def input_layernorm_tokens(predictor, batch, layer_id: int | None) -> int:
    """Keep the dense-to-MoE boundary full width; later inputs stay split."""

    tokens = batch.get_effective_total_tokens_rounded(predictor._cluster_type)
    if layer_id is None or layer_id == 0:
        return tokens
    model = predictor._model_config
    replica = predictor._replica_config
    if (
        model.get_model_architecture_profile().uses_sequence_parallel_moe(
            replica.attn_tensor_parallel_size, replica.attn_dp,
            replica.num_pipeline_stages, replica.moe_expert_parallel_size,
        )
        and model.is_moe_layer(layer_id - 1)
    ):
        return sequence_parallel_moe_token_widths(
            [tokens], replica.attn_tensor_parallel_size,
        ).local_tokens[0]
    return tokens
