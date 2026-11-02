"""Checkpoint-driven vLLM quantization resolution, with worker-only imports."""

from types import SimpleNamespace


def checkpoint_hf_config(model_config):
    """Expose the checkpoint fields read by vLLM, without a model-name switch."""

    checkpoint = model_config.to_dict()["quantization_config"]
    if checkpoint is not None:
        checkpoint = dict(checkpoint)
        checkpoint["fmt"] = model_config.quantization_config.fmt
        if checkpoint.get("weight_block_size") is not None:
            checkpoint["weight_block_size"] = list(checkpoint["weight_block_size"])
    hf_config = SimpleNamespace(
        model_type=model_config.model_type,
        expert_dtype=model_config.expert_dtype,
        quantization_config=checkpoint,
    )
    return SimpleNamespace(dtype=model_config.dtype, hf_config=hf_config,
                           hf_text_config=hf_config)


def resolve_checkpoint_quantization(hf_config):
    """Detect the checkpoint method with vLLM's own rule, then build its config.

    ``ModelConfig._verify_quantization`` reads only the fields set on ``state``;
    the method it selects is left in ``state.quantization``.
    """

    from vllm.config.model import ModelConfig as VllmModelConfig
    from vllm.model_executor.layers.quantization import get_quantization_config

    checkpoint = hf_config.quantization_config
    state = SimpleNamespace(
        quantization=None, hf_config=hf_config, allow_deprecated_quantization=False,
        model_arch_config=SimpleNamespace(quantization_config=checkpoint),
    )
    VllmModelConfig._verify_quantization(state)
    return state.quantization, get_quantization_config(state.quantization).from_config(checkpoint)
