"""Small compatibility helpers for standalone use of current vLLM custom ops."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional


@contextmanager
def vllm_config_context(*, model_config: Optional[Any] = None):
    """Provide a standalone vLLM config only when none is active.

    ``model_config`` is used by current quantized kernels whose backend oracle
    reads dtype and quantization choices from the active vLLM configuration.
    """
    try:
        from vllm.config import (
            VllmConfig,
            get_current_vllm_config_or_none,
            set_current_vllm_config,
        )
    except ImportError:
        yield
        return

    if get_current_vllm_config_or_none() is not None:
        yield
        return

    standalone_config = VllmConfig()
    if model_config is not None:
        standalone_config.model_config = model_config
    with set_current_vllm_config(standalone_config):
        yield


__all__ = ["vllm_config_context"]
