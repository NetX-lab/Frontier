"""Attention backends for profiling."""

from enum import Enum
from typing import Union


class AttentionBackend(Enum):
    """Attention backend types for profiling."""

    FLASHINFER = "FLASHINFER"
    NO_OP = "NO_OP"


ATTENTION_BACKEND = AttentionBackend.NO_OP


def set_attention_backend(backend: Union[str, AttentionBackend]):
    """Set the global attention backend.

    Args:
        backend: Either a string name or AttentionBackend enum value.

    Raises:
        ValueError: If the backend is not supported.
    """
    if isinstance(backend, str):
        backend = backend.upper()
        if backend not in AttentionBackend.__members__:
            raise ValueError(f"Unsupported attention backend: {backend}")
        backend = AttentionBackend[backend]
    elif not isinstance(backend, AttentionBackend):
        raise ValueError(f"Unsupported attention backend: {backend}")

    global ATTENTION_BACKEND
    ATTENTION_BACKEND = backend


def get_attention_wrapper():
    """Get the attention wrapper instance for the current backend.

    Returns:
        The singleton instance of the appropriate attention wrapper.

    Raises:
        ValueError: If the backend is not supported.
    """
    if ATTENTION_BACKEND == AttentionBackend.FLASHINFER:
        from frontier.profiling.attention.backends.flashinfer_attention_wrapper import (
            FlashinferAttentionWrapper,
        )

        return FlashinferAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.NO_OP:
        from frontier.profiling.attention.backends.no_op_attention_wrapper import (
            NoOpAttentionWrapper,
        )

        return NoOpAttentionWrapper.get_instance()

    raise ValueError(f"Unsupported attention backend: {ATTENTION_BACKEND}")


def __getattr__(name: str):
    if name == "BaseAttentionWrapper":
        from frontier.profiling.attention.backends.base_attention_wrapper import (
            BaseAttentionWrapper,
        )

        return BaseAttentionWrapper
    if name == "FlashinferAttentionWrapper":
        from frontier.profiling.attention.backends.flashinfer_attention_wrapper import (
            FlashinferAttentionWrapper,
        )

        return FlashinferAttentionWrapper
    if name == "NoOpAttentionWrapper":
        from frontier.profiling.attention.backends.no_op_attention_wrapper import (
            NoOpAttentionWrapper,
        )

        return NoOpAttentionWrapper
    raise AttributeError(name)


__all__ = [
    "AttentionBackend",
    "BaseAttentionWrapper",
    "FlashinferAttentionWrapper",
    "NoOpAttentionWrapper",
    "set_attention_backend",
    "get_attention_wrapper",
]
