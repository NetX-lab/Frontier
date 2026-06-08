"""Custom activation functions for profiling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[1] // 2.

    Shapes:
        x: (num_tokens, 2 * d)
        return: (num_tokens, d)
    """

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(torch.ops, "_C") or not hasattr(
            torch.ops._C, "silu_and_mul"
        ):
            raise ImportError(
                "vLLM silu_and_mul kernel is unavailable for profiling."
            )
        self._op = torch.ops._C.silu_and_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using vLLM's silu_and_mul kernel."""
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self._op(out, x)
        return out


class NewGELU(nn.Module):
    """GELU activation function with tanh approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


class FastGELU(nn.Module):
    """Fast GELU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_fast": FastGELU(),
    "gelu_new": NewGELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
}


def get_act_fn(act_fn: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn = act_fn.lower()
    if act_fn in _ACTIVATION_REGISTRY:
        return _ACTIVATION_REGISTRY[act_fn]
    raise ValueError(f"Activation function {act_fn!r} is not supported.")
