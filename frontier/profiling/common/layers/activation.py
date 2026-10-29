"""Custom activation functions for profiling."""

from functools import partial

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


class SwigluStepAndMul(nn.Module):
    """Native Step SwiGLU, with a CPU reference for its CUDA arithmetic."""

    native_backend = "vllm.model_executor.layers.activation.swiglustep_and_mul_triton"

    def __init__(self, limit: float | None = None) -> None:
        super().__init__()
        if limit is None:
            raise ValueError("SwigluStepAndMul requires limit to be set.")
        self.limit = limit
        self._op = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            # The native Triton kernel computes both halves in FP32 and casts
            # only the final product, not the intermediate SiLU, back to BF16.
            gate, up = x.float().chunk(2, dim=-1)
            gate = F.silu(gate).clamp(max=self.limit)
            return (gate * up.clamp(-self.limit, self.limit)).to(x.dtype)
        if self._op is None:
            try:
                from vllm.model_executor.layers.activation import (
                    swiglustep_and_mul_triton,
                )
            except ImportError as exc:
                raise ImportError(
                    "Step SwiGLU profiling requires the native "
                    "swiglustep_and_mul_triton kernel from the Step vLLM image."
                ) from exc
            self._op = swiglustep_and_mul_triton
        out = torch.empty(
            x.shape[:-1] + (x.shape[-1] // 2,), dtype=x.dtype, device=x.device,
        )
        self._op(out, x, self.limit)
        return out


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU,
    "gelu_fast": FastGELU,
    "gelu_new": NewGELU,
    "gelu_pytorch_tanh": partial(nn.GELU, approximate="tanh"),
    "relu": nn.ReLU,
    "silu": SiluAndMul,
    "swiglustep": SwigluStepAndMul,
}


def get_act_fn(act_fn: str, *, limit: float | None = None) -> nn.Module:
    """Get an activation function by name."""
    act_fn = act_fn.lower()
    if act_fn in _ACTIVATION_REGISTRY:
        factory = _ACTIVATION_REGISTRY[act_fn]
        return factory() if limit is None else factory(limit=limit)
    raise ValueError(f"Activation function {act_fn!r} is not supported.")
