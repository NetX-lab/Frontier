"""Custom normalization layers for profiling."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.profiling.common.vllm_compat import vllm_config_context

VllmRMSNorm = None
try:
    from vllm.model_executor.layers.layernorm import (
        GemmaRMSNorm as VllmGemmaRMSNorm,
        rms_norm as vllm_rms_norm,
        fused_add_rms_norm as vllm_fused_add_rms_norm,
    )

    HAS_VLLM_RMSNORM = True
except ImportError:
    try:
        from vllm.model_executor.layers.layernorm import (
            GemmaRMSNorm as VllmGemmaRMSNorm,
            RMSNorm as VllmRMSNorm,
        )

        HAS_VLLM_RMSNORM = True
    except ImportError:
        HAS_VLLM_RMSNORM = False
        VllmGemmaRMSNorm = None
    vllm_rms_norm = None
    vllm_fused_add_rms_norm = None


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        norm_name: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._impl = None
        if VllmRMSNorm is not None:
            with vllm_config_context():
                self._impl = VllmRMSNorm(hidden_size, eps=eps)
            self.register_parameter("_weight", None)
        else:
            self._weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._norm_timer = CudaTimer(norm_name, layer_id=layer_id)

    @property
    def weight(self) -> nn.Parameter:
        if self._impl is not None:
            return self._impl.weight
        return self._weight

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using vLLM's fused RMSNorm CUDA kernel."""
        with self._norm_timer:
            if not HAS_VLLM_RMSNORM:
                raise ImportError(
                    "vLLM is required for RMSNorm profiling. "
                    "Install vllm or set PYTHONPATH to the vllm source tree."
                )
            if self._impl is not None:
                return self._impl(x, residual)
            if residual is None:
                return vllm_rms_norm(x, self.weight.data, self.variance_epsilon)
            return vllm_fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm wrapper with profiling timer."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        norm_name: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._norm_timer = CudaTimer(norm_name, layer_id=layer_id)
        if not HAS_VLLM_RMSNORM or VllmGemmaRMSNorm is None:
            raise ImportError(
                "vLLM is required for GemmaRMSNorm profiling. "
                "Install vllm or set PYTHONPATH to the vllm source tree."
            )
        with vllm_config_context():
            self._impl = VllmGemmaRMSNorm(hidden_size, eps=eps)

    @property
    def weight(self) -> nn.Parameter:
        return self._impl.weight

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with self._norm_timer:
            return self._impl(x, residual)


class Step5RMSNorm(nn.Module):
    """Step's native mixed-dtype, zero-centered RMSNorm with scope timing."""

    native_backend = "vllm.models.step5.nvidia.layernorm.make_step5_rms_norm"

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        zero_centered: bool,
        dtype: torch.dtype,
        fp32_residual_connection: bool,
        norm_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not zero_centered:
            raise NotImplementedError("Step5 requires zero-centered RMSNorm weights")
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.zero_centered = zero_centered
        self.fp32_residual_connection = fp32_residual_connection
        self._weight_dtype = dtype
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self._impl = None
        self._norm_timer = CudaTimer(norm_name)

    def _apply(self, fn, recurse=True):
        # The enclosing profiling model is converted to BF16. Norm weights
        # must retain the independently configured native norm_dtype.
        weight = self.weight

        def apply(tensor):
            if tensor is weight:
                device = fn(tensor.new_empty(0)).device
                return tensor.to(device=device, dtype=self._weight_dtype)
            return fn(tensor)

        return super()._apply(apply, recurse=recurse)

    def _native_impl(self):
        if self._impl is None:
            try:
                from vllm.models.step5.nvidia.layernorm import make_step5_rms_norm
            except ImportError as exc:
                raise ImportError(
                    "Step RMSNorm profiling requires make_step5_rms_norm "
                    "from the Step vLLM image."
                ) from exc
            self._impl = make_step5_rms_norm(
                self.hidden_size,
                eps=self.variance_epsilon,
                zero_centered=self.zero_centered,
                dtype=self._weight_dtype,
            )
            self._impl.weight = self.weight
        return self._impl

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        out_dtype = torch.bfloat16 if self.fp32_residual_connection else x.dtype
        if residual is not None and self.fp32_residual_connection:
            residual = residual.float()
        impl = self._native_impl() if x.is_cuda else None
        with self._norm_timer:
            if impl is not None:
                return impl(x, residual, out_dtype=out_dtype)
            summed = x if residual is None else x + residual
            compute = summed.float()
            variance = compute.square().mean(dim=-1, keepdim=True)
            normalized = compute * torch.rsqrt(variance + self.variance_epsilon)
            output = (normalized * (self.weight.float() + 1.0)).to(out_dtype)
            return output if residual is None else (output, summed)
