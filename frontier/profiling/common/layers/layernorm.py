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
