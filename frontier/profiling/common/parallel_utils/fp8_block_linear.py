"""Block-FP8 linear kernels of vLLM releases without the block-FP8 custom op.

vLLM 0.10 exposes ``torch.ops.vllm.apply_w8a8_block_fp8_linear``; the
profiling linear layers call it directly. Later releases removed that op and
select a block-scaled GEMM kernel per weight shape inside ``Fp8LinearMethod``.
``Fp8LinearMethodKernel`` binds one ``Fp8LinearMethod`` to one profiled weight
so the timed GEMM is the kernel vLLM itself selects for that shape. Kernel
selection reads the active vLLM config (model dtype and model type), so
construction must run inside ``vllm_config_context``; the forward does not read
it.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, Sequence

import torch

LEGACY_BLOCK_FP8_OP_NAME = "apply_w8a8_block_fp8_linear"


@lru_cache(maxsize=1)
def has_legacy_block_fp8_op() -> bool:
    """Return whether the installed vLLM registers the 0.10 block-FP8 op."""

    import vllm.model_executor.layers.quantization.utils.fp8_utils  # noqa: F401

    return hasattr(torch.ops, "vllm") and hasattr(
        torch.ops.vllm, LEGACY_BLOCK_FP8_OP_NAME
    )


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class Fp8LinearMethodKernel:
    """One vLLM ``Fp8LinearMethod`` bound to one block-FP8 weight shard."""

    def __init__(
        self,
        *,
        weight: torch.Tensor,
        block_size: Sequence[int],
        logical_widths: Sequence[int],
        input_size: int,
        output_size: int,
        tensor_parallel_size: int,
    ) -> None:
        from vllm.model_executor.layers.quantization.fp8 import (
            Fp8Config,
            Fp8LinearMethod,
        )
        from vllm.platforms import current_platform

        if weight.ndim != 2:
            raise ValueError(f"block-FP8 weight must be 2-D, got {tuple(weight.shape)}")
        output_size_per_partition, input_size_per_partition = weight.shape
        if sum(logical_widths) != output_size_per_partition:
            raise ValueError(
                f"logical widths {list(logical_widths)} do not sum to the weight "
                f"output size {output_size_per_partition}"
            )
        layer = torch.nn.Module()
        layer.tp_size = tensor_parallel_size
        with _default_dtype(weight.dtype), weight.device:
            method = Fp8LinearMethod(
                Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    activation_scheme="dynamic",
                    weight_block_size=list(block_size),
                )
            )
            method.create_weights(
                layer,
                input_size_per_partition=input_size_per_partition,
                output_partition_sizes=list(logical_widths),
                input_size=input_size,
                output_size=output_size,
                params_dtype=weight.dtype,
                weight_loader=None,
            )
            fp8_dtype = current_platform.fp8_dtype()
            fp8_limit = torch.finfo(fp8_dtype).max
            with torch.no_grad():
                layer.weight.copy_(weight.clamp(-fp8_limit, fp8_limit).to(fp8_dtype))
                layer.weight_scale_inv.fill_(1.0)
            method.process_weights_after_loading(layer)
        self._layer = layer
        self._method = method
        self.backend_name = type(method.fp8_linear).__name__

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return self._method.apply(self._layer, x, bias)


__all__ = [
    "Fp8LinearMethodKernel",
    "LEGACY_BLOCK_FP8_OP_NAME",
    "has_legacy_block_fp8_op",
]
