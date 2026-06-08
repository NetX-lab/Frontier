# Copyright 2023 The Sarathi team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
"""Tensor parallel layers for profiling."""

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from frontier.config import PrecisionType, get_quantization_manager
from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.profiling.common.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from frontier.profiling.common.parallel_utils.tensor_parallel_mappings import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from frontier.profiling.common.parallel_utils.tensor_parallel_utils import (
    VocabUtility,
    divide,
)


HAS_VLLM_GEMM = False
dispatch_unquantized_gemm = None
_VLLM_GEMM_IMPORT_ERROR = None

HAS_VLLM_FP8 = False
vllm_ops = None
GroupShape = None
scaled_quantize = None
Fp8LinearOp = None
cutlass_block_fp8_supported = None
should_use_deepgemm_for_fp8_linear = None
current_platform = None
_VLLM_FP8_IMPORT_ERROR = None


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def _load_dispatch_unquantized_gemm():
    """Lazily import vLLM GEMM dispatch for non-CUDA compatibility paths."""
    global HAS_VLLM_GEMM
    global dispatch_unquantized_gemm
    global _VLLM_GEMM_IMPORT_ERROR

    if dispatch_unquantized_gemm is not None:
        return dispatch_unquantized_gemm
    if _VLLM_GEMM_IMPORT_ERROR is not None:
        return None

    try:
        from vllm.model_executor.layers.utils import (
            dispatch_unquantized_gemm as loaded_dispatch_unquantized_gemm,
        )
    except Exception as exc:
        HAS_VLLM_GEMM = False
        _VLLM_GEMM_IMPORT_ERROR = exc
        return None

    dispatch_unquantized_gemm = loaded_dispatch_unquantized_gemm
    HAS_VLLM_GEMM = True
    return dispatch_unquantized_gemm


def _load_vllm_fp8_symbols():
    """Lazily import vLLM FP8 symbols only when FP8 kernels are required."""
    global HAS_VLLM_FP8
    global vllm_ops
    global GroupShape
    global scaled_quantize
    global Fp8LinearOp
    global cutlass_block_fp8_supported
    global should_use_deepgemm_for_fp8_linear
    global current_platform
    global _VLLM_FP8_IMPORT_ERROR

    if HAS_VLLM_FP8 and (
        GroupShape is not None
        or Fp8LinearOp is not None
        or vllm_ops is not None
        or scaled_quantize is not None
        or current_platform is not None
    ):
        return {
            "vllm_ops": vllm_ops,
            "GroupShape": GroupShape,
            "scaled_quantize": scaled_quantize,
            "Fp8LinearOp": Fp8LinearOp,
            "cutlass_block_fp8_supported": cutlass_block_fp8_supported,
            "should_use_deepgemm_for_fp8_linear": should_use_deepgemm_for_fp8_linear,
            "current_platform": current_platform,
        }
    if _VLLM_FP8_IMPORT_ERROR is not None:
        return None

    try:
        import vllm._custom_ops as loaded_vllm_ops
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            GroupShape as loaded_group_shape,
            scaled_quantize as loaded_scaled_quantize,
        )
        import vllm.model_executor.layers.quantization.utils.fp8_utils  # noqa: F401
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            Fp8LinearOp as loaded_fp8_linear_op,
            cutlass_block_fp8_supported as loaded_cutlass_block_fp8_supported,
        )
        from vllm.platforms import current_platform as loaded_current_platform
        from vllm.utils.deep_gemm import (
            should_use_deepgemm_for_fp8_linear as loaded_should_use_deepgemm,
        )
    except Exception as exc:
        HAS_VLLM_FP8 = False
        _VLLM_FP8_IMPORT_ERROR = exc
        return None

    HAS_VLLM_FP8 = True
    vllm_ops = loaded_vllm_ops
    GroupShape = loaded_group_shape
    scaled_quantize = loaded_scaled_quantize
    Fp8LinearOp = loaded_fp8_linear_op
    cutlass_block_fp8_supported = loaded_cutlass_block_fp8_supported
    should_use_deepgemm_for_fp8_linear = loaded_should_use_deepgemm
    current_platform = loaded_current_platform
    return {
        "vllm_ops": vllm_ops,
        "GroupShape": GroupShape,
        "scaled_quantize": scaled_quantize,
        "Fp8LinearOp": Fp8LinearOp,
        "cutlass_block_fp8_supported": cutlass_block_fp8_supported,
        "should_use_deepgemm_for_fp8_linear": should_use_deepgemm_for_fp8_linear,
        "current_platform": current_platform,
    }


def _require_vllm_fp8_symbols():
    symbols = _load_vllm_fp8_symbols()
    if symbols is None:
        raise ImportError(
            "vLLM FP8 utilities are required for FP8 profiling. "
            "Install vllm or set PYTHONPATH to the vllm source tree."
        ) from _VLLM_FP8_IMPORT_ERROR
    return symbols


def _use_fp8_gemm_surrogate() -> bool:
    return os.environ.get("FRONTIER_FP8_GEMM_SURROGATE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def param_is_not_tensor_parallel_duplicate(param):
    return (
        hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
    ) or (get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _run_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run unquantized GEMM without forcing vLLM imports in memory probes."""
    if x.is_cuda and torch.version.hip is None:
        return torch.nn.functional.linear(x, weight, bias)

    dispatch_factory = _load_dispatch_unquantized_gemm()
    if dispatch_factory is None:
        if _use_fp8_gemm_surrogate():
            return torch.nn.functional.linear(x, weight, bias)
        raise ImportError(
            "vLLM is required for TP linear profiling. "
            "Install vllm or set PYTHONPATH to the vllm source tree."
        ) from _VLLM_GEMM_IMPORT_ERROR

    gemm = dispatch_factory()
    if gemm is torch.nn.functional.linear:
        return gemm(x, weight, bias)
    return gemm(layer, x, weight, bias)


def _maybe_adjust_fp8_weight_scale_layout(
    fp8_weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Match vLLM block-FP8 scale layout expectations on Hopper/H800.

    vLLM 0.10 block-FP8 CUTLASS path expects row-major (transposed) weight
    scales on SM90 when DeepGEMM is not selected.

    Args:
        fp8_weight: The FP8-quantized weight tensor.
        weight_scale_inv: The inverse scale tensor for the FP8 weight.
        output_dtype: The model's compute dtype (e.g. torch.bfloat16 or
            torch.float16). DeepGEMM only supports bfloat16 output, so
            float16 models always take the CUTLASS path and need the
            transposed scale layout.
    """
    symbols = _load_vllm_fp8_symbols()
    if symbols is None:
        return weight_scale_inv
    loaded_current_platform = symbols["current_platform"]
    loaded_cutlass_supported = symbols["cutlass_block_fp8_supported"]
    loaded_should_use_deepgemm = symbols["should_use_deepgemm_for_fp8_linear"]

    if loaded_current_platform is None:
        return weight_scale_inv
    if not loaded_current_platform.is_device_capability(90):
        return weight_scale_inv
    if loaded_cutlass_supported is None or not loaded_cutlass_supported():
        return weight_scale_inv
    if loaded_should_use_deepgemm is None:
        return weight_scale_inv
    if loaded_should_use_deepgemm(output_dtype, fp8_weight):
        return weight_scale_inv
    return weight_scale_inv.T.contiguous()


def _create_tensorwise_fp8_linear_op() -> "Fp8LinearOp":
    symbols = _require_vllm_fp8_symbols()
    fp8_linear_op = symbols["Fp8LinearOp"]
    group_shape = symbols["GroupShape"]
    if fp8_linear_op is None or group_shape is None:
        raise ImportError(
            "vLLM FP8 utilities are required for tensorwise FP8 profiling. "
            "Install vllm or set PYTHONPATH to the vllm source tree."
        )
    return fp8_linear_op(
        act_quant_static=False,
        act_quant_group_shape=group_shape.PER_TOKEN,
    )


def _quantize_fp8_weight(
    *,
    weight: torch.Tensor,
    fp8_weight_block_size: Optional[Tuple[int, int]],
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    symbols = _require_vllm_fp8_symbols()

    if fp8_weight_block_size is None:
        loaded_vllm_ops = symbols["vllm_ops"]
        if loaded_vllm_ops is None:
            raise ImportError(
                "vLLM FP8 custom ops are required for tensorwise FP8 profiling."
            )
        fp8_weight, weight_scale = loaded_vllm_ops.scaled_fp8_quant(
            weight,
            scale=None,
        )
        return fp8_weight.T, weight_scale

    if not hasattr(torch.ops, "vllm") or not hasattr(
        torch.ops.vllm, "apply_w8a8_block_fp8_linear"
    ):
        raise ImportError(
            "vLLM FP8 kernel apply_w8a8_block_fp8_linear is unavailable."
        )

    block_m, block_n = fp8_weight_block_size
    if weight.shape[0] % block_m != 0 or weight.shape[1] % block_n != 0:
        raise ValueError(
            "FP8 block size must divide weight shape: "
            f"weight={tuple(weight.shape)} "
            f"block_size={fp8_weight_block_size}"
        )

    loaded_group_shape = symbols["GroupShape"]
    loaded_current_platform = symbols["current_platform"]
    loaded_scaled_quantize = symbols["scaled_quantize"]
    if (
        loaded_group_shape is None
        or loaded_current_platform is None
        or loaded_scaled_quantize is None
    ):
        raise ImportError(
            "vLLM FP8 block quantization utilities are required for FP8 profiling."
        )
    group_shape = loaded_group_shape(block_m, block_n)
    fp8_dtype = loaded_current_platform.fp8_dtype()
    fp8_weight, weight_scale_inv = loaded_scaled_quantize(
        weight,
        group_shape,
        fp8_dtype,
    )
    weight_scale_inv = _maybe_adjust_fp8_weight_scale_layout(
        fp8_weight,
        weight_scale_inv,
        output_dtype,
    )
    return fp8_weight, weight_scale_inv


def _apply_fp8_linear(
    *,
    input_tensor: torch.Tensor,
    fp8_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    fp8_weight_block_size: Optional[list[int]],
    output_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
    tensorwise_fp8_linear_op: Optional["Fp8LinearOp"],
) -> torch.Tensor:
    if fp8_weight_block_size is not None:
        return torch.ops.vllm.apply_w8a8_block_fp8_linear(
            input=input_tensor,
            weight=fp8_weight,
            block_size=fp8_weight_block_size,
            weight_scale=weight_scale,
            input_scale=None,
            bias=bias,
        )

    if tensorwise_fp8_linear_op is None:
        raise RuntimeError("Tensorwise FP8 profiling op was not initialized.")

    return tensorwise_fp8_linear_op.apply(
        input=input_tensor,
        weight=fp8_weight,
        weight_scale=weight_scale,
        out_dtype=output_dtype,
        input_scale=None,
        bias=bias,
    )


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Arguments:
        init_method: method to initialize weights.
        params_dtype
        use_cpu_initialization
        perform_initialization
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method=init.xavier_normal_,
        params_dtype: torch.dtype = None,
        use_cpu_initialization: bool = False,
        perform_initialization: bool = False,
        linear_metric_name: Optional[str] = None,
        communication_metric_name: Optional[str] = None,
        reduce_results: Optional[bool] = True,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        pad_vocab_size: bool = False,
    ):
        super(VocabParallelEmbedding, self).__init__()
        assert not perform_initialization
        assert not use_cpu_initialization

        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = (
            get_tensor_model_parallel_world_size() if world_size is None else world_size
        )
        self.rank = get_tensor_model_parallel_rank() if rank is None else rank
        self.reduce_results = reduce_results
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                self.rank,
                self.tensor_model_parallel_size,
                pad_vocab_size=pad_vocab_size,
            )
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        self.weight = Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )

        # Keep the public emb metric at the module-forward boundary so the
        # timed region includes local TP masking plus the embedding lookup.
        self._emb_forward_timer = CudaTimer(linear_metric_name)
        self._linear_timer = CudaTimer(None)
        self._communication_timer = CudaTimer(communication_metric_name)

    def forward(self, input_):
        with self._emb_forward_timer:
            if self.tensor_model_parallel_size > 1:
                # Build the mask.
                input_mask = (input_ < self.vocab_start_index) | (
                    input_ >= self.vocab_end_index
                )
                # Mask the input.
                masked_input = input_.clone() - self.vocab_start_index
                masked_input[input_mask] = 0
            else:
                masked_input = input_

            output_parallel = F.embedding(
                masked_input,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

            # Mask the output embedding.
            if self.tensor_model_parallel_size > 1:
                output_parallel[input_mask, :] = 0.0
        if self.reduce_results:
            # Reduce across all the model parallel GPUs.
            with self._communication_timer:
                output = reduce_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class ReplicatedLinear(torch.nn.Module):
    """Linear layer without TP sharding.

    The linear layer is defined as Y = XA + b. Weight and bias are replicated
    on every TP rank and no collective communication is used.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        bias=True,
        skip_bias_add=False,
        params_dtype=None,
        use_cpu_initialization=False,
        perform_initialization=False,
        linear_metric_name: Optional[str] = None,
        precision_op_name: Optional[str] = None,
        fp8_weight_block_size: Optional[Tuple[int, int]] = None,
        world_size: Optional[int] = None,
        layer_id: Optional[int] = None,
    ):
        super(ReplicatedLinear, self).__init__()
        assert not perform_initialization
        assert not use_cpu_initialization

        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        # Keep for API compatibility with TP linear layers.
        self.world_size = (
            get_tensor_model_parallel_world_size() if world_size is None else world_size
        )

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )

        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self._linear_timer = CudaTimer(linear_metric_name, layer_id=layer_id)
        self._precision_op_name = precision_op_name or linear_metric_name
        self._fp8_weight_block_size = fp8_weight_block_size
        if fp8_weight_block_size is None:
            self._fp8_block_size_list = None
        else:
            self._fp8_block_size_list = list(fp8_weight_block_size)
        self._fp8_weight: Optional[torch.Tensor] = None
        self._fp8_weight_scale: Optional[torch.Tensor] = None
        self._tensorwise_fp8_linear_op: Optional["Fp8LinearOp"] = None

    def _get_precision(self) -> Optional[PrecisionType]:
        if self._precision_op_name is None:
            return None
        quant_manager = get_quantization_manager()
        if not quant_manager.is_operation_supported(self._precision_op_name):
            return None
        return quant_manager.get_precision(self._precision_op_name)

    def _get_fp8_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fp8_weight is not None and self._fp8_weight_scale is not None:
            return self._fp8_weight, self._fp8_weight_scale

        fp8_weight, weight_scale = _quantize_fp8_weight(
            weight=self.weight.data,
            fp8_weight_block_size=self._fp8_weight_block_size,
            output_dtype=self.weight.dtype,
        )
        self._fp8_weight = fp8_weight
        self._fp8_weight_scale = weight_scale
        if (
            self._fp8_block_size_list is None
            and self._tensorwise_fp8_linear_op is None
        ):
            self._tensorwise_fp8_linear_op = _create_tensorwise_fp8_linear_op()
        return fp8_weight, weight_scale

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        precision = self._get_precision()
        if precision == PrecisionType.FP8:
            if _use_fp8_gemm_surrogate():
                with self._linear_timer:
                    return _run_unquantized_gemm(self, x, self.weight, bias)
            fp8_weight, weight_scale = self._get_fp8_weights()
            with self._linear_timer:
                return _apply_fp8_linear(
                    input_tensor=x,
                    fp8_weight=fp8_weight,
                    weight_scale=weight_scale,
                    fp8_weight_block_size=self._fp8_block_size_list,
                    output_dtype=self.weight.dtype,
                    bias=bias,
                    tensorwise_fp8_linear_op=self._tensorwise_fp8_linear_op,
                )
        with self._linear_timer:
            return _run_unquantized_gemm(self, x, self.weight, bias)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        output = self.apply_weights(input_, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        params_dtype=None,
        use_cpu_initialization=False,
        perform_initialization=False,
        linear_metric_name: Optional[str] = None,
        communication_metric_name: Optional[str] = None,
        precision_op_name: Optional[str] = None,
        fp8_weight_block_size: Optional[Tuple[int, int]] = None,
        world_size: Optional[int] = None,
        layer_id: Optional[int] = None,
    ):
        super(ColumnParallelLinear, self).__init__()
        assert not perform_initialization
        assert not use_cpu_initialization

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.world_size = (
            get_tensor_model_parallel_world_size() if world_size is None else world_size
        )
        self.output_size_per_partition = divide(output_size, self.world_size)
        self.skip_bias_add = skip_bias_add

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.create_weights(params_dtype)

        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self._linear_timer = CudaTimer(linear_metric_name, layer_id=layer_id)
        self._communication_timer = CudaTimer(
            communication_metric_name, layer_id=layer_id
        )
        self._precision_op_name = precision_op_name or linear_metric_name
        self._fp8_weight_block_size = fp8_weight_block_size
        if fp8_weight_block_size is None:
            self._fp8_block_size_list = None
        else:
            self._fp8_block_size_list = list(fp8_weight_block_size)
        self._fp8_weight: Optional[torch.Tensor] = None
        self._fp8_weight_scale: Optional[torch.Tensor] = None
        self._tensorwise_fp8_linear_op: Optional["Fp8LinearOp"] = None

    def create_weights(self, dtype: torch.dtype) -> None:
        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
        )

    def _get_precision(self) -> Optional[PrecisionType]:
        if self._precision_op_name is None:
            return None
        quant_manager = get_quantization_manager()
        if not quant_manager.is_operation_supported(self._precision_op_name):
            return None
        return quant_manager.get_precision(self._precision_op_name)

    def _get_fp8_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fp8_weight is not None and self._fp8_weight_scale is not None:
            return self._fp8_weight, self._fp8_weight_scale

        fp8_weight, weight_scale = _quantize_fp8_weight(
            weight=self.weight.data,
            fp8_weight_block_size=self._fp8_weight_block_size,
            output_dtype=self.weight.dtype,
        )
        self._fp8_weight = fp8_weight
        self._fp8_weight_scale = weight_scale
        if (
            self._fp8_block_size_list is None
            and self._tensorwise_fp8_linear_op is None
        ):
            self._tensorwise_fp8_linear_op = _create_tensorwise_fp8_linear_op()
        return fp8_weight, weight_scale

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        precision = self._get_precision()
        if precision == PrecisionType.FP8:
            if _use_fp8_gemm_surrogate():
                with self._linear_timer:
                    return _run_unquantized_gemm(self, x, self.weight, bias)
            fp8_weight, weight_scale = self._get_fp8_weights()
            with self._linear_timer:
                return _apply_fp8_linear(
                    input_tensor=x,
                    fp8_weight=fp8_weight,
                    weight_scale=weight_scale,
                    fp8_weight_block_size=self._fp8_block_size_list,
                    output_dtype=self.weight.dtype,
                    bias=bias,
                    tensorwise_fp8_linear_op=self._tensorwise_fp8_linear_op,
                )
        with self._linear_timer:
            return _run_unquantized_gemm(self, x, self.weight, bias)

    def forward(self, input_):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        input_parallel = input_
        # Matrix multiply.
        output_parallel = self.apply_weights(input_parallel, bias)
        if self.gather_output:
            # All-gather across the partitions.
            with self._communication_timer:
                output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
        reduce_results:
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        params_dtype=None,
        use_cpu_initialization=False,
        perform_initialization=False,
        reduce_results=True,
        linear_metric_name: Optional[str] = None,
        communication_metric_name: Optional[str] = None,
        precision_op_name: Optional[str] = None,
        fp8_weight_block_size: Optional[Tuple[int, int]] = None,
        world_size: Optional[int] = None,
        layer_id: Optional[int] = None,
    ):
        super(RowParallelLinear, self).__init__()
        assert not perform_initialization
        assert not use_cpu_initialization

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Divide the weight matrix along the last dimension.
        self.world_size = (
            get_tensor_model_parallel_world_size() if world_size is None else world_size
        )
        self.input_size_per_partition = divide(input_size, self.world_size)
        self.skip_bias_add = skip_bias_add

        self.create_weights(params_dtype)

        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self._linear_timer = CudaTimer(linear_metric_name, layer_id=layer_id)
        self._communication_timer = CudaTimer(
            communication_metric_name, layer_id=layer_id
        )
        self._precision_op_name = precision_op_name or linear_metric_name
        self._fp8_weight_block_size = fp8_weight_block_size
        if fp8_weight_block_size is None:
            self._fp8_block_size_list = None
        else:
            self._fp8_block_size_list = list(fp8_weight_block_size)
        self._fp8_weight: Optional[torch.Tensor] = None
        self._fp8_weight_scale: Optional[torch.Tensor] = None
        self._tensorwise_fp8_linear_op: Optional["Fp8LinearOp"] = None

    def create_weights(self, dtype: torch.dtype) -> None:
        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
        )

    def _get_precision(self) -> Optional[PrecisionType]:
        if self._precision_op_name is None:
            return None
        quant_manager = get_quantization_manager()
        if not quant_manager.is_operation_supported(self._precision_op_name):
            return None
        return quant_manager.get_precision(self._precision_op_name)

    def _get_fp8_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fp8_weight is not None and self._fp8_weight_scale is not None:
            return self._fp8_weight, self._fp8_weight_scale

        fp8_weight, weight_scale = _quantize_fp8_weight(
            weight=self.weight.data,
            fp8_weight_block_size=self._fp8_weight_block_size,
            output_dtype=self.weight.dtype,
        )
        self._fp8_weight = fp8_weight
        self._fp8_weight_scale = weight_scale
        if (
            self._fp8_block_size_list is None
            and self._tensorwise_fp8_linear_op is None
        ):
            self._tensorwise_fp8_linear_op = _create_tensorwise_fp8_linear_op()
        return fp8_weight, weight_scale

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        precision = self._get_precision()
        if precision == PrecisionType.FP8:
            if _use_fp8_gemm_surrogate():
                with self._linear_timer:
                    return _run_unquantized_gemm(self, x, self.weight, None)
            fp8_weight, weight_scale = self._get_fp8_weights()
            with self._linear_timer:
                return _apply_fp8_linear(
                    input_tensor=x,
                    fp8_weight=fp8_weight,
                    weight_scale=weight_scale,
                    fp8_weight_block_size=self._fp8_block_size_list,
                    output_dtype=self.weight.dtype,
                    bias=None,
                    tensorwise_fp8_linear_op=self._tensorwise_fp8_linear_op,
                )
        with self._linear_timer:
            return _run_unquantized_gemm(self, x, self.weight, None)

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = self.apply_weights(input_parallel)
        if self.reduce_results and self.world_size > 1:
            with self._communication_timer:
                output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        else:
            output_ = output_parallel

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
