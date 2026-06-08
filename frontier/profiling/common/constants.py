"""File to store names for different metrics captured during profiling."""

import enum


class OperationMetrics(enum.Enum):
    """Enum for operation metrics used in profiling."""
    
    # MLP operations
    MLP_UP_PROJ = "mlp_up_proj"
    MLP_UP_PROJ_ALL_GATHER = "mlp_up_proj_all_gather"
    MLP_ACTIVATION = "mlp_activation"
    MLP_DOWN_PROJ = "mlp_down_proj"
    MLP_DOWN_PROJ_ALL_REDUCE = "mlp_down_proj_all_reduce"
    
    # Attention operations
    ATTN_PRE_PROJ = "attn_pre_proj"
    ATTN_PRE_PROJ_ALL_GATHER = "attn_pre_proj_all_gather"
    ATTN_POST_PROJ = "attn_post_proj"
    ATTN_POST_PROJ_ALL_REDUCE = "attn_post_proj_all_reduce"
    ATTN_KV_CACHE_SAVE = "attn_kv_cache_save"
    ATTN = "attn"
    ATTN_PREFILL = "attn_prefill"
    ATTN_DECODE = "attn_decode"
    ATTN_ROPE = "attn_rope"
    ATTN_INPUT_RESHAPE = "attn_input_reshape"
    ATTN_OUTPUT_RESHAPE = "attn_output_reshape"
    
    # Embedding operations
    EMBED_LINEAR = "embed_linear"
    EMBED_ALL_REDUCE = "embed_all_reduce"
    MTP_FUSION_PROJ = "mtp_fusion_proj"
    LM_HEAD_LINEAR = "lm_head_linear"
    LM_HEAD_ALL_GATHER = "lm_head_all_gather"
    
    # Normalization operations
    INPUT_LAYERNORM = "input_layernorm"
    POST_ATTENTION_LAYERNORM = "post_attention_layernorm"
    NORM = "norm"
    
    # Other operations
    ADD = "add"
    NCCL_SEND = "nccl_send"
    NCCL_RECV = "nccl_recv"


class CpuOperationMetrics(enum.Enum):
    """Enum for CPU operation metrics used in profiling."""
    
    SCHEDULE = "schedule"
    SAMPLER_E2E = "sample_e2e"
    PREPARE_INPUTS_E2E = "prepare_inputs_e2e"
    MODEL_EXECUTION_E2E = "model_execution_e2e"
    PROCESS_MODEL_OUTPUTS = "process_model_outputs"
