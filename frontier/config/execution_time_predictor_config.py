"""Execution-time predictor configuration and its calibration scales."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from frontier.config.base_poly_config import BasePolyConfig
from frontier.types import ExecutionTimePredictorType


@dataclass
class BaseExecutionTimePredictorConfig(BasePolyConfig):
    linear_op_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/linear_op.csv",
        metadata={"help": "Path to the linear operation profiling input file."},
    )
    # Backward compatibility alias
    mlp_input_file: str = field(
        default="",
        metadata={"help": "[DEPRECATED] Use linear_op_input_file instead."},
    )
    atten_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv",
        metadata={"help": "Path to the attention input file."},
    )
    gdn_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/gdn.csv",
        metadata={"help": "Path to the standard GDN profiling input file."},
    )
    all_reduce_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv",
        metadata={"help": "Path to the all reduce input file."},
    )
    send_recv_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/send_recv.csv",
        metadata={"help": "Path to the send recv input file."},
    )
    cpu_overhead_input_file: str = field(
        default="./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv",
        metadata={"help": "Path to the cpu overhead input file."},
    )
    cpu_overhead_kernel_only_input_file: str = field(
        default="./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads_kernel_only.csv",
        metadata={"help": "Path to the kernel-only cpu overhead input file."},
    )
    pp_stage_boundary_input_file: str = field(
        default="./data/profiling/other_overhead/{DEVICE}/{MODEL}/pp_stage_boundary.csv",
        metadata={"help": "Path to the pipeline stage-boundary overhead input file."},
    )
    pp_receiver_head_input_file: str = field(
        default="./data/profiling/other_overhead/{DEVICE}/{MODEL}/pp_receiver_head.csv",
        metadata={"help": "Path to the PP receiver-head overhead input file."},
    )
    pp_producer_send_path_input_file: str = field(
        default="./data/profiling/other_overhead/{DEVICE}/{MODEL}/pp_producer_send_path.csv",
        metadata={"help": "Path to the PP producer send-path overhead input file."},
    )
    pp_prefill_consumer_active_input_file: str = field(
        default="./data/profiling/other_overhead/{DEVICE}/{MODEL}/pp_prefill_consumer_active.csv",
        metadata={
            "help": "Path to the PP prefill consumer-active overhead input file."
        },
    )
    moe_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/moe.csv",
        metadata={"help": "Path to the MoE profiling input file."},
    )
    linear_op_kernel_only_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/linear_op_kernel_only.csv",
        metadata={"help": "Path to the kernel-only linear operation profiling input file."},
    )
    atten_kernel_only_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/attention_kernel_only.csv",
        metadata={"help": "Path to the kernel-only attention input file."},
    )
    moe_kernel_only_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/moe_kernel_only.csv",
        metadata={"help": "Path to the kernel-only MoE profiling input file."},
    )
    k_fold_cv_splits: int = field(
        default=10,
        metadata={"help": "Number of k fold cross validation splits."},
    )
    no_cache: bool = field(
        default=False,
        metadata={"help": "Whether to cache prediction models."},
    )
    kv_cache_prediction_granularity: int = field(
        default=64,
        metadata={"help": "KV cache prediction granularity."},
    )
    prediction_max_prefill_chunk_size: int = field(
        default=4096,
        metadata={"help": "Max prefill chunk size for prediction."},
    )
    prediction_max_batch_size: int = field(
        default=128,
        metadata={"help": "Max batch size for prediction."},
    )
    prediction_max_tokens_per_request: int = field(
        default=4096,
        metadata={"help": "Max tokens per request for prediction."},
    )
    attention_decode_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention decode batching overhead fraction."},
    )
    attention_prefill_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention prefill batching overhead fraction."},
    )
    attn_pre_proj_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for attn_pre_proj prediction. Must be > 0."
        },
    )
    prefill_phase_attn_pre_proj_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for attn_pre_proj "
                "prediction when the batch includes prefill tokens. Must be > 0."
            )
        },
    )
    attn_post_proj_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for attn_post_proj prediction. Must be > 0."
        },
    )
    prefill_phase_attn_post_proj_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for attn_post_proj "
                "prediction when the batch includes prefill tokens. Must be > 0."
            )
        },
    )
    attn_decode_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for attn_decode prediction. Must be > 0."
        },
    )
    attn_decode_in_mixed_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for attn_decode_in_mixed "
                "prediction when a co-location batch contains both prefill and decode "
                "tokens. Must be > 0."
            )
        },
    )
    late_decode_attn_decode_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for attn_decode "
                "prediction when every decode request in the batch has already "
                "completed the first pure decode token. Must be > 0."
            )
        },
    )
    attn_kv_cache_save_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for attn_kv_cache_save prediction. Must be > 0."
        },
    )
    prefill_phase_attn_kv_cache_save_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for attn_kv_cache_save "
                "prediction when the batch includes prefill tokens. Must be > 0."
            )
        },
    )
    mlp_up_proj_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for mlp_up_proj prediction. Must be > 0."
        },
    )
    prefill_phase_mlp_up_proj_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for mlp_up_proj "
                "prediction when the batch includes prefill tokens. Must be > 0."
            )
        },
    )
    mlp_down_proj_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for mlp_down_proj prediction. Must be > 0."
        },
    )
    decode_phase_mlp_down_proj_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for mlp_down_proj "
                "prediction when the batch contains decode tokens but no "
                "prefill tokens. Must be > 0."
            )
        },
    )
    nccl_cpu_launch_overhead_ms: float = field(
        default=0.02,
        metadata={"help": "NCCL CPU launch overhead in ms."},
    )
    nccl_cpu_skew_overhead_per_device_ms: float = field(
        default=0.0,
        metadata={"help": "NCCL CPU skew overhead per device in ms."},
    )
    num_training_job_threads: int = field(
        default=-1,
        metadata={"help": "Number of training job threads."},
    )
    skip_cpu_overhead_modeling: bool = field(
        default=True,
        metadata={"help": "Whether to skip CPU overhead modeling."},
    )

    # Dummy mode configuration for fast testing and development
    enable_dummy_mode: bool = field(
        default=False,
        metadata={
            "help": "Enable dummy mode to skip ML model training and return fixed execution times."
        },
    )
    dummy_execution_time_ms: float = field(
        default=1.0,
        metadata={
            "help": "Fixed execution time in milliseconds to return in dummy mode."
        },
    )

    def __post_init__(self) -> None:
        for field_name in (
            "attn_pre_proj_calibration_scale",
            "prefill_phase_attn_pre_proj_calibration_scale",
            "attn_post_proj_calibration_scale",
            "prefill_phase_attn_post_proj_calibration_scale",
            "attn_decode_calibration_scale",
            "attn_decode_in_mixed_calibration_scale",
            "late_decode_attn_decode_calibration_scale",
            "attn_kv_cache_save_calibration_scale",
            "prefill_phase_attn_kv_cache_save_calibration_scale",
            "mlp_up_proj_calibration_scale",
            "prefill_phase_mlp_up_proj_calibration_scale",
            "mlp_down_proj_calibration_scale",
            "decode_phase_mlp_down_proj_calibration_scale",
        ):
            raw_value = getattr(self, field_name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if value <= 0.0:
                raise ValueError(
                    f"{self.__class__.__name__}.{field_name} must be > 0, got={value!r}"
                )



@dataclass
class LinearRegressionExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    polynomial_degree: List[int] = field(
        default_factory=lambda: list(range(1, 6)),
        metadata={"help": "Polynomial degree for linear regression."},
    )
    polynomial_include_bias: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial include bias for linear regression."},
    )
    polynomial_interaction_only: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial interaction only for linear regression."},
    )
    fit_intercept: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Fit intercept for linear regression."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.LINEAR_REGRESSION


@dataclass
class RandomForrestExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    num_estimators: List[int] = field(
        default_factory=lambda: [250, 500, 750],
        metadata={"help": "Number of estimators for random forest."},
    )
    max_depth: List[int] = field(
        default_factory=lambda: [8, 16, 32],
        metadata={"help": "Maximum depth for random forest."},
    )
    min_samples_split: List[int] = field(
        default_factory=lambda: [2, 5, 10],
        metadata={"help": "Minimum samples split for random forest."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.RANDOM_FORREST
