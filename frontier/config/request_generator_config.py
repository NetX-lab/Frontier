"""Workload generation configuration: arrival intervals, lengths, generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from frontier.config.base_poly_config import BasePolyConfig
from frontier.types import (
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
)


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens."},
    )


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace request interval generator file."},
    )
    start_time: str = field(
        default="1970-01-04 12:00:00",
        metadata={"help": "Start time of the trace request interval generator."},
    )
    end_time: str = field(
        default="1970-01-04 15:00:00",
        metadata={"help": "End time of the trace request interval generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={
            "help": "Time scale factor for the trace request interval generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.5,
        metadata={"help": "Queries per second for Poisson Request Interval Generator."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.2,
        metadata={"help": "Queries per second for Gamma Request Interval Generator."},
    )
    cv: float = field(
        default=0.5,
        metadata={
            "help": "Coefficient of variation for Gamma Request Interval Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace request length generator file."},
    )
    prefill_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Prefill scale factor for the trace request length generator."
        },
    )
    decode_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Decode scale factor for the trace request length generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = field(
        default=0.6,
        metadata={"help": "Theta for Zipf Request Length Generator."},
    )
    scramble: bool = field(
        default=False,
        metadata={"help": "Scramble for Zipf Request Length Generator."},
    )
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Zipf Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={"help": "Prefill to decode ratio for Zipf Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Uniform Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={
            "help": "Prefill to decode ratio for Uniform Request Length Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=2048,
        metadata={"help": "Prefill tokens for Fixed Request Length Generator."},
    )
    decode_tokens: int = field(
        default=512,
        metadata={"help": "Decode tokens for Fixed Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED

    def __post_init__(self):
        if self.decode_tokens < 1:
            raise ValueError(f"decode_tokens must be >= 1, got {self.decode_tokens}")
        if self.prefill_tokens < 2:
            raise ValueError(f"prefill_tokens must be >1, got {self.prefill_tokens}")


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    num_decode_bound_requests: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of generated requests that require decode-cluster work. "
            "Derived by request generation and used by offline pd-disaggregation scheduling."
        },
    )


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig,
        metadata={"help": "Length generator config for Synthetic Request Generator."},
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig,
        metadata={"help": "Interval generator config for Synthetic Request Generator."},
    )
    num_requests: Optional[int] = field(
        default=128,
        metadata={"help": "Number of requests for Synthetic Request Generator."},
    )
    duration: Optional[float] = field(
        default=None,
        metadata={"help": "Duration of the synthetic request generator."},
    )
    default_priority: int = field(
        default=0,
        metadata={
            "help": "Default priority for all generated requests. "
            "Lower value = higher priority (0 = highest). "
            "Matches vLLM v1 semantics."
        },
    )

    def __post_init__(self):
        self.max_tokens = self.length_generator_config.max_tokens

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/splitwise_conv.csv",
        metadata={"help": "Path to the trace request generator file."},
    )
    prefill_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Prefill scale factor for the trace request generator."},
    )
    decode_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Decode scale factor for the trace request generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Time scale factor for the trace request generator."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens for the trace request generator."},
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE_REPLAY
