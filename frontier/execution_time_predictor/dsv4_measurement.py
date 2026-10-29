"""Measurement and host-time contract of the DeepSeek-V4 operator route."""

from frontier.attention.model_binding import resolve_runtime_attention_family
from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.types import MeasurementType


def dsv4_measurement_type(model_config):
    """V4 uses kernel-only tables in every role and runtime graph mode."""
    if resolve_runtime_attention_family(model_config) is DEEPSEEK_V4_ATTENTION_FAMILY:
        return MeasurementType.KERNEL_ONLY
    return None


def validate_dsv4_host_time(config, model_config):
    if dsv4_measurement_type(model_config) is not None and not config.skip_cpu_overhead_modeling:
        raise DeepSeekV4CoverageError(
            "V4 requires skip_cpu_overhead_modeling=True; host time is the explicit zero"
        )
