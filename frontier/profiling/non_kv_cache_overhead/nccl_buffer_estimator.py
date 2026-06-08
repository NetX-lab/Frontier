"""Mechanism-based NCCL buffer estimation for non-torch memory overhead.

Models vLLM's actual non-torch allocation patterns (NCCL channel buffers,
communicator overhead, CustomAllreduce buffers) parameterized by tp_size
and NCCL configuration.  Single GPU, single process — no real NCCL init.

Empirical finding (A800, NCCL 2.x, vLLM v1):
  vLLM non_torch overhead for TP>1 is approximately constant (~0.72 GiB)
  regardless of tp_size.  NCCL pre-allocates a fixed buffer pool.
  The mechanism-based formula provides per-component breakdown, but the
  total is clamped to an empirical floor to match observed behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional, Tuple

_MiB = 1024 * 1024


@dataclass
class NCCLBufferEstimationConfig:
    """Configuration for mechanism-based NCCL buffer estimation.

    Defaults calibrated against A800 TP=8 empirical measurement (645 MiB).
    Override for different hardware/NCCL versions.

    Empirical finding: NCCL pre-allocates a fixed buffer pool for TP>1,
    so nccl_min_pool_bytes acts as a floor on the total estimate.
    """

    nccl_buffsize_bytes: int = 4 * _MiB
    nccl_channels_per_peer: int = 2
    nccl_max_channels: int = 32
    nccl_num_communicators: int = 2
    nccl_comm_base_overhead_bytes: int = 100 * _MiB
    nccl_per_peer_overhead_bytes: int = 15 * _MiB
    custom_ar_enabled: bool = False
    custom_ar_max_size_bytes: int = 8 * _MiB
    # Empirical floor: NCCL allocates at least this much for any TP>1.
    # Measured on A800 with NCCL 2.x: vLLM non_torch is ~0.72 GiB constant
    # across TP=2/4/8.  645 MiB is the original calibrated constant.
    nccl_min_pool_bytes: int = 645 * _MiB
    # Domain-specific vLLM worker extras. Defaults stay zero; callers must pass
    # validated case-local addends explicitly.
    vllm_worker_base_extra_bytes: int = 0
    pp_final_stage_extra_bytes: int = 0
    dp_communicator_extra_bytes: int = 0
    ep_all2all_extra_bytes: int = 0

    def cache_fingerprint(self) -> Tuple:
        """Return a deterministic tuple of all config fields for cache key use."""
        return (
            self.nccl_buffsize_bytes,
            self.nccl_channels_per_peer,
            self.nccl_max_channels,
            self.nccl_num_communicators,
            self.nccl_comm_base_overhead_bytes,
            self.nccl_per_peer_overhead_bytes,
            self.custom_ar_enabled,
            self.custom_ar_max_size_bytes,
            self.nccl_min_pool_bytes,
            self.vllm_worker_base_extra_bytes,
            self.pp_final_stage_extra_bytes,
            self.dp_communicator_extra_bytes,
            self.ep_all2all_extra_bytes,
        )


@dataclass(frozen=True)
class NCCLBufferEstimate:
    """Per-component breakdown of estimated NCCL non-torch memory overhead."""

    tp_size: int
    nccl_channel_bytes: int
    nccl_comm_overhead_bytes: int
    custom_ar_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class VLLMWorkerNonTorchEstimate:
    """Domain-aware vLLM worker non-torch memory estimate."""

    tp_size: int
    pp_size: int
    dp_size: int
    ep_size: int
    pipeline_stage_idx: int
    is_moe: bool
    tp_nccl_bytes: int
    nccl_channel_bytes: int
    nccl_comm_overhead_bytes: int
    custom_ar_bytes: int
    vllm_worker_base_extra_bytes: int
    pp_final_stage_extra_bytes: int
    dp_communicator_extra_bytes: int
    ep_all2all_extra_bytes: int
    total_bytes: int


def validate_nccl_buffer_config(config: NCCLBufferEstimationConfig) -> None:
    """Validate config fields.  Raises ValueError for invalid values."""
    if config.nccl_buffsize_bytes <= 0:
        raise ValueError(
            f"nccl_buffsize_bytes must be > 0, got={config.nccl_buffsize_bytes}"
        )
    if config.nccl_channels_per_peer <= 0:
        raise ValueError(
            f"nccl_channels_per_peer must be > 0, got={config.nccl_channels_per_peer}"
        )
    if config.nccl_max_channels <= 0:
        raise ValueError(
            f"nccl_max_channels must be > 0, got={config.nccl_max_channels}"
        )
    if config.nccl_num_communicators <= 0:
        raise ValueError(
            f"nccl_num_communicators must be > 0, got={config.nccl_num_communicators}"
        )
    if config.nccl_comm_base_overhead_bytes < 0:
        raise ValueError(
            f"nccl_comm_base_overhead_bytes must be >= 0, got={config.nccl_comm_base_overhead_bytes}"
        )
    if config.nccl_per_peer_overhead_bytes < 0:
        raise ValueError(
            f"nccl_per_peer_overhead_bytes must be >= 0, got={config.nccl_per_peer_overhead_bytes}"
        )
    if config.custom_ar_max_size_bytes < 0:
        raise ValueError(
            f"custom_ar_max_size_bytes must be >= 0, got={config.custom_ar_max_size_bytes}"
        )
    if config.nccl_min_pool_bytes < 0:
        raise ValueError(
            f"nccl_min_pool_bytes must be >= 0, got={config.nccl_min_pool_bytes}"
        )
    if config.vllm_worker_base_extra_bytes < 0:
        raise ValueError(
            "vllm_worker_base_extra_bytes must be >= 0, "
            f"got={config.vllm_worker_base_extra_bytes}"
        )
    if config.pp_final_stage_extra_bytes < 0:
        raise ValueError(
            "pp_final_stage_extra_bytes must be >= 0, "
            f"got={config.pp_final_stage_extra_bytes}"
        )
    if config.dp_communicator_extra_bytes < 0:
        raise ValueError(
            "dp_communicator_extra_bytes must be >= 0, "
            f"got={config.dp_communicator_extra_bytes}"
        )
    if config.ep_all2all_extra_bytes < 0:
        raise ValueError(
            "ep_all2all_extra_bytes must be >= 0, "
            f"got={config.ep_all2all_extra_bytes}"
        )


def _resolve_env_nccl_buffsize_bytes() -> Optional[int]:
    """Resolve NCCL_BUFFSIZE override from environment with strict validation."""
    env_buffsize = os.environ.get("NCCL_BUFFSIZE")
    if env_buffsize is None:
        return None

    try:
        value = int(env_buffsize)
    except ValueError as exc:
        raise ValueError(
            "NCCL_BUFFSIZE must be an integer number of bytes, "
            f"got={env_buffsize!r}"
        ) from exc

    if value <= 0:
        raise ValueError(f"NCCL_BUFFSIZE must be > 0, got={value!r}")

    return value


def get_effective_nccl_buffer_config(
    config: Optional[NCCLBufferEstimationConfig] = None,
) -> NCCLBufferEstimationConfig:
    """Return effective config after validation and env override application."""
    if config is None:
        config = NCCLBufferEstimationConfig()

    validate_nccl_buffer_config(config)

    env_override = _resolve_env_nccl_buffsize_bytes()
    if env_override is None:
        return config

    overridden_config = replace(config, nccl_buffsize_bytes=env_override)
    validate_nccl_buffer_config(overridden_config)
    return overridden_config


def estimate_nccl_non_torch_bytes(
    tp_size: int,
    config: Optional[NCCLBufferEstimationConfig] = None,
) -> NCCLBufferEstimate:
    """Estimate NCCL non-torch memory overhead for a given tp_size.

    Args:
        tp_size: Tensor parallel size (must be > 0).
        config: Optional estimation config.  Uses defaults if None.

    Returns:
        NCCLBufferEstimate with per-component breakdown.

    Raises:
        ValueError: If tp_size <= 0 or config has invalid fields.
    """
    if tp_size <= 0:
        raise ValueError(f"tp_size must be > 0, got={tp_size}")

    if tp_size == 1:
        return NCCLBufferEstimate(
            tp_size=1,
            nccl_channel_bytes=0,
            nccl_comm_overhead_bytes=0,
            custom_ar_bytes=0,
            total_bytes=0,
        )

    config = get_effective_nccl_buffer_config(config)

    num_peers = tp_size - 1
    effective_channels = min(
        num_peers * config.nccl_channels_per_peer,
        config.nccl_max_channels,
    )

    nccl_channel_bytes = (
        effective_channels
        * config.nccl_buffsize_bytes
        * 2  # send + recv
        * config.nccl_num_communicators
    )

    nccl_comm_overhead_bytes = (
        config.nccl_comm_base_overhead_bytes
        + config.nccl_per_peer_overhead_bytes * num_peers
    ) * config.nccl_num_communicators

    if config.custom_ar_enabled:
        custom_ar_bytes = config.custom_ar_max_size_bytes * 2 + 8 * _MiB
    else:
        custom_ar_bytes = 0

    total_bytes = nccl_channel_bytes + nccl_comm_overhead_bytes + custom_ar_bytes

    # Apply empirical floor: NCCL pre-allocates a fixed buffer pool for TP>1.
    # The mechanism-based formula may underestimate for small TP sizes because
    # NCCL allocates the same pool regardless of peer count.
    total_bytes = max(total_bytes, config.nccl_min_pool_bytes)

    return NCCLBufferEstimate(
        tp_size=tp_size,
        nccl_channel_bytes=nccl_channel_bytes,
        nccl_comm_overhead_bytes=nccl_comm_overhead_bytes,
        custom_ar_bytes=custom_ar_bytes,
        total_bytes=total_bytes,
    )


def estimate_vllm_worker_non_torch_bytes(
    *,
    tp_size: int,
    pp_size: int,
    dp_size: int,
    ep_size: int,
    pipeline_stage_idx: int,
    is_moe: bool,
    config: Optional[NCCLBufferEstimationConfig] = None,
) -> VLLMWorkerNonTorchEstimate:
    """Estimate vLLM worker-domain non-torch bytes for one rank/stage."""
    for name, value in {
        "tp_size": tp_size,
        "pp_size": pp_size,
        "dp_size": dp_size,
        "ep_size": ep_size,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be > 0, got={value!r}")

    if int(pipeline_stage_idx) < 0 or int(pipeline_stage_idx) >= int(pp_size):
        raise ValueError(
            "pipeline_stage_idx must satisfy 0 <= stage_idx < pp_size, "
            f"got stage_idx={pipeline_stage_idx}, pp_size={pp_size}"
        )

    effective_config = get_effective_nccl_buffer_config(config)
    tp_estimate = estimate_nccl_non_torch_bytes(
        int(tp_size),
        config=effective_config,
    )

    distributed_domain_active = (
        int(tp_size) > 1
        or int(pp_size) > 1
        or int(dp_size) > 1
        or int(ep_size) > 1
    )
    worker_base_extra = (
        int(effective_config.vllm_worker_base_extra_bytes)
        if distributed_domain_active
        else 0
    )
    pp_final_extra = (
        int(effective_config.pp_final_stage_extra_bytes)
        if int(pp_size) > 1 and int(pipeline_stage_idx) == int(pp_size) - 1
        else 0
    )
    dp_extra = (
        int(effective_config.dp_communicator_extra_bytes)
        if int(dp_size) > 1
        else 0
    )
    ep_all2all_extra = (
        int(effective_config.ep_all2all_extra_bytes)
        if bool(is_moe) and int(dp_size) > 1 and int(ep_size) > 1
        else 0
    )

    total_bytes = (
        int(tp_estimate.total_bytes)
        + worker_base_extra
        + pp_final_extra
        + dp_extra
        + ep_all2all_extra
    )

    return VLLMWorkerNonTorchEstimate(
        tp_size=int(tp_size),
        pp_size=int(pp_size),
        dp_size=int(dp_size),
        ep_size=int(ep_size),
        pipeline_stage_idx=int(pipeline_stage_idx),
        is_moe=bool(is_moe),
        tp_nccl_bytes=int(tp_estimate.total_bytes),
        nccl_channel_bytes=int(tp_estimate.nccl_channel_bytes),
        nccl_comm_overhead_bytes=int(tp_estimate.nccl_comm_overhead_bytes),
        custom_ar_bytes=int(tp_estimate.custom_ar_bytes),
        vllm_worker_base_extra_bytes=worker_base_extra,
        pp_final_stage_extra_bytes=pp_final_extra,
        dp_communicator_extra_bytes=dp_extra,
        ep_all2all_extra_bytes=ep_all2all_extra,
        total_bytes=int(total_bytes),
    )
