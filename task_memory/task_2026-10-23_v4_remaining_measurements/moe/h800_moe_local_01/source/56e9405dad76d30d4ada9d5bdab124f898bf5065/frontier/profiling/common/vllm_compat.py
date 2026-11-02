"""Small compatibility helpers for standalone use of current vLLM custom ops."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional


@contextmanager
def vllm_config_context(*, model_config: Optional[Any] = None):
    """Provide a standalone vLLM config only when none is active."""

    try:
        from vllm.config import (
            VllmConfig,
            get_current_vllm_config_or_none,
            set_current_vllm_config,
        )
    except ImportError:
        yield
        return

    if get_current_vllm_config_or_none() is not None:
        yield
        return

    standalone_config = VllmConfig()
    if model_config is not None:
        standalone_config.model_config = model_config
    with set_current_vllm_config(standalone_config):
        yield


def ensure_vllm_single_rank_parallel_state() -> None:
    """Join vLLM's process groups as the only rank of this process.

    vLLM layers and parameters read the tensor-parallel rank and size from
    vLLM's process groups when they are built, which a vLLM worker creates
    before it builds the model. A profiling process that builds vLLM layers
    outside an engine forms the same groups with one rank. The profiled
    tensor-parallel shard comes from the layer shapes, and Frontier's own
    simulated tensor-parallel state is separate from these groups.

    Call this outside any ``vllm_config_context``: the groups are laid out
    from a one-rank default config, not from the config of a profiled model.
    """

    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.platforms import current_platform

    try:
        # vLLM 0.11 and later moved the network helpers out of vllm.utils.
        from vllm.utils.network_utils import get_open_port
    except ImportError:
        # vLLM 0.10.x defines them in the package root.
        from vllm.utils import get_open_port

    with set_current_vllm_config(VllmConfig()):
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
                local_rank=(
                    torch.cuda.current_device()
                    if current_platform.is_cuda_alike()
                    else 0
                ),
                backend=current_platform.dist_backend,
            )
        ensure_model_parallel_initialized(1, 1)


__all__ = ["ensure_vllm_single_rank_parallel_state", "vllm_config_context"]
