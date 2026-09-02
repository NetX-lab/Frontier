from __future__ import annotations

from pathlib import Path

import torch

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.moe import moe_wrapper as moe_wrapper_module
from frontier.profiling.moe.moe_wrapper import MoEWrapper


class _CaptureModule(torch.nn.Module):
    instances: list["_CaptureModule"] = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def forward(self, *args, **kwargs):
        del args, kwargs
        return torch.empty(0)


def _step3_like_config() -> ModelConfig:
    config = ModelConfig.from_model_name("step3-moe-noquant")
    # Keep the legacy scalar deliberately different so the producer must use
    # the profile-owned routed width instead of silently reusing it.
    config.mlp_hidden_dim = 9999
    return config


def test_moe_wrapper_uses_profile_owned_routed_width(monkeypatch, tmp_path: Path) -> None:
    """MoE producer construction must bind grouped GEMM to routed width."""

    _CaptureModule.instances = []
    monkeypatch.setattr(moe_wrapper_module, "MoEGatingNetwork", _CaptureModule)
    monkeypatch.setattr(moe_wrapper_module, "MoETokenShuffler", _CaptureModule)
    monkeypatch.setattr(moe_wrapper_module, "MoEGroupedGEMM", _CaptureModule)
    monkeypatch.setattr(
        MoEWrapper,
        "_init_gating_runtime_context_state",
        lambda self: None,
    )
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, device=None: self)

    MoEWrapper(
        model_config=_step3_like_config(),
        num_tensor_parallel_workers=1,
        expert_parallel_size=8,
        profile_method="cuda_event",
        rank=0,
        output_dir=str(tmp_path),
        use_vllm_kernel=False,
    )

    grouped_gemm = _CaptureModule.instances[2]
    shuffler = _CaptureModule.instances[1]
    assert grouped_gemm.kwargs["expert_hidden_dim"] == 5120
    assert shuffler.kwargs["expert_hidden_dim"] == 5120
