"""Collected ROCm/gfx950 acceptance for the actual isolated vLLM GDN producer."""

import os
from pathlib import Path

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.gdn.inputs import GDNProfileInput
from frontier.profiling.gdn.vllm_wrapper import VllmQwen35GDNWrapper


@pytest.fixture(scope="module")
def native_gdn_wrapper():
    torch = pytest.importorskip("torch", reason="AMD GDN requires a ROCm PyTorch environment")
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("AMD/MI355X hardware and ROCm PyTorch unavailable")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if not properties.gcnArchName.startswith("gfx950"):
        pytest.skip("Pinned GDN producer requires AMD gfx950/MI355X hardware")
    pytest.importorskip("vllm", reason="Pinned vLLM GDN runtime unavailable")
    pytest.importorskip("triton", reason="ROCm Triton runtime unavailable")
    model_path = os.environ.get("FRONTIER_GDN_MODEL_PATH")
    if not model_path or not Path(model_path).is_dir():
        pytest.skip("Set FRONTIER_GDN_MODEL_PATH to an existing Qwen3.5 checkpoint directory")
    model = ModelConfig.from_model_name(os.environ.get(
        "FRONTIER_GDN_MODEL_NAME", "Qwen3.8-2.4T-A95B-Quark-MXFP4",
    ))
    with VllmQwen35GDNWrapper(
        frontier_model_config=model, model_path=model_path, device_name="mi355x",
        profile_method="device_event", max_model_len=128, max_batch_size=4,
        tensor_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ) as wrapper:
        yield wrapper


@pytest.mark.parametrize("workload", [
    GDNProfileInput.prefill(seq_len=16, batch_size=2),
    GDNProfileInput.prefill(seq_len=8, batch_size=2, context_len=16),
    GDNProfileInput.prefill(seq_len=1, batch_size=2, context_len=16),
    GDNProfileInput.decode(batch_size=2, context_len=16),
], ids=["cold-prefill", "continuation-prefill", "one-token-prefill", "decode"])
def test_native_gdn_decomposition_state_and_measurements(native_gdn_wrapper, workload):
    if workload.has_initial_state:
        native_gdn_wrapper.validate_prefix_continuation(workload)
    result = native_gdn_wrapper.profile(workload, warmup_iterations=1, profile_iterations=2)
    assert result["measurement_type"] == "DEVICE_EVENT"
    for name in ("gdn_input_projections", f"gdn_core_{workload.phase}",
                 "gdn_output_projection", "gdn_layer_e2e"):
        assert result["time_stats"][name]["count"] == 2
        assert result["time_stats"][name]["mean"] >= 0
