"""Reject invalid complete GDN campaigns before native producer construction."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.gdn import main as campaign
from frontier.profiling.gdn import vllm_wrapper
from frontier.profiling.gdn.inputs import GDNProfileInput
from frontier.profiling.utils import build_profiling_output_path


MODEL_NAME = "Qwen3.8-2.4T-A95B-Quark-MXFP4"


class NativeConstructionReached(AssertionError):
    """Stop at the native boundary without importing or allocating GPU state."""


@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    model = ModelConfig.from_model_name(MODEL_NAME)
    assert model.get_num_gdn_layers() > 0
    assert model.get_gdn_config() is not None
    return model


@pytest.fixture
def campaign_boundary(monkeypatch, tmp_path: Path, model_config: ModelConfig):
    calls = []

    def forbidden_constructor(**kwargs):
        calls.append(kwargs)
        raise NativeConstructionReached("Native GDN producer construction reached")

    def resolve_model(name):
        assert name == MODEL_NAME
        return model_config

    monkeypatch.setattr(campaign.ModelConfig, "from_model_name", resolve_model)
    monkeypatch.setattr(vllm_wrapper, "VllmQwen35GDNWrapper", forbidden_constructor)
    output = Path(
        build_profiling_output_path(
            output_root=tmp_path,
            profiling_type="compute",
            hardware="mi355x",
            model_name=MODEL_NAME,
            op_name="gdn",
        )
    )
    output.parent.mkdir(parents=True)
    sentinel = b"existing profiling evidence must remain unchanged\n"
    output.write_bytes(sentinel)
    original_files = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    arguments = [
        "gdn-profile",
        "--model", MODEL_NAME,
        "--model-path", str(tmp_path / "checkpoint"),
        "--output-dir", str(tmp_path),
        "--prefill-seq-lens", "8",
        "--prefill-batch-sizes", "1",
        "--decode-batch-sizes", "1",
        "--decode-context-len", "16",
        "--continuation-context-len", "16",
        "--max-model-len", "128",
        "--max-batch-size", "4",
        "--warmup-iterations", "0",
        "--profile-iterations", "1",
    ]

    def run(extra_arguments):
        monkeypatch.setattr(sys, "argv", arguments + list(extra_arguments))
        try:
            campaign.main()
        finally:
            assert output.read_bytes() == sentinel
            assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == original_files

    return run, calls


@pytest.mark.parametrize(
    "extra_arguments",
    [
        pytest.param(["--include-mixed"], id="mixed-after-valid-workloads"),
        pytest.param(["--prefill-batch-sizes", "0"], id="zero-prefill-batch"),
        pytest.param(["--prefill-batch-sizes", "1", "5"], id="later-prefill-batch-overflow"),
        pytest.param(["--decode-batch-sizes", "1", "5"], id="later-decode-batch-overflow"),
        pytest.param(["--prefill-seq-lens", "8", "0"], id="later-zero-prefill-query"),
        pytest.param(["--prefill-seq-lens", "8", "129"], id="later-prefill-query-overflow"),
        pytest.param(["--max-batch-size", "0"], id="zero-batch-capacity"),
        pytest.param(["--max-model-len", "0"], id="zero-sequence-capacity"),
        pytest.param(["--tensor-parallel-size", "0"], id="zero-tensor-parallel-size"),
        pytest.param(["--decode-context-len", "0"], id="zero-decode-context"),
        pytest.param(["--decode-context-len", "-1"], id="negative-decode-context"),
        pytest.param(["--decode-context-len", "128"], id="decode-context-plus-query-overflow"),
        pytest.param(
            ["--include-continuation-prefill", "--continuation-context-len", "-1"],
            id="negative-continuation-context",
        ),
        pytest.param(
            ["--include-continuation-prefill", "--continuation-context-len", "121"],
            id="continuation-context-plus-query-overflow",
        ),
        pytest.param(["--warmup-iterations", "-1"], id="negative-warmup-count"),
        pytest.param(["--profile-iterations", "0"], id="zero-profile-count"),
        pytest.param(["--profile-iterations", "-1"], id="negative-profile-count"),
        *[
            pytest.param(["--profile-method", method], id=f"unsupported-method-{method}")
            for method in (
                "cuda_event", "cuda", "record_function", "kernel_only",
                "perf_counter", "kineto", "unknown",
            )
        ],
    ],
)
def test_invalid_campaign_rejects_before_native_construction(
    campaign_boundary, extra_arguments
):
    run, calls = campaign_boundary
    with pytest.raises((ValueError, SystemExit)):
        run(extra_arguments)
    assert calls == []


def test_physical_batch_overflow_rejects_before_native_construction(
    monkeypatch, campaign_boundary
):
    inputs = [
        GDNProfileInput.prefill(seq_len=8),
        GDNProfileInput(
            query_lens=(1,), context_lens=(16,), logical_phase="decode",
            physical_batch_size=5,
        ),
    ]
    monkeypatch.setattr(campaign, "build_profile_inputs", lambda args: inputs)
    run, calls = campaign_boundary
    with pytest.raises((ValueError, SystemExit)):
        run([])
    assert calls == []


@pytest.mark.parametrize(
    "extra_arguments",
    [
        pytest.param([], id="cold-prefill-and-decode"),
        pytest.param(
            ["--include-continuation-prefill", "--prefill-seq-lens", "1",
             "--continuation-context-len", "127", "--decode-context-len", "127",
             "--prefill-batch-sizes", "4", "--decode-batch-sizes", "4"],
            id="one-token-continuation-at-capacity",
        ),
    ],
)
def test_valid_campaign_reaches_native_boundary(campaign_boundary, extra_arguments):
    run, calls = campaign_boundary
    with pytest.raises(NativeConstructionReached):
        run(extra_arguments)
    assert len(calls) == 1
    assert isinstance(calls[0]["frontier_model_config"], ModelConfig)
    assert calls[0]["profile_method"] == "device_event"
