import gzip
import json

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.gdn.sglang_trace import (
    build_sglang_profile_row,
    extract_sglang_gdn_samples,
    parse_sglang_trace_shape,
)


def _event(name: str, ts: float, dur: float) -> dict:
    return {
        "cat": "kernel",
        "ph": "X",
        "name": name,
        "ts": ts,
        "dur": dur,
    }


def _write_trace(tmp_path, phase: str, repeats: int = 1):
    events = []
    ts = 0.0
    for _ in range(repeats):
        anchor = (
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
            if phase == "prefill"
            else "fused_recurrent_gated_delta_rule_packed_decode_kernel"
        )
        for name, dur in (
            ("_gemma_fused_add_rmsnorm_kernel", 100.0),
            ("projection_gemm", 10.0),
            ("projection_ba_gemm", 20.0),
            ("elementwise_kernel_manual_unroll", 3.0),
            ("_causal_conv1d_update_kernel", 4.0),
            (anchor, 5.0),
            ("core_output_copy", 6.0),
            ("_layer_norm_fwd_1pass_kernel", 7.0),
            ("output_projection_gemm", 8.0),
            ("quickreduce::allreduce", 200.0),
        ):
            events.append(_event(name, ts, dur))
            ts += dur
    path = tmp_path / f"profile_batch32_input1024_output64_{phase}.trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"traceEvents": events}, stream)
    return path


def test_extract_sglang_trace_excludes_input_norm_and_all_reduce(tmp_path) -> None:
    path = _write_trace(tmp_path, "decode", repeats=69)
    samples = extract_sglang_gdn_samples(path)

    assert len(samples["gdn_layer_e2e"]) == 69
    assert samples["gdn_input_projections"][0] == pytest.approx(0.030)
    assert samples["gdn_core_decode"][0] == pytest.approx(0.018)
    assert samples["gdn_output_projection"][0] == pytest.approx(0.015)
    assert samples["gdn_layer_e2e"][0] == pytest.approx(0.063)


def test_sglang_trace_shape_and_profile_contract(tmp_path) -> None:
    path = _write_trace(tmp_path, "prefill", repeats=69)
    assert parse_sglang_trace_shape(path) == {
        "batch_size": 32,
        "input_len": 1024,
        "output_len": 64,
        "phase": "prefill",
    }

    row = build_sglang_profile_row(
        path,
        model_config=ModelConfig.from_model_name(
            "Qwen3.8-2.4T-A95B-Quark-MXFP4"
        ),
        device="mi355x",
        tensor_parallel_size=8,
        runtime_stack_signature="sglang=test;torch=test;rocm=test;aiter=test",
    )
    assert row["measurement_type"] == "KERNEL_ONLY"
    assert row["gdn_runtime_backend"] == "sglang_rocm_aiter_trace"
    assert row["num_tensor_parallel_workers"] == 8
    assert row["batch_num_tokens"] == 32 * 1024
    assert row["batch_num_prefill_tokens"] == 32 * 1024
    assert row["batch_num_decode_tokens"] == 0
    assert row["trace_profiled_iterations"] == 1
    assert row["time_stats"]["gdn_core_decode"]["median"] == 0.0


def test_sglang_trace_rejects_partial_model_pass(tmp_path) -> None:
    path = _write_trace(tmp_path, "decode", repeats=68)
    with pytest.raises(ValueError, match="positive multiple"):
        build_sglang_profile_row(
            path,
            model_config=ModelConfig.from_model_name(
                "Qwen3.8-2.4T-A95B-Quark-MXFP4"
            ),
            device="mi355x",
            tensor_parallel_size=8,
            runtime_stack_signature="test",
        )
