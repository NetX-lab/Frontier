"""CPU contracts for the selected experimental SGLang integration.

These tests deliberately avoid importing or executing SGLang, AITER, CUDA, or
ROCm.  They verify metadata, planning, deterministic replay inputs, and trace
provenance only.
"""

from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path

import pytest

from frontier.profiling.common.model_config import ModelConfig


MODEL_NAME = "Qwen3.8-2.4T-A95B-Quark-MXFP4"


def test_package_is_lightweight_and_runtime_dependencies_are_lazy():
    package = importlib.import_module("frontier.profiling.experimental.sglang")
    assert package.__all__ == ()
    assert not any(name.startswith(("torch", "sglang", "aiter")) for name in package.__dict__)


def test_selected_shape_contracts_are_cpu_safe():
    model = ModelConfig.from_model_name(MODEL_NAME)
    dense = importlib.import_module("frontier.profiling.experimental.sglang.dense")
    gdn = importlib.import_module("frontier.profiling.experimental.sglang.gdn")
    attention = importlib.import_module("frontier.profiling.experimental.sglang.attention")
    moe = importlib.import_module("frontier.profiling.experimental.sglang.moe")

    dense_spec = dense.dense_primitive_spec("gdn_input_projections", model, 8)
    assert dense_spec["local_weight_shapes"] == ((4608, 8192), (32, 8192))
    dense.validate_dense_spec(dense_spec, 8, "gdn_input_projections")

    gdn_spec = gdn.gdn_core_spec(model, 8)
    assert gdn_spec["conv_state_shape"] == (2560, 3)
    assert gdn_spec["recurrent_state_shape"] == (16, 128, 128)
    gdn.validate_gdn_core_spec(gdn_spec, 8)

    attention_spec = attention.attention_decode_spec(model, 8)
    assert attention_spec["q_heads"] == 8
    assert attention_spec["kv_heads"] == 1
    attention.validate_attention_decode_spec(attention_spec, 8)

    routed = moe.moe_routed_spec(model, 8)
    assert routed["w13_packed_shape"] == (512, 512, 4096)
    assert routed["w13_scale_shape"] == (512, 512, 256)
    moe.validate_moe_routed_spec(routed, 8)


def test_invalid_tp_and_workloads_fail_fast():
    model = ModelConfig.from_model_name(MODEL_NAME)
    dense = importlib.import_module("frontier.profiling.experimental.sglang.dense")
    gdn = importlib.import_module("frontier.profiling.experimental.sglang.gdn")
    attention = importlib.import_module("frontier.profiling.experimental.sglang.attention")
    moe = importlib.import_module("frontier.profiling.experimental.sglang.moe")

    with pytest.raises(ValueError):
        dense.dense_primitive_spec("gdn_input_projections", model, 3)
    with pytest.raises(ValueError):
        gdn.gdn_core_spec(model, 3)
    with pytest.raises(ValueError):
        attention.attention_decode_spec(model, 3)
    with pytest.raises(ValueError):
        moe.moe_routed_spec(model, 3)
    spec = moe.moe_routed_spec(model, 8)
    with pytest.raises(ValueError, match="physical expert counts"):
        moe.validate_moe_route_workload(8, (80,) + (0,) * 511, spec)


def test_routing_replay_reuses_frontier_distribution_helper():
    replay = importlib.import_module(
        "frontier.profiling.experimental.sglang.routed_moe_replay"
    )
    expected = replay.frontier_routing_ratios(
        total_expert_num=8, distribution_type="random", seed=17, layer_id=3
    )
    from frontier.moe_ep_workload import generate_moe_routing_ratios

    assert expected == generate_moe_routing_ratios(
        total_expert_num=8, distribution_type="random", seed=17, layer_id=3
    )
    assignments = replay.reconstruct_topk_ids(
        8, (4,) * 20 + (0,) * 492, num_experts=512, top_k=10
    )
    assert len(assignments) == 8
    assert all(len(row) == 10 and len(set(row)) == 10 for row in assignments)


def test_explicit_route_json_and_visibility_plans(tmp_path):
    replay = importlib.import_module(
        "frontier.profiling.experimental.sglang.routed_moe_replay"
    )
    route_path = tmp_path / "routes.json"
    route_path.write_text(json.dumps({"num_experts": 4, "top_k": 2, "counts": [2, 2, 2, 2]}))
    route = replay.load_expert_count_json(route_path)
    assert route == {"num_experts": 4, "top_k": 2, "counts": (2, 2, 2, 2)}

    graph = importlib.import_module(
        "frontier.profiling.experimental.sglang.graph_replay"
    )
    plan = graph.build_replay_plan(
        primitive="attn_decode", sizes=(8,), invocations=(2,), repetitions=5,
        split="calibration", logical_sizes=(4,), context_lengths=(32,),
        environment={"ROCR_VISIBLE_DEVICES": "3", "HIP_VISIBLE_DEVICES": "3"},
        rank=0,
    )
    assert plan["visibility"] == {"ROCR_VISIBLE_DEVICES": "3", "HIP_VISIBLE_DEVICES": "3"}
    assert plan["local_rank"] == 0
    with pytest.raises(ValueError):
        graph.validate_replay_plan({**plan, "repetitions": 1})


def _event(name: str, ts: float, dur: float) -> dict[str, object]:
    return {"cat": "kernel", "ph": "X", "name": name, "ts": ts, "dur": dur}


def test_trace_json_and_gzip_fail_closed_on_anchor_and_layer_count(tmp_path):
    trace = {
        "traceEvents": [
            _event("_gemma_fused_add_rmsnorm_kernel", 0.0, 100.0),
            _event("projection_gemm", 100.0, 10.0),
            _event("elementwise_kernel_manual_unroll", 110.0, 3.0),
            _event("fused_recurrent_gated_delta_rule_packed_decode_kernel", 113.0, 5.0),
            _event("_layer_norm_fwd_1pass_kernel", 118.0, 7.0),
            _event("output_projection_gemm", 125.0, 8.0),
            _event("quickreduce::allreduce", 133.0, 20.0),
        ]
    }
    path = tmp_path / "trace_batch1_input8_output1_decode.trace.json"
    path.write_text(json.dumps(trace))
    gz_path = Path(str(path) + ".gz")
    with gzip.open(gz_path, "wt", encoding="utf-8") as stream:
        json.dump(trace, stream)

    importer = importlib.import_module(
        "frontier.profiling.experimental.sglang.gdn_trace"
    )
    assert importer.parse_sglang_trace_shape(gz_path)["phase"] == "decode"
    samples = importer.extract_sglang_gdn_samples(gz_path)
    assert samples["gdn_layer_e2e"] == [pytest.approx(0.033)]
    row = importer.build_sglang_profile_row_from_samples(
        {key: value * 69 for key, value in samples.items()},
        shape={"phase": "decode", "batch_size": 1, "input_len": 8, "output_len": 1},
        trace_path=str(gz_path), model_config=ModelConfig.from_model_name(MODEL_NAME),
        device="mi355x", tensor_parallel_size=8, runtime_stack_signature="test",
    )
    assert row["measurement_source"] == "sglang_kineto_trace"
    assert row["evidence_kind"] == "in_situ_kernel_sum"
    csv_path, json_path = importer.write_trace_summary([row], tmp_path / "summary")
    assert csv_path.name == "gdn-trace-summary.csv"
    assert json.loads(json_path.read_text())["experimental_trace"] is True
    with pytest.raises(ValueError, match="anchor"):
        importer.extract_sglang_gdn_event_samples(trace["traceEvents"][:-4], phase="decode")
    with pytest.raises(ValueError, match="positive multiple"):
        importer.build_sglang_profile_row_from_samples(
            samples,
            shape={"phase": "decode", "batch_size": 1, "input_len": 8, "output_len": 1},
            trace_path=str(gz_path), model_config=ModelConfig.from_model_name(MODEL_NAME),
            device="mi355x", tensor_parallel_size=8, runtime_stack_signature="test",
        )


def test_source_does_not_retain_discarded_runtime_cost_or_manifest_paths():
    root = Path("frontier/profiling/experimental/sglang")
    source = "\n".join(path.read_text() for path in root.glob("*.py"))
    for forbidden in (
        "RuntimeIdentity", "CostQuery", "SGLangCostContract", "ExactRuntimeCostTable",
        "capture_manifest", "frontier.runtime_cost", "frontier.validation.sglang_",
    ):
        assert forbidden not in source
