"""V4 G5h call-list and literal-byte checks against design.md §3-4."""

from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import pytest

from frontier.profiling.collectives.vllm_dsv4_forward_collectives import (
    DeepseekV4Shape,
    dsv4_forward_collectives,
    dsv4_plan,
)
from frontier.profiling.collectives.vllm_forward_collectives import (
    ForwardScenario,
    ServingParallelism,
    call_bytes,
)

V4_CONFIG = {
    "hidden_size": 4096, "num_hidden_layers": 43, "n_routed_experts": 256,
    "num_experts_per_tok": 6, "vocab_size": 129280, "hc_mult": 4,
}
ENGINE = {
    "tensor_parallel_size": 8, "data_parallel_size": 2, "pipeline_parallel_size": 1,
    "enable_expert_parallel": True, "dtype": "bfloat16", "async_scheduling": True,
}
GRAPH = [1, 2, 4, 8, 16, 24, 32]
GRID = {"graph_tokens_per_lane": GRAPH, "eager_lane_tokens": [6144, 1],
        "logits_rows_per_lane": list(range(1, 25))}
CALLS = ["embedding_all_reduce", "sp_attention_all_gather", "sp_attention_reduce_scatter",
         "sp_final_mhc_all_gather", "moe_dispatch_all_gatherv", "moe_combine_reduce_scatterv",
         "logits_all_gather"]
EP_CALLS = CALLS[4:6]
LAUNCH = (Path(__file__).resolve().parents[2] /
          "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch")


def _forward(lanes=(8, 8), logits=None, graph=False):
    parallel = ServingParallelism.from_engine({**ENGINE, "data_parallel_size": len(lanes)})
    scenario = ForwardScenario("s", lanes, logits or (0,) * len(lanes), graph)
    calls = dsv4_forward_collectives(DeepseekV4Shape.from_hf_config(V4_CONFIG), parallel, scenario)
    return parallel, {call.name: call for call in calls}


def test_call_list_order_counts_and_capture():
    parallel, calls = _forward(logits=(8, 8), graph=True)
    assert list(calls) == CALLS
    assert [call.calls_per_forward for call in calls.values()] == [1, 43, 43, 1, 43, 43, 1]
    assert [(call.op, call.group, call.dim) for call in calls.values()] == [
        ("all_reduce", "tp", 0), ("sp_all_gather", "tp", 0),
        ("sp_reduce_scatter", "tp", 0), ("sp_all_gather", "tp", 0),
        ("all_gatherv", "ep", 0), ("reduce_scatterv", "ep", 0), ("all_gather", "tp", -1)]
    assert {call.name for call in calls.values() if not call.captured} == {"logits_all_gather"}
    assert not any(call.group == "dp" for call in calls.values())
    assert calls["moe_dispatch_all_gatherv"].columns == (4096, 6, 6)
    assert calls["moe_dispatch_all_gatherv"].dtypes == ("bfloat16", "float32", "int32")
    assert calls["sp_final_mhc_all_gather"].columns == (16384,)
    assert parallel.expert_parallel_size == 16
    _, eager = _forward()
    assert list(eager) == CALLS[:-1] and not any(call.captured for call in eager.values())


@pytest.mark.parametrize("tokens, embedding, gather, mhc", [
    (1, 8_192, 8_192, 32_768), (2, 16_384, 8_192, 32_768),
    (4, 32_768, 8_192, 32_768), (8, 65_536, 8_192, 32_768),
    (16, 131_072, 16_384, 65_536), (24, 196_608, 24_576, 98_304),
    (32, 262_144, 32_768, 131_072),
])
@pytest.mark.parametrize("dp", [2, 4])
def test_graph_literal_bytes(tokens, embedding, gather, mhc, dp):
    parallel, calls = _forward((tokens,) * dp, graph=True)
    r = (tokens + 7) // 8
    for rank in (0, 8):
        assert call_bytes(calls["embedding_all_reduce"], parallel, rank) == (embedding, embedding)
        assert call_bytes(calls["sp_attention_all_gather"], parallel, rank) == (gather, gather * 8)
        assert call_bytes(calls["sp_attention_reduce_scatter"], parallel, rank) == (embedding, gather)
        padded = call_bytes(calls["sp_attention_reduce_scatter"], parallel, rank)[1] * 8
        assert padded == 65_536 * r
        assert call_bytes(calls["sp_final_mhc_all_gather"], parallel, rank) == (mhc, mhc * 8)
        assert call_bytes(calls["moe_dispatch_all_gatherv"], parallel, rank) == (
            r * 8_240, r * 8_240 * dp * 8)
        assert call_bytes(calls["moe_combine_reduce_scatterv"], parallel, rank) == (
            (131_072 if dp == 2 else 262_144) * r, r * 8_192)


@pytest.mark.parametrize("lanes, combine", [
    ((6144, 6144), 100_663_296), ((6144, 1), 50_397_184),
    ((1, 6144), 50_397_184), ((1, 1), 131_072),
    ((6144, 6144, 6144, 1), 151_060_480),
])
def test_eager_literal_bytes(lanes, combine):
    parallel, calls = _forward(lanes)
    for rank in (0, 8):
        active = lanes[rank // 8] == 6144
        r = 768 if active else 1
        embedding = 50_331_648 if active else 8_192
        gather = 6_291_456 if active else 8_192
        mhc = 25_165_824 if active else 32_768
        assert call_bytes(calls["embedding_all_reduce"], parallel, rank) == (embedding, embedding)
        assert call_bytes(calls["sp_attention_all_gather"], parallel, rank) == (gather, 8 * gather)
        assert call_bytes(calls["sp_attention_reduce_scatter"], parallel, rank) == (embedding, gather)
        assert call_bytes(calls["sp_final_mhc_all_gather"], parallel, rank) == (mhc, 8 * mhc)
        assert call_bytes(calls["moe_dispatch_all_gatherv"], parallel, rank)[0] == r * 8_240
        assert call_bytes(calls["moe_combine_reduce_scatterv"], parallel, rank) == (
            combine, r * 8_192)
    if lanes == (6144, 1):
        dispatch = calls["moe_dispatch_all_gatherv"]
        assert dispatch.shard_rows == (768,) * 8 + (1,) * 8
        assert call_bytes(dispatch, parallel, 0) == (6_328_320, 50_692_480)
        assert call_bytes(dispatch, parallel, 8) == (8_240, 50_692_480)
        assert calls["sp_attention_reduce_scatter"].shard_rows == (6144,) * 8 + (1,) * 8


@pytest.mark.parametrize("rows", range(1, 25))
def test_logits_literal_bytes(rows):
    parallel, calls = _forward((rows, rows), (rows, rows))
    for rank in (0, 8):
        assert call_bytes(calls["logits_all_gather"], parallel, rank) == (
            32_320 * rows, 258_560 * rows)


@pytest.mark.parametrize("dp, count", [(2, 82), (4, 46)])
def test_stage_case_counts_and_selection(dp, count):
    config = json.loads((LAUNCH / f"stage_g5h{dp*8}_tp8_dp{dp}_ep{dp*8}.json").read_text())
    parallel = ServingParallelism.from_engine(config["engine"])
    assert config["engine"]["data_parallel_size"] == dp
    plan = dsv4_plan(DeepseekV4Shape.from_hf_config(V4_CONFIG), parallel, config["grid"])
    assert len(plan) == count
    assert config["grid"]["graph_tokens_per_lane"] == GRAPH
    assert config["grid"]["eager_lane_tokens"] == [6144, 1]
    assert config["grid"]["logits_rows_per_lane"] == (list(range(1, 25)) if dp == 2 else [])
    eager = {scenario.name: scenario.tokens_per_dp_rank for scenario, _ in plan
             if scenario.name.startswith("eager_")}
    assert list(eager.values()) == list(product([6144, 1], repeat=dp))
    for scenario, call in plan:
        if len(set(scenario.tokens_per_dp_rank)) > 1 or dp == 4:
            assert call.group == "ep" and call.name in EP_CALLS
        if scenario.name.startswith("logits_"):
            assert call.name == "logits_all_gather" and not call.captured
        if scenario.cuda_graph:
            assert not any(scenario.logit_rows_per_dp_rank) and call.captured
    if dp == 2:
        assert list(dict.fromkeys(s.name for s, _ in plan)) == (
            [f"graph_t{t}" for t in GRAPH] + list(eager) + [f"logits_b{b}" for b in range(1, 25)])


def test_optional_call_filter_applies_to_all_scenarios():
    parallel = ServingParallelism.from_engine(ENGINE)
    plan = dsv4_plan(DeepseekV4Shape.from_hf_config(V4_CONFIG), parallel,
                     {**GRID, "calls": ["moe_dispatch_all_gatherv", "logits_all_gather"]})
    assert len(plan) == 35
    assert {c.name for _, c in plan} == {"moe_dispatch_all_gatherv", "logits_all_gather"}
    assert not dsv4_plan(DeepseekV4Shape.from_hf_config(V4_CONFIG), parallel, {**GRID, "calls": []})


def test_shape_config_and_vocab_padding():
    model = DeepseekV4Shape.from_hf_config({**V4_CONFIG, "hidden_size": 16,
                                          "num_hidden_layers": 2, "hc_mult": 3, "vocab_size": 65})
    parallel = ServingParallelism.from_engine({**ENGINE, "tensor_parallel_size": 2})
    calls = dsv4_forward_collectives(model, parallel, ForwardScenario("s", (3, 1), (1, 0), False))
    assert calls[1].calls_per_forward == 2 and calls[3].columns == (48,)
    assert calls[-1].columns == (64,) and calls[-1].shard_rows == (1, 1, 0, 0)
    with pytest.raises(ValueError, match="divisible"):
        dsv4_forward_collectives(model, replace(parallel, tensor_parallel_size=3),
                                 ForwardScenario("s", (3, 1), (1, 0), False))


def test_rejects_missing_experts_and_unknown_selection():
    with pytest.raises(ValueError, match="no routed experts"):
        DeepseekV4Shape.from_hf_config({k: v for k, v in V4_CONFIG.items() if k != "n_routed_experts"})
    parallel = ServingParallelism.from_engine(ENGINE)
    model = DeepseekV4Shape.from_hf_config(V4_CONFIG)
    for grid, message in [({"unknown": []}, "Unknown grid keys"),
                          ({"calls": ["dp_batch_sync_all_reduce"]}, "Unknown calls")]:
        with pytest.raises(ValueError, match=message):
            dsv4_plan(model, parallel, grid)


@pytest.mark.parametrize("scenario, message", [
    (ForwardScenario("s", (4, 1), (0, 0), True), "DP-padded"),
    (ForwardScenario("s", (1,), (0, 0), False), "one token and logits"),
    (ForwardScenario("s", (1, 1), (0,), False), "one token and logits"),
    (ForwardScenario("s", (0, 1), (0, 0), False), "at least one token"),
])
def test_scenario_rejections(scenario, message):
    with pytest.raises(ValueError, match=message):
        dsv4_forward_collectives(DeepseekV4Shape.from_hf_config(V4_CONFIG),
                                 ServingParallelism.from_engine(ENGINE), scenario)


@pytest.mark.parametrize("override", [
    {"pipeline_parallel_size": 2}, {"tensor_parallel_size": 1},
    {"data_parallel_size": 1}, {"enable_expert_parallel": False},
    {"all2all_backend": "deepep_low_latency"},
])
def test_topology_rejections(override):
    parallel = ServingParallelism.from_engine({**ENGINE, **override})
    scenario = ForwardScenario("s", (1,) * parallel.data_parallel_size,
                               (0,) * parallel.data_parallel_size, False)
    with pytest.raises(ValueError):
        dsv4_forward_collectives(DeepseekV4Shape.from_hf_config(V4_CONFIG), parallel, scenario)
