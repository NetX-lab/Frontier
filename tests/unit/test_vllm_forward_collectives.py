"""Unit checks for the vLLM v0.28 forward-pass collective shapes."""

import pytest

from frontier.profiling.collectives.vllm_forward_collectives import (
    ForwardScenario,
    ModelShape,
    ServingParallelism,
    call_bytes,
    forward_collectives,
    scenarios_from_grid,
)

GLM_CONFIG = {
    "hidden_size": 6144,
    "num_hidden_layers": 78,
    "first_k_dense_replace": 3,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "vocab_size": 154880,
}

ENGINE = {
    "tensor_parallel_size": 8,
    "data_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "enable_expert_parallel": True,
    "dtype": "bfloat16",
    "async_scheduling": True,
}


def _calls(engine=ENGINE, config=GLM_CONFIG, scenario=None):
    parallel = ServingParallelism.from_engine(engine)
    scenario = scenario or ForwardScenario("s", (100, 1), (3, 0), False)
    calls = forward_collectives(ModelShape.from_hf_config(config), parallel, scenario)
    return parallel, {call.name: call for call in calls}


def test_call_counts_follow_layer_split():
    _, calls = _calls()
    assert calls["dense_layer_all_reduce"].calls_per_forward == 6
    assert calls["moe_attention_reduce_scatter"].calls_per_forward == 75
    assert calls["moe_layer_input_all_gather"].calls_per_forward == 74
    assert calls["final_hidden_residual_all_gather"].calls_per_forward == 1
    assert calls["moe_dispatch_all_gatherv"].calls_per_forward == 75
    assert calls["moe_combine_reduce_scatterv"].calls_per_forward == 75


def test_sequence_parallel_rows_and_bytes():
    parallel, calls = _calls()
    dispatch = calls["moe_dispatch_all_gatherv"]
    # ceil(100/8)=13 rows on DP0 ranks, ceil(1/8)=1 on DP1 ranks.
    assert dispatch.shard_rows == (13,) * 8 + (1,) * 8
    row = 6144 * 2 + 8 * 4 + 8 * 4
    assert call_bytes(dispatch, parallel, 0) == (13 * row, (13 * 8 + 8) * row)
    combine = calls["moe_combine_reduce_scatterv"]
    assert call_bytes(combine, parallel, 9) == ((13 * 8 + 8) * 6144 * 2, 6144 * 2)
    reduce_scatter = calls["moe_attention_reduce_scatter"]
    assert call_bytes(reduce_scatter, parallel, 0) == (104 * 6144 * 2, 13 * 6144 * 2)
    assert calls["embedding_all_reduce"].shard_rows[0] == 100
    assert call_bytes(calls["final_hidden_residual_all_gather"], parallel, 0)[1] == (
        104 * 2 * 6144 * 2)


def test_logits_and_dp_sync():
    parallel, calls = _calls()
    logits = calls["logits_all_gather"]
    assert logits.dim == -1 and logits.columns == (154880 // 8,)
    assert logits.shard_rows == (3,) * 8 + (0,) * 8
    sync = calls["dp_batch_sync_all_reduce"]
    assert sync.op == "cpu_all_reduce" and sync.columns == (2,)
    _, eager_sync = _calls(engine={**ENGINE, "async_scheduling": False})
    assert eager_sync["dp_batch_sync_all_reduce"].op == "device_group_all_reduce"


def test_shapes_follow_config_and_parallelism():
    config = {**GLM_CONFIG, "hidden_size": 4096, "first_k_dense_replace": 1,
              "num_experts_per_tok": 4}
    engine = {**ENGINE, "data_parallel_size": 4}
    scenario = ForwardScenario("s", (8, 8, 8, 8), (8, 8, 8, 8), True)
    parallel, calls = _calls(engine, config, scenario)
    assert parallel.expert_parallel_size == 32
    assert calls["dense_layer_all_reduce"].calls_per_forward == 2
    dispatch = calls["moe_dispatch_all_gatherv"]
    assert dispatch.columns == (4096, 4, 4)
    assert len(dispatch.shard_rows) == 32
    assert call_bytes(dispatch, parallel, 31)[1] == 32 * (4096 * 2 + 32)


@pytest.mark.parametrize("override, message", [
    ({"data_parallel_size": 1}, "Sequence-parallel MoE"),
    ({"enable_expert_parallel": False}, "Sequence-parallel MoE"),
    ({"pipeline_parallel_size": 2}, "Sequence-parallel MoE"),
    ({"all2all_backend": "deepep_low_latency"}, "not modeled"),
])
def test_rejects_topologies_outside_the_modeled_path(override, message):
    engine = {**ENGINE, **override}
    scenario = ForwardScenario("s", (4,) * engine["data_parallel_size"],
                               (1,) * engine["data_parallel_size"], False)
    with pytest.raises(ValueError, match=message):
        _calls(engine, scenario=scenario)


def test_only_the_model_forward_is_captured():
    _, calls = _calls(scenario=ForwardScenario("s", (4, 4), (4, 4), True))
    assert {name for name, call in calls.items() if not call.captured} == {
        "logits_all_gather", "dp_batch_sync_all_reduce"}
    _, eager = _calls()
    assert not any(call.captured for call in eager.values())


def test_graph_steps_must_be_dp_padded():
    with pytest.raises(ValueError, match="DP-padded"):
        _calls(scenario=ForwardScenario("s", (4, 1), (4, 1), True))


def test_grid_builds_uniform_single_active_and_graph_steps():
    parallel = ServingParallelism.from_engine(ENGINE)
    scenarios = scenarios_from_grid(parallel, {"eager": [[512, 2]],
                                               "cuda_graph_batch_sizes": [16]})
    assert [(s.tokens_per_dp_rank, s.logit_rows_per_dp_rank, s.cuda_graph)
            for s in scenarios] == [((512, 512), (2, 2), False),
                                    ((512, 1), (2, 0), False),
                                    ((16, 16), (16, 16), True)]
    with pytest.raises(ValueError, match="Invalid eager point"):
        scenarios_from_grid(parallel, {"eager": [[4, 8]], "cuda_graph_batch_sizes": []})
