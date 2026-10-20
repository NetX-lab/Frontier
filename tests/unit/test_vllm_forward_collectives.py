"""CPU checks for native SP and Step5 forward-pass collective shapes."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from frontier.profiling.collectives.vllm_forward_collectives import (
    ForwardScenario,
    ModelShape,
    ServingParallelism,
    call_bytes,
    forward_collectives,
    group_members,
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


STEP5_CONFIG = {
    "model_type": "step5_vl",
    "text_config": {
        "model_type": "step5",
        "hidden_size": 4096,
        "num_hidden_layers": 92,
        "num_nextn_predict_layers": 3,
        "moe_num_experts": 352,
        "moe_top_k": 8,
        "vocab_size": 128896,
        "moe_layer_list": list(range(3, 91)),
        "moe_layers_enum": ",".join(str(i) for i in range(3, 91)),
    },
    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            f"model.layers.{i}.moe.experts.{weight}"
            for i in range(88, 91) for weight in ("w13_weight", "w2_weight")
        ] + ["model.layers.3.self_attn.o_proj", "model.layers.3.share_expert.down_proj"],
    },
}

STEP5_ENGINE = {**ENGINE, "data_parallel_size": 4, "async_scheduling": False,
                "moe_backend": "auto"}


def _step5_grid():
    """The 50 measurement scenarios in design01 sections 4.1-4.2."""
    entries = []

    def add(name, wire, useful, graph=False, rails=None):
        entries.append({
            "name": name,
            "tokens_per_dp_rank": list(wire),
            "useful_tokens_per_dp_rank": list(useful),
            "logit_rows_per_dp_rank": list(useful) if graph else [int(n > 0) for n in useful],
            "cuda_graph": graph,
            "active_dp_rails": rails,
        })

    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        add(f"graph_uniform_b{batch}", (batch,) * 4, (batch,) * 4, True)
    for tokens in (16, 32, 64, 65, 128, 256, 512, 1024, 2048, 4096, 8192, 8193):
        add(f"eager_uniform_t{tokens}", (tokens,) * 4, (tokens,) * 4)
    for tokens in (256, 512, 2048, 8192):
        patterns = ((tokens, 0, 0, 0), (tokens, tokens, 0, 0),
                    (tokens, tokens, tokens, 0), (tokens, tokens // 2, tokens // 4, 0),
                    (tokens, 64, 64, 64))
        for index, useful in enumerate(patterns):
            add(f"eager_uneven_t{tokens}_p{index}", tuple(max(1, n) for n in useful), useful)
    for batch in (8, 64, 512):
        add(f"graph_idle_b{batch}", (batch,) * 4, (batch, batch, batch, 0), True)
    for graph, tokens in ((True, 8), (True, 64), (True, 512), (False, 512), (False, 4096)):
        add(f"rail0_{'graph' if graph else 'eager'}_t{tokens}", (tokens,) * 4,
            (tokens,) * 4, graph, [0])
    return {"scenarios": entries}


def test_step5_shape_reads_the_retained_hf_config():
    path = Path(__file__).resolve().parents[2] / "data/config/models/stepfun-ai__Step-5-Preview.json"
    model = ModelShape.from_hf_config(json.loads(path.read_text()))
    assert model.model_type == "step5"
    assert (model.hidden_size, model.num_hidden_layers, model.num_experts_per_tok) == (4096, 92, 8)
    assert model.moe_layer_ids() == list(range(3, 91))
    assert model.fp8_moe_layers == tuple(range(3, 88))
    assert model.fp8_group_size == 128
    assert model.vocab_size == 128896


@pytest.mark.parametrize("entry", _step5_grid()["scenarios"], ids=lambda s: s["name"])
def test_step5_full_forward_counts_and_bytes_for_50_scenarios(entry):
    parallel = ServingParallelism.from_engine(STEP5_ENGINE)
    scenario, = scenarios_from_grid(parallel, {"scenarios": [entry]})
    model = ModelShape.from_hf_config(STEP5_CONFIG)
    calls = {c.name: c for c in forward_collectives(model, parallel, scenario)}
    expected_counts = {
        "embedding_all_reduce": 1,
        "attention_all_reduce": 92,
        "dense_down_proj_all_reduce": 4,
        "shared_expert_down_proj_all_reduce": 88,
        "moe_dispatch_all_gatherv_fp8": 85,
        "moe_dispatch_all_gatherv_bf16": 3,
        "moe_combine_reduce_scatterv": 88,
        "routed_expert_all_reduce": 88,
        "logits_all_gather": 1,
        "dp_batch_sync_all_reduce": 1,
    }
    assert {name: c.calls_per_forward for name, c in calls.items()} == expected_counts
    assert sum(c.calls_per_forward for c in calls.values() if c.op == "all_reduce") == 273
    assert sum(c.calls_per_forward for c in calls.values()) == 451
    assert {c.group for c in calls.values()} == {"tp", "dp"}
    fp8 = calls["moe_dispatch_all_gatherv_fp8"]
    bf16 = calls["moe_dispatch_all_gatherv_bf16"]
    combine = calls["moe_combine_reduce_scatterv"]
    logits = calls["logits_all_gather"]
    sync = calls["dp_batch_sync_all_reduce"]
    assert (fp8.columns, fp8.dtypes, fp8.row_bytes()) == (
        (4096, 8, 8, 32), ("float8_e4m3fn", "float32", "int32", "float32"), 4288)
    assert (bf16.columns, bf16.dtypes, bf16.row_bytes()) == (
        (4096, 8, 8), ("bfloat16", "float32", "int32"), 8256)
    assert combine.group == "dp" and combine.dtypes == ("bfloat16",) and combine.row_bytes() == 8192
    assert logits.dim == -1 and logits.row_bytes() == 32224
    assert sync.op == "cpu_all_reduce" and sync.row_bytes() == 16
    assert sync.shard_rows == (6,) * 32
    assert {name for name, c in calls.items() if c.captured} == (
        set(calls) - {"logits_all_gather", "dp_batch_sync_all_reduce"} if scenario.cuda_graph else set())
    total_tokens = sum(scenario.tokens_per_dp_rank)
    for rank in range(parallel.world_size):
        lane, rail = divmod(rank, 8)
        tokens, rows = scenario.tokens_per_dp_rank[lane], scenario.logit_rows_per_dp_rank[lane]
        active = scenario.active_dp_rails is None or rail in scenario.active_dp_rails
        assert group_members(parallel, "dp", rank) == [rail, rail + 8, rail + 16, rail + 24]
        assert call_bytes(fp8, parallel, rank) == ((tokens * 4288, total_tokens * 4288) if active else (0, 0))
        assert call_bytes(bf16, parallel, rank) == ((tokens * 8256, total_tokens * 8256) if active else (0, 0))
        assert call_bytes(combine, parallel, rank) == ((total_tokens * 8192, tokens * 8192) if active else (0, 0))
        assert call_bytes(logits, parallel, rank) == (rows * 32224, rows * 257792)
        assert call_bytes(sync, parallel, rank) == (96, 96)
        for c in calls.values():
            if c.op == "all_reduce":
                assert c.group == "tp" and c.row_bytes() == 8192
                assert call_bytes(c, parallel, rank) == (tokens * 8192, tokens * 8192)
        expected_input = (273 * tokens * 8192 + rows * 32224 + 96
                          + active * (85 * tokens * 4288 + 3 * tokens * 8256 + 88 * total_tokens * 8192))
        expected_output = (273 * tokens * 8192 + rows * 257792 + 96
                           + active * (85 * total_tokens * 4288 + 3 * total_tokens * 8256 + 88 * tokens * 8192))
        weighted_bytes = tuple(sum(call_bytes(c, parallel, rank)[side] * c.calls_per_forward
                                   for c in calls.values()) for side in (0, 1))
        assert weighted_bytes == (expected_input, expected_output)


def test_step5_measurement_grid_coverage_and_legacy_grid_can_be_combined():
    parallel = ServingParallelism.from_engine(STEP5_ENGINE)
    explicit = scenarios_from_grid(parallel, _step5_grid())
    assert len(explicit) == len({s.name for s in explicit}) == 50
    assert sum(s.cuda_graph for s in explicit) == 16
    assert sum(s.active_dp_rails == (0,) for s in explicit) == 5
    assert sum(not all(s.useful_tokens_per_dp_rank) for s in explicit) == 19
    combined = scenarios_from_grid(parallel, {**_step5_grid(), "eager": [[512, 2]],
                                              "cuda_graph_batch_sizes": [16]})
    assert len(combined) == 53
    assert [s.name for s in combined[:3]] == ["eager_uniform_t512_r2",
                                              "eager_single_active_t512_r2", "graph_decode_b16"]
    assert combined[1].tokens_per_dp_rank == (512, 1, 1, 1)
    assert combined[1].useful_tokens_per_dp_rank == (512, 0, 0, 0)


@pytest.mark.parametrize("batch", [8, 64, 512])
def test_graph_idle_rank_keeps_padded_device_communication(batch):
    parallel = ServingParallelism.from_engine(STEP5_ENGINE)
    model = ModelShape.from_hf_config(STEP5_CONFIG)
    uniform = ForwardScenario("uniform", (batch,) * 4, (batch,) * 4, True)
    idle = ForwardScenario("idle", (batch,) * 4, (batch, batch, batch, 0), True,
                           (batch, batch, batch, 0))
    uniform_calls = forward_collectives(model, parallel, uniform)
    idle_calls = forward_collectives(model, parallel, idle)
    assert [c for c in uniform_calls if c.name != "logits_all_gather"] == [
        c for c in idle_calls if c.name != "logits_all_gather"]
    assert all(c.shard_rows[24] == batch for c in idle_calls if c.captured)


def test_graph_wire_padding_does_not_add_useful_tokens_or_sampled_rows():
    parallel = ServingParallelism.from_engine(STEP5_ENGINE)
    model = ModelShape.from_hf_config(STEP5_CONFIG)
    scenario, = scenarios_from_grid(parallel, {"scenarios": [{
        "name": "graph_padded", "cuda_graph": True,
        "tokens_per_dp_rank": [16, 16, 16, 16],
        "useful_tokens_per_dp_rank": [7, 2, 9, 0],
        "logit_rows_per_dp_rank": [7, 2, 9, 0],
    }]})
    calls = {c.name: c for c in forward_collectives(model, parallel, scenario)}
    assert scenario.useful_tokens_per_dp_rank == (7, 2, 9, 0)
    assert calls["moe_dispatch_all_gatherv_fp8"].shard_rows == (16,) * 32
    assert call_bytes(calls["moe_dispatch_all_gatherv_fp8"], parallel, 24) == (16 * 4288, 64 * 4288)
    assert call_bytes(calls["logits_all_gather"], parallel, 0) == (7 * 32224, 7 * 257792)
    assert call_bytes(calls["logits_all_gather"], parallel, 24) == (0, 0)


@pytest.mark.parametrize("tp,dp", [(4, 2), (8, 2), (4, 4), (8, 4)])
@pytest.mark.parametrize("tokens", [1, 64, 65, 512, 8192, 8193])
def test_step5_sizes_follow_topology_and_communicator_breaks(tp, dp, tokens):
    parallel = ServingParallelism.from_engine({**STEP5_ENGINE, "tensor_parallel_size": tp,
                                              "data_parallel_size": dp})
    scenario = ForwardScenario("size", (tokens,) * dp, (1,) * dp, False)
    calls = {c.name: c for c in forward_collectives(ModelShape.from_hf_config(STEP5_CONFIG), parallel, scenario)}
    assert calls["attention_all_reduce"].shard_rows == (tokens,) * (tp * dp)
    assert call_bytes(calls["attention_all_reduce"], parallel, 0) == (tokens * 8192,) * 2
    assert call_bytes(calls["moe_dispatch_all_gatherv_fp8"], parallel, 0) == (tokens * 4288, dp * tokens * 4288)
    assert call_bytes(calls["moe_combine_reduce_scatterv"], parallel, 0) == (dp * tokens * 8192, tokens * 8192)
    assert call_bytes(calls["dp_batch_sync_all_reduce"], parallel, 0) == (6 * dp * 4,) * 2
    assert calls["logits_all_gather"].row_bytes() == 128896 // tp * 2


def test_step5_schedule_quantization_and_width_are_config_driven():
    config = deepcopy(STEP5_CONFIG)
    text = config["text_config"]
    text.update(hidden_size=2048, num_hidden_layers=8, moe_top_k=4,
                moe_layer_list=[1, 4, 6, 8], moe_layers_enum="1,4,6,8")
    quant = config["quantization_config"]
    quant.update(weight_block_size=[64, 64], modules_to_not_convert=["model.layers.4.moe.experts.w2_weight"])
    model = ModelShape.from_hf_config(config)
    assert model.moe_layer_ids() == [1, 4, 6]  # Index 8 is MTP, not a main-model layer.
    assert model.fp8_moe_layers == (1, 6)
    parallel, calls = _calls(STEP5_ENGINE, config, ForwardScenario("s", (8,) * 4, (1,) * 4, False))
    assert calls["attention_all_reduce"].calls_per_forward == 8
    assert calls["dense_down_proj_all_reduce"].calls_per_forward == 5
    assert calls["shared_expert_down_proj_all_reduce"].calls_per_forward == 3
    assert calls["routed_expert_all_reduce"].calls_per_forward == 3
    assert calls["moe_dispatch_all_gatherv_fp8"].calls_per_forward == 2
    assert calls["moe_dispatch_all_gatherv_bf16"].calls_per_forward == 1
    assert calls["moe_dispatch_all_gatherv_fp8"].columns == (2048, 4, 4, 32)
    assert calls["moe_dispatch_all_gatherv_fp8"].row_bytes() == 2208
    assert calls["moe_dispatch_all_gatherv_bf16"].row_bytes() == 4128
    assert call_bytes(calls["moe_combine_reduce_scatterv"], parallel, 0) == (4 * 8 * 4096, 8 * 4096)


def test_step5_unquantized_text_config_and_native_schedule_default():
    text = {k: v for k, v in STEP5_CONFIG["text_config"].items()
            if k not in ("moe_layers_enum", "moe_layer_list")}
    model = ModelShape.from_hf_config(text)
    assert model.moe_layer_ids() == list(range(1, 92))
    assert not model.fp8_moe_layers
    _, calls = _calls(STEP5_ENGINE, text, ForwardScenario("s", (2,) * 4, (1,) * 4, False))
    assert "moe_dispatch_all_gatherv_fp8" not in calls
    assert calls["moe_dispatch_all_gatherv_bf16"].calls_per_forward == 91


def test_step5_flattened_config_and_ignored_layer_alias():
    text = deepcopy(STEP5_CONFIG["text_config"])
    text["model_type"] = "Step5"
    quant = deepcopy(STEP5_CONFIG["quantization_config"])
    quant["ignored_layers"] = quant.pop("modules_to_not_convert")
    text["quantization_config"] = quant
    assert ModelShape.from_hf_config(text) == ModelShape.from_hf_config(STEP5_CONFIG)


@pytest.mark.parametrize("override,message", [
    ({"moe_backend": "deep_gemm_mega_moe"}, "sequence-parallel collectives are not modeled"),
    ({"data_parallel_size": 1}, "Step5 DP MoE"),
    ({"enable_expert_parallel": False}, "Step5 DP MoE"),
    ({"pipeline_parallel_size": 2}, "Step5 DP MoE"),
    ({"all2all_backend": "deepep_low_latency"}, "not modeled"),
    ({"dtype": "float32"}, "bfloat16 activations"),
])
def test_step5_rejects_different_native_paths(override, message):
    engine = {**STEP5_ENGINE, **override}
    dp = engine["data_parallel_size"]
    with pytest.raises(ValueError, match=message):
        _calls(engine, STEP5_CONFIG, ForwardScenario("s", (2,) * dp, (1,) * dp, False))


@pytest.mark.parametrize("override,message", [
    ({"tokens_per_dp_rank": [4, 4]}, "one token and logits count"),
    ({"tokens_per_dp_rank": [4, 4, 4, 0]}, "at least one token"),
    ({"tokens_per_dp_rank": [4, 4, 4, 1], "cuda_graph": True}, "DP-padded"),
    ({"useful_tokens_per_dp_rank": [4, 4, 4]}, "Useful token counts"),
    ({"useful_tokens_per_dp_rank": [4, 4, 4, 5]}, "Useful token counts"),
    ({"useful_tokens_per_dp_rank": [0, 0, 0, 0]}, "at least one DP rank"),
    ({"useful_tokens_per_dp_rank": [4, 4, 4, 0]}, "one-token dummy"),
    ({"useful_tokens_per_dp_rank": [4, 4, 4, 0], "cuda_graph": True}, "Logits rows"),
    ({"active_dp_rails": []}, "distinct TP indices"),
    ({"active_dp_rails": [0, 0]}, "distinct TP indices"),
    ({"active_dp_rails": [-1]}, "distinct TP indices"),
    ({"active_dp_rails": [8]}, "distinct TP indices"),
])
def test_explicit_scenario_validation(override, message):
    entry = {"name": "invalid", "tokens_per_dp_rank": [4] * 4,
             "logit_rows_per_dp_rank": [1] * 4, "cuda_graph": False, **override}
    with pytest.raises(ValueError, match=message):
        scenarios_from_grid(ServingParallelism.from_engine(STEP5_ENGINE), {"scenarios": [entry]})


@pytest.mark.parametrize("field,value,message", [
    ("moe_layers_enum", "3,4", "different layers"),
    ("moe_layer_list", [3, 3], "duplicate"),
    ("moe_layer_list", [-1], "different layers"),
])
def test_step5_rejects_conflicting_or_duplicate_schedules(field, value, message):
    config = deepcopy(STEP5_CONFIG)
    config["text_config"][field] = value
    with pytest.raises(ValueError, match=message):
        ModelShape.from_hf_config(config)


@pytest.mark.parametrize("indices", [[-1], [95]])
def test_step5_rejects_out_of_range_schedules(indices):
    config = deepcopy(STEP5_CONFIG)
    config["text_config"].update(moe_layer_list=indices, moe_layers_enum=None)
    with pytest.raises(ValueError, match="MoE layer schedule must be"):
        ModelShape.from_hf_config(config)


@pytest.mark.parametrize("quant", [
    {"quant_method": "fp4"}, {"fmt": "e5m2"}, {"activation_scheme": "static"},
    {"weight_block_size": None}, {"weight_block_size": [128, 0]},
])
def test_step5_rejects_unmodeled_dispatch_quantization(quant):
    config = deepcopy(STEP5_CONFIG)
    config["quantization_config"].update(quant)
    with pytest.raises(ValueError, match="FP8"):
        ModelShape.from_hf_config(config)


@pytest.mark.parametrize("model_type", ["deepseek_v2", "glm_moe_dsa"])
@pytest.mark.parametrize("tp,dp", [(2, 2), (4, 2), (8, 4)])
@pytest.mark.parametrize("tokens", [1, 65, 512, 8193])
def test_existing_sp_shapes_are_unchanged(model_type, tp, dp, tokens):
    engine = {**ENGINE, "tensor_parallel_size": tp, "data_parallel_size": dp}
    config = {**GLM_CONFIG, "model_type": model_type}
    scenario = ForwardScenario("sp", (tokens,) + (1,) * (dp - 1), (1,) + (0,) * (dp - 1), False)
    parallel, calls = _calls(engine, config, scenario)
    assert [c.name for c in calls.values()] == [
        "embedding_all_reduce", "dense_layer_all_reduce", "moe_attention_reduce_scatter",
        "moe_layer_input_all_gather", "final_hidden_residual_all_gather",
        "moe_dispatch_all_gatherv", "moe_combine_reduce_scatterv", "logits_all_gather",
        "dp_batch_sync_all_reduce"]
    assert [c.calls_per_forward for c in calls.values()] == [1, 6, 75, 74, 1, 75, 75, 1, 1]
    sp = (tokens + tp - 1) // tp
    total = (sp + dp - 1) * tp
    dispatch = calls["moe_dispatch_all_gatherv"]
    assert dispatch.group == "ep" and dispatch.shard_rows == (sp,) * tp + (1,) * (tp * (dp - 1))
    assert call_bytes(dispatch, parallel, 0) == (sp * 12352, total * 12352)
    assert call_bytes(calls["moe_combine_reduce_scatterv"], parallel, 0) == (total * 12288, sp * 12288)
    assert call_bytes(calls["moe_attention_reduce_scatter"], parallel, 0) == (tp * sp * 12288, sp * 12288)
    assert call_bytes(calls["final_hidden_residual_all_gather"], parallel, 0) == (sp * 24576, tp * sp * 24576)
    assert call_bytes(calls["dp_batch_sync_all_reduce"], parallel, 0) == (4 * dp * 4,) * 2
    assert calls["dp_batch_sync_all_reduce"].op == "cpu_all_reduce"
