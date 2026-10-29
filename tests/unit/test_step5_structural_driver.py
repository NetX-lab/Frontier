"""Acceptance faults for the manual Step controlled CLI driver."""

import json

import pytest

from tests.integration import run_step5_structural as driver


def test_cases_pin_the_controlled_cli_and_keep_native_execution_out(tmp_path):
    cases = driver.build_cases(tmp_path)
    assert [case.case_id for case in cases] == [
        "offline-none", "online-piecewise", "offline-full_decode_only",
    ]
    for case in cases:
        assert case.env["MODEL_NAME"] == "stepfun-ai/Step-5-Preview"
        assert (case.env["ATTN_TP"], case.env["MOE_TP"], case.env["MOE_EP"], case.env["PP"]) == (
            "16", "16", "1", "1",
        )
        assert (case.env["NUM_REQUESTS"], case.env["PREFILL_TOKENS"], case.env["DECODE_TOKENS"]) == (
            "1", "4105", "9",
        )
        args = case.extra_args
        for flag, value in {
            "--replica_config_attention_backend": "STEP5_DSA",
            "--replica_config_kv_cache_dtype": "auto",
            "--vllm_v1_scheduler_config_num_blocks": "0",
            "--vllm_v1_scheduler_config_num_blocks_mode": "memory_planner",
            "--vllm_v1_scheduler_config_max_model_len": "8192",
            "--vllm_v1_scheduler_config_batch_size_cap": "1",
        }.items():
            assert args[args.index(flag) + 1] == value
        assert "--replica_config_indexer_kv_cache_dtype" not in args
        assert "--no-vllm_v1_scheduler_config_enable_runtime_non_kv_cache_overhead_profiling" in args
        assert case.env["FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH"] == str(
            tmp_path / "cases" / case.case_id / "schedule_decisions.jsonl"
        )
        assert not case.uses_trained_predictor
    assert cases[0].script.endswith("offline/moe_model_basic.sh")
    assert cases[1].script.endswith("online/moe_model_basic_online.sh")
    for case in cases[1:]:
        assert case.extra_args[-2:] == ("--cudagraph_capture_sizes", "8")


def stage_row(query=512, prefill=512):
    return {"batch_id": 17, "cluster_type": "MONOLITHIC", "stage_id": 0, "replica_id": 0,
            "replica_local_id": 0, "execution_scope": "ATTN_DP_LANE", "request_ids": ["0"],
            "request_num_tokens": [query], "request_num_prefill_tokens": [prefill],
            "stage_start_ts": 1.0, "stage_end_ts": 1.38925}


def test_decode_padding_cannot_be_counted_as_eight_effective_request_tokens():
    stage = stage_row(query=1, prefill=0)
    driver.check_stage(stage, 1, 0)
    stage["request_num_tokens"] = [8]
    with pytest.raises(ValueError, match="padding credited"):
        driver.check_stage(stage, 1, 0)


def trace_names():
    """Independent source order; tensor details are irrelevant to omission faults."""
    rows = [("schedule", -1, .25), ("prepare_inputs_e2e", -1, .25),
            ("emb", -1, .25), ("embedding_tensor_parallel_allreduce", -1, .25), ("input_cast", -1, .25)]
    for layer in range(92):
        names = ["input_layernorm", "step5_qkv_projection", "step5_attention_prepare"]
        if layer % 4 == 3:
            names.extend(("step5_indexer_weights", "step5_indexer_prepare", "step5_sparse_attention"))
        else:
            names.extend(("step5_swa_attention", "step5_gate_projection", "step5_headwise_gate"))
        names.extend(("step5_output_projection", "attn_tensor_parallel_allreduce", "post_attention_layernorm"))
        if 3 <= layer <= 90:
            names.extend(("share_expert_up_proj", "share_expert_act", "share_expert_down_proj", "moe_gating_linear",
                          "moe_gating_routing_topk", "moe_shuffling", "moe_grouped_gemm", "step5_ffn_merge",
                          "moe_tensor_parallel_allreduce"))
        else:
            names.extend(("mlp_up_proj", "mlp_act", "mlp_down_proj", "mlp_tensor_parallel_allreduce"))
        rows.extend((name, layer, .125 if name in ("moe_gating_linear", "moe_gating_routing_topk") else .25)
                    for name in names)
    rows.extend((name, -1, .25) for name in ("final_residual", "sampler_e2e", "process_model_outputs", "ray_comm_time"))
    return rows


def split_scope_events(names):
    wave_names = {"post_attention_layernorm", "share_expert_up_proj", "share_expert_act",
                  "share_expert_down_proj", "moe_gating_linear", "moe_gating_routing_topk",
                  "moe_shuffling", "moe_grouped_gemm", "step5_ffn_merge", "moe_tensor_parallel_allreduce"}
    stage, waves = [], []
    for name, layer, cost in names:
        row = {"name": name, "duration_ms": cost, "meta": {"global_layer_id": layer}}
        (waves if 3 <= layer <= 90 and name in wave_names else stage).append(row)
    return stage, waves


@pytest.mark.parametrize("fault", ["missing_layer91", "merge_missing", "merge_duplicate", "cpu_duplicate",
                                   "reordered_preparation", "cost_moved"])
def test_real_batch_cannot_pass_with_missing_duplicate_or_reordered_work(fault):
    names = trace_names()
    driver.check_scope_order(*split_scope_events(names), fused=False)
    target = next(i for i, (name, layer, _) in enumerate(names) if name == "step5_ffn_merge" and layer == 90)
    if fault == "missing_layer91":
        names = [row for row in names if row[1] != 91]
    elif fault == "merge_missing":
        names.pop(target)
    elif fault == "merge_duplicate":
        names.insert(target, names[target])
    elif fault == "cpu_duplicate":
        names.insert(0, names[0])
    elif fault == "reordered_preparation":
        i = next(i for i, (name, layer, _) in enumerate(names) if name == "step5_indexer_prepare" and layer == 3)
        names[i], names[i - 1] = names[i - 1], names[i]
    else:
        name, layer, cost = names[target]
        names[target] = (name, layer, cost - .125)
        name, layer, cost = names[target + 1]
        names[target + 1] = (name, layer, cost + .125)
    with pytest.raises(ValueError, match="missing/duplicate/out-of-order"):
        driver.check_scope_order(*split_scope_events(names), fused=False)


def merge_trace():
    return {"name": "step5_ffn_merge", "type": "COMPUTE", "meta": {
        "global_layer_id": 90, "source_geometry": {"global_layer_id": 90}, "evidence_kind": "source_structural_metadata", "attention_family_id": "step5_attention",
        "attention_variant_id": "swa", "attention_operator_names": ["step5_qkv_projection", "step5_attention_prepare",
            "step5_swa_attention", "step5_gate_projection", "step5_headwise_gate", "step5_output_projection"],
        "num_layers": 92, "request_ids": ["0"], "total_tokens": 1, "num_tokens": [1],
        "effective_total_tokens_compute": 8, "effective_total_tokens_rounded": 8, "effective_total_tokens_transfer": 1,
        "parallel_context": {"PP": 1, "ATTN_TP": 16, "ATTN_DP": 1, "MOE_TP": 16, "MOE_EP": 1, "ROUTER_TOPK": 8},
        "tensor_shape": {"routed": [8, 4096], "shared": [8, 4096], "output": [8, 4096]},
        "tensor_dtype": {"routed": "BF16", "shared": "BF16", "output": "FP32"},
        "tensor_size_bytes": {"routed": 65536, "shared": 65536, "output": 131072}}}


@pytest.mark.parametrize("fault", ["shape", "precision", "byte_cost", "physical_padding", "claim", "layer_identity"])
def test_merge_tensor_faults_cannot_hide_behind_correct_total_cost(fault):
    row = merge_trace()
    driver.check_tensor(row, 8, 64, False)
    meta = row["meta"]
    if fault == "shape":
        meta["tensor_shape"]["routed"] = [1, 4096]
        meta["tensor_size_bytes"]["routed"] = 8192
    elif fault == "precision":
        meta["tensor_dtype"]["output"] = "BF16"
        meta["tensor_size_bytes"]["output"] = 65536
    elif fault == "byte_cost":
        meta["tensor_size_bytes"]["shared"] -= 2
    elif fault == "physical_padding":
        meta["effective_total_tokens_compute"] = 1
    elif fault == "claim":
        meta["evidence_kind"] = "native_measured"
    else:
        meta["global_layer_ids"] = [88]
    with pytest.raises(ValueError, match="shape/precision|conservation|attribution|qualification|layer identity"):
        driver.check_tensor(row, 8, 64, False)


def test_per_rank_weight_shape_and_fp8_exclusions_are_checked_in_artifacts():
    row = merge_trace()
    row["name"] = "moe_grouped_gemm"
    meta = row["meta"]
    meta["global_layer_id"] = 88
    meta["source_geometry"]["global_layer_id"] = 88
    meta["num_layers"] = 1
    meta["tokens_are_post_routing"] = True
    meta["effective_total_tokens_compute"] = 64
    meta["tensor_shape"] = {"input": [8, 4096], "output": [8, 4096]}
    meta["internal_tensor_shape"] = {"gate_up": [8, 8, 192], "activation": [64, 96], "down": [8, 8, 4096]}
    meta["covered_scopes"] = ["gate_up_gemm", "swiglu", "down_gemm", "topk_reduce"]
    meta["tensor_dtype"] = {"input": "BF16", "output": "BF16"}
    meta["tensor_size_bytes"] = {"input": 65536, "output": 65536}
    meta["projection_modules"] = [{"weight_shape": [352, 192, 4096], "weight_dtype": "BF16"},
                                  {"weight_shape": [352, 4096, 96], "weight_dtype": "BF16"}]
    driver.check_tensor(row, 8, 64, False, wave=True)
    meta["projection_modules"][0]["weight_dtype"] = "FP8"
    with pytest.raises(ValueError, match="per-layer FP8/BF16"):
        driver.check_tensor(row, 8, 64, False, wave=True)
    meta["projection_modules"][0]["weight_dtype"] = "BF16"
    meta["projection_modules"][1]["weight_shape"][-1] = 1536
    with pytest.raises(ValueError, match="per-rank projection"):
        driver.check_tensor(row, 8, 64, False, wave=True)


def test_existing_output_is_rejected_before_the_harness_can_clear_a_case(tmp_path):
    marker = tmp_path / "old_run.json"
    marker.write_text("retained evidence\n")
    with pytest.raises(SystemExit) as error:
        driver.main(["--output-root", str(tmp_path)])
    assert error.value.code == 2
    assert marker.read_text() == "retained evidence\n"


def test_runtime_and_driver_source_drift_is_reported():
    before = {"source_revision": "a" * 40, "harness_revision": "a" * 40,
              "files_sha256": {"frontier/main.py": "b" * 64, "tests/integration/run_step5_structural.py": "c" * 64},
              "python": {"version": "3.10.6"}}
    after = {**before, "files_sha256": {**before["files_sha256"], "frontier/main.py": "d" * 64}}
    assert driver.source_changes(before, before) == []
    assert driver.source_changes(before, after) == ["files_sha256"]


def stage_timing():
    # Worked 92-layer, multi-query source oracle, independent of the driver.
    positive = {"attn_norm_time": 23, "mlp_norm_time": 23, "attention_all_reduce_time": 23,
                "mlp_all_reduce_time": 23, "step5_qkv_projection": 23, "step5_attention_prepare": 23,
                "step5_output_projection": 23, "step5_sparse_attention": 5.75, "step5_indexer_weights": 5.75,
                "step5_indexer_prepare": 5.75, "step5_swa_attention": 17.25, "step5_gate_projection": 17.25,
                "step5_headwise_gate": 17.25, "mlp_layer_up_proj_execution_time": 1,
                "mlp_layer_act_execution_time": 1, "mlp_layer_down_proj_execution_time": 1,
                "moe_gating_linear_time": 11, "moe_gating_routing_topk_time": 11,
                "moe_shuffling_time": 22, "moe_grouped_gemm_time": 22, "step5_ffn_merge": 22,
                "share_expert_up_proj_time": 22, "share_expert_act_time": 22, "share_expert_down_proj_time": 22,
                "schedule_time": .25, "prepare_inputs_e2e_time": .25, "sampler_e2e_time": .25,
                "process_model_outputs_time": .25, "ray_comm_time": .25,
                "emb": .25, "embedding_tensor_parallel_allreduce": .25, "input_cast": .25, "final_residual": .25}
    return {"component_ledger_ms": {**positive, "pipeline_parallel_communication_time": 0},
            "total_time_ms": 389.25, "model_time_ms": 388, "diagnostic_total_time_ms": 389.25,
            "diagnostic_component_ledger_ms": {"pp_stage_boundary_handoff_time": 0}}


def test_correct_stage_total_cannot_hide_a_charge_in_an_inactive_component():
    timing = stage_timing()
    driver.check_costs(timing, trace_names())
    timing["component_ledger_ms"]["step5_sparse_attention"] -= .25
    timing["component_ledger_ms"]["attention_prefill_execution_time"] = .25
    with pytest.raises(ValueError, match="component ownership"):
        driver.check_costs(timing, trace_names())


def test_ep_wave_trace_joins_the_source_stage_instead_of_its_synthetic_batch():
    stages = [stage_row()]
    header = {"meta": {"simulation_info": {}}}
    stage_event = {"batch_id": 17, "type": "COMPUTE", "name": "input_layernorm", "meta": {"global_layer_id": 0}}
    wave_event = {"batch_id": 73, "type": "COMPUTE", "name": "step5_ffn_merge", "meta": {
        "source_batch_ids": [17], "global_layer_ids": [3]}}
    wave = {"batch_id": 73, "source_batch_ids": [17], "global_layer_ids": [3]}
    stage_events, wave_events, wave_rows = driver.join_wave_evidence(
        [header, wave_event, stage_event], stages, [wave],
    )
    assert stage_events == {17: [stage_event]}
    assert wave_events == {17: [wave_event]}
    assert wave_rows == {17: [wave]}
    wave_event["meta"]["source_batch_ids"] = [99]
    with pytest.raises(ValueError, match="unattributed wave source"):
        driver.join_wave_evidence([header, wave_event, stage_event], stages, [wave])


def wave_row():
    pre = {"post_attention_layernorm": .25, "share_expert_up_proj": .25, "share_expert_act": .25,
           "share_expert_down_proj": .25, "moe_gating_linear": .125, "moe_gating_routing_topk": .125,
           "moe_shuffling": .25}
    return {"batch_id": 73, "ep_id": 0, "stage_id": 0, "replica_id": 0, "cluster": "MONOLITHIC",
            "record_type": "ep_wave_lane", "source_batch_ids": [17], "global_layer_ids": [3],
            "wave_start_time_s": 1.0, "wave_end_time_s": 1.00225,
            "phases": {
                "pre_dispatch": {"start_time_s": 1.0, "duration_ms": 1.5, "operators_ms": pre},
                "dispatch": {"start_time_s": 1.0015, "duration_ms": 0, "operators_ms": {}},
                "routed_compute": {"start_time_s": 1.0015, "duration_ms": .25,
                                   "operators_ms": {"moe_grouped_gemm": .25}},
                "combine": {"start_time_s": 1.00175, "duration_ms": 0, "operators_ms": {}},
                "post_combine": {"start_time_s": 1.00175, "duration_ms": .5,
                                 "operators_ms": {"step5_ffn_merge": .25, "moe_tensor_parallel_allreduce": .25}}}}


@pytest.mark.parametrize("fault", ["merge_in_routed_phase", "duration", "missing_dispatch"])
def test_wave_phase_ownership_conserves_cost_without_hiding_missing_or_repeated_work(fault):
    row = wave_row()
    driver.check_wave_costs(row, 3)
    if fault == "merge_in_routed_phase":
        row["phases"]["routed_compute"]["operators_ms"]["step5_ffn_merge"] = .25
        row["phases"]["post_combine"]["operators_ms"].pop("step5_ffn_merge")
    elif fault == "duration":
        row["phases"]["post_combine"]["duration_ms"] += .25
    else:
        row["phases"].pop("dispatch")
    with pytest.raises(ValueError, match="wave phase"):
        driver.check_wave_costs(row, 3)


def controlled_config():
    """Published controlled config; the serializer omits optional None fields."""
    return {
        "simulation_mode": "offline", "sys_arch": "co-location", "enable_parallel_clusters": False,
        "decode_cuda_graph_mode": "none", "use_cuda_graph": False, "cudagraph_capture_sizes": None,
        "cluster_config": {
            "num_replicas": 1, "cc_backend_config": {"name": "analytical"},
            "replica_config": {
                "model_name": "stepfun-ai/Step-5-Preview", "device": "h800", "network_device": "h800_dgx",
                "attn_tensor_parallel_size": 16, "attn_dp": 1, "moe_tensor_parallel_size": 16,
                "moe_expert_parallel_size": 1, "num_pipeline_stages": 1, "world_size": 16,
                "attention_backend": "STEP5_DSA", "kv_cache_dtype": "auto", "total_expert_num": 352,
                "router_topk": 8, "speculative_decoding_config": {"enabled": False},
                "model_config": {"model_type": "step5", "num_layers": 92, "num_experts": 352,
                    "moe_layers_enum": ",".join(map(str, range(3, 91))), "num_nextn_predict_layers": 3,
                    "layer_types": (["sliding_attention"] * 3 + ["full_attention"]) * 23}
            },
            "replica_scheduler_config": {
                "name": "vllm_v1", "block_size": 64, "num_blocks": 0, "num_blocks_mode": "memory_planner",
                "max_model_len": 8192, "batch_size_cap": 1, "max_tokens_in_batch": 512,
                "long_prefill_token_threshold": 512, "enable_chunked_prefill": True,
                "enable_prefix_caching": False, "enable_runtime_non_kv_cache_overhead_profiling": False,
            },
            "execution_time_predictor_config": {"enable_dummy_mode": True, "dummy_execution_time_ms": .25},
        },
        "request_generator_config": {
            "num_requests": 1, "length_generator_config": {"prefill_tokens": 4105, "decode_tokens": 9},
        },
    }


def test_unset_indexer_cache_dtype_survives_optional_field_serialization():
    config = controlled_config()
    driver.check_config(config, "offline-none")
    config["cluster_config"]["replica_config"]["indexer_kv_cache_dtype"] = "fp8"
    with pytest.raises(ValueError, match="indexer_kv_cache_dtype"):
        driver.check_config(config, "offline-none")


def test_wave_phase_order_uses_timestamps_after_sorted_json_export():
    row = json.loads(json.dumps(wave_row(), sort_keys=True))
    driver.check_wave_costs(row, 3)
    row["phases"]["dispatch"]["start_time_s"] = row["phases"]["routed_compute"]["start_time_s"] + .00025
    with pytest.raises(ValueError, match="wave phase boundary"):
        driver.check_wave_costs(row, 3)


def test_attention_scope_cannot_omit_its_query_tensor_boundary():
    row = merge_trace()
    row["name"] = "step5_attention_prepare"
    meta = row["meta"]
    meta["global_layer_id"] = meta["source_geometry"]["global_layer_id"] = 0
    meta["tensor_shape"] = {"q": [8, 4, 192], "k": [8, 1, 192], "v": [8, 1, 192], "positions": [8]}
    meta["tensor_dtype"] = {"q": "BF16", "k": "BF16", "v": "BF16", "positions": "INT64"}
    meta["tensor_size_bytes"] = {"q": 12288, "k": 3072, "v": 3072, "positions": 64}
    driver.check_tensor(row, 8, 64, False)
    for field in ("tensor_shape", "tensor_dtype", "tensor_size_bytes"):
        meta[field].pop("q")
    with pytest.raises(ValueError, match="Q/K/V shape"):
        driver.check_tensor(row, 8, 64, False)


@pytest.mark.parametrize("operator", ["emb", "embedding_tensor_parallel_allreduce", "input_cast", "final_residual"])
@pytest.mark.parametrize("fault", ["omitted", "duplicated"])
def test_model_endpoints_are_required_once_at_their_source_position(operator, fault):
    names = trace_names()
    driver.check_scope_order(*split_scope_events(names), fused=False)
    index = next(i for i, (name, _, _) in enumerate(names) if name == operator)
    if fault == "omitted":
        names.pop(index)
    else:
        names.insert(index, names[index])
    with pytest.raises(ValueError, match="missing/duplicate/out-of-order"):
        driver.check_scope_order(*split_scope_events(names), fused=False)


def stage_scope_trace(name):
    """Worked TP16, eight padded rows, one valid request source boundary."""
    tensors = {
        "emb": ({"input_ids": [8], "weight": [8056, 4096], "output": [8, 4096]},
                {"input_ids": "INT32", "weight": "BF16", "output": "BF16"},
                {"input_ids": 32, "weight": 65_994_752, "output": 65_536}),
        "embedding_tensor_parallel_allreduce": ({"data": [8, 4096]}, {"data": "BF16"}, {}),
        "input_cast": ({"input": [8, 4096], "output": [8, 4096]},
                       {"input": "BF16", "output": "FP32"}, {"input": 65_536, "output": 131_072}),
        "final_residual": ({"input": [8, 4096], "residual": [8, 4096], "output": [8, 4096]},
                           {"input": "BF16", "residual": "FP32", "output": "FP32"},
                           {"input": 65_536, "residual": 131_072, "output": 131_072}),
    }
    shapes, dtypes, sizes = tensors[name]
    meta = {"scope": "stage", "evidence_kind": "source_structural_metadata", "num_layers": 92,
            "request_ids": ["0"], "num_tokens": [1], "total_tokens": 1,
            "effective_total_tokens_compute": 8, "effective_total_tokens_rounded": 8,
            "effective_total_tokens_transfer": 1,
            "parallel_context": {"PP": 1, "ATTN_TP": 16, "ATTN_DP": 1,
                                 "MOE_TP": 16, "MOE_EP": 1, "ROUTER_TOPK": 8},
            "tensor_shape": shapes, "tensor_dtype": dtypes, "tensor_size_bytes": sizes,
            "dtype": "BF16", "dtype_bytes": 2,
            "precision_op": "emb" if name == "embedding_tensor_parallel_allreduce" else name}
    if name == "embedding_tensor_parallel_allreduce":
        meta.pop("tensor_size_bytes")
        meta.update(element_count=32768, base_size_bytes=65536, data_size_bytes=65536)
    return {"name": name, "type": "COMM" if "allreduce" in name else "COMPUTE", "layer_id": -1, "meta": meta}


@pytest.mark.parametrize(("name", "fault"), [
    ("emb", "vocabulary_shard"), ("emb", "byte_cost"), ("input_cast", "precision"),
    ("final_residual", "valid_instead_of_physical"), ("final_residual", "layer_owner"),
    ("embedding_tensor_parallel_allreduce", "payload"), ("embedding_tensor_parallel_allreduce", "precision"),
])
def test_stage_model_scopes_cannot_pass_with_wrong_shape_precision_or_owner(name, fault):
    row = stage_scope_trace(name)
    driver.check_stage_scope(row, 8, 1)
    meta = row["meta"]
    if fault == "vocabulary_shard":
        meta["tensor_shape"]["weight"] = [128896, 4096]
        meta["tensor_size_bytes"]["weight"] = 1_055_916_032
    elif fault == "byte_cost":
        meta["tensor_size_bytes"]["input_ids"] -= 4
    elif fault == "precision":
        key = "data" if "allreduce" in name else "output"
        meta["tensor_dtype"][key] = "FP32" if key == "data" else "BF16"
    elif fault == "valid_instead_of_physical":
        meta["tensor_shape"]["output"] = [1, 4096]
        meta["tensor_size_bytes"]["output"] = 16384
    elif fault == "layer_owner":
        row["layer_id"] = 91
        meta["global_layer_id"] = 91
    else:
        meta["data_size_bytes"] -= 2
    with pytest.raises(ValueError, match="stage.*shape|stage.*byte|stage.*precision|stage.*owner|stage.*payload"):
        driver.check_stage_scope(row, 8, 1)


def sampler_scope_trace():
    row = stage_scope_trace("final_residual")
    row["name"], row["type"] = "sampler_e2e", "OVERHEAD"
    meta = row["meta"]
    for key in ("tensor_dtype", "tensor_size_bytes", "dtype", "dtype_bytes", "precision_op"):
        meta.pop(key)
    meta.update(time_ownership="inclusive_sampler_e2e", logits_rows=1,
                model_precision="BF16", norm_weight_precision="FP32",
                covered_scopes=["select_hidden_states", "logits_input_cast", "final_norm",
                                "lm_head_linear", "logits_gather", "sampling"],
                tensor_shape={"selected_hidden": [1, 4096], "local_logits": [1, 8056]})
    return row


@pytest.mark.parametrize("fault", ["omitted_norm", "duplicate_logits", "padded_logits", "norm_precision", "exclusive_cost"])
def test_sampler_owns_all_logits_scopes_once_and_uses_actual_request_rows(fault):
    row = sampler_scope_trace()
    driver.check_stage_scope(row, 8, 1)
    meta = row["meta"]
    if fault == "omitted_norm":
        meta["covered_scopes"].remove("final_norm")
    elif fault == "duplicate_logits":
        meta["covered_scopes"].append("lm_head_linear")
    elif fault == "padded_logits":
        meta["logits_rows"] = 8
        meta["tensor_shape"] = {"selected_hidden": [8, 4096], "local_logits": [8, 8056]}
    elif fault == "norm_precision":
        meta["norm_weight_precision"] = "BF16"
    else:
        meta["time_ownership"] = "exclusive_sampling"
    with pytest.raises(ValueError, match="sampler.*scope|sampler.*rows|sampler.*precision|sampler.*ownership"):
        driver.check_stage_scope(row, 8, 1)


@pytest.mark.parametrize("missing_ms", [.75, .25, -.25])
def test_actual_stage_time_cannot_hide_missing_input_or_terminal_work(missing_ms):
    stage = stage_row()
    driver.check_stage_cost_span(stage, 389.25)
    stage["stage_end_ts"] -= missing_ms / 1000
    with pytest.raises(ValueError, match="actual stage duration"):
        driver.check_stage_cost_span(stage, 389.25)


def test_expert_assignment_rows_cannot_be_presented_as_branch_outputs():
    row = merge_trace()
    row["name"] = "moe_grouped_gemm"
    meta = row["meta"]
    meta["num_layers"] = 1
    meta["tokens_are_post_routing"] = True
    meta["effective_total_tokens_compute"] = 64
    meta["tensor_shape"] = {"input": [8, 4096], "output": [8, 4096]}
    meta["tensor_dtype"] = {"input": "BF16", "output": "BF16"}
    meta["tensor_size_bytes"] = {"input": 65536, "output": 65536}
    meta["internal_tensor_shape"] = {"gate_up": [8, 8, 192], "activation": [64, 96], "down": [8, 8, 4096]}
    meta["covered_scopes"] = ["gate_up_gemm", "swiglu", "down_gemm", "topk_reduce"]
    meta["projection_modules"] = [{"weight_shape": [352, 192, 4096], "weight_dtype": "BF16"},
                                  {"weight_shape": [352, 4096, 96], "weight_dtype": "BF16"}]
    driver.check_tensor(row, 8, 64, False, wave=True)
    meta["tensor_shape"]["output"] = [64, 4096]
    meta["tensor_size_bytes"]["output"] = 524288
    with pytest.raises(ValueError, match="branch boundary"):
        driver.check_tensor(row, 8, 64, False, wave=True)
