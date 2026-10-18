"""Run manual Step controlled CPU cases through the existing example harness.

Evidence is source_structural: complete model scopes with inclusive sampling,
synthetic timing, source storage and simulator padding. Native timing, capacity
and full physical E2E fidelity remain unverified.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

from tests.e2e.refactor_fidelity.cases import (
    COLOCATION_OFFLINE_MOE, COLOCATION_ONLINE_MOE, FidelityCase,
)
from tests.e2e.refactor_fidelity.compare import file_digest
from tests.e2e.refactor_fidelity.run_matrix import _git, _run_case, source_provenance


REPO = Path(__file__).resolve().parents[2]
PYTHON = "/data/ycfeng/envs/frontier-py310/bin/python"
CASES = {"offline-none": ("offline", "none"),
         "online-piecewise": ("online", "piecewise"),
         "offline-full_decode_only": ("offline", "full_decode_only")}
WEIGHT_BYTES = 40_599_958_208
PARAMETERS = 37_750_873_536
# Independent constructor arithmetic in test_step5_parameter_memory._source_oracle.
WEIGHT_PARTS = {"attention_memory_bytes": 1_740_394_176,
                "ffn_memory_bytes": 38_724_543_488, "other_memory_bytes": 135_020_544}
OVERHEAD = ("schedule", "prepare_inputs_e2e", "sampler_e2e", "process_model_outputs", "ray_comm_time")
STAGE_MODEL_OPS = ("emb", "embedding_tensor_parallel_allreduce", "input_cast", "final_residual")
LIMITS = ["Synthetic controlled costs at 0.25 ms; no native timing or predictor accuracy evidence.",
          "Complete model scopes with inclusive sampling only; no full physical E2E fidelity claim.",
          "sampler_e2e includes hidden selection, logits cast, final norm, lm head, logits gather and sampling; no exclusive costs.",
          "Source-required parameter storage; native allocation and hardware fit unverified.",
          "Simulator graph dispatch and padding; no native capture or replay evidence.",
          "Stage trace cursors omit EP wave spans; conservation joins both existing cost owners.",
          "One request, TP16/DP1/EP1/PP1 MONOLITHIC/vllm_v1, prefix/MTP/PD off.",
          "Source hashes at endpoints detect drift, not continuous loaded-source attestation.",
          "metrics_ground_truth.jsonl describes simulator events, not native ground truth."]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def build_cases(output_root):
    env = {"MODEL_NAME": "stepfun-ai/Step-5-Preview", "SYS_ARCH": "co-location",
           "CC_BACKEND": "analytical", "NUM_REPLICAS": "1", "ATTN_TP": "16", "MOE_TP": "16",
           "MOE_EP": "1", "PP": "1", "TOTAL_EXPERTS": "352", "ROUTER_TOPK": "8",
           "REPLICA_SCHEDULER": "vllm_v1", "NUM_REQUESTS": "1", "PREFILL_TOKENS": "4105",
           "DECODE_TOKENS": "9", "QPS": "1", "ENABLE_DUMMY_MODE": "true",
           "DUMMY_EXEC_TIME_MS": "0.25", "ENABLE_CHUNKED_PREFILL": "true",
           "MAX_TOKENS_IN_BATCH": "512", "LONG_PREFILL_TOKEN_THRESHOLD": "512",
           "MOE_ROUTING_DISTRIBUTION_TYPE": "balanced", "MOE_ROUTING_SEED": "42",
           "FRONTIER_LOG_LEVEL": "warning", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "TMPDIR": "/data/ycfeng/tmp", "FRONTIER_TMP_ROOT": "/data/ycfeng/tmp"}
    flags = ("--log_level", "warning", "--seed", "42", "--no-enable_parallel_clusters",
             "--replica_config_device", "h800", "--replica_config_network_device", "h800_dgx",
             "--replica_config_attn_dp", "1", "--replica_config_attention_backend", "STEP5_DSA",
             "--replica_config_kv_cache_dtype", "auto", "--no-speculative_decoding_config_enabled",
             "--no-vllm_v1_scheduler_config_enable_prefix_caching", "--vllm_v1_scheduler_config_block_size", "64",
             "--vllm_v1_scheduler_config_num_blocks", "0", "--vllm_v1_scheduler_config_num_blocks_mode", "memory_planner",
             "--vllm_v1_scheduler_config_max_model_len", "8192", "--vllm_v1_scheduler_config_batch_size_cap", "1",
             "--no-vllm_v1_scheduler_config_enable_runtime_non_kv_cache_overhead_profiling",
             "--metrics_config_store_operation_metrics", "--metrics_config_keep_individual_batch_metrics",
             "--metrics_config_store_frontier_stage_batch_ledger", "--metrics_config_store_frontier_stage_batch_ledger_summary",
             "--metrics_config_enable_op_level_tracing", "--metrics_config_enable_per_layer_expansion",
             "--metrics_config_num_requests_to_trace_per_layer", "1", "--metrics_config_enable_metrics_ground_truth_trace")
    return [FidelityCase(
        case_id=name, group="step5_source_structural",
        script=COLOCATION_OFFLINE_MOE if mode == "offline" else COLOCATION_ONLINE_MOE,
        purpose="Step source_structural complete model scopes with inclusive sampling and exclusive cost owners",
        env={**env, "DECODE_CUDA_GRAPH_MODE": graph,
             "FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH":
             str(output_root / "cases" / name / "schedule_decisions.jsonl")},
        extra_args=flags + ("--metrics_config_cache_dir", str(output_root / "cases" / name / "cache"))
        + (() if graph == "none" else ("--cudagraph_capture_sizes", "8")),
    ) for name, (mode, graph) in CASES.items()]


def snapshot():
    inputs = ("frontier", "data/config", COLOCATION_OFFLINE_MOE, COLOCATION_ONLINE_MOE,
              "tests/e2e/refactor_fidelity", "tests/integration/run_step5_structural.py")
    names = _git(REPO, "ls-files", "--cached", "--others", "--exclude-standard", "--", *inputs).splitlines()
    return {**source_provenance(REPO), "files_sha256": {
        name: file_digest(REPO / name) for name in sorted(set(names))
        if (REPO / name).is_file() and Path(name).suffix in {".py", ".json", ".yaml", ".yml", ".sh"}
    }, "python": {"executable": str(Path(sys.executable).resolve()), "version": sys.version,
                  "sha256": file_digest(Path(sys.executable).resolve())}}


def source_changes(before, after):
    return [field for field in ("source_revision", "harness_revision", "files_sha256", "python")
            if before.get(field) != after.get(field)]


def check_config(config, case):
    mode, graph = CASES[case]
    for key, value in {"simulation_mode": mode, "sys_arch": "co-location", "enable_parallel_clusters": False,
                       "decode_cuda_graph_mode": graph, "use_cuda_graph": False,
                       "cudagraph_capture_sizes": None if graph == "none" else [8]}.items():
        require(config[key] == value, f"wrong configuration: {key}")
    cluster = config["cluster_config"]
    replica, scheduler = cluster["replica_config"], cluster["replica_scheduler_config"]
    require(cluster["num_replicas"] == 1 and cluster["cc_backend_config"]["name"] == "analytical", "wrong cluster")
    for key, value in {"model_name": "stepfun-ai/Step-5-Preview", "device": "h800", "network_device": "h800_dgx",
                       "attn_tensor_parallel_size": 16, "attn_dp": 1, "moe_tensor_parallel_size": 16,
                       "moe_expert_parallel_size": 1, "num_pipeline_stages": 1, "world_size": 16,
                       "attention_backend": "STEP5_DSA", "kv_cache_dtype": "auto",
                       "total_expert_num": 352, "router_topk": 8}.items():
        require(replica[key] == value, f"wrong replica configuration: {key}")
    require(replica.get("indexer_kv_cache_dtype") is None, "wrong replica configuration: indexer_kv_cache_dtype")
    require(not replica["speculative_decoding_config"]["enabled"], "MTP must be off")
    model = replica["model_config"]
    require(model["model_type"] == "step5" and model["num_layers"] == 92 and model["num_experts"] == 352,
            "incomplete Step model representation")
    require(set(map(int, model["moe_layers_enum"].split(","))) == set(range(3, 91)), "wrong FFN schedule")
    require(Counter(model["layer_types"][:92]) == {"sliding_attention": 69, "full_attention": 23}, "wrong attention schedule")
    require(model["num_nextn_predict_layers"] == 3, "declared MTP inventory lost")
    for key, value in {"name": "vllm_v1", "block_size": 64, "num_blocks": 0, "num_blocks_mode": "memory_planner",
                       "max_model_len": 8192, "batch_size_cap": 1, "max_tokens_in_batch": 512,
                       "long_prefill_token_threshold": 512, "enable_chunked_prefill": True,
                       "enable_prefix_caching": False, "enable_runtime_non_kv_cache_overhead_profiling": False}.items():
        require(scheduler[key] == value, f"wrong scheduler configuration: {key}")
    predictor = cluster["execution_time_predictor_config"]
    require(predictor["enable_dummy_mode"] and predictor["dummy_execution_time_ms"] == .25, "wrong synthetic timing")
    generator = config["request_generator_config"]
    length = generator["length_generator_config"]
    require(generator["num_requests"] == 1 and (length["prefill_tokens"], length["decode_tokens"]) == (4105, 9),
            "wrong controlled request")


def attention_names(layer, fused):
    if layer % 4 != 3:
        return ["step5_qkv_projection", "step5_attention_prepare", "step5_swa_attention",
                "step5_gate_projection", "step5_headwise_gate", "step5_output_projection"]
    return (["step5_qkv_projection", "step5_indexer_weights", "step5_attention_prepare"] if fused else
            ["step5_qkv_projection", "step5_attention_prepare", "step5_indexer_weights", "step5_indexer_prepare"]) + [
                "step5_sparse_attention", "step5_output_projection"]


def stage_sequence(fused):
    sequence = [(name, -1, .25) for name in OVERHEAD[:2] + STAGE_MODEL_OPS[:3]]
    for layer in range(92):
        names = ["input_layernorm", *attention_names(layer, fused), "attn_tensor_parallel_allreduce", "post_attention_layernorm"]
        names += (["share_expert_up_proj", "share_expert_act", "share_expert_down_proj", "moe_gating_linear",
                   "moe_gating_routing_topk", "moe_shuffling", "moe_grouped_gemm", "step5_ffn_merge",
                   "moe_tensor_parallel_allreduce"] if 3 <= layer <= 90 else
                  ["mlp_up_proj", "mlp_act", "mlp_down_proj", "mlp_tensor_parallel_allreduce"])
        sequence.extend((name, layer, .125 if name in ("moe_gating_linear", "moe_gating_routing_topk") else .25) for name in names)
    return sequence + [(name, -1, .25) for name in STAGE_MODEL_OPS[3:] + OVERHEAD[2:]]


def check_stage(row, query, prefill):
    require(row["cluster_type"] == "MONOLITHIC" and row["stage_id"] == row["replica_id"] == row["replica_local_id"] == 0,
            "wrong stage placement")
    require(row["execution_scope"] == "ATTN_DP_LANE", "wrong stage execution scope")
    require(row["request_ids"] == ["0"] and row["request_num_tokens"] == [query]
            and row["request_num_prefill_tokens"] == [prefill], "missing tokens or padding credited as request progress")
    start, end = row["stage_start_ts"], row["stage_end_ts"]
    require(math.isfinite(start) and math.isfinite(end) and 0 <= start < end, "invalid stage boundaries")



def check_stage_cost_span(stage, total_ms):
    """Compare actual request-stage time with the joined exclusive cost owners."""
    actual_ms = (stage["stage_end_ts"] - stage["stage_start_ts"]) * 1000
    require(math.isclose(actual_ms, total_ms, rel_tol=0, abs_tol=1e-7),
            "actual stage duration differs from the joined stage/wave costs")


def check_tensor(row, tokens, routed_tokens, fused, *, wave=False):
    name, meta = row["name"], row["meta"]
    layer = trace_layer(row)
    require(type(layer) is int and 0 <= layer < 92 and meta["source_geometry"]["global_layer_id"] == layer,
            "missing or inconsistent layer identity")
    require(meta["evidence_kind"] == "source_structural_metadata", "missing source_structural qualification")
    require(meta["attention_family_id"] == "step5_attention" and meta["attention_variant_id"] ==
            ("sparse_gqa" if layer % 4 == 3 else "swa"), "wrong layer attention binding")
    require(meta["num_layers"] == (1 if wave else 92), "wrong trace layer coverage")
    compute_tokens = routed_tokens if wave and name == "moe_grouped_gemm" else tokens
    require(meta["effective_total_tokens_compute"] == compute_tokens, "wrong physical/valid token attribution")
    if wave:
        require(meta["tokens_are_post_routing"] == (name == "moe_grouped_gemm"), "wrong routed tensor attribution")
    else:
        require(meta["attention_operator_names"] == attention_names(layer, fused), "wrong attention scope/fusion metadata")
        require(meta["request_ids"] == ["0"] and meta["total_tokens"] == sum(meta["num_tokens"]), "wrong trace request")
        require(meta["effective_total_tokens_rounded"] == tokens and
                meta["effective_total_tokens_transfer"] == meta["total_tokens"], "wrong physical/valid token attribution")
    require(meta["parallel_context"] == {"PP": 1, "ATTN_TP": 16, "ATTN_DP": 1, "MOE_TP": 16, "MOE_EP": 1, "ROUTER_TOPK": 8},
            "wrong trace parallel shape")
    shapes, dtypes = meta["tensor_shape"], meta["tensor_dtype"]
    if row["type"] == "COMM":
        dtype = "FP32" if name == "moe_tensor_parallel_allreduce" else "BF16"
        require(shapes == {"data": [tokens, 4096]} and dtypes == {"data": dtype}, "wrong TP reduction tensor shape/precision")
        require(meta["data_size_bytes"] == tokens * 4096 * (4 if dtype == "FP32" else 2), "wrong TP payload cost")
        return
    sizes = meta["tensor_size_bytes"]
    require(set(shapes) == set(dtypes) == set(sizes), "missing tensor size/dtype metadata")
    for key, shape in shapes.items():
        require(all(type(n) is int and n >= 0 for n in shape) and sizes[key] == math.prod(shape) *
                {"BF16": 2, "FP32": 4, "FP8": 1, "INT32": 4, "INT64": 8}[dtypes[key]], "tensor byte conservation differs")
    modules = meta.get("projection_modules", [])
    expected_weights = {"step5_qkv_projection": [[2692 if layer % 4 == 3 else 1152, 4096]],
                        "step5_gate_projection": [[4, 4096]], "step5_indexer_weights": [[4, 4096]],
                        "step5_output_projection": [[4096, 768]], "mlp_up_proj": [[1728, 4096]],
                        "mlp_down_proj": [[4096, 864]], "share_expert_up_proj": [[192, 4096]],
                        "share_expert_down_proj": [[4096, 96]], "moe_gating_linear": [[352, 4096]],
                        "moe_grouped_gemm": [[352, 192, 4096], [352, 4096, 96]]}
    if name in expected_weights:
        require([module["weight_shape"] for module in modules] == expected_weights[name], "wrong per-rank projection weight shape")
        dtype = "FP32" if name == "moe_gating_linear" else "FP8" if name == "moe_grouped_gemm" and layer < 88 else "BF16"
        require(all(module["weight_dtype"] == dtype for module in modules), "wrong per-layer FP8/BF16 weight precision")
    if name == "step5_ffn_merge":
        require(shapes == {key: [tokens, 4096] for key in ("routed", "shared", "output")} and
                dtypes == {"routed": "BF16", "shared": "BF16", "output": "FP32"}, "wrong FFN merge shape/precision")
    elif name == "moe_grouped_gemm":
        require(shapes == {"input": [tokens, 4096], "output": [tokens, 4096]}, "wrong routed branch boundary")
        require(meta["internal_tensor_shape"] == {"gate_up": [tokens, 8, 192],
                "activation": [routed_tokens, 96], "down": [tokens, 8, 4096]},
                "wrong internal routed assignment shape")
        require(meta["covered_scopes"] == ["gate_up_gemm", "swiglu", "down_gemm", "topk_reduce"],
                "missing routed reduction owner")
    elif name == "moe_shuffling":
        require(shapes == {"expert_ids": [tokens, 8]} and dtypes == {"expert_ids": "INT32"} and
                meta["covered_scopes"] == ["expert_assignment_alignment"], "wrong alignment boundary")
        require(meta["alignment_outputs"] == ["sorted_token_ids", "expert_ids", "num_tokens_post_padded"] and
                meta["alignment_output_shapes"] == "unresolved native dispatch and alignment padding" and
                meta["sorted_token_ids_nullable"] is True, "alignment padding/dispatch falsely resolved")
    elif name in ("input_layernorm", "post_attention_layernorm"):
        input_dtype = "FP32" if name == "input_layernorm" and (layer == 0 or 4 <= layer <= 91) else "BF16"
        require(shapes["input"] == shapes["output"] == [tokens, 4096] and dtypes["input"] == input_dtype
                and dtypes["output"] == "BF16" and meta["internal_tensor_dtype"]["norm_weight"] == "FP32",
                "wrong norm input/output precision")
    elif name in ("step5_attention_prepare", "step5_indexer_prepare", "step5_sparse_attention", "step5_swa_attention"):
        has_index = name == "step5_indexer_prepare" or (name == "step5_attention_prepare" and layer % 4 == 3 and fused)
        require(("index_q" in shapes) == has_index, "wrong fused indexer tensor boundary")
        if has_index:
            require(shapes["index_q"] == [tokens, 4, 256] and meta["internal_tensor_dtype"] ==
                    {"quantized_index_q": "FP8", "scaled_weights": "FP32"}, "wrong indexer mixed precision")
        if name != "step5_indexer_prepare":
            require(shapes.get("q") == [tokens, 4, 192] and shapes.get("k") == shapes.get("v") == [tokens, 1, 192],
                    "wrong replicated Q/K/V shape")


def check_stage_scope(row, physical, query):
    """Audit once-only model boundaries and the existing inclusive sampler."""
    name, meta = row["name"], row["meta"]
    require(row["layer_id"] == trace_layer(row) == -1 and meta["scope"] == "stage",
            "stage scope has a layer owner")
    require(meta["evidence_kind"] == "source_structural_metadata", "missing stage source_structural qualification")
    require(meta["num_layers"] == 92 and meta["request_ids"] == ["0"] and
            meta["num_tokens"] == [query] and meta["total_tokens"] == query and
            meta["effective_total_tokens_compute"] == meta["effective_total_tokens_rounded"] == physical and
            meta["effective_total_tokens_transfer"] == query, "wrong stage physical/valid token attribution")
    require(meta["parallel_context"] == {"PP": 1, "ATTN_TP": 16, "ATTN_DP": 1,
            "MOE_TP": 16, "MOE_EP": 1, "ROUTER_TOPK": 8}, "wrong stage parallel shape")
    if name == "sampler_e2e":
        require(row["type"] == "OVERHEAD" and meta["time_ownership"] == "inclusive_sampler_e2e",
                "wrong sampler inclusive cost ownership")
        require(meta["covered_scopes"] == ["select_hidden_states", "logits_input_cast", "final_norm",
                "lm_head_linear", "logits_gather", "sampling"], "missing/duplicate/out-of-order sampler scope")
        require(meta["logits_rows"] == len(meta["request_ids"]) == 1 and meta["tensor_shape"] ==
                {"selected_hidden": [1, 4096], "local_logits": [1, 8056]}, "sampler uses wrong logits rows or rank shape")
        require(meta["model_precision"] == "BF16" and meta["norm_weight_precision"] == "FP32",
                "wrong sampler model/norm precision")
        return
    require(name in STAGE_MODEL_OPS and row["type"] == ("COMM" if "allreduce" in name else "COMPUTE"),
            "wrong stage model scope type")
    require(meta["dtype"] == "BF16" and meta["dtype_bytes"] == 2 and
            meta["precision_op"] == ("emb" if "allreduce" in name else name), "wrong stage model precision")
    shapes, dtypes = meta["tensor_shape"], meta["tensor_dtype"]
    if name == "embedding_tensor_parallel_allreduce":
        require(shapes == {"data": [physical, 4096]} and dtypes == {"data": "BF16"},
                "wrong stage embedding reduction shape/precision")
        require(meta["data_size_bytes"] == meta["base_size_bytes"] == physical * 4096 * 2 and
                meta["element_count"] == physical * 4096, "wrong stage embedding reduction payload")
        return
    expected = {
        "emb": ({"input_ids": [physical], "weight": [8056, 4096], "output": [physical, 4096]},
                {"input_ids": "INT32", "weight": "BF16", "output": "BF16"}),
        "input_cast": ({"input": [physical, 4096], "output": [physical, 4096]},
                       {"input": "BF16", "output": "FP32"}),
        "final_residual": ({"input": [physical, 4096], "residual": [physical, 4096], "output": [physical, 4096]},
                           {"input": "BF16", "residual": "FP32", "output": "FP32"}),
    }
    require((shapes, dtypes) == expected[name], "wrong stage model shape/precision")
    require(meta["tensor_size_bytes"] == {key: math.prod(shape) * {"INT32": 4, "BF16": 2, "FP32": 4}[dtypes[key]]
            for key, shape in shapes.items()}, "stage tensor byte conservation differs")


def check_costs(timing, expected):
    field_names = {"input_layernorm": "attn_norm_time", "post_attention_layernorm": "mlp_norm_time",
                   "attn_tensor_parallel_allreduce": "attention_all_reduce_time",
                   "mlp_tensor_parallel_allreduce": "mlp_all_reduce_time", "moe_tensor_parallel_allreduce": "mlp_all_reduce_time",
                   "mlp_up_proj": "mlp_layer_up_proj_execution_time", "mlp_act": "mlp_layer_act_execution_time",
                   "mlp_down_proj": "mlp_layer_down_proj_execution_time"}
    costs = Counter()
    for name, _, cost in expected:
        field = name if name.startswith("step5_") or name in STAGE_MODEL_OPS or name == "ray_comm_time" else name + "_time"
        costs[field_names.get(name, field)] += cost
    ledger = timing["component_ledger_ms"]
    require(all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in ledger.values()),
            "invalid stage component cost")
    require({name: cost for name, cost in ledger.items() if cost} == dict(costs), "stage component ownership differs")
    total = sum(costs.values())
    require(timing["total_time_ms"] == sum(ledger.values()) == total and timing["model_time_ms"] == total - 1.25,
            "stage cost conservation or overhead ownership differs")
    require(timing["diagnostic_total_time_ms"] == total and timing["diagnostic_component_ledger_ms"] ==
            {"pp_stage_boundary_handoff_time": 0} and ledger["pipeline_parallel_communication_time"] == 0,
            "PP1/diagnostic cost differs")


def wave_sequence(layer):
    """Independent native FFN order inside the existing EP phase owners."""
    pre = ("post_attention_layernorm", "share_expert_up_proj", "share_expert_act", "share_expert_down_proj",
           "moe_gating_linear", "moe_gating_routing_topk", "moe_shuffling")
    return {"pre_dispatch": [(name, layer, .125 if name in pre[4:6] else .25) for name in pre],
            "dispatch": [], "routed_compute": [("moe_grouped_gemm", layer, .25)], "combine": [],
            "post_combine": [("step5_ffn_merge", layer, .25), ("moe_tensor_parallel_allreduce", layer, .25)]}


def trace_layer(row):
    meta = row["meta"]
    ids = meta.get("global_layer_ids", [])
    if "global_layer_id" in meta:
        layer = meta["global_layer_id"]
        require("global_layer_ids" not in meta or ids == [layer], "inconsistent layer identity")
        return layer
    require(isinstance(ids, list) and len(ids) <= 1, "inconsistent layer identity")
    return ids[0] if ids else -1


def check_scope_order(stage_events, wave_events, *, fused):
    wave_expected = [entry for layer in range(3, 91) for phase in wave_sequence(layer).values() for entry in phase]
    wave_keys = {(name, layer) for name, layer, _ in wave_expected}
    stage_expected = [entry for entry in stage_sequence(fused) if entry[:2] not in wave_keys]
    for actual, expected in ((stage_events, stage_expected), (wave_events, wave_expected)):
        require([(row["name"], trace_layer(row), row["duration_ms"]) for row in actual] == expected,
                "missing/duplicate/out-of-order scope or wrong cost")
    return stage_expected


def check_wave_costs(row, layer):
    phases = row["phases"]
    expected = wave_sequence(layer)
    require(set(phases) == set(expected), "wave phase inventory differs")
    cursor = row["wave_start_time_s"]
    for name, operators in expected.items():
        phase = phases[name]
        require({op: cost for op, cost in phase["operators_ms"].items() if cost} ==
                {op: cost for op, _, cost in operators}, "wave phase operator ownership differs")
        require(phase["duration_ms"] == sum(phase["operators_ms"].values()) == sum(cost for _, _, cost in operators),
                "wave phase cost conservation differs")
        require(math.isclose(phase["start_time_s"], cursor, rel_tol=0, abs_tol=1e-9), "wave phase boundary differs")
        cursor += phase["duration_ms"] / 1000
    require(math.isclose(cursor, row["wave_end_time_s"], rel_tol=0, abs_tol=1e-9), "wave phase end differs")


def join_wave_evidence(events, stages, waves):
    """Join synthetic EP batches to their authoritative source stage."""
    stage_events, wave_events, wave_rows = ({row["batch_id"]: [] for row in stages} for _ in range(3))
    headers = [row for row in events if "type" not in row]
    require(len(headers) == 1 and "simulation_info" in headers[0].get("meta", {}), "missing trace source header")

    def source_id(row):
        sources = row.get("source_batch_ids", row.get("meta", {}).get("source_batch_ids"))
        require(isinstance(sources, list) and len(sources) == 1 and sources[0] in stage_events,
                "unattributed wave source or unsupported multiple source lanes")
        require(row["batch_id"] not in stage_events, "wave synthetic batch duplicates a source stage")
        return sources[0]

    for row in events:
        if "type" not in row:
            continue
        if "source_batch_ids" in row.get("meta", {}):
            wave_events[source_id(row)].append(row)
        else:
            require(row["batch_id"] in stage_events, "unattributed trace batch")
            stage_events[row["batch_id"]].append(row)
    for row in waves:
        wave_rows[source_id(row)].append(row)
    return stage_events, wave_events, wave_rows


def check_traces(events, stages, waves, case):
    stage_events, wave_events, wave_rows = join_wave_evidence(events, stages, waves)
    stats = Counter()
    for index, stage in enumerate(stages):
        batch_id = stage["batch_id"]
        query, prefill = (512, 512) if index < 8 else (9, 9) if index == 8 else (1, 0)
        graph = CASES[case][1]
        physical = 8 if not prefill and graph != "none" else query
        fused = query == 1 and graph != "piecewise"
        expected = stage_sequence(fused)
        actual_stage, actual_waves = stage_events[batch_id], wave_events[batch_id]
        owned_expected = check_scope_order(actual_stage, actual_waves, fused=fused)
        rows = wave_rows[batch_id]
        require([row["global_layer_ids"] for row in rows] == [[layer] for layer in range(3, 91)] and
                len({row["batch_id"] for row in rows}) == 88, "missing/duplicate/out-of-order layer wave")
        cursor = stage["stage_start_ts"]
        wave_starts, wave_ends = {}, {}
        for name, layer, cost in expected:
            if name == "post_attention_layernorm" and 3 <= layer <= 90:
                wave_starts[layer] = cursor
            cursor += cost / 1000
            if name == "moe_tensor_parallel_allreduce":
                wave_ends[layer] = cursor
        require(math.isclose(cursor, stage["stage_end_ts"], abs_tol=1e-9, rel_tol=0), "full stage boundary conservation differs")
        cursor = stage["stage_start_ts"]
        for row in actual_stage:
            require(math.isclose(row["ts_start"], cursor, abs_tol=1e-9, rel_tol=0), "stage owned trace cursor differs")
            cursor += row["duration_ms"] / 1000
        events_by_layer = {}
        for event in actual_waves:
            events_by_layer.setdefault(trace_layer(event), []).append(event)
        for row in rows:
            layer = row["global_layer_ids"][0]
            require(row["record_type"] == "ep_wave_lane" and row["cluster"] == "MONOLITHIC" and
                    row["replica_id"] == row["stage_id"] == row["ep_id"] == 0, "wrong wave placement")
            check_wave_costs(row, layer)
            require(math.isclose(row["wave_start_time_s"], wave_starts[layer], abs_tol=1e-9, rel_tol=0) and
                    math.isclose(row["wave_end_time_s"], wave_ends[layer], abs_tol=1e-9, rel_tol=0), "wave/stage ordering differs")
            layer_events = iter(events_by_layer[layer])
            for phase, operators in wave_sequence(layer).items():
                cursor = row["phases"][phase]["start_time_s"]
                for name, _, _ in operators:
                    event = next(layer_events)
                    require(event["batch_id"] == row["batch_id"] and event["meta"]["phase"] == phase and
                            math.isclose(event["ts_start"], cursor, abs_tol=1e-9, rel_tol=0), "wave trace ownership/boundary differs")
                    cursor += event["duration_ms"] / 1000
        for is_wave, actual in ((False, actual_stage), (True, actual_waves)):
            for row in actual:
                name, layer = row["name"], trace_layer(row)
                require(row["cluster"] == "MONOLITHIC" and row["replica_id"] == 0 and row["layer_id"] in (layer, -1),
                        "wrong trace placement")
                require(row["type"] == ("OVERHEAD" if name in OVERHEAD else "COMM" if "allreduce" in name else "COMPUTE"), "wrong scope type")
                if layer >= 0:
                    if not is_wave:
                        require(row["meta"]["num_tokens"] == [query], "padding credited to trace request progress")
                    check_tensor(row, physical, physical * 8, fused, wave=is_wave)
                elif name in STAGE_MODEL_OPS or name == "sampler_e2e":
                    check_stage_scope(row, physical, query)
        timing = stage["execution_time"]
        check_costs(timing, owned_expected)
        stage_ms = sum(row["duration_ms"] for row in actual_stage)
        wave_ms = sum(phase["duration_ms"] for row in rows for phase in row["phases"].values())
        total_ms = sum(cost for _, _, cost in expected)
        check_stage_cost_span(stage, total_ms)
        require(timing["total_time_ms"] == stage_ms and wave_ms == 198 and
                stage_ms + wave_ms == total_ms == sum(row["duration_ms"] for row in actual_stage + actual_waves),
                "joined stage/wave ledger/trace cost conservation differs")
        stats.update({"trace_events": len(actual_stage) + len(actual_waves), "full_layer_executions": 92,
                      "ffn_merges": 88, "ep_wave_rows": 88, "backbone_ms": total_ms - 2.25,
                      "stage_model_scope_ms": 1.0, "host_overhead_ms": 1.0, "inclusive_sampler_ms": .25,
                      "controlled_total_ms": total_ms, "stage_trace_ms": stage_ms, "ep_wave_ms": wave_ms,
                      "valid_prefill_tokens": prefill, "valid_decode_inputs": 0 if prefill else query,
                      "padded_compute_rows": physical - query, "padded_decode_batches": int(physical != query)})
    return dict(stats)


def check_cache_lifetime(events, stages, completion):
    """Join actual scheduler state to valid inputs and request completion."""
    wanted = {"grouped_cache_allocation", "grouped_cache_batch_end", "grouped_cache_release"}
    rows = [event for event in events if event["event"] in wanted]
    require([row["event"] for row in rows] ==
            [name for _ in stages for name in ("grouped_cache_allocation", "grouped_cache_batch_end")]
            + ["grouped_cache_release"], "missing/duplicate/out-of-order cache lifecycle evidence")
    request_id = str(completion["request_id"])
    require(all(stage["request_ids"] == [request_id] for stage in stages), "cache request/stage identity differs")
    groups = ("swa_0", "swa_1", "swa_2", "sparse")
    previous_tables = {group: [] for group in groups}
    previous_frontier = 0
    for index, stage in enumerate(stages):
        allocation, ended = rows[2 * index:2 * index + 2]
        query = stage["request_num_tokens"][0]
        frontier = previous_frontier + query
        output_tokens = max(0, index - 7)
        for event, event_time in ((allocation, stage["stage_start_ts"]), (ended, stage["stage_end_ts"])):
            require(event["source"] == "frontier" and event["scheduler"] == "vllm_v1" and
                    event["cluster_type"] == "MONOLITHIC" and event["request_id"] == request_id and
                    event["grouped_cache_schema_version"] == 1 and
                    event["evidence_kind"] == "source_structural_state", "wrong cache state identity/evidence")
            require(math.isclose(event["simulation_time"], event_time, rel_tol=0, abs_tol=1e-9),
                    "cache state uses wrong scheduling/completion time")
            state = event["grouped_cache_state"]
            require(state["pool_num_blocks"] == 29967 and state["pool_reserved_blocks"] == 1 and
                    state["physical_block_size_bytes"] == 1224980 and
                    state["fixed_cache_reservation_bytes"] == 94484, "wrong pool/fixed scratch reservation")
            tables = state["group_block_ids"]
            require(state["has_grouped_allocation"] and set(tables) == set(groups), "missing request group ownership")
            page_count = math.ceil(frontier / 64)
            retired = max(0, previous_frontier - 511) // 64
            require(all(len(table) == page_count for table in tables.values()), "group frontier counts padding/output bonus")
            for group in groups[:-1]:
                require(tables[group][:retired] == [None] * retired and
                        all(block is not None for block in tables[group][retired:]),
                        "SWA pages not retired against settled progress")
            require(all(block is not None for block in tables["sparse"]) and
                    tables["sparse"][:len(previous_tables["sparse"])] == previous_tables["sparse"],
                    "sparse historical state released or replaced")
            assigned = [block for table in tables.values() for block in table if block is not None]
            require(len(assigned) == len(set(assigned)) and 0 not in assigned and
                    state["block_ref_counts"] == {str(block): 1 for block in assigned},
                    "duplicate/reserved/dangling cache owner")
            require(state["request_allocated_blocks"] == state["scheduler_allocated_blocks"] ==
                    state["pool_used_blocks"] == len(assigned) and
                    state["pool_used_blocks"] + state["pool_free_blocks"] + 1 == 29967 and
                    event["available_blocks"] == state["pool_free_blocks"], "cache pool conservation differs")
        initial = allocation["grouped_cache_state"]
        final = ended["grouped_cache_state"]
        require(initial["group_block_ids"] == final["group_block_ids"] and
                allocation["num_tokens"] == query and ended["batch_id"] == stage["batch_id"],
                "cache batch/valid query join differs")
        require(initial["settled_num_computed_tokens"] == previous_frontier and
                initial["scheduled_num_computed_tokens"] == (None if index == 0 else previous_frontier) and
                initial["reclamation_num_computed_tokens"] == (None if index == 0 else previous_frontier),
                "allocation uses optimistic or padded KV progress")
        require(final["settled_num_computed_tokens"] == final["scheduled_num_computed_tokens"] == frontier and
                final["num_processed_tokens"] == frontier + (1 if index >= 8 else 0) and
                final["num_emitted_decode_tokens"] == output_tokens and
                final["completed"] == (index == 16), "completed KV/output frontier differs")
        require(initial["num_emitted_decode_tokens"] == max(0, index - 8) and
                initial["num_processed_tokens"] == previous_frontier + (1 if index > 8 else 0),
                "allocation counts output bonus as a KV input")
        new_ids = {group: [block for block in table[len(previous_tables[group]):] if block is not None]
                   for group, table in initial["group_block_ids"].items()}
        require(allocation["group_new_block_ids"] == new_ids and allocation["zero_required_block_ids"] == [],
                "group allocation/zeroing evidence differs")
        previous_tables = final["group_block_ids"]
        previous_frontier = frontier
    released = rows[-1]
    state = released["grouped_cache_state"]
    require(released["request_id"] == request_id and released["batch_id"] == stages[-1]["batch_id"] and
            math.isclose(released["simulation_time"], completion["completed_at"], rel_tol=0, abs_tol=1e-9),
            "release does not join actual completion")
    require(state["completed"] and state["settled_num_computed_tokens"] == 4113 and
            state["num_processed_tokens"] == 4114 and state["num_emitted_decode_tokens"] == 9 and
            state["scheduled_num_computed_tokens"] is None and state["reclamation_num_computed_tokens"] is None,
            "completed scheduler/KV state not released")
    require(state["group_block_ids"] == state["block_ref_counts"] == {} and not state["has_grouped_allocation"] and
            state["request_allocated_blocks"] == state["scheduler_allocated_blocks"] == state["pool_used_blocks"] == 0 and
            state["pool_num_blocks"] == 29967 and state["pool_free_blocks"] == released["available_blocks"] == 29966 and
            state["pool_reserved_blocks"] == 1 and state["physical_block_size_bytes"] == 1224980 and
            state["fixed_cache_reservation_bytes"] == 94484, "request release/fixed scratch conservation differs")
    return {"cache_state_events": len(rows), "kv_input_tokens": previous_frontier,
            "pool_num_blocks": 29967, "released_free_blocks": 29966, "fixed_cache_reservation_bytes": 94484}


def validate(output, case, log):
    config = json.loads((output / "config.json").read_text())
    check_config(config, case)
    system = json.loads((output / "system_metrics.json").read_text())
    require(system["simulation_metadata"] == {"total_requests": 1, "completed_requests": 1,
            "system_architecture": "co-location", "server_count": 2}, "request completion/topology differs")
    require(system["preemption_statistics"]["total_preemption_events"] == 0, "unexpected request preemption")
    memory = system["model_weight_memory"]["MONOLITHIC"]
    require(memory["total_memory_bytes"] == WEIGHT_BYTES and memory["total_parameters"] == PARAMETERS and
            all(memory["breakdown"][key] == value for key, value in WEIGHT_PARTS.items()), "source storage oracle differs")
    require(memory["memory_accounting"] == "source-defined weight storage estimate; native allocation unverified", "wrong memory evidence label")
    requests = read_rows(output / "metrics_ground_truth.jsonl")
    arrivals = [row for row in requests if row["event_type"] == "request_arrival"]
    completions = [row for row in requests if row["event_type"] == "request_completion"]
    require(len(arrivals) == len(completions) == 1 and arrivals[0]["request_id"] == completions[0]["request_id"] == 0, "missing/duplicate request completion")
    completion = completions[0]
    require((completion["num_prefill_tokens"], completion["num_decode_tokens"]) == (4105, 9) and
            0 <= completion["arrived_at"] < completion["prefill_completed_at"] < completion["completed_at"], "wrong completed token counts/boundaries")
    require(completion["arrived_at"] == arrivals[0]["arrived_at"] and
            (completion["arrived_at"] == 0 if CASES[case][0] == "offline" else completion["arrived_at"] > 0), "wrong offline/online arrival")
    stages = read_rows(output / "frontier_stage_batch_ledger.jsonl")
    require(len(stages) == 17 and len({row["batch_id"] for row in stages}) == 17, "missing/duplicate stage batch")
    stages.sort(key=lambda row: row["stage_start_ts"])
    for index, row in enumerate(stages):
        check_stage(row, 512 if index < 8 else 9 if index == 8 else 1, 512 if index < 8 else 9 if index == 8 else 0)
        require(index == 0 or stages[index - 1]["stage_end_ts"] <= row["stage_start_ts"] + 1e-9, "overlapping request stages")
    require(math.isclose(stages[8]["stage_end_ts"], completion["prefill_completed_at"], abs_tol=1e-9) and
            math.isclose(stages[-1]["stage_end_ts"], completion["completed_at"], abs_tol=1e-9), "stage/token completion join differs")
    require(json.loads((output / "frontier_stage_batch_ledger_summary.json").read_text())["total_rows"] == 17, "incomplete stage summary")
    stats = check_traces(read_rows(output / "op_traces.jsonl"), stages,
                         read_rows(output / "frontier_ep_wave_lane_ledger.jsonl"), case)
    cache_stats = check_cache_lifetime(read_rows(log.parent / "schedule_decisions.jsonl"), stages, completion)
    text = log.read_text()
    require(not any(value in text for value in ("Traceback", "ERROR ", "FRONTIER_MEMORY_OOM", "Error during metrics output")), "runtime/metrics failure in run.log")
    return {"evidence_kind": "source_structural", "execution_coverage": "complete_model_scopes_with_inclusive_sampling",
            "completed_requests": 1, "output_tokens": 9, "stage_batches": 17,
            "weight_bytes_per_rank": WEIGHT_BYTES, "total_parameters_per_rank": PARAMETERS, **stats, **cache_stats}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="Fresh destination; existing directories are rejected")
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    if root.exists():
        parser.error(f"output root already exists; preserve it and choose a new path: {root}")
    require(Path(sys.executable).resolve() == Path(PYTHON).resolve() and sys.version_info[:3] == (3, 10, 6), "use the approved Python 3.10.6 runtime")
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for case in build_cases(root):
        if args.case and case.case_id not in args.case:
            continue
        before = snapshot()
        outcome = _run_case(case, REPO, root, PYTHON, 300)
        after = snapshot()
        record = {**outcome.as_record(), "evidence_kind": "source_structural", "limits": LIMITS,
                  "source_before": before, "source_after": after, "source_changed_fields": source_changes(before, after), "status": "FAIL"}
        try:
            require(outcome.returncode == 0 and outcome.error is None, f"CLI failed: {outcome.error or outcome.log_tail}")
            record["validation"] = validate(root / outcome.artifact_dir, case.case_id, root / "cases" / case.case_id / "run.log")
            record["status"] = "UNVERIFIED_SOURCE_CHANGED" if record["source_changed_fields"] else "PASS"
        except (ValueError, KeyError, OSError, TypeError) as error:
            record["validation_error"] = f"{type(error).__name__}: {error}"
        write_json(root / "cases" / case.case_id / "structural_result.json", record)
        results.append(record)
        print(f"{case.case_id}: {record['status']} {record.get('validation_error', '')}", flush=True)
    write_json(root / "summary.json", {"evidence_kind": "source_structural", "limits": LIMITS, "cases": results})
    return 0 if all(record["status"] == "PASS" for record in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
