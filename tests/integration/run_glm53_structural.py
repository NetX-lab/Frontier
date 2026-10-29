"""Run three manual CPU GLM structural checks through the real Frontier CLI.

These checks use synthetic operation durations and source-defined memory
estimates. They do not establish native timing, CUDA graph capture, hardware
fit, checkpoint loading, or 64-GPU execution.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
CASES = {
    "offline-none": ("offline", "none", 4, 96, 4),
    "online-piecewise": ("online", "piecewise", 2, 64, 2),
    "offline-full_decode_only": ("offline", "full_decode_only", 2, 64, 2),
}
FULL_INDEX = {0, 1, 2, *range(6, 78, 4)}
DUMMY_MS = 0.25
ALL_EP = (1 << 64) - 1
EP_PHASE_COSTS = {
    "pre_dispatch": {
        "add_attn_residual": .125, "dp_input_allreduce": 0,
        "moe_gating_linear": .125, "moe_gating_routing_topk": .125,
        "moe_shuffling": .25, "moe_tensor_parallel_allgather": 0,
        "post_attention_layernorm": .25, "share_expert_up_proj": .25,
        "share_expert_act": .25, "share_expert_down_proj": .25,
        "share_expert_tensor_parallel_allreduce": 0,
    },
    "dispatch": {"expert_parallel_alltoall_dispatch": .25},
    "routed_compute": {"moe_grouped_gemm": .25, "moe_tensor_parallel_allreduce": 0},
    "combine": {"expert_parallel_alltoall_combine": .25},
    "post_combine": {"add_ffn_residual": .125, "dp_output_allreduce": 0},
}
STAGE_COSTS = {
    **dict.fromkeys(("attention_pre_proj_time", "attention_post_proj_time",
                    "attention_rope_execution_time", "attention_all_reduce_time",
                    "attn_norm_time", "attn_dsa_mla"), 78 * DUMMY_MS),
    "attn_dsa_indexer": 21 * DUMMY_MS,
    **dict.fromkeys(("mlp_norm_time", "mlp_all_reduce_time", "mlp_layer_up_proj_execution_time",
                    "mlp_layer_act_execution_time", "mlp_layer_down_proj_execution_time"), 3 * DUMMY_MS),
    **dict.fromkeys(("add_attn_residual_time", "add_ffn_residual_time"), 3 * DUMMY_MS / 2),
    **dict.fromkeys(("pipeline_parallel_communication_time", "schedule_time", "prepare_inputs_e2e_time",
                    "sampler_e2e_time", "process_model_outputs_time", "ray_comm_time"), DUMMY_MS),
    **dict.fromkeys(("attention_prefill_execution_time", "attention_decode_execution_time",
                    "attention_kv_cache_save_execution_time", "attn_mla_decode_time", "attn_mla_prefill_time",
                    "attn_mla_kv_cache_save_time", "attn_mla_v_up_proj_time", "attn_mla_decode_q_latent_proj_time",
                    "attn_mla_prefill_kv_up_proj_time", "dp_input_allreduce_time", "dp_output_allreduce_time",
                    "expert_parallel_communication_time", "moe_gating_linear_time", "moe_gating_routing_topk_time",
                    "moe_grouped_gemm_time", "moe_shuffling_time", "moe_tensor_parallel_allgather_time",
                    "share_expert_up_proj_time", "share_expert_act_time", "share_expert_down_proj_time",
                    "share_expert_tensor_parallel_allreduce_time", "decode_draft_proposer_time",
                    "mtp_terminal_overshoot_time", "pp_prefill_consumer_active_runtime_time",
                    "pp_producer_send_path_runtime_time", "pp_receiver_head_runtime_time",
                    "pp_stage_boundary_residual_runtime_time"), 0),
}
# Independent source oracle also exercised in test_glm53_parameter_memory.py.
# TP8/DP8/EP64 keeps one full shared expert per rank (sequence-parallel MoE).
WEIGHT_BYTES = 18_172_004_352
PARAMETERS = 17_503_822_848
WEIGHT_PARTS = {
    "attention_memory_bytes": 3_214_089_984,
    "ffn_memory_bytes": 14_480_193_792,
    "other_memory_bytes": 477_720_576,
}
LIMITS = [
    "Dummy structural evidence; operation durations are synthetic.",
    "Source-defined weight/cache storage; native allocation and hardware fit unverified.",
    "Graph configuration only; runtime graph activation, capture/replay and padding remain unverified.",
    "EP source IDs identify a representative batch, not every real request lane; real coverage joins cohort completion boundaries.",
    "metrics_ground_truth.jsonl contains simulator request records, not native ground truth.",
    "The existing simulator CPU CSV publisher may emit an empty table; CPU ownership is checked in the stage ledger.",
    "Source snapshots identify HEAD plus local changes; they are not clean-commit qualification.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def snapshot():
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPO)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "frontier", "data/config/models"],
        cwd=REPO, text=True,
    ).splitlines()
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "dirty_tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_runtime_file_sha256": {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in untracked
        },
        "status_porcelain": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO, text=True,
        ).splitlines(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualification": "HEAD and working-tree snapshot; not a clean commit",
    }


def check_source_identity(before, after):
    fields = ("head", "dirty_tracked_diff_sha256", "untracked_runtime_file_sha256", "driver_sha256")
    for field in fields:
        require(field in before and field in after, f"missing source identity: {field}")
        require(before[field] == after[field], f"source changed during execution: {field}")
    for field, size in (("head", 40), ("dirty_tracked_diff_sha256", 64), ("driver_sha256", 64)):
        value = before[field]
        require(isinstance(value, str) and len(value) == size and
                all(char in "0123456789abcdef" for char in value), f"invalid source identity: {field}")
    require(isinstance(before["untracked_runtime_file_sha256"], dict), "invalid untracked source identity")
    return {"stable": True, "compared_fields": list(fields)}


def command(case, directory):
    mode, graph, count, prompt, decode = CASES[case]
    argv = [sys.executable, "-m", "frontier.main", *shlex.split("""
        --sys_arch co-location --no-enable_parallel_clusters
        --log_level warning --cc_backend_config_type analytical
        --cluster_config_num_replicas 1
        --replica_config_model_name zai-org/GLM-5.3
        --replica_config_device h800 --replica_config_network_device h800_dgx
        --replica_config_num_pipeline_stages 1
        --replica_config_attn_tensor_parallel_size 8 --replica_config_attn_dp 8
        --replica_config_moe_tensor_parallel_size 1
        --replica_config_moe_expert_parallel_size 64
        --replica_config_total_expert_num 256 --replica_config_router_topk 8
        --replica_config_moe_routing_distribution_type balanced
        --replica_config_moe_routing_seed 42
        --replica_scheduler_config_type vllm_v1
        --vllm_v1_scheduler_config_batch_size_cap 4
        --vllm_v1_scheduler_config_block_size 64
        --vllm_v1_scheduler_config_num_blocks 0
        --vllm_v1_scheduler_config_num_blocks_mode memory_planner
        --vllm_v1_scheduler_config_max_model_len 128
        --vllm_v1_scheduler_config_max_tokens_in_batch 32
        --vllm_v1_scheduler_config_long_prefill_token_threshold 32
        --vllm_v1_scheduler_config_enable_chunked_prefill
        --no-vllm_v1_scheduler_config_enable_runtime_non_kv_cache_overhead_profiling
        --random_forrest_execution_time_predictor_config_enable_dummy_mode
        --random_forrest_execution_time_predictor_config_dummy_execution_time_ms 0.25
        --request_generator_config_type synthetic --length_generator_config_type fixed
        --interval_generator_config_type poisson
        --poisson_request_interval_generator_config_qps 100 --seed 42
        --metrics_config_write_metrics --metrics_config_store_request_metrics
        --metrics_config_store_batch_metrics --metrics_config_store_token_completion_metrics
        --metrics_config_store_utilization_metrics --metrics_config_store_operation_metrics
        --metrics_config_keep_individual_batch_metrics
        --metrics_config_store_frontier_stage_batch_ledger
        --metrics_config_store_frontier_stage_batch_ledger_summary
        --metrics_config_enable_op_level_tracing
        --metrics_config_enable_per_layer_expansion
        --metrics_config_num_requests_to_trace_per_layer 4
        --metrics_config_enable_metrics_ground_truth_trace
        --no-metrics_config_store_plots --no-metrics_config_enable_chrome_trace
        --no-metrics_config_write_json_trace
    """)]
    for name, value in {
        "simulation_mode": mode, "decode_cuda_graph_mode": graph,
        "synthetic_request_generator_config_num_requests": count,
        "fixed_request_length_generator_config_prefill_tokens": prompt,
        "fixed_request_length_generator_config_decode_tokens": decode,
        "metrics_config_output_dir": directory / "metrics",
        "metrics_config_cache_dir": directory / "cache",
        "metrics_config_run_id": case,
    }.items():
        argv.extend([f"--{name}", str(value)])
    if graph != "none":
        argv.extend(["--cudagraph_capture_sizes", "1", "2", "4", "8", "16", "32"])
    return argv


def check_config(config, case):
    mode, graph, count, prompt, decode = CASES[case]
    require(config["simulation_mode"] == mode and config["sys_arch"] == "co-location",
            "wrong serving mode or architecture")
    require(config["decode_cuda_graph_mode"] == graph and not config["use_cuda_graph"],
            "wrong graph configuration")
    require(not config["enable_parallel_clusters"], "parallel clusters must be disabled")
    require(config["cudagraph_capture_sizes"] == (None if graph == "none" else [1, 2, 4, 8, 16, 32]),
            "wrong graph capture size configuration")
    cluster = config["cluster_config"]
    require(cluster["num_replicas"] == 1 and cluster["cc_backend_config"]["name"] == "analytical",
            "wrong replica count or communication backend")
    replica = cluster["replica_config"]
    for key, value in {
        "model_name": "zai-org/GLM-5.3", "device": "h800", "network_device": "h800_dgx",
        "num_pipeline_stages": 1, "attn_tensor_parallel_size": 8, "attn_dp": 8,
        "moe_tensor_parallel_size": 1, "moe_expert_parallel_size": 64,
        "total_expert_num": 256, "router_topk": 8, "world_size": 64,
        "memory_margin_fraction": 0.1,
    }.items():
        require(replica[key] == value, f"wrong replica setting: {key}")
    model = replica["model_config"]
    require(model["num_layers"] == 78 and model["num_experts"] == 256,
            "model representation must retain 78 layers and 256 experts")
    require(set(map(int, model["moe_layers_enum"].split(","))) == set(range(3, 78)),
            "model representation must retain 3 dense and 75 routed layers")
    scheduler = cluster["replica_scheduler_config"]
    for key, value in {
        "block_size": 64, "num_blocks_mode": "memory_planner", "max_model_len": 128,
        "num_blocks": 0, "non_kv_cache_overhead_bytes": 0,
        "max_tokens_in_batch": 32, "long_prefill_token_threshold": 32,
        "enable_chunked_prefill": True, "enable_runtime_non_kv_cache_overhead_profiling": False,
    }.items():
        require(scheduler[key] == value, f"wrong scheduler setting: {key}")
    predictor = cluster["execution_time_predictor_config"]
    require(predictor["enable_dummy_mode"] and predictor["dummy_execution_time_ms"] == DUMMY_MS,
            "synthetic predictor durations are required")
    generator = config["request_generator_config"]
    require(generator["num_requests"] == count, "wrong generated request count")
    length = generator["length_generator_config"]
    require((length["prefill_tokens"], length["decode_tokens"]) == (prompt, decode),
            "wrong generated request lengths")


def check_requests(output, case):
    mode, _, count, prompt, decode = CASES[case]
    records = list(rows(output / "metrics_ground_truth.jsonl"))
    arrivals = [r for r in records if r["event_type"] == "request_arrival"]
    completed = [r for r in records if r["event_type"] == "request_completion"]
    expected = set(range(count))
    require(len(arrivals) == len(completed) == count, "missing or duplicate request events")
    require({r["request_id"] for r in arrivals} == {r["request_id"] for r in completed} == expected,
            "original request identities were not preserved")
    for row in completed:
        require((row["num_prefill_tokens"], row["num_decode_tokens"]) == (prompt, decode),
                "completed request lengths differ")
        require(0 <= row["arrived_at"] <= row["prefill_completed_at"] <= row["completed_at"],
                "request completion timestamps are out of order")
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in
                    ("ttft_ms", "tpot_ms", "request_e2e_time_ms")), "invalid request latency")
    times = {row["arrived_at"] for row in arrivals}
    require(times == {0.0} if mode == "offline" else len(times) == count,
            "offline/online arrival semantics differ")
    return expected


def check_stage_costs(row):
    require(row["execution_scope"] == "ATTN_DP_LANE", "wrong source stage execution scope")
    timing = row["execution_time"]
    require(timing["component_ledger_ms"] == STAGE_COSTS, "source stage scope keys or costs differ")
    require(timing["diagnostic_component_ledger_ms"] == {"pp_stage_boundary_handoff_time": 0},
            "source stage diagnostic scope differs")
    require(timing["total_time_ms"] == timing["diagnostic_total_time_ms"] == sum(STAGE_COSTS.values()) == 128.25
            and timing["model_time_ms"] == 127, "source stage totals or CPU ownership differ")


def check_stages(output, case, request_ids):
    _, _, count, prompt, decode = CASES[case]
    stage_rows = list(rows(output / "frontier_stage_batch_ledger.jsonl"))
    expected_batches = count * (prompt // 32 + decode - 1)
    require(len(stage_rows) == expected_batches, "wrong number of real source stage batches")
    chunks, prefill = defaultdict(Counter), Counter()
    stages = {}
    for row in stage_rows:
        require(row["cluster_type"] == "MONOLITHIC", "unexpected stage cluster")
        require(row["batch_id"] not in stages, "duplicate source stage batch")
        require(row["replica_id"] == row["stage_id"] == 0 and row["replica_local_id"] in range(8),
                "wrong source stage placement")
        start, end = row["stage_start_ts"], row["stage_end_ts"]
        require(math.isfinite(start) and math.isfinite(end) and 0 <= start < end, "invalid stage boundary")
        stages[row["batch_id"]] = row
        check_stage_costs(row)
        require(len(row["request_ids"]) == 1 == len(row["request_num_tokens"]) ==
                len(row["request_num_prefill_tokens"]), "stage request columns differ")
        for rid, tokens, pref in zip(row["request_ids"], row["request_num_tokens"], row["request_num_prefill_tokens"]):
            chunks[int(rid)][tokens] += 1
            prefill[int(rid)] += pref
    require(set(chunks) == request_ids, "stage request identities differ")
    for rid in request_ids:
        require(chunks[rid] == Counter({32: prompt // 32, 1: decode - 1}) and prefill[rid] == prompt,
                f"chunked prefill/decode ownership differs for request {rid}")
    return stages


def source_cohorts(stages, case):
    cohorts, source_to_cohort = {}, {}
    for batch, row in stages.items():
        key = (row["replica_id"], row["stage_id"], row["stage_end_ts"])
        cohort = cohorts.setdefault(key, {"lanes": {}, "start": row["stage_start_ts"], "layers": set()})
        lane = row["replica_local_id"]
        require(lane not in cohort["lanes"], "duplicate real lane in a source cohort")
        cohort["lanes"][lane] = sum(row["request_num_tokens"])
        cohort["start"] = min(cohort["start"], row["stage_start_ts"])
        source_to_cohort[batch] = key
    for key, cohort in cohorts.items():
        widths = list(cohort["lanes"].values()) + [1] * (8 - len(cohort["lanes"]))
        require(len(widths) == 8 and all(type(width) is int and width > 0 for width in widths),
                "invalid aggregate source-workload shape")
        # ep_wave_inputs.py uses a common maximum when configured metadata is
        # present. This is an input-shape oracle, not runtime graph evidence.
        cohort["physical_tokens"] = sum(widths) if CASES[case][1] == "none" else 8 * max(widths)
    ordered = sorted(cohorts)
    for previous, current in zip(ordered, ordered[1:]):
        if previous[:2] == current[:2]:
            require(previous[2] <= cohorts[current]["start"] + 1e-12, "overlapping source cohorts")
    return cohorts, source_to_cohort


def check_waves(output, stages, case):
    cohorts, source_to_cohort = source_cohorts(stages, case)
    waves = {}
    count = 0
    for row in rows(output / "frontier_ep_wave_lane_ledger.jsonl"):
        count += 1
        require(row["record_type"] == "ep_wave_lane" and row["cluster"] == "MONOLITHIC",
                "unexpected EP lane record")
        require(len(row["global_layer_ids"]) == 1, "EP lane must identify one physical layer")
        layer = row["global_layer_ids"][0]
        require(layer in range(3, 78), "routed EP work was assigned to a dense layer")
        require(row["attention_family_id"] == "dsa_attention" and row["attention_variant_id"] ==
                ("full_index" if layer in FULL_INDEX else "shared_index"), "wrong EP layer binding")
        start, end = row["wave_start_time_s"], row["wave_end_time_s"]
        require(math.isfinite(start) and math.isfinite(end) and 0 <= start <= end, "invalid wave boundary")
        require(len(row["source_batch_ids"]) == 1, "EP wave must identify one representative source")
        source = row["source_batch_ids"][0]
        key = (row["replica_id"], row["stage_id"], layer, start, end, source)
        if key not in waves:
            cohort_key = source_to_cohort.get(source)
            if cohort_key is not None:
                cohort = cohorts[cohort_key]
                require(key[:2] == cohort_key[:2] and start >= cohort["start"] - 1e-12 and
                        end <= cohort_key[2] + 1e-12, "EP representative source has incompatible cohort boundaries")
                require(layer not in cohort["layers"], "duplicate routed layer wave in a real source cohort")
                cohort["layers"].add(layer)
            waves[key] = {"cohort": cohort_key, "lane_mask": 0, "batch_ids": [None] * 64,
                          "physical_tokens": cohorts[cohort_key]["physical_tokens"] if cohort_key is not None else 8,
                          "gating_mask": 0, "routed_mask": 0, "routed_cost_mask": 0, "routed_tokens": 0,
                          "phase_starts": {name: phase["start_time_s"] for name, phase in row["phases"].items()},
                          "phase_max_durations": dict.fromkeys(EP_PHASE_COSTS, 0)}
        wave = waves[key]
        ep = row["ep_id"]
        require(type(ep) is int and ep in range(64), "invalid EP participant")
        require(not wave["lane_mask"] & (1 << ep), "duplicate EP lane in a wave")
        wave["lane_mask"] |= 1 << ep
        wave["batch_ids"][ep] = row["batch_id"]
        require(set(row["phases"]) == set(EP_PHASE_COSTS),
                "missing EP phase")
        routed_cost = row["phases"]["routed_compute"]["operators_ms"].get("moe_grouped_gemm")
        require(routed_cost in (0, DUMMY_MS), "invalid routed EP lane cost")
        if routed_cost:
            wave["routed_cost_mask"] |= 1 << ep
        for name, base_costs in EP_PHASE_COSTS.items():
            phase = row["phases"][name]
            expected = {**base_costs}
            # The dummy predictor skips both routed operations on an empty
            # EP lane. Gating, shared experts and barriers still execute.
            if name == "pre_dispatch":
                expected["moe_shuffling"] = routed_cost
            elif name == "routed_compute":
                expected["moe_grouped_gemm"] = routed_cost
            require(phase["operators_ms"] == expected, f"EP phase scope keys or costs differ: {name}")
            require(phase.get("kernel_gap_ms", 0) == 0 and phase["duration_ms"] == sum(expected.values()),
                    "EP phase duration or kernel gap differs")
            require(phase["start_time_s"] == wave["phase_starts"][name], "inconsistent EP phase barrier")
            wave["phase_max_durations"][name] = max(wave["phase_max_durations"][name], phase["duration_ms"])
    require(waves and all(wave["lane_mask"] == ALL_EP for wave in waves.values()), "missing EP participants")
    for key, wave in waves.items():
        require(wave["routed_cost_mask"], "EP wave has no routed assignments")
        phase_start = key[3]
        for name in EP_PHASE_COSTS:
            require(math.isclose(wave["phase_starts"][name], phase_start, rel_tol=0, abs_tol=1e-12),
                    "EP phase boundary differs from maximum lane cost")
            phase_start += wave["phase_max_durations"][name] / 1000
        require(math.isclose(phase_start, key[4], rel_tol=0, abs_tol=1e-12), "EP wave end differs from barrier costs")
    require(cohorts and all(cohort["layers"] == set(range(3, 78)) for cohort in cohorts.values()),
            "real source cohort missing routed EP layer coverage")
    return waves, {"lane_records": count, "waves": len(waves), "layers": 75, "lanes_per_wave": 64,
                   "real_source_cohorts": len(cohorts), "real_source_waves": len(cohorts) * 75,
                   "representative_source_limit": "EP source IDs identify a representative; cohort membership uses shared completion boundaries"}


def check_ep_trace(row, waves):
    name, meta = row["name"], row["meta"]
    require(len(meta["source_batch_ids"]) == 1 and len(meta["global_layer_ids"]) == 1,
            "EP token trace must identify one representative and layer")
    layer = meta["global_layer_ids"][0]
    key = (row["replica_id"], meta["stage_id"], layer, meta["wave_start_time_s"],
           meta["wave_end_time_s"], meta["source_batch_ids"][0])
    require(key in waves, "EP token trace has no exact source/wave/layer ledger join")
    wave, ep = waves[key], meta["ep_id"]
    require(type(ep) is int and ep in range(64) and row["batch_id"] == wave["batch_ids"][ep],
            "EP token trace participant or virtual batch differs")
    require(row["cluster"] == "MONOLITHIC" and row["layer_id"] in (layer, -1) and meta["num_layers"] == 1
            and meta["attention_family_id"] == "dsa_attention" and meta["attention_variant_id"] ==
            ("full_index" if layer in FULL_INDEX else "shared_index"), "wrong EP trace layer binding")
    require(meta["parallel_context"] == {"PP": 1, "ATTN_TP": 8, "ATTN_DP": 8, "MOE_TP": 1,
                                         "MOE_EP": 64, "ROUTER_TOPK": 8}, "wrong EP trace parallel shape")
    gating = name == "moe_gating_linear"
    mask = "gating_mask" if gating else "routed_mask"
    require(not wave[mask] & (1 << ep), f"duplicate positive EP token trace: {name}")
    require(row["duration_ms"] == (DUMMY_MS / 2 if gating else DUMMY_MS) and
            meta["phase"] == ("pre_dispatch" if gating else "routed_compute"), "EP token trace cost or phase differs")
    tokens = meta["effective_total_tokens_compute"]
    require(type(tokens) is int and tokens >= 0 and meta["tokens_are_post_routing"] is (not gating),
            "EP trace token count or routing attribution differs")
    require(meta["tensor_shape"] == {"input": [tokens, 6144], "output": [tokens, 256 if gating else 2048]},
            "EP token trace tensor shape differs")
    if gating:
        require(tokens == wave["physical_tokens"], "unattributed EP workload or real cohort physical width differs")
    else:
        require(tokens > 0 and wave["routed_cost_mask"] & (1 << ep), "positive routed trace on empty EP lane")
        wave["routed_tokens"] += tokens
    wave[mask] |= 1 << ep


def check_wave_tokens(waves, stages):
    temporal = Counter({"before_real_cohorts": 0, "within_real_cohort_span": 0, "after_real_cohorts": 0})
    first = min(row["stage_start_ts"] for row in stages.values())
    last = max(row["stage_end_ts"] for row in stages.values())
    real_tokens = dummy_tokens = 0
    for key, wave in waves.items():
        require(wave["gating_mask"] == ALL_EP and wave["routed_mask"] == wave["routed_cost_mask"],
                "missing EP gating or routed token traces")
        require(wave["routed_tokens"] == wave["physical_tokens"] * 8, "EP routed-token conservation differs")
        if wave["cohort"] is not None:
            real_tokens += wave["routed_tokens"]
        else:
            # An absent source ID alone does not establish dummy work. These
            # checks qualify only the observed eight-lane dummy workload shape.
            require(wave["physical_tokens"] == 8 and wave["routed_tokens"] == 64, "unattributed non-request EP wave")
            bucket = "before_real_cohorts" if key[4] <= first else (
                "after_real_cohorts" if key[3] >= last else "within_real_cohort_span")
            temporal[bucket] += 1
            dummy_tokens += wave["routed_tokens"]
    return {"real_cohort_routed_assignments": real_tokens, "dummy_shaped_routed_assignments": dummy_tokens,
            "validated_dummy_shaped_waves": sum(temporal.values()), "dummy_shaped_wave_temporal_counts": dict(temporal),
            "unattributed_non_request_waves": 0,
            "shape_qualification": "aggregate EP source-workload shape under configured metadata; graph activation unverified"}


def check_traces(output, stages, waves):
    batch_ids = set(stages)
    identities, costs = defaultdict(lambda: defaultdict(set)), defaultdict(Counter)
    selected = {"attn_dsa_indexer", "attn_dsa_mla", "mlp_up_proj", "moe_gating_linear", "moe_grouped_gemm"}
    count = 0
    with (output / "op_traces.jsonl").open() as stream:
        for line in stream:
            count += 1
            if not any(f'"{name}"' in line for name in selected):
                continue
            row = json.loads(line)
            name = row.get("name")
            if name not in selected:
                continue
            if name in {"moe_gating_linear", "moe_grouped_gemm"}:
                check_ep_trace(row, waves)
                continue
            batch, meta = row["batch_id"], row["meta"]
            require(batch in batch_ids and meta["attention_family_id"] == "dsa_attention",
                    "trace operator has no real source stage or wrong family")
            require(meta["attention_variant_id"] in {"full_index", "shared_index"} and
                    (name == "attn_dsa_mla" or meta["attention_variant_id"] == "full_index"),
                    "wrong trace attention variant")
            ids = meta.get("global_layer_ids", [row["layer_id"]])
            require(ids and len(ids) == len(set(ids)), "missing or duplicate trace layer identities")
            expected = FULL_INDEX if name == "attn_dsa_indexer" else (
                {0, 1, 2} if name == "mlp_up_proj" else
                FULL_INDEX if meta["attention_variant_id"] == "full_index" else set(range(78)) - FULL_INDEX
            )
            require(set(ids) <= expected, f"positive trace contains inactive layer identities: {name}, batch={batch}")
            require(not identities[batch][name].intersection(ids), "duplicate operator charge for one layer")
            require(math.isfinite(row["duration_ms"]) and row["duration_ms"] > 0, "invalid positive trace duration")
            identities[batch][name].update(ids)
            costs[batch][name] += row["duration_ms"]
    require(set(identities) == batch_ids, "missing source batch traces")
    for batch in batch_ids:
        for name, ids in {"attn_dsa_indexer": FULL_INDEX, "attn_dsa_mla": set(range(78)),
                          "mlp_up_proj": {0, 1, 2}}.items():
            require(identities[batch][name] == ids and costs[batch][name] == len(ids) * DUMMY_MS,
                    f"trace coverage or scope cost differs: {name}, batch={batch}")
    return {"streamed_lines": count, "real_attention_batches": len(batch_ids),
            "mla_layers": 78, "full_index_layers": 21, "shared_index_layers": 57, "dense_layers": 3,
            "ep_token_conservation": check_wave_tokens(waves, stages)}


def validate(output, case, log, *, source_before, source_after):
    source_identity = check_source_identity(source_before, source_after)
    for name in ("config.json", "system_metrics.json", "request_metrics.csv", "metrics_ground_truth.jsonl",
                 "frontier_stage_batch_ledger.jsonl", "frontier_stage_batch_ledger_summary.json",
                 "frontier_ep_wave_lane_ledger.jsonl", "op_traces.jsonl", "monolithic_operation_metrics.csv"):
        require((output / name).is_file() and (output / name).stat().st_size > 0, f"missing artifact: {name}")
    config = json.loads((output / "config.json").read_text())
    check_config(config, case)
    system = json.loads((output / "system_metrics.json").read_text())
    count = CASES[case][2]
    require(system["simulation_metadata"] == {"total_requests": count, "completed_requests": count,
            "system_architecture": "co-location", "server_count": 8}, "system completion/topology differs")
    require(system["preemption_statistics"]["total_preemption_events"] == 0, "unexpected preemption")
    memory = system["model_weight_memory"]["MONOLITHIC"]
    require(memory["total_parameters"] == PARAMETERS and memory["total_memory_bytes"] == WEIGHT_BYTES,
            "weight storage differs from independent source oracle")
    require(memory["memory_accounting"] == "source-defined weight storage estimate; native allocation unverified",
            "source memory estimate must retain its qualification")
    require(all(memory["breakdown"][k] == v for k, v in WEIGHT_PARTS.items()), "weight storage breakdown differs")
    require(math.isclose(memory["total_memory_gb"], WEIGHT_BYTES / 1024**3), "weight GiB conversion differs")
    request_ids = check_requests(output, case)
    batches = check_stages(output, case, request_ids)
    require(json.loads((output / "frontier_stage_batch_ledger_summary.json").read_text())["total_rows"] == len(batches),
            "stage ledger summary count differs")
    waves, wave_stats = check_waves(output, batches, case)
    traces = check_traces(output, batches, waves)
    warnings = []
    with log.open() as stream:
        for line in stream:
            require(not any(s in line for s in ("Traceback", "FRONTIER_MEMORY_OOM",
                    "Failed to calculate model weight memory", "Error during metrics output", "ERROR ")),
                    f"runtime or metrics error: {line.strip()}")
            if "WARNING" in line and len(warnings) < 10:
                warnings.append(line.strip())
    cpu_csv = output / "monolithic_cpu_operation_metrics.csv"
    return {"completed_requests": count, "real_stage_batches": len(batches), "ep": wave_stats, "traces": traces,
            "source_identity": source_identity,
            "weight_bytes_per_device": WEIGHT_BYTES, "cache_budget_qualification": "source-derived; not native allocator evidence",
            "source_cache_bytes_per_common_block": 64 * (78 * 576 * 2 + 21 * 132),
            "source_cache_budget_bytes": int(80 * 1024**3 * .9) - WEIGHT_BYTES,
            "cpu_csv_bytes": cpu_csv.stat().st_size if cpu_csv.exists() else None, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="New destination; existing paths are rejected")
    parser.add_argument("--case", choices=tuple(CASES), action="append", help="Select a case; default runs all three sequentially")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        parser.error(f"output root already exists; preserve it and select a new path: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    for case in dict.fromkeys(args.case or CASES):
        directory = output_root / case
        directory.mkdir(exist_ok=False)
        argv = command(case, directory)
        env = {"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1", "FRONTIER_LOG_LEVEL": "WARNING",
               "WANDB_DISABLED": "true", "VIDUR_DISABLE_WANDB": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        source_before = snapshot()
        write_json(directory / "invocation.json", {"argv": argv, "cwd": str(REPO), "environment_overrides": env,
                   "python_version": sys.version, "source_before": source_before, "limits": LIMITS})
        result = {"case": case, "status": "FAIL", "returncode": None, "limits": LIMITS}
        log = directory / "run.log"
        try:
            with log.open("x") as stream:
                process = subprocess.run(argv, cwd=REPO, env={**os.environ, **env}, stdout=stream,
                                         stderr=subprocess.STDOUT, timeout=1800)
            result["returncode"] = process.returncode
            result["source_after"] = snapshot()
            require(process.returncode == 0, f"Frontier returned {process.returncode}; inspect {log}")
            mode = "offline_batch" if CASES[case][0] == "offline" else "online_serving"
            metrics = directory / "metrics" / "zai_org_glm_5_3" / mode / case.replace("-", "_")
            result["metrics_directory"] = str(metrics)
            result["validation"] = validate(metrics, case, log, source_before=source_before,
                                            source_after=result["source_after"])
            result["status"] = "PASS"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        if "source_after" not in result:
            result["source_after"] = snapshot()
        write_json(directory / "result.json", result)
        results.append(result)
        print(f"{case}: {result['status']} {result.get('error', '')}", flush=True)
    write_json(output_root / "summary.json", {"results": results, "limits": LIMITS})
    return int(any(row["status"] != "PASS" for row in results))


if __name__ == "__main__":
    sys.exit(main())
