"""Exercise full 43-layer V4 serving with explicitly synthetic dummy timing.

Eager PP1 co-location includes the existing request-free idle DP forwards.
One grouped EP dispatch owns hidden, weights and int64 ids; embedding and
logits collectives are priced by the configured communication backend.
Source SP geometry is derived from scheduled/dummy rows; destination expert
ratios are synthetic balanced inputs. Native routing, latency, overlap,
workspace capacity, CUDA graphs and PD remain unqualified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


SOURCE_FILES = (
    "data/config/models/deepseek-ai__DeepSeek-V4-Flash.json",
    "frontier/attention/deepseek_v4.py", "frontier/attention/families.py",
    "frontier/attention/memory.py", "frontier/kv_cache/base_kv_cache_manager.py",
    "frontier/kv_cache/kv_cache_block_pool.py", "frontier/scheduler/utils/memory_planner.py",
    "frontier/config/quantization_manager.py", "frontier/entities/execution_time.py",
    "frontier/entities/stage_execution_time.py", "frontier/entities/time_components.py",
    "frontier/entities/request.py",
    "frontier/entities/batch.py", "frontier/utils/param_counter.py",
    "frontier/execution_time_predictor/deepseek_v4.py",
    "frontier/execution_time_predictor/sklearn_execution_time_predictor.py",
    "frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py",
    "frontier/metrics/constants.py", "frontier/metrics/ep_wave_metrics.py",
    "frontier/metrics/metrics_store.py", "frontier/metrics/op_trace_utils.py",
    "frontier/operators/families.py",
    "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py",
    "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py",
    "frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py",
    "frontier/scheduler/utils/ep_wave.py", "frontier/scheduler/utils/expert_parallel.py",
    "frontier/scheduler/utils/layer_workload.py",
    "tests/integration/run_dsv4_structural.py",
)


def source_identity():
    return {path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            for path in SOURCE_FILES}


def validate_attention_boundary(operators, *, query, context, layer):
    """Check source logical calls; this does not witness physical cache contents.

    vLLM 2cf0a691: attention.py:908-930 selects eager CSA fill from the
    batch maximum; compressor.py:48-58,374-396 saves state before the HCA
    boundary branch. This driver deliberately schedules one request per batch.
    """
    config = json.loads((REPO / "data/config/models/deepseek-ai__DeepSeek-V4-Flash.json").read_text())
    ratio = config["compress_ratios"][layer]
    sequence = context + query
    history = sequence // ratio if ratio > 1 else 0
    writes = history - context // ratio if ratio > 1 else 0

    def check(name, **expected):
        require(name in operators, f"missing boundary scope: {name}")
        calls = [call for call in operators[name]
                 if call.get("global_layer_id", layer) == layer]
        require(len(calls) == 1, f"duplicate boundary call: {name}")
        for field, value in expected.items():
            actual = json.loads(json.dumps(calls[0].get(field)))
            require(actual == json.loads(json.dumps(value)),
                    f"boundary shape mismatch: layer={layer} context={context} query={query} {name}.{field}")

    hidden, head = config["hidden_size"], config["head_dim"]
    check("dsv4_fused_qkv_projection", input_shape=(query, hidden),
          output_shape=(query, config["q_lora_rank"] + head))
    check("dsv4_fused_q_kv_prepare", output_shape=(query, 64, head),
          kv_cache_write_rows=query)
    check("dsv4_sparse_attention", input_shape=(query, 64, head),
          compression_ratio=ratio, compressed_history_rows=(history,),
          swa_logical_window_span_rows=(query + min(context, config["sliding_window"] - 1),),
          query_intervals=((context, sequence),))
    store = ratio > 1
    require(("dsv4_compressor_fused_compress_store" in operators) == store,
            "wrong boundary scope: dsv4_compressor_fused_compress_store")
    if ratio > 1:
        width = 4 * head if ratio == 4 else 2 * head
        check("dsv4_compressor_projection", output_shape=(query, width))
        check("dsv4_compressor_save_partial_states", input_shape=(query, width),
              state_width=width, kv_shape=(query, width // 2), score_shape=(query, width // 2))
        if store:
            check("dsv4_compressor_fused_compress_store", new_compressed_rows=(writes,))
    if ratio == 4:
        index_dim = config["index_head_dim"]
        check("dsv4_index_weights_projection", output_shape=(query, config["index_n_heads"]))
        check("dsv4_index_compressor_projection", output_shape=(query, 4 * index_dim))
        check("dsv4_index_compressor_save_partial_states", state_width=4 * index_dim)
        check("dsv4_index_compressor_fused_compress_store", new_compressed_rows=(writes,))
        short = history <= config["index_topk"]
        for name in ("dsv4_index_query_projection", "dsv4_index_fused_query_prepare",
                     "dsv4_index_scores", "dsv4_index_topk"):
            require((name in operators) == (not short), f"wrong CSA threshold scope: {name}")
        require(("dsv4_index_short_context_fill" in operators) == short, "wrong CSA fill branch")
        if short:
            check("dsv4_index_short_context_fill", output_shape=(query, config["index_topk"]))
        else:
            shape = (query, config["index_n_heads"], index_dim)
            check("dsv4_index_query_projection", output_shape=shape)
            check("dsv4_index_fused_query_prepare", input_shape=shape, output_shape=shape)
            check("dsv4_index_scores", compressed_history_rows=(history,))
            check("dsv4_index_topk", output_shape=(query, config["index_topk"]),
                  compressed_history_rows=(history,))


def validate_cache_zeroing(decisions, stages, traces, *, group_ids,
                          physical_block_size_bytes, dummy_execution_time_ms):
    """Reconcile allocation occurrences, batch handoff and structural worker work.

    Group order comes from the canonical capacity. Repeated assignments survive;
    distinct slab reservation extent does not describe native store traffic.
    """
    pending = {}
    handed_steps = set()
    handoffs = {}
    for row in decisions:
        if row["event"] not in {"grouped_cache_allocation", "grouped_cache_batch"}:
            continue
        key = tuple(row[field] for field in (
            "cluster_type", "replica_id", "replica_local_id", "iteration_id"))
        require(key not in handed_steps, "allocation or handoff repeated after step drain")
        groups = pending.setdefault(key, {group: [] for group in group_ids})
        if row["event"] == "grouped_cache_allocation":
            new = row["group_new_block_ids"]
            require(set(new) == set(group_ids), "allocation groups differ from canonical layout")
            raw = [block for group in group_ids for block in new[group]]
            require(all(type(block) is int and block > 0 for block in raw),
                    "allocation includes null or invalid physical block IDs")
            require(raw == row["zero_required_block_ids"], "allocation zero assignments differ")
            for group in group_ids:
                groups[group].extend(new[group])
            continue
        ids = [block for group in group_ids for block in groups[group]]
        require(ids == row["zero_required_block_ids"], "batch zero IDs differ from allocation stream")
        require(row["physical_block_size_bytes"] == physical_block_size_bytes,
                "batch physical stride differs from canonical capacity")
        require(row["batch_id"] not in handoffs, "batch has duplicate cache handoffs")
        handoffs[row["batch_id"]] = row
        handed_steps.add(key)
        pending.pop(key)
    require(not pending, "allocation step has no cache batch handoff")
    stage_by_batch = {row["batch_id"]: row for row in stages}
    require(set(stage_by_batch) == set(handoffs), "stage/cache batch identity coverage differs")
    zeros = defaultdict(list)
    embeddings = defaultdict(list)
    for row in traces:
        if row.get("name") == "kv_cache_block_zero":
            require(row["batch_id"] in stage_by_batch, "idle or orphan cache zero trace")
            require("phase" not in row.get("meta", {}), "cache zero duplicated in EP phase")
            zeros[row["batch_id"]].append(row)
        elif row.get("name") == "dsv4_embedding" and "phase" not in row.get("meta", {}):
            embeddings[row["batch_id"]].append(row)
    counts = Counter()
    for batch_id, row in handoffs.items():
        stage = stage_by_batch[batch_id]
        require(all(stage[field] == row[field] for field in (
            "cluster_type", "replica_id", "replica_local_id")), "stage/cache lane identity differs")
        ids = row["zero_required_block_ids"]
        require(len(zeros[batch_id]) == bool(ids), "cache zero must occur once only for nonempty IDs")
        if not ids:
            continue
        zero, = zeros[batch_id]
        require(zero["type"] == "COMPUTE" and
                zero.get("meta", {}).get("global_layer_ids", [zero["layer_id"]]) == [0],
                "cache zero must use the before-forward anchor")
        require(abs(zero["duration_ms"] - dummy_execution_time_ms) < 1e-9,
                "cache zero duration must be one synthetic invocation")
        require(len(embeddings[batch_id]) == 1, "cache zero has no unique embedding boundary")
        embedding, = embeddings[batch_id]
        require(zero["ts_start"] + zero["duration_ms"] / 1000 <= embedding["ts_start"] + 1e-12,
                "cache zero must finish before embedding")
        meta = zero["meta"]
        require("synthetic" in meta["cost_source"], "cache zero cost has no synthetic provenance")
        require(len(meta["source_tensor_calls"]) == 1, "cache zero workload counted more than once")
        call, = meta["source_tensor_calls"]
        extent = len(set(ids)) * physical_block_size_bytes
        expected = dict(block_ids=ids, block_assignment_count=len(ids),
                        unique_block_count=len(set(ids)),
                        physical_block_size_bytes=physical_block_size_bytes,
                        physical_reservation_extent_bytes=extent,
                        native_store_traffic_bytes="UNKNOWN", native_h2d_cost="UNKNOWN")
        require(all(call.get(field) == value for field, value in expected.items()),
                "cache zero trace workload differs from raw assignments or canonical extent")
        counts["cache_zero_invocations"] += 1
        counts["cache_zero_block_assignments"] += len(ids)
        counts["cache_zero_reservation_extent_bytes"] += extent
    return {**counts, "cache_zero_scope":
            "one per physical worker per nonempty step; distinct reservation extent is not native store traffic"}


def validate_artifacts(run_dir, dp, *, decision_path, prefill_tokens=17,
                       chunk_size=16, decode_tokens=2):
    from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
    from frontier.attention.memory import AttentionKVCacheSelection, get_attention_kv_cache_capacity
    from frontier.config.model_config import BaseModelConfig
    from frontier.types import DeviceSKUType

    capacity = get_attention_kv_cache_capacity(
        DEEPSEEK_V4_ATTENTION_FAMILY,
        BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash"), block_size=256,
        selection=AttentionKVCacheSelection(
            device=str(DeviceSKUType.H800), gpu_platform="cuda", block_size=256,
            attention_backend="FLASHMLA_SPARSE_DSV4", kv_cache_dtype="fp8",
            indexer_kv_cache_dtype="fp8"))
    system = json.loads((run_dir / "system_metrics.json").read_text())
    metadata = system["simulation_metadata"]
    require(metadata["total_requests"] == metadata["completed_requests"] == dp,
            "all and only submitted requests must complete")
    with (run_dir / "request_metrics.csv").open(newline="") as stream:
        requests = list(csv.DictReader(stream))
    require(len(requests) == dp, "request metrics coverage differs from input")
    require({int(r["request_num_prefill_tokens"]) for r in requests} == {prefill_tokens}, "wrong prompts")
    require({int(r["request_num_decode_tokens"]) for r in requests} == {decode_tokens}, "wrong decode length")
    request_ids = {str(row["Request Id"]) for row in requests}
    stage_rows = list(read_jsonl(run_dir / "frontier_stage_batch_ledger.jsonl"))
    cache_traces = (row for row in read_jsonl(run_dir / "op_traces.jsonl")
                    if row.get("name") in {"kv_cache_block_zero", "dsv4_embedding"})
    cache_zero = validate_cache_zeroing(
        read_jsonl(decision_path), stage_rows, cache_traces,
        group_ids=tuple(group.group_id for group in capacity.groups),
        physical_block_size_bytes=capacity.physical_block_size_bytes,
        dummy_execution_time_ms=0.01)
    stage_by_batch = {row["batch_id"]: row for row in stage_rows}
    stage_ops = defaultdict(Counter)
    phase_ops = defaultdict(Counter)
    attention_layers = defaultdict(dict)
    attention_calls = defaultdict(dict)
    stage_inputs = {}
    processed = Counter()
    for row in sorted(stage_rows, key=lambda value: value["stage_start_ts"]):
        require(len(row["request_ids"]) == 1, "boundary driver requires one source request per batch")
        request_id = row["request_ids"][0]
        stage_inputs[row["batch_id"]] = (row["request_num_tokens"][0], processed[request_id])
        processed[request_id] += row["request_num_tokens"][0]
    counts = Counter()
    aliases = {"attn_pre_proj", "attn_post_proj", "attn_prefill", "attn_decode",
               "attn_rope", "attn_kv_cache_save", "add_attn_residual", "add_ffn_residual",
               "input_layernorm", "post_attention_layernorm", "moe_all_to_all"}
    for row in read_jsonl(run_dir / "op_traces.jsonl"):
        if "name" not in row:
            continue
        counts["trace_rows"] += 1
        name, meta = row["name"], row.get("meta", {})
        require(name not in aliases, f"legacy alias would duplicate physical work: {name}")
        require(row["duration_ms"] > 0, "trace contains non-positive work")
        if "phase" in meta:
            key = (row["batch_id"], meta["ep_id"], meta["global_layer_ids"][0], meta["phase"])
            phase_ops[key][name] += row["duration_ms"]
            require(meta["native_routed_assignments"] == "UNKNOWN", "synthetic routes claimed native")
            require(meta["destination_provenance"] == "configured synthetic expert ratios", "missing ratio origin")
            require(set(meta["request_ids"]) <= request_ids, "virtual destination identities credited as requests")
        elif row["batch_id"] in stage_by_batch:
            stage_ops[row["batch_id"]][name] += row["duration_ms"]
            if "source_tensor_calls" in meta:
                for layer in meta.get("global_layer_ids", [row["layer_id"]]):
                    calls = attention_calls[(row["batch_id"], layer)]
                    require(name not in calls, f"physical attention scope counted twice: {name}")
                    calls[name] = meta["source_tensor_calls"]
            if name == "dsv4_sparse_attention":
                layers = meta.get("global_layer_ids", [row["layer_id"]])
                for layer in layers:
                    require(layer not in attention_layers[row["batch_id"]], "attention double ownership")
                    attention_layers[row["batch_id"]][layer] = meta["attention_variant_id"]
    cpu_names = {"schedule_time": "schedule", "sampler_e2e_time": "sampler_e2e",
                 "prepare_inputs_e2e_time": "prepare_inputs_e2e",
                 "process_model_outputs_time": "process_model_outputs", "ray_comm_time": "ray_comm_time"}
    scheduled = defaultdict(list)
    max_stage_error = 0.0
    for row in stage_rows:
        require(set(row["request_ids"]) <= request_ids, "idle forward credited as real request")
        components = row["execution_time"]["component_ledger_ms"]
        error = abs(sum(components.values()) - row["execution_time"]["total_time_ms"])
        max_stage_error = max(max_stage_error, error)
        require(error < 1e-6, "stage component ledger is not conserved")
        observed = stage_ops[row["batch_id"]]
        for name, duration in components.items():
            require(abs(observed[cpu_names.get(name, name)] - duration) < 1e-6,
                    f"stage trace/component disagreement: batch={row['batch_id']} {name}")
        require(set(attention_layers[row["batch_id"]]) == set(range(43)), "stage lacks43physicalattentionlayers")
        require(Counter(attention_layers[row["batch_id"]].values()) ==
                {"uncompressed": 2, "csa": 21, "hca": 20}, "wrong43-layer variants")
        query, context = stage_inputs[row["batch_id"]]
        for layer in range(43):
            validate_attention_boundary(attention_calls[(row["batch_id"], layer)],
                                        query=query, context=context, layer=layer)
        require(abs(components["dsv4_mhc_first_pre"] - 0.01) < 1e-9, "first mHC counted incorrectly")
        require(abs(components["dsv4_mhc_ffn_post_attn_pre"] - 0.42) < 1e-9, "cross-layer mHC counted incorrectly")
        for name in cpu_names:
            require(abs(components[name] - 0.01) < 1e-9, "stage CPU overhead must occur once")
        require(abs(components["dsv4_embedding"] - 0.01) < 1e-9,
                "local embedding must occur once")
        # The backend prices the reduction apart from the local dummy scope;
        # its single trace is already reconciled with the stage ledger above.
        require(components["dsv4_embedding_allreduce"] > 0,
                "embedding reduction must be priced by the communication backend")
        for request_id, tokens in zip(row["request_ids"], row["request_num_tokens"]):
            scheduled[request_id].append((row["stage_start_ts"], tokens))
    require(set(scheduled) == request_ids, "stage/request identity coverage differs")
    prefill_schedule = [min(chunk_size, prefill_tokens - offset)
                        for offset in range(0, prefill_tokens, chunk_size)]
    expected_schedule = prefill_schedule + [1] * (decode_tokens - 1)
    for values in scheduled.values():
        require([tokens for _, tokens in sorted(values)] == expected_schedule,
                f"wrong chunked-prefill/decode schedule: expected {expected_schedule}")
    waves = defaultdict(list)
    max_phase_error = 0.0
    for row in read_jsonl(run_dir / "frontier_ep_wave_lane_ledger.jsonl"):
        layer = row["global_layer_ids"][0]
        waves[(row["wave_start_time_s"], layer)].append(row)
        local = (row["source_rows"] + 7) // 8
        valid = max(0, min(local, row["source_rows"] - row["source_tp_rank"] * local))
        require(row["physical_local_rows"] == local and row["valid_local_rows"] == valid,
                "source SP geometry differs from its batch")
        idle = row["source_forward_kind"] == "request-free dummy"
        require(row["request_valid_local_rows"] == (0 if idle else valid), "dummy request credit")
        require(not idle or (row["source_rows"] == 1 and not row["request_ids"]), "wrong idle contract")
        counts["idle_ep_lanes" if idle else "real_ep_lanes"] += 1
        require(row["routing_operator_variant"] == ("hash" if layer < 3 else "score"), "wrong routing variant")
        for phase, details in row["phases"].items():
            predicted = details["operators_ms"]
            observed = phase_ops.pop((row["batch_id"], row["ep_id"], layer, phase), Counter())
            error = abs(sum(predicted.values()) + details.get("kernel_gap_ms", 0) - details["duration_ms"])
            max_phase_error = max(max_phase_error, error)
            require(error < 1e-6, "EP phase ledger is not conserved")
            for name, duration in predicted.items():
                require(abs(observed[name] - duration) < 1e-6, "EP trace/phase disagreement")
        order = ("pre_dispatch", "dispatch", "routed_compute", "combine", "post_combine")
        for previous, following in zip(order, order[1:]):
            end = row["phases"][previous]["start_time_s"] + row["phases"][previous]["duration_ms"] / 1000
            require(end <= row["phases"][following]["start_time_s"] + 1e-12,
                    "structural serial phase boundaries overlap")
        pre = row["phases"]["pre_dispatch"]["operators_ms"]
        for name in ("dsv4_mhc_attn_post_ffn_pre", "share_expert_up_proj", "share_expert_act",
                     "share_expert_down_proj", "moe_gating_linear", "moe_gating_routing_topk"):
            require(abs(pre[name] - 0.01) < 1e-9, "missing/duplicated source FFN physical scope")
        dispatch = row["phases"]["dispatch"]["operators_ms"]
        require(set(dispatch) == {"dsv4_ep_dispatch_allgather"},
                "one grouped EP dispatch must own hidden, weights and int64 ids")
        post = row["phases"]["post_combine"]["operators_ms"]
        require(("dsv4_mhc_final_post" in post) == (layer == 42), "final mHC placement")
        require(("dsv4_lm_head" in post) == (layer == 42 and not idle), "selected LM-row ownership")
        require(("dsv4_logits_gather" in post) == (layer == 42 and not idle),
                "backend logits gather must follow selected LM rows only")
    require(not phase_ops, "EP trace has work without a phase-ledger owner")
    for rows in waves.values():
        require(len(rows) == dp * 8, "EP wave is missing source-rank participants")
        physical = sum(row["physical_local_rows"] for row in rows)
        require(sum(row["synthetic_destination_assignments"] for row in rows) == physical * 6,
                "synthetic destination assignments do not conserve physical source rows")
    require(counts["real_ep_lanes"] > 0 and counts["idle_ep_lanes"] > 0, "real/idle scope not exercised")
    return {**counts, **cache_zero, "stage_rows": len(stage_rows), "ep_waves": len(waves),
            "attention_boundaries": [{"query": q, "context": c, "sequence": q + c}
                                     for q, c in sorted(set(stage_inputs.values()), key=lambda value: value[1])],
            "physical_cache_retirement": "UNKNOWN; logical source writes and existing grouped allocation only",
            "idle_attention_trace_coverage": "existing request-free path emits no request/stage attention trace",
            "max_stage_conservation_error_ms": max_stage_error,
            "max_phase_conservation_error_ms": max_phase_error}


def command(case, output_root, dp, *, prefill_tokens=17, chunk_size=16,
            decode_tokens=2, num_blocks=256):
    require(all(type(value) is int and value > 0
                for value in (prefill_tokens, chunk_size, decode_tokens, num_blocks)),
            "workload token and block counts must be positive integers")
    max_model_len = max(256, prefill_tokens + decode_tokens)
    return [sys.executable, "-m", "frontier.main",
        "--simulation_mode", case, "--sys_arch", "co-location",
        "--log_level", "warning", "--decode_cuda_graph_mode", "none",
        "--cc_backend_config_type", "analytical",
        "--replica_config_model_name", "deepseek-ai/DeepSeek-V4-Flash",
        "--replica_config_device", "h800", "--replica_config_network_device", "h800_dgx",
        "--replica_config_num_pipeline_stages", "1",
        "--replica_config_attn_tensor_parallel_size", "8",
        "--replica_config_attn_dp", str(dp),
        "--replica_config_moe_tensor_parallel_size", "1",
        "--replica_config_moe_expert_parallel_size", str(8 * dp),
        "--replica_config_total_expert_num", "256", "--replica_config_router_topk", "6",
        "--replica_config_moe_routing_distribution_type", "balanced",
        "--replica_config_attention_backend", "FLASHMLA_SPARSE_DSV4",
        "--replica_config_kv_cache_dtype", "fp8",
        "--replica_config_indexer_kv_cache_dtype", "fp8",
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_batch_size_cap", "1",
        "--vllm_v1_scheduler_config_block_size", "256",
        "--vllm_v1_scheduler_config_num_blocks", str(num_blocks),
        "--vllm_v1_scheduler_config_num_blocks_mode", "explicit",
        "--vllm_v1_scheduler_config_max_model_len", str(max_model_len),
        "--vllm_v1_scheduler_config_max_tokens_in_batch", str(chunk_size),
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", str(chunk_size),
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms", "0.01",
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", str(output_root / "requests.csv"),
        "--trace_request_generator_config_max_tokens", str(max_model_len), "--seed", "42",
        "--metrics_config_output_dir", str(output_root / "metrics"),
        "--metrics_config_cache_dir", str(output_root / "cache"),
        "--metrics_config_run_id", f"dsv4_structural_{case}",
        "--metrics_config_write_metrics", "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics", "--metrics_config_store_operation_metrics",
        "--metrics_config_store_frontier_stage_batch_ledger",
        "--metrics_config_store_frontier_stage_batch_ledger_summary",
        "--metrics_config_enable_op_level_tracing", "--metrics_config_enable_per_layer_expansion",
        "--metrics_config_num_requests_to_trace_per_layer", str(dp),
        "--no-metrics_config_store_plots", "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace"]


def run_case(case, output_root, dp, *, prefill_tokens=17, chunk_size=16,
             decode_tokens=2, num_blocks=256):
    root = output_root / case
    root.mkdir(parents=True, exist_ok=False)
    with (root / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "arrived_at", "num_prefill_tokens", "num_decode_tokens"))
        writer.writeheader()
        for _ in range(dp):
            writer.writerow(dict(arrived_at=0.0, num_prefill_tokens=prefill_tokens,
                                 num_decode_tokens=decode_tokens))
    workload = dict(prefill_tokens=prefill_tokens, chunk_size=chunk_size, decode_tokens=decode_tokens)
    argv = command(case, root, dp, **workload, num_blocks=num_blocks)
    (root / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    before = source_identity()
    decision_path = root / "schedule_decisions.jsonl"
    with (root / "run.log").open("w") as stream:
        completed = subprocess.run(argv, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
                 "FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH": str(decision_path)},
            timeout=900)
    (root / "source_identity.json").write_text(json.dumps({
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "c17_sources_before": before,
        "c17_sources_after": source_identity(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    require(before == source_identity(), "C17 source files changed during the run")
    require(completed.returncode == 0, f"simulator failed: {root / 'run.log'}")
    runs = list((root / "metrics").rglob("config.json"))
    require(len(runs) == 1, "one metrics run is required")
    run_dir = runs[0].parent
    validation = validate_artifacts(run_dir, dp, decision_path=decision_path, **workload)
    return {"case": case, "logical_gpus": dp * 8, "run_dir": str(run_dir),
            "requests_completed": dp, "workload": workload, "num_blocks": num_blocks, **validation,
            "evidence_level": "CPU dummy structural; native assignments and latency UNKNOWN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dp", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--case", choices=("offline", "online", "both"), default="both")
    parser.add_argument("--prefill-tokens", type=int, default=17)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=256)
    args = parser.parse_args()
    cases = ("offline", "online") if args.case == "both" else (args.case,)
    workload = dict(prefill_tokens=args.prefill_tokens, chunk_size=args.chunk_size,
                    decode_tokens=args.decode_tokens)
    results = [run_case(case, args.output_root, args.dp, **workload,
                        num_blocks=args.num_blocks) for case in cases]
    (args.output_root / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
