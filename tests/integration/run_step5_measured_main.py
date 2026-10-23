"""Run the actual main CLI with explicitly synthetic CPU Step P16/D16 inputs.

The supplied profile tables come from step5_measured_compute_fixture.py. This
runner never patches simulator classes and establishes no native/accuracy gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
import hashlib
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

from frontier.attention.memory import AttentionKVCacheSelection, get_attention_kv_transfer_payload
from frontier.attention.step5_memory import resolve_step5_kv_cache_capacity
from frontier.config import ReplicaConfig


REPO = Path(__file__).resolve().parents[2]
LIMITATIONS = [
    "Synthetic CPU compute profiles and 3.5 ms READ cost; no native admission or accuracy claim.",
    "One request per mode, not formal 128 requests, 16 warmups, or three repetitions.",
    "CPU overhead is explicitly skipped; collective_sim uses the built-in H800 topology.",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    return dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        runner_sha256=file_hash(Path(__file__)),
        fixture_sha256=file_hash(REPO / "tests/integration/step5_measured_compute_fixture.py"),
    )


def read_profile(path):
    fixture = json.loads(path.read_text())
    require(fixture["schema"] in ("step5_measured_compute_cpu_fixture.v2","step5_position_compute_cpu_fixture.v1"),
            "Historical full-prompt fixture cannot cover the corrected P target; generate a new fixture")
    for key, value in (("evidence_kind", "cpu_fixture"), ("native_admitted", False),
                       ("device", "h800"), ("attention_tp", 8), ("attention_dp", 2),
                        ("moe_tp", 1), ("moe_ep", 16), ("max_model_len", 7168),
                       ("block_size", 16), ("batch_width", 1)):
        require(fixture[key] == value, f"Unsupported fixture {key}: {fixture[key]}")
    require(fixture["decode_cuda_graph_mode"] == "full_and_piecewise", "Fixture graph policy must be native")
    require(fixture["cases"][0] == [6128, 0, "prefill", "none"], "Fixture must cover actual P6128")
    expected="position_cyclic" if fixture["schema"] == "step5_position_compute_cpu_fixture.v1" else "balanced"
    require(fixture["routing_distribution_type"] == expected and fixture["routing_seed"] == 42,
             "Fixture must retain its explicit routing policy")
    return fixture


def write_read_cost(path):
    from tests.integration.step5_measured_compute_fixture import PREFILL_CASES

    replica = ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview", device="h800", network_device="h800_dgx",
        attn_tensor_parallel_size=8, attn_dp=2, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=16, total_expert_num=352, router_topk=8,
        attention_backend="STEP5_DSA", kv_cache_dtype="auto",
    )
    capacity = resolve_step5_kv_cache_capacity(
        replica.model_config, 16, AttentionKVCacheSelection.from_replica(replica, block_size=16),
    )
    measurements = []
    for prompt, prefix, _, _ in sorted(PREFILL_CASES, key=lambda case: case[0] != 6144):
        payload = get_attention_kv_transfer_payload(
            replica.model_config, capacity, prompt_tokens=prefix, tensor_parallel_size=8,
            target_reserved_blocks=len(capacity.groups) * (prefix // 16),
        )
        measurements.append(dict(num_prefill_tokens=prompt, num_computed_tokens=prefix,
                                 transfer_bytes_per_rank=payload.transfer_bytes_per_rank, time_ms=3.5))
    document = dict(
        schema="nixl_read_cost.v1", model_architecture_profile="step5",
        profile_identity="CPU_CONTRACT_FIXTURE_NOT_NATIVE",
        source_device="h800", target_device="h800",
        source_attention_tp=8, target_attention_tp=8, duration_scope="read_to_gpu_ready",
        cache_capacity=asdict(capacity), measurements=measurements,
    )
    write_json(path, document)
    return document


def command(mode, root, fixture, read_cost):
    args = [
        "--simulation_mode", mode, "--sys_arch", "pd-disaggregation",
        "--no-enable_parallel_clusters", "--no-enable_thinking_mode",
        "--no-speculative_decoding_config_enabled", "--no-use_cuda_graph",
        "--decode_cuda_graph_mode", fixture["decode_cuda_graph_mode"],
        "--cudagraph_capture_sizes", *map(str, fixture["cudagraph_capture_sizes"]),
        "--cc_backend_config_type", "collective_sim",
        "--collective_sim_cc_backend_config_scenario_profile", "h800_rail",
        "--collective_sim_cc_backend_config_cache_dir", str(root / "cc_cache"),
        "--cluster_scheduler_config_type", "vllm_load_balancing",
        "--vllm_load_balancing_cluster_scheduler_config_policy_version", "v0.28.0",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", "1",
        "--replica_config_model_name", "stepfun-ai/Step-5-Preview",
        "--replica_config_device", "h800", "--replica_config_network_device", "h800_dgx",
        "--replica_config_attn_dp", "2", "--replica_config_attention_backend", "STEP5_DSA",
        "--replica_config_kv_cache_dtype", "auto",
        "--replica_config_moe_routing_distribution_type", fixture.get("routing_distribution_type","balanced"),
        "--replica_config_moe_routing_seed", "42",
        "--replica_scheduler_config_type", "vllm_v1",
        "--no-vllm_v1_scheduler_config_enable_prefix_caching",
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        "--vllm_v1_scheduler_config_batch_size_cap", "128",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_max_model_len", str(fixture["max_model_len"]),
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "8192",
        "--vllm_v1_scheduler_config_gpu_memory_utilization", "0.92",
        "--vllm_v1_scheduler_config_num_blocks", "0",
        "--vllm_v1_scheduler_config_num_blocks_mode", "memory_planner",
        "--no-vllm_v1_scheduler_config_enable_runtime_non_kv_cache_overhead_profiling",
        "--analytical_kv_cache_transfer_config_kv_connector", "nixl",
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", str(root / "requests.csv"),
        "--trace_request_generator_config_max_tokens", "8192",
        "--seed", "42", "--log_level", "warning",
        "--metrics_config_output_dir", str(root / "metrics"),
        "--metrics_config_cache_dir", fixture["cache_dir"],
        "--metrics_config_run_id", f"step5_measured_cpu_main_{mode}",
        "--metrics_config_write_metrics", "--metrics_config_store_request_metrics",
        "--metrics_config_enable_metrics_ground_truth_trace",
        "--metrics_config_keep_individual_batch_metrics",
        "--metrics_config_store_frontier_stage_batch_ledger",
        "--metrics_config_store_frontier_stage_batch_ledger_summary",
        "--no-metrics_config_enable_op_level_tracing", "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace", "--metrics_config_write_json_trace",
    ]
    if fixture.get("routing_distribution_type","balanced") == "position_cyclic":
        args.append("--metrics_config_enable_op_level_tracing")
        args.remove("--no-metrics_config_enable_op_level_tracing")
        args.extend(("--collective_sim_cc_backend_config_profiling_data_dir",str(Path(fixture["collective_plan_input_file"]).parents[1])))
        args.extend(("--collective_sim_cc_backend_config_placement_order","TP,DP"))
    for role in ("prefill", "decode"):
        prefix = f"--cluster_config_{role}_replica_config_"
        for key, value in dict(num_pipeline_stages=1, attn_tensor_parallel_size=8,
                               moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
                               total_expert_num=352, router_topk=8, device="h800").items():
            args.extend((prefix + key, str(value)))
    options = {**fixture["predictor_config"], "nixl_read_profile": str(read_cost)}
    prefix = "random_forrest_execution_time_predictor_config_"
    for key, value in options.items():
        if isinstance(value, bool):
            args.append(f"--{'' if value else 'no-'}{prefix}{key}")
        else:
            args.extend((f"--{prefix}{key}", *map(str, value if isinstance(value, list) else [value])))
    return [sys.executable, "-B", "-m", "frontier.main", *args]


def validate(root, mode, document):
    [config_path] = list((root / "metrics").rglob("config.json"))
    run = config_path.parent
    config = json.loads(config_path.read_text())
    cluster = config["cluster_config"]
    require(config["simulation_mode"] == mode, "Wrong mode")
    require(config["decode_cuda_graph_mode"] == "full_and_piecewise", "Wrong graph mode")
    require(not config["use_cuda_graph"] and config["cudagraph_capture_sizes"] == [1, 2, 4, *range(8, 256, 8), 256],
            "Wrong capture configuration")
    for role in ("prefill", "decode"):
        replica = cluster[f"{role}_replica_config"]
        require([replica[k] for k in ("attn_tensor_parallel_size", "attn_dp",
                "moe_tensor_parallel_size", "moe_expert_parallel_size", "world_size")]
                == [8, 2, 1, 16, 16], f"Wrong {role} topology")
    scheduler = cluster["replica_scheduler_config"]
    require(scheduler["max_model_len"] == 7168 and scheduler["batch_size_cap"] == 128,
            "Run must retain native capacity and workload")
    require(not scheduler["enable_prefix_caching"] and not scheduler["enable_chunked_prefill"],
            "Prefix/chunked prefill must stay disabled")
    system = json.loads((run / "system_metrics.json").read_text())
    metadata = system["simulation_metadata"]
    require(metadata["total_requests"] == metadata["completed_requests"] == 1, "Incomplete run")
    with (run / "request_metrics.csv").open() as stream:
        [request] = list(csv.DictReader(stream))
    require(int(request["request_num_prefill_tokens"]) == 6144, "Wrong input count")
    require(int(request["request_num_decode_tokens"]) == 1024, "Wrong output count")
    for key in ("request_e2e_time", "client_ttft", "tpot"):
        require(math.isfinite(float(request[key])) and float(request[key]) > 0, f"Invalid final metric {key}")
    [transfer] = [json.loads(line) for line in (run / "frontier_transfer_ledger.jsonl").read_text().splitlines()]
    require(transfer["status"] == "completed" and transfer["target_bound"], "Incomplete READ")
    require(transfer["bytes"] == document["measurements"][0]["transfer_bytes_per_rank"], "Wrong READ bytes")
    require(abs(transfer["duration_ms"] - 3.5) < 1e-6, "Wrong READ cost")
    events = json.loads((run / "event_trace.json").read_text())
    [read_end] = [event for event in events if event["event_type"] == "KV_CACHE_TRANSFER_END"]
    require(read_end["payload_kind"] == "prefix_pages_per_worker_lane", "Wrong READ prefix semantics")
    require(read_end["target_ready_time"] == read_end["source_release_time"] == read_end["time"],
            "READ must make the reserved D lane ready and release the source")
    require(read_end["target_reserved_common_blocks"] == 4 * (6128 // 16), "Wrong D prefix reservation")
    require(read_end["source_prompt_common_blocks"] == 6128 // 16, "Wrong computed P prefix ownership")
    rows = [json.loads(line) for line in (run / "frontier_stage_batch_ledger.jsonl").read_text().splitlines()]
    useful = [row for row in rows if row["request_ids"]]
    prefill = [row for row in useful if row["cluster_type"] == "PREFILL"]
    decode = [row for row in useful if row["cluster_type"] == "DECODE"]
    require(len(prefill) == 1 and prefill[0]["request_num_tokens"] == [6128], "Wrong useful P forward census")
    require(len(decode) == 1024 and Counter(tuple(row["request_num_tokens"]) for row in decode)
            == Counter({(16,): 1, (1,): 1023}), "Wrong D tail/forward census")
    first_decode = min(decode, key=lambda row: row["stage_start_ts"])
    require(first_decode["stage_start_ts"] >= read_end["target_ready_time"], "D launched before READ readiness")
    if cluster["prefill_replica_config"]["moe_routing_distribution_type"] == "position_cyclic":
        lanes=[json.loads(line) for line in (run/"frontier_ep_wave_lane_ledger.jsonl").read_text().splitlines()]
        require(lanes and all("routing_source" in row for row in lanes),"Missing actual position source census")
        empty=0
        for row in lanes:
            source=row["routing_source"]
            require(sum(s["physical_rows"] for s in source["source_lanes"])==source["gathered_physical_rows"],"Physical source conservation failed")
            from frontier.moe_routing_policy import source_lane_index_for_ep
            from frontier.operators.families import get_moe_profiling_operators
            from frontier.operators.binding import bind_operator_query
            replica=cluster[row["cluster"].lower()+"_replica_config"]
            owner=source["source_lanes"][source_lane_index_for_ep(row["ep_id"],attention_tp_size=replica["attn_tensor_parallel_size"],
                source_lane_count=len(source["source_lanes"]),participant_ep_ids=range(replica["world_size"]))]
            context=row["phases"]["dispatch"]["collective_proofs"]["step5_moe_dispatch"]["plan"]["buffers"]["compute_context"]
            applicable={op.name for op in get_moe_profiling_operators(context)}
            require({name for phase in row["phases"].values() for name in phase["operators_ms"]} >= applicable,"Missing applicable compute source operators")
            empty+=int(owner["live_rows"]==0)
            for phase in row["phases"].values():
                require("moe_shuffling" not in phase["operators_ms"],"Position path charged legacy shuffle")
                compute={name for name,duration in phase["operators_ms"].items() if (duration>0 or name in applicable) and bind_operator_query(name).operator.trace_kind.value=="COMPUTE"}
                require(set(phase.get("compute_proofs",{})) == compute,"Missing actual compute source proof correspondence")
                require(all(bind_operator_query(name).family_id!="moe" or name in applicable for name in compute),"Inapplicable compute source operator")
                for name,proof in phase.get("compute_proofs",{}).items():
                    require(proof["exact_lookup"] and math.isfinite(proof["duration_ms"]) and proof["duration_ms"]>0 and
                            not isinstance(proof["kernel_count"],bool) and math.isfinite(proof["kernel_count"]) and proof["kernel_count"]>0 and proof["kernel_count"]==int(proof["kernel_count"]) and
                            proof["duration_ms"]==phase["operators_ms"].get(name),"Missing exact positive compute pair")
                for proof in phase.get("collective_proofs",{}).values():
                    plan,cost=proof["plan"],proof["cost"]
                    require(plan["sizes"]==[s["physical_rows"] for s in source["source_lanes"]],"Collective physical sizes mismatch")
                    require(len(cost["kernel_counts"])==16 and all(n>0 for n in cost["kernel_counts"]),"Missing collective count pairs")
                    require(json.loads(cost["count_receipt_json"])["evidence_kind"]=="cpu_fixture","CPU actual-main must remain CPU evidence")
                    require(abs(cost["duration_ms"]-cost["network_ms"]-max(cost["local_rank_ms"])-cost["remainder_ms"])<1e-9,"Collective completion equation mismatch")
        require(empty>0,"No empty physical source scope retained")
    require(all(math.isfinite(row["execution_time"]["total_time_ms"])
                and row["execution_time"]["total_time_ms"] > 0 for row in useful), "Invalid stage cost")
    return dict(mode=mode, status="PASS", evidence_kind="cpu_fixture", native_admitted=False,
                 run_dir=str(run), request=request, transfer=transfer, system_metrics=system,
                read_completion=read_end, useful_forward_counts=dict(prefill=len(prefill), decode=len(decode)),
                event_counts=dict(Counter(str(event["event_type"]) for event in events)), limitations=LIMITATIONS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("offline", "online", "all"), default="all")
    args = parser.parse_args()
    fixture = read_profile(args.fixture_manifest)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    document = write_read_cost(root / "synthetic_read_cost.json")
    write_json(root / "environment.json", dict(python=sys.version, executable=sys.executable,
               platform=platform.platform(), task_id="D23-ACTUAL-MAIN", limitations=LIMITATIONS,
               source=source_identity()))
    write_json(root / "input_hashes.json", {
        str(path): file_hash(path) for path in [args.fixture_manifest, root / "synthetic_read_cost.json",
            *[Path(value) for key, value in fixture["predictor_config"].items()
              if key.endswith("input_file") and isinstance(value, str)]]
    })
    results = []
    for mode in (("offline", "online") if args.mode == "all" else (args.mode,)):
        case = root / mode
        case.mkdir()
        (case / "requests.csv").write_text(
            "arrived_at,num_prefill_tokens,num_decode_tokens\n"
            f"{0.0 if mode == 'offline' else 0.25},6144,1024\n")
        argv = command(mode, case, fixture, root / "synthetic_read_cost.json")
        write_json(case / "command.json", argv)
        (case / "command.bash").write_text(shlex.join(argv) + "\n")
        env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "WANDB_DISABLED": "true",
                "VIDUR_DISABLE_WANDB": "1", "FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH": str(case / "scheduler.jsonl")}
        source = source_identity()
        started = time.monotonic()
        with (case / "run.log").open("w") as stream:
            try:
                result = subprocess.run(argv, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=2400)
            except subprocess.TimeoutExpired:
                result = None
        returncode = None if result is None else result.returncode
        write_json(case / "execution.json", dict(returncode=returncode,
                   timed_out=result is None, wall_s=time.monotonic() - started, source=source))
        require(source_identity() == source, "Source changed during actual CLI")
        if returncode != 0:
            receipt = dict(mode=mode, status="FAIL", returncode=returncode, timed_out=result is None,
                           run_log=str(case / "run.log"), limitations=LIMITATIONS)
            write_json(case / "validation.json", receipt)
            results.append(receipt)
            continue
        try:
            receipt = validate(case, mode, document)
        except (AssertionError, KeyError, ValueError, FileNotFoundError) as error:
            receipt = dict(mode=mode, status="FAIL", returncode=returncode,
                           validation_error=f"{type(error).__name__}: {error}",
                           run_log=str(case / "run.log"), limitations=LIMITATIONS)
        write_json(case / "validation.json", receipt)
        results.append(receipt)
    write_json(root / "summary.json", results)
    print(json.dumps([dict(mode=r["mode"], status=r["status"], run_dir=r.get("run_dir")) for r in results]))
    require(all(receipt["status"] == "PASS" for receipt in results), f"Actual main failed; inspect {root / 'summary.json'}")


if __name__ == "__main__":
    main()
