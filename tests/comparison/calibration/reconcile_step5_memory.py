"""Reconstruct retained native per-rank budgets without importing vLLM or Torch.

This is a case-specific evidence reader, not a simulator capacity estimator.
The topology and before-change pins refer to baseline 7abd2fa. Native logs and
the exact tensor census remain the authority; rounded logs are not exact bytes.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

from frontier.config import ReplicaConfig
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType


GIB = 1024**3
PROFILE = re.compile(
    r"Actual usage is ([\d.]+) GiB for consumed memory.*?, ([\d.]+) GiB for "
    r"peak activation, and ([\d.]+) GiB for CUDAGraph memory.*?"
    r"Current kv cache memory in use is ([\d.]+) GiB"
)
RANK = re.compile(r"Worker_(?:DP(\d+)_)?TP(\d+)(?:_EP(\d+))? ")


def read_ranks(path, role, lane):
    workers, selected = {}, []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        rank = RANK.search(line)
        if rank is None:
            continue
        worker = workers.setdefault(int(rank[2]), {"role": role, "dp": lane, "rank": int(rank[2])})
        if "Model loading took" in line:
            worker["native_load_gib"] = float(re.search(r"Model loading took ([\d.]+)", line)[1])
        if "Graph capturing finished" in line:
            worker.setdefault("native_graph_gib", []).append(float(re.search(r"took ([\d.]+) GiB", line)[1]))
        profile = PROFILE.search(line)
        if profile:
            keys = ("native_consumed_gib", "native_peak_including_graph_estimate_gib",
                    "native_final_graph_gib", "native_available_gib")
            worker.update(zip(keys, map(float, profile.groups())))
        if any(word in line for word in ("Model loading took", "Actual usage is", "Graph capturing finished", "Available KV cache memory")):
            selected.append({"line": number, "text": line})
    return workers, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "selected_lines": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--pd-log-root", type=Path, required=True)
    parser.add_argument("--engine-json", type=Path, required=True)
    parser.add_argument("--visible-gpu-gib", type=float, required=True)
    parser.add_argument("--nominal-gpu-gib", type=float, required=True)
    parser.add_argument("--concurrent-batches", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = json.loads(args.engine_json.read_text())["engine"]
    block, length = engine["block_size"], engine["max_model_len"]
    utilization = engine["gpu_memory_utilization"]
    entries = [
        ("single", 0, args.task_dir / f"native_request_physical_audit_20261020_04/cloud_selected/node_{i}/native.stdout",
         (16, 1, 16, 1), 40_599_958_208, 1_036_282, 101_579)
        for i in range(2)
    ]
    for root in sorted(path for path in args.pd_log_root.iterdir() if path.is_dir()):
        plan = json.loads((root / "result.json").read_text())["plan"]
        entries.append((plan["role"], plan["lane"], root / "server.log",
                        (8, 2, 1, 16), 42_056_012_480, 979_459, 96_009))
    rows, sources = [], []
    for role, lane, path, topology, before, native_tokens, native_blocks in entries:
        workers, source = read_ranks(path, role, lane)
        sources.append(source)
        replica = ReplicaConfig(
            model_name="stepfun-ai/Step-5-Preview", device="h800",
            attn_tensor_parallel_size=topology[0], attn_dp=topology[1],
            moe_tensor_parallel_size=topology[2], moe_expert_parallel_size=topology[3],
            attention_backend="STEP5_DSA", kv_cache_dtype="auto",
        )
        planner = MemoryPlanner(replica, SimpleNamespace(total_memory_gb=args.nominal_gpu_gib),
                                ClusterType.MONOLITHIC, max_num_seqs=engine["max_num_seqs"])
        after = planner._get_parameter_memory_per_device()
        capacity = planner._get_component_kv_cache_capacity(block)
        # Native max-admission formula, not nominal max_model_len for SWA.
        window = replica.model_config.sliding_window
        inflight = args.concurrent_batches * engine["max_num_batched_tokens"]
        swa = -(-min(window - 1 + inflight, length) // block) + 1
        required = (len(capacity.groups) - 1) * swa + -(-length // block)
        requested = int(args.nominal_gpu_gib * GIB * utilization)
        for rank, worker in sorted(workers.items()):
            consumed = worker["native_consumed_gib"]
            peak = worker["native_peak_including_graph_estimate_gib"]
            overhead = round((args.nominal_gpu_gib - args.visible_gpu_gib) * utilization * GIB
                             + (consumed + peak) * GIB - after)
            blocks = planner.get_num_blocks(block_size=block, gpu_memory_utilization=utilization,
                                            non_kv_cache_overhead_bytes=overhead)
            native_load = 41_993_472_000 if role == "single" else round(worker["native_load_gib"] * GIB)
            worker.update(
                topology=dict(zip(("tp", "dp", "moe_tp", "ep"), topology)), source=str(path),
                native_load_bytes=native_load,
                native_load_bytes_precision="exact_census" if role == "single" else "rounded_GiB_log_point",
                frontier_before_bytes=before, frontier_after_bytes=after,
                frontier_profile_overhead_bytes=overhead,
                frontier_profile_available_bytes=requested - after - overhead,
                frontier_profile_rank_blocks=blocks,
                frontier_profile_rank_equivalent_tokens=int(blocks / required * length),
                native_common_pool_blocks=native_blocks, native_equivalent_tokens=native_tokens,
                native_non_torch_gib=consumed - native_load / GIB,
                native_activation_without_graph_estimate_gib=peak - worker["native_graph_gib"][0],
                native_group_block_counts=[native_blocks] * len(capacity.groups),
                frontier_group_block_counts=[blocks] * len(capacity.groups),
            )
            for label, weights in (("before", before), ("after_source_only", after)):
                available = requested - weights
                count = (available - capacity.fixed_bytes_per_request_slot * engine["max_num_seqs"]) // capacity.physical_block_size_bytes
                worker[f"frontier_{label}_available_bytes"] = available
                worker[f"frontier_{label}_blocks"] = count
                worker[f"frontier_{label}_tokens"] = int(count / required * length)
            for label, weights in (("before", before), ("after", after)):
                worker[f"{label}_load_error_bytes"] = weights - native_load
                worker[f"{label}_load_relative_error"] = weights / native_load - 1
            rows.append(worker)
    assert len(rows) == 48, len(rows)
    census = args.task_dir / "native_request_physical_audit_20261020_04/cloud_selected/node_0/device_preparation.json"
    registered = {worker["rank"]: worker["device_observation"]["state_inventory"]["resident_parameters"]["unique_registered_storage_bytes"]
                  for worker in json.loads(census.read_text())["workers"]}
    assert len(registered) == 16 and set(registered.values()) == {41_909_084_704}
    for row in rows:
        if row["role"] == "single":
            row["native_registered_storage_bytes"] = registered[row["rank"]]
    summary = []
    for role in ("single", "prefill", "decode"):
        group = [row for row in rows if row["role"] == role]
        count = min(row["frontier_profile_rank_blocks"] for row in group)
        tokens = int(count / required * length)
        reference = group[0]["native_equivalent_tokens"]
        summary.append(dict(role=role, native_blocks=group[0]["native_common_pool_blocks"],
                            frontier_profile_global_min_blocks=count, native_tokens=reference,
                            frontier_profile_global_min_tokens=tokens, absolute_error_tokens=abs(tokens - reference),
                            relative_error=tokens / reference - 1,
                            frontier_available_bytes_at_min=min(row["frontier_profile_available_bytes"] for row in group),
                            native_global_min_profile_overhead_bytes=max(row["frontier_profile_overhead_bytes"] for row in group)))
    result = dict(date="2026-10-20", scope="retained_native_capacity_reconstruction_not_predictive_accuracy",
                  per_rank=rows, global_min_comparison=summary, retained_log_lines=sources,
                  registered_census=dict(path=str(census), sha256=hashlib.sha256(census.read_bytes()).hexdigest(), bytes_by_rank=registered),
                  engine=engine, native_max_admission_blocks=required)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
