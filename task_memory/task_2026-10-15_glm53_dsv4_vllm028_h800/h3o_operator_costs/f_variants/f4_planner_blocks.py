"""F4: the decode KV block count from Frontier's memory planner, against the native value.

Usage: f4_planner_blocks.py <case> <output root>   # p2_32 or p2_64

Builds the A10 configuration with the planner mode instead of the native explicit
value, initializes the simulator only, and writes planner_blocks.json: per decode
lane the planner num_blocks, plus the planner terms
(requested memory, parameter memory, physical block bytes) and the native values.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "a10"))
import a10_driver  # noqa: E402

PLANNER_FIELDS = {
    "vllm_v1_scheduler_config_num_blocks_mode": "memory_planner_profiled",
    "vllm_v1_scheduler_config_num_blocks": 0,
}


def main(name, root):
    from frontier.scheduler.utils.memory_planner import MemoryPlanner
    from frontier.simulator import Simulator
    from frontier.types import ClusterType
    from frontier.utils.random import set_seeds

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config, changes = a10_driver.build_config(name, root, PLANNER_FIELDS)
    set_seeds(config.seed)
    simulator = Simulator(config)
    scheduler = simulator._global_scheduler.get_cluster_scheduler(ClusterType.DECODE)
    replica = next(iter(scheduler._replica_schedulers.values()))
    # The scheduler keeps no planner; rebuild it from the same inputs to report its terms.
    planner = MemoryPlanner(replica_config=replica._replica_config, replica=replica._replica,
                            cluster_type=replica._cluster_type,
                            max_num_seqs=replica._admitted_request_capacity)
    capacity = replica._kv_cache_capacity
    result = dict(
        case=name, fields=PLANNER_FIELDS, override_diff=changes["override_diff"],
        native_num_blocks=a10_driver.CASES[name]["num_blocks"],
        planner=dict(
            requested_memory_bytes=planner._get_requested_memory_bytes(
                replica._config.gpu_memory_utilization),
            parameter_memory_per_device_bytes=planner.get_parameter_memory_per_device_bytes(),
            non_kv_cache_overhead_bytes=int(replica._config.non_kv_cache_overhead_bytes),
            physical_block_size_bytes=capacity.physical_block_size_bytes,
            reserved_blocks=capacity.reserved_blocks,
            total_memory_gb=replica._replica.total_memory_gb,
        ),
        # h4a4.kv_admission matches the pool against the native logged concurrency,
        # which only the native pool can satisfy; report the planner pool per lane.
        decode_lanes={f"replica{replica_id}_lane{lane}": int(scheduler_lane._config.num_blocks)
                      for (replica_id, lane), scheduler_lane in sorted(scheduler._replica_schedulers.items())},
    )
    (root / "planner_blocks.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps(result["planner"]), json.dumps(result["decode_lanes"]))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in a10_driver.CASES:
        raise SystemExit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]))
