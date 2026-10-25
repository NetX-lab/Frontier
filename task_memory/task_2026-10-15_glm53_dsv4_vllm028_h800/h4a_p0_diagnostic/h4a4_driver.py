"""H4a-4: run the native06 Frontier input on the step-cost path and record what was changed.

The input is native06's retained flat config (case/inputs/frontier.json, digest pinned below).
The driver rebuilds SimulationConfig from it through the existing flat dataclass, applies only
the BINDING overrides and new output paths, compares the decode KV admission with vLLM's 18.59x,
and runs the simulator. receipt.json records the source commit, the override diff, the fields the
current code adds with their defaults, and the KV admission check.
"""
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import MISSING
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[1]
INPUT = TASK / "v4_p0_native_20261021/candidate_h800_p0_06/case/inputs/frontier.json"
INPUT_SHA256 = "747a08e5da6759f8189d29b2c8419ab48eb215ac2db4a2596af4f1d15ef34897"
TABLE = REPO / "data/profiling/compute/h800/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json"
# vLLM: "GPU KV cache size: 133,280 tokens, Maximum concurrency for 7,168 tokens per request: 18.59x".
NATIVE_MAX_CONCURRENCY = 18.59

# Design section 5 (KV binding) and the H4a-2 role inheritance; output paths are new.
BINDING = {
    "vllm_v1_scheduler_config_num_blocks_mode": "explicit",
    "vllm_v1_scheduler_config_num_blocks": 54740,
    "replica_config_attention_backend": "FLASHMLA_SPARSE_DSV4",
    "replica_config_kv_cache_dtype": "fp8",
    "replica_config_indexer_kv_cache_dtype": "fp8",
    # Per-step ledger: 64 MB for 3 requests of 4 tokens in the H4a-2 probe.
    "metrics_config_store_frontier_stage_batch_ledger": False,
    # The V4 idle-forward guard accepts only eager idle forwards. On the step-cost path the
    # graph mode is not a pricing input: the decode entries were measured on native graph
    # steps, and idle forwards, sklearn models and kernel-only tables are not used.
    "decode_cuda_graph_mode": "none",
}


def outputs(root):
    return {
        "metrics_config_output_dir": str(root / "metrics"),
        "metrics_config_cache_dir": str(root / "cache"),
        "metrics_config_run_id": "h4a4_native06_step_cost",
        "measured_cc_backend_config_cache_dir": str(root / "cc_cache"),
    }


def build_config(root):
    from frontier.config import SimulationConfig
    from frontier.config.flat_dataclass import create_flat_dataclass

    raw = INPUT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA256:
        raise SystemExit(f"{INPUT} digest differs from the native06 retained input")
    values = json.loads(raw)
    flat_cls = create_flat_dataclass(SimulationConfig)
    current = {field.name: field for field in flat_cls.__dataclass_fields__.values()}
    removed = sorted(set(values) - set(current))
    if removed:
        raise SystemExit(f"native06 fields no longer in SimulationConfig: {removed}")
    added = {}
    for name in sorted(set(current) - set(values)):
        field = current[name]
        if field.default is not MISSING:
            added[name] = field.default
        elif field.default_factory is not MISSING:
            added[name] = field.default_factory()
        else:
            raise SystemExit(f"required field {name} is not in the native06 input")
    overrides = {**BINDING, **outputs(root)}
    diff = {name: {"native06": values[name], "h4a4": value} for name, value in overrides.items()
            if values[name] != value}
    flat = flat_cls(**{**added, **values, **overrides})
    config = flat.reconstruct_original_dataclass()
    config.__flat_config__ = flat
    return config, {"added_fields": added, "override_diff": diff}


def kv_admission(simulator):
    from frontier.types import ClusterType

    scheduler = simulator._global_scheduler.get_cluster_scheduler(ClusterType.DECODE)
    rows = {}
    for (replica_id, lane), replica in sorted(scheduler._replica_schedulers.items()):
        blocks = replica._kv_cache_manager.get_num_admission_blocks(
            int(replica._max_model_len), max_model_len=int(replica._max_model_len),
            max_in_flight_tokens=int(replica._config.max_tokens_in_batch),
        )
        pool = int(replica._config.num_blocks)
        rows[f"decode_replica{replica_id}_lane{lane}"] ={"num_blocks": pool, "admission_blocks_at_max_model_len": blocks,
                                     "max_concurrency": pool / blocks}
    pools = {row["num_blocks"] for row in rows.values()}
    demands = {row["admission_blocks_at_max_model_len"] for row in rows.values()}
    [pool], [demand] = pools, demands
    # Per-request block counts for which vLLM's two-decimal log reads 18.59x.
    native = [b for b in range(1, pool + 1) if round(pool / b, 2) == NATIVE_MAX_CONCURRENCY]
    return {**rows, "native": NATIVE_MAX_CONCURRENCY, "native_admission_blocks": [min(native), max(native)],
            "status": "MATCH" if demand in native else "DIFFERS"}


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    from frontier.utils.random import set_seeds

    config, changes = build_config(root)
    receipt = {
        "status": "STARTED", "label": "DIAGNOSTIC",
        "source_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, check=True).stdout.strip(),
        "source_dirty_paths": subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "frontier", "data"],
                                             capture_output=True, text=True, check=True).stdout.splitlines(),
        "input": str(INPUT), "input_sha256": INPUT_SHA256,
        "step_cost_table": str(TABLE), "step_cost_table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest(),
        "python": sys.version.split()[0], **changes,
    }
    write = lambda: (root / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    write()
    set_seeds(config.seed)
    from frontier.simulator import Simulator

    simulator = Simulator(config)
    receipt["kv_admission"] = kv_admission(simulator)
    write()
    started = time.monotonic()
    simulator.run()
    receipt.update(status="FINISHED", run_wall_s=time.monotonic() - started,
                   metrics_dir=str(config.metrics_config.output_dir))
    write()


if __name__ == "__main__":
    main()
