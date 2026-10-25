"""H4a-4: run a native reference's Frontier input on the step-cost path and record what was changed.

    python h4a4_driver.py <case> <output root>     # case: native06 (H800 P0), p2_32 or p2_64 (H200 P2)

The input is the reference's retained flat config (case/inputs/frontier.json, digest pinned in CASES).
The driver rebuilds SimulationConfig from it through the existing flat dataclass, applies only
the BINDING and case overrides and new output paths, compares the decode KV admission with vLLM's
logged maximum concurrency, and runs the simulator. receipt.json records the source commit, the
override diff, the fields the current code adds with their defaults, and the KV admission check.
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
H200 = TASK / "h200_mainline"
DP4_TABLE = H200 / "diagnostic/step_cost_attn_dp4/dsv4_step_cost.json"
# num_blocks and max_concurrency are the reference's vLLM values: serving/rank*.json inventory.cache.num_blocks and
# "Maximum concurrency for 7,168 tokens per request: X" in server.log, equal on every node of both roles.
CASES = {
    "native06": dict(
        input=TASK / "v4_p0_native_20261021/candidate_h800_p0_06/case/inputs/frontier.json",
        input_sha256="747a08e5da6759f8189d29b2c8419ab48eb215ac2db4a2596af4f1d15ef34897",
        table=REPO / "data/profiling/compute/h800/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json",
        num_blocks=54740, max_concurrency=18.59, overrides={}),
    "p2_32": dict(
        input=H200 / "p2_32_reference/package_01/case/inputs/frontier.json",
        input_sha256="5ff79a515c470f3ea537c9b96777a805352660ec262ab00f018e49504cd88b43",
        table=REPO / "data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json",
        num_blocks=109934, max_concurrency=37.34, overrides={}),
    "p2_64": dict(
        input=H200 / "p2_64_reference/package_01/case/inputs/frontier.json",
        input_sha256="3aac0a3f6b829a478b4a350614722d386398271d5eb0cf670299bc9a533c8cf4",
        table=DP4_TABLE, num_blocks=112914, max_concurrency=38.35,
        # The loader reads the table beside the resolved attention input and accepts one topology per directory;
        # the attn_dp=4 table therefore has a directory of its own.
        overrides={"random_forrest_execution_time_predictor_config_atten_input_file": str(DP4_TABLE.parent / "attention.csv")}),
}

# Design section 5 (KV binding) and the H4a-2 role inheritance; output paths are new.
BINDING = {
    "vllm_v1_scheduler_config_num_blocks_mode": "explicit",
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


def outputs(root, name):
    return {
        "metrics_config_output_dir": str(root / "metrics"),
        "metrics_config_cache_dir": str(root / "cache"),
        "metrics_config_run_id": f"h4a4_{name}_step_cost",
        "measured_cc_backend_config_cache_dir": str(root / "cc_cache"),
    }


def build_config(name, case, root):
    from frontier.config import SimulationConfig
    from frontier.config.flat_dataclass import create_flat_dataclass

    raw = case["input"].read_bytes()
    if hashlib.sha256(raw).hexdigest() != case["input_sha256"]:
        raise SystemExit(f"{case['input']} digest differs from the {name} retained input")
    values = json.loads(raw)
    flat_cls = create_flat_dataclass(SimulationConfig)
    current = {field.name: field for field in flat_cls.__dataclass_fields__.values()}
    removed = sorted(set(values) - set(current))
    if removed:
        raise SystemExit(f"{name} fields no longer in SimulationConfig: {removed}")
    added = {}
    for field_name in sorted(set(current) - set(values)):
        field = current[field_name]
        if field.default is not MISSING:
            added[field_name] = field.default
        elif field.default_factory is not MISSING:
            added[field_name] = field.default_factory()
        else:
            raise SystemExit(f"required field {field_name} is not in the {name} input")
    overrides = {**BINDING, "vllm_v1_scheduler_config_num_blocks": case["num_blocks"], **case["overrides"],
                 **outputs(root, name)}
    diff = {field_name: {"input": values[field_name], "h4a4": value} for field_name, value in overrides.items()
            if values[field_name] != value}
    flat = flat_cls(**{**added, **values, **overrides})
    config = flat.reconstruct_original_dataclass()
    config.__flat_config__ = flat
    return config, {"added_fields": added, "override_diff": diff}


def kv_admission(simulator, native_max_concurrency):
    from frontier.types import ClusterType

    scheduler = simulator._global_scheduler.get_cluster_scheduler(ClusterType.DECODE)
    rows = {}
    for (replica_id, lane), replica in sorted(scheduler._replica_schedulers.items()):
        blocks = replica._kv_cache_manager.get_num_admission_blocks(
            int(replica._max_model_len), max_model_len=int(replica._max_model_len),
            max_in_flight_tokens=int(replica._config.max_tokens_in_batch),
        )
        pool = int(replica._config.num_blocks)
        rows[f"decode_replica{replica_id}_lane{lane}"] = {"num_blocks": pool, "admission_blocks_at_max_model_len": blocks,
                                                         "max_concurrency": pool / blocks}
    pools = {row["num_blocks"] for row in rows.values()}
    demands = {row["admission_blocks_at_max_model_len"] for row in rows.values()}
    [pool], [demand] = pools, demands
    # Per-request block counts for which vLLM's two-decimal log reads the native concurrency.
    native = [b for b in range(1, pool + 1) if round(pool / b, 2) == native_max_concurrency]
    return {**rows, "native": native_max_concurrency, "native_admission_blocks": [min(native), max(native)],
            "status": "MATCH" if demand in native else "DIFFERS"}


def main(name, root):
    case = CASES[name]
    root.mkdir(parents=True, exist_ok=True)
    from frontier.utils.random import set_seeds

    config, changes = build_config(name, case, root)
    receipt = {
        "status": "STARTED", "label": "DIAGNOSTIC", "case": name,
        "source_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, check=True).stdout.strip(),
        "source_dirty_paths": subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "frontier", "data"],
                                             capture_output=True, text=True, check=True).stdout.splitlines(),
        "input": str(case["input"]), "input_sha256": case["input_sha256"],
        "step_cost_table": str(case["table"]),
        "step_cost_table_sha256": hashlib.sha256(case["table"].read_bytes()).hexdigest(),
        "python": sys.version.split()[0], **changes,
    }
    write = lambda: (root / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    write()
    set_seeds(config.seed)
    from frontier.simulator import Simulator

    simulator = Simulator(config)
    receipt["kv_admission"] = kv_admission(simulator, case["max_concurrency"])
    write()
    started = time.monotonic()
    simulator.run()
    receipt.update(status="FINISHED", run_wall_s=time.monotonic() - started,
                   metrics_dir=str(config.metrics_config.output_dir))
    write()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in CASES:
        raise SystemExit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]))
