"""A10: replay P2-32/P2-64 on V4 operator costs, with a reproducible receipt.

Usage: a10_driver.py <case> <output root>   # p2_32 or p2_64
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from a10_common import REPO, TASK, cache_inventory, load_module, table_record, sha256

h4a4 = load_module("h4a4_driver", TASK / "h4a_p0_diagnostic/h4a4_driver.py")
CASES = {name: h4a4.CASES[name] for name in ("p2_32", "p2_64")}
GIT_TIMEOUT_S = 30
# A10 report section 3.5, observation 4: measured launch-queue depth,
# native P2-64 wave run exp-1030-144848-164721; shared by both PDD roles.
DP_MAX_FORWARDS_IN_FLIGHT = 19


def operator_fields(name, root):
    fields = h4a4.outputs(root, name)
    fields.update(
        metrics_config_run_id=f"a10_{name}_operator_path",
        decode_cuda_graph_mode="full_and_piecewise",
        vllm_v1_scheduler_config_max_forwards_in_flight=DP_MAX_FORWARDS_IN_FLIGHT,
        replica_config_network_device="h200_dgx",
        analytical_kv_cache_transfer_config_nixl_read_completion="engine_step",
        cluster_config_decode_replica_scheduler_config_max_concurrent_batches=2,
        cudagraph_capture_sizes=json.loads(
            (REPO / "tests/fixtures/dsv4_native_dp_sync_rows.json").read_text()
        )["capture_sizes"],
    )
    return fields


def build_config(name, root, fields=None):
    case = {key: value for key, value in CASES[name].items() if key not in ("table", "overrides")}
    case["overrides"] = {}
    return h4a4.build_config(name, case, root, {**operator_fields(name, root), **(fields or {})})


def resolved_inputs(config):
    """Use the production family resolver and V4 loaders' sibling-table owners."""
    from frontier.execution_time_predictor.dsv4_measurement import dsv4_measurement_type
    from frontier.execution_time_predictor.dsv4_predictor import dsv4_input_path
    from frontier.execution_time_predictor.dsv4_runner import BLOCK_TABLE, REPLAY_TABLE
    from frontier.execution_time_predictor.measurement_input_paths import (
        resolve_measurement_input_paths, resolve_training_file_paths,
    )

    tables, inputs, dummy = {}, {}, {}
    for role, cluster in config.get_clusters().items():
        replica = cluster.replica_config
        predictor = cluster.execution_time_predictor_config
        binding = dict(device=replica.device, model=replica.model_config.get_name(),
                       network_device=replica.network_device)
        paths = resolve_training_file_paths(predictor, **binding)
        # Include the configured inputs in A1's sibling check, even inputs
        # whose family is inactive (the diagnostic loader checks CUDA_EVENT).
        inputs[role.name] = sorted({str(Path(path).resolve()) for path in paths.values() if path})
        dummy[role.name] = predictor.enable_dummy_mode
        family = resolve_measurement_input_paths(
            predictor, dsv4_measurement_type(replica.model_config), **binding,
        )
        consumed = {key: value for key, value in asdict(family).items()
                    if key in ("compute", "attention", "moe")}
        consumed["attention"] = dsv4_input_path(consumed["attention"])
        consumed["kernel_gap"] = paths["kernel_gap_input_file"]
        runner_root = Path(family.attention).parent
        consumed.update(runner_block=str(runner_root / BLOCK_TABLE),
                        runner_replay=str(runner_root / REPLAY_TABLE))
        cc = cluster.cc_backend_config
        consumed["collectives"] = cc.summary_csv_path.format(
            profiling_data_dir=cc.profiling_data_dir, NETWORK_DEVICE=replica.network_device,
        )
        for owner, path in consumed.items():
            resolved = str(Path(path).resolve())
            tables.setdefault(resolved, []).append(f"{role.name}/{owner}")
        inputs[role.name] = sorted(set(inputs[role.name]) | {str(Path(path).resolve()) for path in consumed.values()})
    records = [{**table_record(path), "consumers": owners} for path, owners in sorted(tables.items())]
    from frontier.execution_time_predictor.dsv4_step_cost_input import FILE_NAME
    step_tables = sorted({str(Path(path).parent / FILE_NAME)
                          for paths in inputs.values() for path in paths
                          if (Path(path).parent / FILE_NAME).exists()})
    return records, {
        "enable_dummy_mode": dummy,
        "resolved_inputs": inputs,
        "step_cost_files_beside_inputs": step_tables,
        "no_step_cost_file_beside_inputs": not step_tables,
    }


def git_output(*arguments):
    return subprocess.run(["git", "-C", str(REPO), *arguments], capture_output=True,
                          text=True, check=True, timeout=GIT_TIMEOUT_S).stdout


def failure_batch(error):
    from frontier.entities import Batch
    trace = error.__traceback__
    found = None
    while trace is not None:
        batch = trace.tb_frame.f_locals.get("batch")
        if isinstance(batch, Batch):
            metadata = batch.decode_cuda_graph_metadata
            found = dict(batch_id=batch.id, role=batch._cluster_type.name,
                         request_ids=[request.id for request in batch.requests],
                         query_lengths=batch.num_tokens, context_lengths=batch.num_context_tokens,
                         request_is_decoding=batch.request_is_decoding,
                         graph_metadata=asdict(metadata) if metadata is not None else None)
        trace = trace.tb_next
    return found


def main(name, root, *, observe=True, fields=None):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)  # A warm run may have a pre-seeded cache.
    config, changes = build_config(name, root, fields)
    receipt = dict(
        status="STARTED", label="A10", case=name, run_root=str(root),
        source_commit=git_output("rev-parse", "HEAD").strip(),
        source_dirty_paths=git_output("status", "--porcelain", "--untracked-files=all", "--", "frontier", "data").splitlines(),
        python=sys.version.split()[0], input=str(CASES[name]["input"]),
        input_sha256=CASES[name]["input_sha256"], **changes,
        metrics_dir=str(config.metrics_config.output_dir), cache_dir=str(config.metrics_config.cache_dir),
        init_wall_s=None, run_wall_s=None,
    )

    def write():
        (root / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")

    write()
    observer = None
    receipt["observer_enabled"] = observe
    trace = Path(config.__flat_config__.trace_request_generator_config_trace_file)
    receipt["trace"] = {"path": str(trace), "sha256": sha256(trace)}
    phase = "initialization"
    started = time.monotonic()
    try:
        receipt["profiling_tables"], receipt["a1"] = resolved_inputs(config)
        write()
        if any(receipt["a1"]["enable_dummy_mode"].values()) or not receipt["a1"]["no_step_cost_file_beside_inputs"]:
            raise ValueError(f"A10 requires non-dummy operator inputs: {receipt['a1']}")
        from frontier.utils.random import set_seeds
        from frontier.simulator import Simulator

        set_seeds(config.seed)
        started = time.monotonic()
        simulator = Simulator(config)
        receipt["init_wall_s"] = time.monotonic() - started
        receipt["a1"]["step_cost_table_bound_by_predictor"] = {
            role.name: predictor._dsv4_step_costs is not None
            for role, predictor in simulator._predictors.items()
        }
        receipt["kv_admission"] = h4a4.kv_admission(simulator, CASES[name])
        receipt["metrics_dir"] = str(config.metrics_config.output_dir)
        receipt["status"] = "INITIALIZED"
        write()
        if any(receipt["a1"]["step_cost_table_bound_by_predictor"].values()):
            raise ValueError("A10 predictor bound a diagnostic step-cost table")
        phase = "run"
        started = time.monotonic()
        if observe:
            from a10_observer import ForwardObserver
            observer = ForwardObserver(root)
            with observer:
                started = time.monotonic()
                simulator.run()
                run_wall_s = time.monotonic() - started
        else:
            simulator.run()
            run_wall_s = time.monotonic() - started
        receipt.update(status="FINISHED", run_wall_s=run_wall_s)
    except Exception as error:
        receipt.update(status="FAILED", error={"type": type(error).__name__, "message": str(error),
                                                "phase": phase, "batch": failure_batch(error)})
        receipt["init_wall_s" if phase == "initialization" else "run_wall_s"] = time.monotonic() - started
        raise
    finally:
        if observer is not None and (root / "collectives.json").exists():
            receipt["a11"] = observer.receipt()
        receipt["cache_files"] = cache_inventory(config.metrics_config.cache_dir)
        write()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("root", type=Path)
    parser.add_argument("--without-observer", action="store_true")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--without-run-ahead", action="store_true",
                        help="Keep the legacy DP path (max_forwards_in_flight=None).")
    args = parser.parse_args()
    fields = {"trace_request_generator_config_trace_file": str(args.trace.resolve())} if args.trace else {}
    if args.without_run_ahead:
        fields["vllm_v1_scheduler_config_max_forwards_in_flight"] = None
    main(args.case, args.root, observe=not args.without_observer, fields=fields)
