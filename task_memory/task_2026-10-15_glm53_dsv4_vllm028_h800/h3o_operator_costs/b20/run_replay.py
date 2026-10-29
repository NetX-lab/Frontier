"""B20 CPU identity/progress driver, using A10's fields and h4a4's extra-field API.

Usage: run_replay.py <repo> <run-root> [--window SECONDS] [--case p2_32|p2_64]
Outputs and caches belong to run-root. Run each cold identity in a fresh root.
"""
import argparse
import cProfile
from dataclasses import asdict
import importlib.util
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

from generate_trace import generate


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class WallLimit(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--window", type=int)
    parser.add_argument("--observe-dp", action="store_true")
    parser.add_argument("--case", default="p2_32", choices=("p2_32", "p2_64"))
    args = parser.parse_args()
    repo, root = args.repo.resolve(), args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    task = repo / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800"
    sys.path.insert(0, str(task / "h3o_operator_costs/a10"))
    a10 = load("b20_a10", task / "h3o_operator_costs/a10/a10_driver.py")
    case = {key: value for key, value in a10.CASES[args.case].items()
            if key not in ("table", "overrides")}
    case["overrides"] = {}
    fields = a10.operator_fields(args.case, root)
    trace_sha = None
    if args.window is None:
        trace, trace_sha = generate(root)
        fields["trace_request_generator_config_trace_file"] = str(trace)
    config, changes = a10.h4a4.build_config(args.case, case, root, fields)
    receipt = dict(status="STARTED", repo=str(repo), run_root=str(root),
                   python=sys.version, case=args.case, trace_sha256=trace_sha,
                   window_s=args.window, **changes)
    receipt["source_sha256"] = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in ("frontier/moe_ep_workload.py", "frontier/execution_time_predictor/deepseek_v4.py",
                     "frontier/execution_time_predictor/dsv4_predictor.py")
    }
    receipt["profiling_tables"], receipt["a1"] = a10.resolved_inputs(config)
    write = lambda: (root / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    write()
    from frontier.utils.random import set_seeds
    from frontier.simulator import Simulator
    set_seeds(config.seed)
    started = time.monotonic()
    simulator = Simulator(config)
    receipt.update(init_wall_s=time.monotonic() - started,
                   metrics_dir=str(config.metrics_config.output_dir),
                   cache_dir=str(config.metrics_config.cache_dir),
                   step_cost_bound={role.name: p._dsv4_step_costs is not None
                                    for role, p in simulator._predictors.items()})
    receipt["status"] = "INITIALIZED"
    write()
    print("initialized", receipt["init_wall_s"], flush=True)
    # Read-only observer: records the actual metadata produced by the native
    # scheduler method. No config or production output behavior is changed.
    observer = None
    dp_observer = None
    if args.window is None:
        from frontier.scheduler.replica_scheduler.vllm_v1_iteration_policy import IterationSchedulingPolicy
        from frontier.entities.batch import DummyForwardBatch
        original = IterationSchedulingPolicy._build_decode_cuda_graph_metadata
        observer = (root / "forward_dispatch.jsonl").open("w")
        def observed(scheduler, batch):
            result = original(scheduler, batch)
            record = dict(role=scheduler._cluster_type.name,
                          lane=getattr(scheduler, "_replica_local_id", None),
                          replica_id=batch.replica_id, batch_id=batch.id,
                          dummy=isinstance(batch, DummyForwardBatch),
                          request_ids=[r.id for r in batch.requests],
                          prompt_lens=[r.num_prefill_tokens for r in batch.requests],
                          computed_tokens=[r.num_processed_tokens for r in batch.requests],
                          query_lens=batch.num_tokens,
                          context_lens=batch.num_context_tokens,
                          new_rows=list(batch.kv_block_table_writes.new_rows) if batch.kv_block_table_writes else [],
                          dispatch=asdict(result) if result is not None else None)
            observer.write(json.dumps(record) + "\n")
            return result
        IterationSchedulingPolicy._build_decode_cuda_graph_metadata = observed
        if args.observe_dp:
            from frontier.scheduler.utils import dp_dummy_forward
            original_sync = dp_dummy_forward._synchronize_dp_descriptors
            dp_observer = (root / "final_forward_dispatch.jsonl").open("w")
            def observed_sync(scheduler, replica_id, batches):
                result = original_sync(scheduler, replica_id, batches)
                for lane, batch in sorted(batches.items()):
                    record = dict(role=scheduler._cluster_type.name, lane=lane,
                                  replica_id=replica_id, batch_id=batch.id,
                                  cohort_id=getattr(batch, "_forward_cohort_id", None),
                                  dummy=isinstance(batch, DummyForwardBatch),
                                  request_ids=[r.id for r in batch.requests],
                                  prompt_lens=[r.num_prefill_tokens for r in batch.requests],
                                  query_lens=batch.num_tokens, context_lens=batch.num_context_tokens,
                                  new_rows=list(batch.kv_block_table_writes.new_rows) if batch.kv_block_table_writes else [],
                                  dispatch=asdict(batch.decode_cuda_graph_metadata) if batch.decode_cuda_graph_metadata else None)
                    dp_observer.write(json.dumps(record) + "\n")
                return result
            dp_dummy_forward._synchronize_dp_descriptors = observed_sync
    profile = cProfile.Profile() if args.window else None
    if args.window:
        def stop(*_):
            raise WallLimit()
        signal.signal(signal.SIGALRM, stop)
        signal.alarm(args.window)
    started = time.monotonic()
    try:
        if profile:
            profile.runcall(simulator.run)
        else:
            simulator.run()
        receipt["status"] = "FINISHED"
    except WallLimit:
        receipt["status"] = "WINDOW_COMPLETE"
    except Exception as exc:
        receipt["status"] = "FAILED"
        receipt["error"] = dict(type=type(exc).__name__, message=str(exc), batch=a10.failure_batch(exc))
        raise
    finally:
        signal.alarm(0)
        receipt.update(run_wall_s=time.monotonic() - started,
                       simulated_time_s=simulator._time,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       event_queue=len(simulator._event_queue),
                       cache_files=a10.cache_inventory(config.metrics_config.cache_dir))
        if profile:
            profile.dump_stats(str(root / "profile.pstats"))
        if observer:
            observer.close()
        if dp_observer:
            dp_observer.close()
        write()
        print(json.dumps({k: receipt[k] for k in ("status", "run_wall_s", "simulated_time_s", "peak_rss_kib")}), flush=True)


if __name__ == "__main__":
    main()
