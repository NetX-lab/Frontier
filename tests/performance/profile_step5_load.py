"""Profile actual Step loader APIs on unchanged source rows, without full setup.

The input is a retained frontier.main config.json. Scratch output must be new.
No full CUDA_EVENT load, production edits, fixture writes, or estimator fit occurs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import cProfile
import csv
from dataclasses import fields
import hashlib
from itertools import islice
import json
from pathlib import Path
import pstats
import shutil
import sys
import time
from types import SimpleNamespace

from frontier.config import BaseModelConfig, RandomForrestExecutionTimePredictorConfig, global_vars
from frontier.config.utils import dataclass_to_dict
from frontier.execution_time_predictor import step5
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.types import ClusterType, MeasurementType


def profile(label, call, output):
    profiler = cProfile.Profile()
    start = time.perf_counter()
    result = profiler.runcall(call)
    wall = time.perf_counter() - start
    profiler.dump_stats(str(output / f"{label}.pstats"))
    stats = pstats.Stats(profiler)
    functions = []
    for (path, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        functions.append({"path": path, "line": line, "name": name,
                          "primitive_calls": primitive, "calls": calls,
                          "self_s": own, "cumulative_s": cumulative})
    functions.sort(key=lambda item: item["cumulative_s"], reverse=True)
    interesting = {"read_csv", "load_step5_attention_rows", "load_step5_attention_input_union",
                   "step5_row_features", "_validate_targets", "operator_contracts",
                   "parse_typed_operator_contracts", "validate_typed_operator_metadata",
                   "describe_step5_layer", "validate_step5_config", "_row_workload",
                   "asdict", "attention_scopes", "_get_model_hash", "to_json",
                   "_load_model_from_cache", "_fit_single_model", "fit",
                   "_train_attn_models_for_cluster", "train_step5_attention_models"}
    selected = [item for item in functions if item["name"] in interesting
                or "pickle.load" in item["name"] or "hashlib" in item["name"]]
    receipt = {"label": label, "wall_s": wall, "selected_functions": selected,
               "top_cumulative": functions[:20], "top_self": sorted(
                   functions, key=lambda item: item["self_s"], reverse=True)[:20]}
    (output / f"{label}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result, receipt


def source_subset(source, destination, count):
    """Copy a prefix with csv semantics unchanged; verify every field after write."""
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(islice(reader, count))
        columns = reader.fieldnames
    if len(rows) != count:
        raise ValueError(f"Expected {count} source rows, found {len(rows)}")
    with destination.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with destination.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == columns
        assert list(reader) == rows
    return {"evidence_kind": "diagnostic_source_subset", "source": str(source),
            "source_bytes": source.stat().st_size, "subset": str(destination),
            "subset_bytes": destination.stat().st_size, "rows": count,
            "columns": len(columns), "all_source_fields_equal": True,
            "subset_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "largest_first_row_cells": sorted(
                ((name, len(value)) for name, value in rows[0].items()),
                key=lambda item: item[1], reverse=True)[:5],
            "shape_counts": [
                {"shape": list(key), "rows": value}
                for key, value in Counter((row["global_layer_id"], row["graph_mode"],
                                           row["native_workload"], row["attention_variant"])
                                          for row in rows).items()]}


def configuration(path):
    recorded = json.loads(path.read_text())
    cluster = recorded["cluster_config"]
    model = BaseModelConfig.create_from_name(cluster["replica_config"]["model_name"])
    model_json = json.loads(json.dumps(dataclass_to_dict(model)))
    assert model_json == cluster["replica_config"]["model_config"]
    names = {field.name for field in fields(RandomForrestExecutionTimePredictorConfig)}
    predictor = RandomForrestExecutionTimePredictorConfig(**{
        key: value for key, value in cluster["execution_time_predictor_config"].items()
        if key in names})
    scheduler = SimpleNamespace(**cluster["replica_scheduler_config"])
    scheduler.get_type = lambda: cluster["replica_scheduler_config"]["name"]
    clusters = {}
    for role, prefix in ((ClusterType.PREFILL, "prefill"), (ClusterType.DECODE, "decode")):
        replica = SimpleNamespace(**cluster[f"{prefix}_replica_config"])
        assert replica.model_config == model_json
        replica.model_config = model
        clusters[role] = SimpleNamespace(replica_config=replica,
                                        replica_scheduler_config=scheduler,
                                        execution_time_predictor_config=predictor)
    global_vars.set_global_vars(recorded["simulation_mode"], recorded["sys_arch"])
    global_vars.set_cuda_graph_config(recorded["use_cuda_graph"], recorded["cudagraph_capture_sizes"],
                                      recorded["decode_cuda_graph_mode"])
    return recorded, clusters, predictor


def partial_manager(clusters, cache):
    """Reach only the actual attention initialization method, not full __init__."""
    manager = ExecutionTimePredictionModelManager.__new__(ExecutionTimePredictionModelManager)
    manager._cluster_configs = clusters
    manager._cache_dir = str(cache)
    manager._all_dummy_mode = False
    for family in ("eager", "device_event", "kernel_only"):
        for prefix in ("_trained_models_", "_models_by_precision_", "_model_profiling_precision_"):
            setattr(manager, prefix + family, {})
        for prefix in ("_trained_models_", "_models_by_precision_"):
            setattr(manager, prefix + family + "_by_contract", {})
    manager._trained_models = {}
    manager._models_by_precision = {}
    manager._model_profiling_precision = {}

    def forbid_fit(_):
        raise RuntimeError("Diagnostic refuses estimator fitting on a cache miss")

    manager._create_estimator_and_params = forbid_fit
    return manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--event-cache-hash",
                        help="Profile one existing CUDA_EVENT root artifact through a scratch copy")
    parser.add_argument("--assert-no-repeated-loads", action="store_true",
                        help="Return red when one unchanged role union reloads source rows")
    args = parser.parse_args()
    if any(count <= 0 or count > 2000 for count in args.rows):
        parser.error("Each diagnostic subset must contain 1..2000 rows")
    if args.event_cache_hash and (len(args.event_cache_hash) != 8 or any(
            character not in "0123456789abcdef" for character in args.event_cache_hash)):
        parser.error("event-cache-hash must be an existing eight-character cache hash")
    args.output.mkdir(parents=True, exist_ok=False)
    recorded, clusters, predictor = configuration(args.config)
    source_event = Path(predictor.atten_input_file)
    source_kernel = Path(predictor.atten_kernel_only_input_file)
    source_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                    for path in (args.config, source_event, source_kernel)}
    role = ClusterType.PREFILL
    cluster = clusters[role]
    manager = partial_manager(clusters, args.output / "unused_cache")
    receipts, subsets = [], []
    plain_loader_timings = []
    for count in args.rows:
        subset = args.output / f"cuda_event_prefix_{count}.csv"
        subsets.append(source_subset(source_event, subset, count))
        start = time.perf_counter()
        frame = step5.load_step5_attention_rows(
            subset, cluster.replica_config, cluster.replica_scheduler_config,
            MeasurementType.CUDA_EVENT)
        plain_loader_timings.append({"rows": len(frame), "wall_s": time.perf_counter() - start,
                                     "profiler_enabled": False})
        del frame
        _, receipt = profile(f"event_load_{count}", lambda: step5.load_step5_attention_rows(
            subset, cluster.replica_config, cluster.replica_scheduler_config,
            MeasurementType.CUDA_EVENT), args.output)
        receipts.append(receipt)
    _, receipt = profile("kernel_load_full184", lambda: step5.load_step5_attention_rows(
        source_kernel, cluster.replica_config, cluster.replica_scheduler_config,
        MeasurementType.KERNEL_ONLY), args.output)
    receipts.append(receipt)

    # Both roles point to the same new subset; all other predictor options stay intact.
    event_subset = Path(subsets[0]["subset"])
    predictor.atten_input_file = str(event_subset)
    _, receipt = profile("event_union_two_roles", lambda: step5.load_step5_attention_input_union(
        manager, role, cluster.replica_config, cluster.replica_scheduler_config,
        predictor, event_subset, MeasurementType.CUDA_EVENT), args.output)
    receipts.append(receipt)
    predictor.atten_input_file = str(source_event)

    # Existing small KERNEL_ONLY artifacts are copied, not hardlinked. Cache lookup's
    # native lock writes and any unexpected migration remain in this scratch dir.
    scratch_cache = args.output / "kernel_cache_copy"
    scratch_cache.mkdir()
    cache_receipt = []
    for source in Path(recorded["metrics_config"]["cache_dir"]).glob("step5_layer_e2e*.pkl"):
        if source.stat().st_size >= 1_000_000:
            continue
        destination = scratch_cache / source.name
        shutil.copyfile(source, destination)
        cache_receipt.append({"source": str(source), "bytes": source.stat().st_size,
                              "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    if not cache_receipt:
        raise ValueError("No existing small Step KERNEL_ONLY cache artifacts found")
    manager = partial_manager(clusters, scratch_cache)
    manager._active_measurement_type = MeasurementType.KERNEL_ONLY
    signatures = set()

    def warm_role(selected_role):
        selected = clusters[selected_role]
        return manager._train_attn_models_for_cluster(
            selected_role, selected.replica_config, predictor,
            selected.replica_scheduler_config, predictor.linear_op_input_file,
            str(source_kernel), trained_model_signatures=signatures)

    models, receipt = profile("kernel_warm_prefill", lambda: warm_role(ClusterType.PREFILL), args.output)
    receipts.append(receipt)
    repeated, receipt = profile("kernel_warm_decode_deduplicated", lambda: warm_role(ClusterType.DECODE), args.output)
    receipts.append(receipt)
    assert len(models) == 2 and repeated == {}
    assert len(signatures) == 1
    for item in cache_receipt:
        source = Path(item["source"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]
        assert hashlib.sha256((scratch_cache / source.name).read_bytes()).hexdigest() == item["sha256"]
    count = sum(item["calls"] for item in receipt["selected_functions"]
                if item["name"] == "load_step5_attention_rows")
    event_cache = None
    if args.event_cache_hash:
        name = f"step5_layer_e2e_{args.event_cache_hash}.pkl"
        source = Path(recorded["metrics_config"]["cache_dir"]) / name
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, scratch_cache / name)
        model, receipt = profile("event_cache_read", lambda: manager._load_model_from_cache(
            "step5_layer_e2e", args.event_cache_hash), args.output)
        receipts.append(receipt)
        event_cache = {"source": str(source), "bytes": source.stat().st_size,
                       "sha256": before, "coverage_keys": len(model._frontier_step5_coverage),
                       "measurement_identity": json.loads(model._frontier_measurement_identity)}
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        assert hashlib.sha256((scratch_cache / name).read_bytes()).hexdigest() == before
        del model
    for name, expected in source_stats.items():
        path = Path(name)
        assert (path.stat().st_size, path.stat().st_mtime_ns) == expected
    summary = {"task_id": "D23-MRV2-LOAD-DIAG", "config": str(args.config),
               "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
               "model_equals_actual_config": True, "subsets": subsets,
               "plain_loader_timings": plain_loader_timings,
               "kernel_source_bytes": source_kernel.stat().st_size,
               "scratch_cache_artifacts": cache_receipt,
               "event_cache_artifact": event_cache, "source_size_and_mtime_unchanged": True,
               "warm_prefill_models": list(models), "deduplicated_decode_models": list(repeated),
               "deduplicated_decode_loader_calls": count,
               "profiles": [{"label": item["label"], "wall_s": item["wall_s"]} for item in receipts],
               "limits": ["Seconds-scale API profile, not full initialization wall profile",
                          "CUDA_EVENT diagnostic prefixes; original timings/columns are unchanged",
                          "No full manager __init__, estimator fit, runtime query, or CLI run"],
               "regression_verdict": "RED_REPEATED_LOAD" if count > 0 else "GREEN_NO_REPEATED_LOAD"}
    (args.output / "receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.assert_no_repeated_loads and count:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
