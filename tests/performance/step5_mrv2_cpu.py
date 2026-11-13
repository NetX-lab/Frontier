"""Read-only real-cache Step prediction latency diagnosis, not native admission."""

from __future__ import annotations

import argparse
from collections import defaultdict
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pickle
import platform
import pstats
import subprocess
import sys
import time
from types import SimpleNamespace

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.execution_time_predictor.step5 import STEP5_FEATURE_COLUMNS
from frontier.execution_time_predictor.step5_cost import (
    predict_step5_attention_time, step5_batch_features, step5_predictions,
)
from frontier.types import ClusterType, MeasurementType


class CacheReplayPredictor(SklearnMoEExecutionTimePredictor):
    """Use production query methods without entering fitting/initialization."""

    def _get_grid_search_params(self):
        raise AssertionError("This read-only replay must not fit estimators")

    def _get_estimator(self):
        raise AssertionError("This read-only replay must not fit estimators")


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_attention(cache, read_profile):
    selected = {}
    artifacts = []
    for bundle_path in sorted(cache.glob("bundle_*.json")):
        bundle = json.loads(bundle_path.read_text())
        if bundle["identity"]["role"].lower() != "decode":
            continue
        if bundle["identity"].get("input_bindings", {}).get("nixl_read_profile", {}).get("path") != read_profile:
            continue
        entries = bundle["models"]
        roots = [(label, entry) for label, entry in entries.items()
                 if entry.get("cache_model_name", label) == "step5_layer_e2e"]
        if not roots:
            continue
        label, entry = roots[0]
        identity = entry["measurement_identity"]
        # Bind the actual05 scheduler capacity identity, not an older cache.
        if not all(binding["shape_policy"].get("max_tokens_in_batch") == 8192
                   for binding in identity["attention_inputs"]):
            continue
        method = MeasurementType.from_string(identity["measurement_type"])
        if method in selected:
            continue
        path = cache / entry["artifact"]
        with path.open("rb") as stream:
            model = pickle.load(stream)
        count_entry = entries[entry["kernel_count_model"]]
        count_path = cache / count_entry["artifact"]
        with count_path.open("rb") as stream:
            count = pickle.load(stream)
        # Restore the paired link exactly as production bundle loading does.
        model._frontier_kernel_count_model = count
        expected = 94392 if method is MeasurementType.CUDA_EVENT else 184
        assert len(model._frontier_exact_lookup) == expected
        assert len(model._frontier_step5_coverage) == expected
        assert identity["evidence_kind"] == "cpu_fixture"
        assert count._frontier_measurement_identity == model._frontier_measurement_identity
        selected[method] = {"step5_layer_e2e": model, "step5_layer_e2e_kernel_count": count}
        digest, count_digest = sha256(path), sha256(count_path)
        assert digest == entry["sha256"] and count_digest == count_entry["sha256"]
        artifacts.append(dict(path=str(path), sha256=digest, bytes=path.stat().st_size,
                              count_path=str(count_path), count_sha256=count_digest,
                              bundle_path=str(bundle_path), bundle_id=bundle["bundle_id"],
                              measurement=method.value, exact_keys=expected,
                              duration_hash=model._frontier_model_hash,
                              count_hash=count._frontier_model_hash))
        if len(selected) == 2:
            break
    assert set(selected) == {MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY}
    return selected, artifacts


def predictor_from_cache(selected, fixture):
    predictor = CacheReplayPredictor.__new__(CacheReplayPredictor)
    predictor._model_config = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    predictor._replica_config = SimpleNamespace(
        device="h800", model_name="stepfun-ai/Step-5-Preview",
        attn_tensor_parallel_size=8, attn_dp=2, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=16, num_pipeline_stages=1,
    )
    predictor._cluster_type = ClusterType.DECODE
    predictor._enable_dummy_mode = False
    predictor._block_size = 16
    predictor._step5_max_model_len = 7168
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._step_measurement_type = MeasurementType.CUDA_EVENT
    # The attention query requires gap availability but does not add a gap here.
    from frontier.execution_time_predictor.kernel_gap import read_kernel_gap_ms
    predictor._kernel_gap_ms = read_kernel_gap_ms(
        str(fixture / "fixture_kernel_gap.csv"), 8,
    )
    for suffix, method in (("eager", MeasurementType.CUDA_EVENT),
                           ("kernel_only", MeasurementType.KERNEL_ONLY)):
        setattr(predictor, f"_models_{suffix}", selected[method])
        setattr(predictor, f"_predictions_{suffix}", step5_predictions(selected[method], method))
        for kind in ("compute", "attention", "moe", "cpu_overhead"):
            setattr(predictor, f"_{kind}_input_file_{suffix}", "")
    predictor._activate_measurement_type(MeasurementType.CUDA_EVENT)
    return predictor


def load_ffn(predictor, attention_artifacts):
    from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager

    manager = ExecutionTimePredictionModelManager.__new__(ExecutionTimePredictionModelManager)
    manager._all_dummy_mode = False
    for family in ("eager", "device_event", "kernel_only"):
        for prefix in ("_trained_models_", "_models_by_precision_", "_model_profiling_precision_"):
            setattr(manager, prefix + family, {})
        setattr(manager, f"_trained_models_{family}_by_contract", {})
        setattr(manager, f"_models_by_precision_{family}_by_contract", {})
    artifacts = []
    diagnostic_models = {}
    for attention in attention_artifacts:
        path = Path(attention["bundle_path"])
        bundle = json.loads(path.read_text())
        models = {}
        for label, entry in bundle["models"].items():
            name = entry.get("cache_model_name", label)
            if name.startswith("step5_layer_e2e"):
                model = predictor._models_eager.get(name) if attention["measurement"] == "CUDA_EVENT" else predictor._models_kernel_only.get(name)
            else:
                artifact = path.parent / entry["artifact"]
                digest = sha256(artifact)
                assert digest == entry["sha256"]
                with artifact.open("rb") as stream:
                    model = pickle.load(stream)
                artifacts.append(dict(path=str(artifact), sha256=digest))
            models[label] = model
            diagnostic_models[model._frontier_model_hash] = model
            registration = bundle["identity"]["registrations"][label]
            family = "eager" if attention["measurement"] == "CUDA_EVENT" else "kernel_only"
            identity = entry["typed_identity"]
            precision = registration["precision"]
            if identity is None:
                getattr(manager, "_trained_models_" + family)[name] = model
                getattr(manager, "_models_by_precision_" + family).setdefault(precision, {})[name] = model
            else:
                getattr(manager, f"_trained_models_{family}_by_contract")[(name, identity)] = model
                getattr(manager, f"_models_by_precision_{family}_by_contract").setdefault(precision, {})[(name, identity)] = model
        for label, entry in bundle["models"].items():
            if "kernel_count_model" in entry:
                count = models[entry["kernel_count_model"]]
                count._frontier_operator_name = entry["target_column"].split(".")[1]
                models[label]._frontier_kernel_count_model = count
    predictor._model_manager = manager
    predictor._diagnostic_models_by_hash = diagnostic_models
    return artifacts


def ffn_queries(predictor, layers):
    from frontier.moe_ep_workload import build_contiguous_expert_ownership, generate_moe_routing_ratios, materialize_layer_ep_workload
    from frontier.moe_load_imbalance import MOE_LOAD_IMBALANCE_FEATURES, MoELoadImbalanceInput

    model = predictor._model_config
    result = []
    for layer in layers:
        workload = materialize_layer_ep_workload(
            routing_ratios=generate_moe_routing_ratios(total_expert_num=352, distribution_type="balanced", seed=42, layer_id=layer),
            target_replica_id=0, global_layer_id=layer, routing_token_count=2,
            router_topk=8, total_expert_num=352, moe_expert_parallel_size=16,
            expert_to_ep=build_contiguous_expert_ownership(352, 16),
        )
        for ep_id in workload.participant_ep_ids:
            lane = workload.lane(ep_id)
            features = MoELoadImbalanceInput(
                num_tokens=2, num_experts_per_device=lane.local_expert_width,
                hidden_dim=model.embedding_dim, expert_hidden_dim=model.mlp_hidden_dim,
                router_topk=8, expert_token_counts=list(lane.local_token_counts),
            ).to_features_dict()
            result.append((layer, ep_id, lane.routed_token_count,
                           {name: features[name] for name in MOE_LOAD_IMBALANCE_FEATURES}))
    return result


def run_ffn(predictor, queries, iterations):
    rows = []
    scalar_ops = ("post_attention_layernorm", "moe_gating_linear", "moe_gating_routing_topk",
                  "share_expert_up_proj", "share_expert_act", "share_expert_down_proj", "step5_ffn_merge")
    for _ in range(iterations):
        for layer, ep_id, routed_tokens, features in queries:
            start = time.perf_counter()
            with predictor._step5_measured_layer(layer):
                metadata_seconds = time.perf_counter() - start
                operator_start = time.perf_counter()
                for name in scalar_ops:
                    predictor._predict_step5_ffn_operator(name, {"num_tokens": 2})
                if routed_tokens:
                    for name in ("moe_shuffling", "moe_grouped_gemm"):
                        predictor._predict_step5_ffn_operator(name, features)
                costs = dict(predictor._step5_ffn_costs)
                operator_seconds = time.perf_counter() - operator_start
            seconds = time.perf_counter() - start
            for name, cost in costs.items():
                assert cost["exact_lookup"] and cost["evidence_kind"] == "cpu_fixture", (name, cost, features)
                assert cost["kernel_count"] == 7
                assert cost["global_layer_id"] == layer
                if name == "moe_grouped_gemm":
                    assert cost["precision"] == ("FP8" if layer < 88 else "BF16")
                model = predictor._diagnostic_models_by_hash[cost["duration_model_hash"]]
                query = features if name in ("moe_shuffling", "moe_grouped_gemm") else {"num_tokens": 2}
                key = tuple(query[column] for column in model._frontier_feature_names)
                assert cost["duration_ms"] == model._frontier_exact_lookup[key]
                assert cost["kernel_count"] == model._frontier_kernel_count_model._frontier_exact_lookup[key]
            rows.append(dict(layer=layer, ep_id=ep_id, routed_tokens=routed_tokens,
                              seconds=seconds, metadata_seconds=metadata_seconds,
                              operator_seconds=operator_seconds, costs=costs))
    return rows


def attention_batch(query, context, graph):
    return SimpleNamespace(
        num_tokens=[query], num_context_tokens=[context],
        num_prefill_tokens=query if query > 1 else 0,
        num_decode_tokens=1 if query == 1 else 0,
        get_effective_total_tokens_for_compute=lambda _: query,
        get_decode_cuda_graph_runtime_mode=lambda: graph.upper(),
    )


def run_attention(predictor, selected, layers, iterations):
    batch = attention_batch(1, 6145, "full")
    expected = []
    source = selected[MeasurementType.CUDA_EVENT]
    for layer in layers:
        features, workload = step5_batch_features(predictor, batch, layer)
        key = tuple(features[name] for name in STEP5_FEATURE_COLUMNS)
        expected.append((key, source["step5_layer_e2e"]._frontier_exact_lookup[key],
                         source["step5_layer_e2e_kernel_count"]._frontier_exact_lookup[key]))
        assert (workload.valid_tokens, workload.max_model_len_tokens,
                workload.block_table_columns) == (1, 7168, 448)
    rows = []
    for _ in range(iterations):
        for layer, (key, duration, count) in zip(layers, expected):
            start = time.perf_counter()
            result = predict_step5_attention_time(predictor, batch, layer)
            seconds = time.perf_counter() - start
            proof = result.operator_costs["step5_layer_e2e"]
            assert result.total_time() == duration
            assert result.kernel_count == count
            assert proof.evidence_kind == "cpu_fixture" and proof.exact_lookup
            rows.append(dict(layer=layer, seconds=seconds, duration_ms=duration,
                             kernel_count=count, feature_key_sha256=hashlib.sha256(
                                 json.dumps(key).encode()).hexdigest()))
    return rows


def run_trace(predictor, rows, output, enable_profile):
    """Exercise actual source-bound trace metadata and JSONL persistence."""
    from frontier.metrics.metrics_store import MetricsStore
    from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta
    from frontier.metrics.trace_store import TraceEvent, TraceStore

    metadata_store = MetricsStore.__new__(MetricsStore)
    predictor._replica_config.model_config = predictor._model_config
    predictor._replica_config.router_topk = 8
    trace = TraceStore(str(output), enabled=True)
    trace.initialize({"task_id": "D23-ACTUAL-MAIN/PERF", "native_admitted": False})
    profile = cProfile.Profile()
    start = time.perf_counter()
    if enable_profile:
        profile.enable()
    expected = []
    for row in rows:
        metadata = metadata_store._get_step5_layer_metadata(predictor._replica_config, row["layer"])
        for name, cost in row["costs"].items():
            routed = name in ("moe_grouped_gemm", "moe_shuffling")
            tokens = row["routed_tokens"] if routed else 2
            context = OpTraceContext(ClusterType.DECODE, predictor._model_config,
                                     predictor._replica_config, 2, tokens, tokens, tokens,
                                     routed, global_layer_id=row["layer"],
                                     layer_metadata=metadata, pre_routing_tokens=2)
            meta = compute_op_trace_meta(name, "COMPUTE", context)
            meta.update(ep_id=row["ep_id"], evidence_kind="cpu_fixture",
                        duration_model_hash=cost["duration_model_hash"],
                        kernel_count_model_hash=cost["kernel_count_model_hash"])
            trace.log_event(TraceEvent(type="COMPUTE", name=name, ts_start=0,
                                      duration_ms=cost["duration_ms"], cluster="DECODE",
                                      layer_id=row["layer"], meta=meta))
            expected.append((name, row["layer"], cost["duration_ms"]))
    trace.close()
    if enable_profile:
        profile.disable()
    measured = time.perf_counter() - start
    actual = [json.loads(line) for line in (output / "op_traces.jsonl").read_text().splitlines()][1:]
    assert [(event["name"], event["layer_id"], event["duration_ms"]) for event in actual] == expected
    if enable_profile:
        profile.dump_stats(str(output / "trace.prof"))
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(20)
        (output / "trace_profile.txt").write_text(stream.getvalue())
    return dict(seconds=measured, event_count=len(expected),
                bytes=(output / "op_traces.jsonl").stat().st_size,
                persisted_truth="PASS", profile_enabled=enable_profile)


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    selected, artifacts = load_attention(Path(args.fixture) / "cache", args.read_profile)
    predictor = predictor_from_cache(selected, Path(args.fixture))
    if args.mode == "ffn":
        artifacts.extend(load_ffn(predictor, list(artifacts)))
        queries = ffn_queries(predictor, args.layers)
    loading = time.perf_counter() - start
    profile = cProfile.Profile()
    start = time.perf_counter()
    if args.profile:
        profile.enable()
    rows = (run_attention(predictor, selected, args.layers, args.iterations)
            if args.mode == "attention" else run_ffn(predictor, queries, args.iterations))
    if args.profile:
        profile.disable()
    measured = time.perf_counter() - start
    stream = io.StringIO()
    if args.profile:
        profile.dump_stats(str(output / "prediction.prof"))
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
        (output / "profile.txt").write_text(stream.getvalue())
    # Conservatively assign the entire remaining original bound to this one
    # component. Attention has two DP lanes; FFN has 16 EP lanes/88 MoE layers.
    budget = ((args.original_bound - args.original_setup) / args.decode_steps
              / (args.dp_lanes if args.mode == "attention" else 16)
              / (args.physical_layers if args.mode == "attention" else 88))
    latencies = [row["seconds"] for row in rows]
    serialization_start = time.perf_counter()
    serialized = json.dumps(rows, sort_keys=True)
    serialization_seconds = time.perf_counter() - serialization_start
    trace = run_trace(predictor, rows, output, args.profile) if args.trace else None
    passed = max(latencies) <= budget
    receipt = dict(
        task_id="D23-ACTUAL-MAIN/PERF", mode=args.mode + "_cache_replay",
        native_admitted=False, source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
        command=sys.argv, python=sys.version, executable=sys.executable,
        platform=platform.platform(), loading_seconds=loading, measured_seconds=measured,
        profile_enabled=args.profile, rows_serialization_seconds=serialization_seconds,
        rows_serialization_bytes=len(serialized),
        trace=trace,
        query_budget_seconds=budget, verdict="GREEN" if passed else "RED",
        source_truth="PASS", artifacts=artifacts, rows=rows,
    )
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({name: receipt[name] for name in (
        "verdict", "source_truth", "loading_seconds", "measured_seconds",
        "query_budget_seconds")}, indent=2))
    print(stream.getvalue())
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="/data/ycfeng/tmp/d23_main_fixture02")
    parser.add_argument("--read-profile", default="/data/ycfeng/tmp/d23_main_actual05_offline/synthetic_read_cost.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("attention", "ffn"), default="attention")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 88])
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace", action="store_true", help="Replay FFN trace metadata and persistence")
    parser.add_argument("--original-bound", type=float, default=2400)
    parser.add_argument("--original-setup", type=float, default=1004)
    parser.add_argument("--decode-steps", type=int, default=1024)
    parser.add_argument("--dp-lanes", type=int, default=2)
    parser.add_argument("--physical-layers", type=int, default=92)
    args = parser.parse_args()
    if args.trace and args.mode != "ffn":
        parser.error("--trace requires --mode ffn")
    raise SystemExit(main(args))
