"""Inspect three bounded production-path shapes using the retained CPU fixture."""

import argparse
import cProfile
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import pstats
import resource
import shutil
import subprocess
import sys
import time
import traceback


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def inspect_case(args, parameters):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    case = parameters["cases"][args.case]
    frozen_raw = Path(parameters["frozen_manifest"]).read_bytes()
    assert sha256(frozen_raw) == parameters["frozen_manifest_sha256"]
    frozen = json.loads(frozen_raw)
    source = Path(frozen["sources"]["baseline"]["root"])
    source_binding = frozen["sources"]["baseline"]
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() == source_binding["head"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)

    def verify_bindings():
        for relative, expected in source_binding["hashes"].items():
            assert sha256((source / relative).read_bytes()) == expected, relative
        for relative, (size, mtime_ns) in frozen["fixture_metadata"].items():
            stat = (Path(frozen["fixture"]) / relative).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (size, mtime_ns), relative
        assert sha256(Path(frozen["protected_body"]).read_bytes()) == frozen["protected_body_sha256"]

    verify_bindings()
    wiring_path = Path(parameters["fixture_wiring"])
    wiring_raw = wiring_path.read_bytes()
    assert sha256(wiring_raw) == parameters["fixture_wiring_sha256"]
    wiring = {"__name__": "d29_fixture_wiring", "__file__": str(wiring_path)}
    exec(compile(wiring_raw, str(wiring_path), "exec"), wiring)
    collective = Path(frozen["fixture"]) / "fixture_collectives/h800/collective_local.json"
    destination = root / "baseline/collective-fixture/h800/collective_local.json"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(collective, destination)
    assert sha256(destination.read_bytes()) == frozen["collective_sha256"]
    dump(root / "stage.json", {"stage": "cached_setup", "case": args.case})
    namespace, _, _, _, _ = wiring["setup"](frozen, "baseline", root)
    p_backend, d_backend, raw, _ = wiring["bind_production_owner"](namespace, root, "baseline")
    assert sha256(raw) == frozen["collective_sha256"]

    from frontier.entities import Batch, DummyForwardBatch, Request
    from frontier.metrics.ep_wave_metrics import record_ep_wave
    from frontier.metrics.trace_store import TraceStore
    from frontier.moe_ep_workload import build_contiguous_expert_ownership, materialize_layer_ep_workload
    from frontier.moe_routing_policy import build_wave_routing_source
    from frontier.scheduler.utils.batch_builders import build_virtual_global_batch
    from frontier.scheduler.utils.ep_wave import EPWavePlan
    from frontier.scheduler.utils.ep_wave_inputs import prepare_ep_wave_inputs
    from frontier.scheduler.utils.expert_parallel import predict_ep_wave_phase_times, get_ep_phase_times_ms, calculate_ep_wave_timing
    from frontier.scheduler.utils.forward_start_agreement import bind_forward_start
    from frontier.types import ClusterType, MeasurementType

    role = ClusterType[case["role"]]
    cluster = namespace["clusters"][role]
    replica = cluster.replica_config
    namespace["role"] = role
    predictor = namespace["predictor"]
    predictor._cluster_type = role
    predictor._replica_config = replica
    predictor._model_config = replica.model_config
    predictor._config = cluster.execution_time_predictor_config
    predictor._cc_backend = p_backend if role == ClusterType.PREFILL else d_backend
    predictor._router_topk = replica.router_topk
    predictor._moe_tp_size = replica.moe_tensor_parallel_size
    predictor._moe_ep_size = replica.moe_expert_parallel_size
    predictor._models_eager = dict(namespace["manager"]._models_view_for_family("eager", role))
    predictor._predictions_eager = {
        name: {"_on_demand_prediction": True, "_model": model,
               "_feature_names": tuple(model._frontier_feature_names)}
        for name, model in predictor._models_eager.items()
    }
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    store = namespace["store"]
    trace_dir = root / "emitted"
    store._trace_store = TraceStore(str(trace_dir), enabled=True)
    store._trace_store.initialize({"task_id": parameters["task_id"], "case": args.case,
                                   "source": source_binding["head"], "evidence_kind": "cpu_fixture"})
    store._frontier_ep_wave_lane_ledger_rows = []
    functions = (predict_ep_wave_phase_times, get_ep_phase_times_ms, record_ep_wave,
                 predictor._step5_position_queries, predictor._predict_step5_ffn_operator)
    paths = {function.__qualname__: function.__code__.co_filename for function in functions}
    assert all(Path(path).resolve().is_relative_to(source) for path in paths.values())
    expected_events = []
    payloads = []
    waves = []

    for layer_id in case["layers"]:
        request = Request(0.0, parameters["request_prefill_tokens"], parameters["request_decode_tokens"],
                          num_processed_tokens=case["logical_start"])
        batch = Batch(case["replica_id"], [request], [case["logical_queries"]], is_moe=True,
                      num_context_tokens=[case["logical_start"]],
                      request_is_decoding=[case["request_is_decoding"]])
        dummy = DummyForwardBatch(case["replica_id"], 0, 0)
        source_batches = {0: batch, 1: dummy}
        assert len(source_batches) == replica.attn_dp
        for lane, source_batch in source_batches.items():
            source_batch.set_global_id(lane)
        bind_forward_start(tuple(source_batches.values()), group_id=namespace["actual"]["source_lanes"][0]["forward_group_id"],
                           request_capacity=cluster.replica_scheduler_config.batch_size_cap)
        inputs = prepare_ep_wave_inputs(source_batches=source_batches, batch=batch, step_id_getter=lambda _: 0,
                                       aggregate_batch_builder=build_virtual_global_batch, cluster_type=role)
        physical_rows = inputs.aggregate_batch.get_effective_total_tokens_for_compute(role)
        routing = build_wave_routing_source(inputs.source_batches, policy_id="position_cyclic", cluster_type=role,
                                           gathered_physical_rows=physical_rows, model_config=replica.model_config)
        assert sum(lane.live_rows for lane in routing.source_lanes) == case["logical_queries"]
        assert routing.source_lanes[0].intervals[0].start == case["logical_start"]
        assert sum(lane.physical_rows for lane in routing.source_lanes) == physical_rows
        workload = materialize_layer_ep_workload(
            policy_id="position_cyclic", routing_source=routing, target_replica_id=case["replica_id"],
            global_layer_id=layer_id, routing_token_count=physical_rows, router_topk=replica.router_topk,
            total_expert_num=replica.total_expert_num, moe_expert_parallel_size=replica.moe_expert_parallel_size,
            expert_to_ep=build_contiguous_expert_ownership(replica.total_expert_num, replica.moe_expert_parallel_size),
        )
        dump(root / "stage.json", {"stage": "pricing", "case": args.case, "layer": layer_id,
                                   "routing_source": asdict(routing)})
        profiler = cProfile.Profile()
        begin = time.perf_counter()
        profiler.enable()
        phases = predict_ep_wave_phase_times(
            layer_workload=workload, source_batch=inputs.aggregate_batch, stage_id=0, layer_id=layer_id,
            cluster_type=role, predictor=predictor, lane_builder=namespace["lane_builder"],
            phase_getter=get_ep_phase_times_ms, workload_logger=namespace["noop"], trace_identity=None,
            batch_id=batch.id, capture_lane_timings=True,
        )
        profiler.disable()
        pricing_s = time.perf_counter() - begin
        profiler.dump_stats(str(root / f"layer{layer_id}.prof"))
        stats = pstats.Stats(profiler)
        calls = {name: sum(row[1] for (_, _, symbol), row in stats.stats.items() if symbol == name)
                 for name in ("_get_on_demand_prediction", "_predict_step5_ffn_operator",
                              "predict_collective_plan", "_step5_position_queries")}
        assert len(phases.lane_records) == replica.moe_expert_parallel_size
        assert [record.ep_id for record in phases.lane_records] == list(workload.participant_ep_ids)
        actual_ops = []
        for record in phases.lane_records:
            layer = record.execution_time.layer_execution_times[0]
            moe = layer.moe_or_mlp_time_component.operator_times
            communication = layer.communication_time_component.operator_times
            frame = moe.physical_frame
            assert layer.global_layer_id == layer_id
            assert frame["routing_source"] == routing
            if args.case == "bf16":
                assert frame["native_context"]["quant_dtype"] is None
                assert "moe_input_quantize" not in moe.op_times
            compute_proofs = normalized(frame["operator_costs"])
            plans = {name: normalized(value.to_dict()) for name, value in communication.plans.items()}
            costs = {name: normalized({field.name: getattr(cost, field.name) for field in fields(cost)})
                     for name, cost in communication.costs.items()}
            assert all(cost["plan_key"] == communication.plans[name].cache_key() for name, cost in costs.items())
            payloads.append({"layer": layer_id, "ep": record.ep_id, "compute_ms": moe.op_times,
                             "communication_ms": communication.op_times, "compute_proofs": compute_proofs,
                             "plans": plans, "collective_costs": costs})
            for phase in ("pre_dispatch", "dispatch", "routed_compute", "combine", "post_combine"):
                for kind, name, duration in layer.moe_phase_operator_times(phase):
                    assert duration > 0, (layer_id, record.ep_id, name, duration)
                    proof = compute_proofs[name] if kind == "COMPUTE" else {"plan": plans[name], "cost": costs[name]}
                    if kind == "COMPUTE":
                        assert proof["global_layer_id"] == layer_id
                        assert proof["duration_ms"] == duration
                    expected_events.append({"layer": layer_id, "ep": record.ep_id, "phase": phase, "kind": kind,
                                            "name": name, "duration_ms": duration, "proof": proof})
                    actual_ops.append(name)
        assert calls["_step5_position_queries"] == replica.moe_expert_parallel_size
        assert calls["_predict_step5_ffn_operator"] == sum(len(value["compute_proofs"]) for value in payloads if value["layer"] == layer_id)
        timing = calculate_ep_wave_timing(
            start_time_s=namespace["captured"]["meta"]["wave_start_time_s"], phases=phases,
            barrier_logger=namespace["noop"], wave_logger=namespace["noop"], cluster_type=role,
            batch_id=batch.id, layer_id=layer_id, participant_ep_ids=workload.participant_ep_ids, trace_identity=None,
        )
        plan = EPWavePlan(None, workload, phases, timing, None)
        dump(root / "stage.json", {"stage": "emission", "case": args.case, "layer": layer_id})
        record_ep_wave(store, plan, time=namespace["captured"]["meta"]["wave_start_time_s"],
                       replica_id=case["replica_id"], stage_id=0, cluster_type=role)
        waves.append({"layer": layer_id, "routing_source": asdict(routing), "profiled_pricing_s": pricing_s,
                      "live_calls": calls, "operators_per_wave": len(actual_ops),
                      "operator_names": sorted(set(actual_ops))})

    store.trace_store.flush()
    trace_path = trace_dir / "op_traces.jsonl"
    trace_raw = trace_path.read_bytes()
    trace_rows = [json.loads(row) for row in trace_raw.splitlines()]
    assert "simulation_info" in trace_rows[0]["meta"]
    events = trace_rows[1:]
    assert len(events) == len(expected_events)
    ledger = normalized(store._frontier_ep_wave_lane_ledger_rows)
    dump(root / "ledger.json", ledger)
    dump(root / "pricing_payload.json", payloads)
    for event, expected in zip(events, expected_events):
        meta = event["meta"]
        actual_identity = (meta["global_layer_ids"], meta["ep_id"], meta["phase"], event["type"], event["name"], event["duration_ms"])
        expected_identity = ([expected["layer"]], expected["ep"], expected["phase"], expected["kind"], expected["name"], expected["duration_ms"])
        assert actual_identity == expected_identity
        assert event["cluster"] == role.name
        lane_ledger = next(row for row in ledger if row["global_layer_ids"] == [expected["layer"]] and row["ep_id"] == expected["ep"])
        phase_ledger = lane_ledger["phases"][expected["phase"]]
        assert phase_ledger["operators_ms"][expected["name"]] == expected["duration_ms"]
        if expected["kind"] == "COMPUTE":
            assert meta["operator_cost"] == expected["proof"]
            assert phase_ledger["compute_proofs"][expected["name"]] == expected["proof"]
        else:
            assert meta["collective_plan"] == expected["proof"]["plan"]
            assert meta["collective_cost"] == expected["proof"]["cost"]
            assert phase_ledger["collective_proofs"][expected["name"]] == expected["proof"]
    verify_bindings()
    result = {"task_id": parameters["task_id"], "case": args.case, "input": case,
              "status": "PASS_CPU_FIXTURE_PRODUCTION_SHAPE", "source": source_binding,
              "production_functions": paths, "waves": waves, "event_count": len(events),
              "ledger_lane_count": len(ledger), "all_values_order_proofs_and_keys_match": True,
              "fixture_metadata_unchanged": True, "source_hashes_unchanged": True,
              "whole_jsonl": {"bytes": len(trace_raw), "sha256": sha256(trace_raw)},
              "ledger_sha256": sha256((root / "ledger.json").read_bytes()),
              "python": sys.version, "rlimit_as": list(resource.getrlimit(resource.RLIMIT_AS)),
              "limits": "One invocation; external60s/30GiB; profiler time is not a speedup benchmark; not native/fullCLI."}
    dump(root / "result.json", result)
    dump(root / "stage.json", {"stage": "complete", "case": args.case})
    print(json.dumps({key: result[key] for key in ("case", "status", "event_count", "ledger_lane_count", "waves")}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    parameters = json.loads(args.parameters.read_bytes())
    assert args.case in parameters["cases"]
    try:
        inspect_case(args, parameters)
    except Exception as error:
        if args.output.exists():
            dump(args.output / "failure.json", {"status": "UNVERIFIED", "type": type(error).__name__,
                 "message": str(error), "traceback": traceback.format_exc()})
        raise
