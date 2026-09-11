"""Audit or evaluate an explicit runtime cost table for one captured decode batch.

The default writes a profile coverage plan, not a prediction. Captured timings
never enter cost queries. Predictions require an explicit complete cost table.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from frontier.runtime_cost.sglang import (
    DecodeWorkload, ExactRuntimeCostTable, RuntimeIdentity, SGLangCostContract, capture_stack_signature,
)
from frontier.validation.batch_record import read_batches
from frontier.validation.routing import read_routing
from frontier.validation.routing_report import audit_routing


def plan_captured_batch(records, routes, *, model_config, identity, padding_context_lens=()):
    """Require all-rank agreement before using one rank's observed load vectors."""
    records, routes = tuple(records), tuple(routes)
    if len({r.batch_id for r in records}) != 1 or any(r.phase != "decode" for r in records):
        raise ValueError("Runtime planning requires exactly one decode cohort")
    audit, _ = audit_routing(records, routes, layer_ids=list(range(model_config.num_layers)),
                            tensor_parallel_size=identity.tensor_parallel_size, model_config=model_config)
    if audit["tp_logical_mismatches"] or audit["tp_padding_mismatches"]:
        raise ValueError("TP routing mismatch prevents rank-0 runtime cost planning")
    reference = next(r for r in records if r.rank == 0)
    padding_context_lens = tuple(padding_context_lens)
    if len(padding_context_lens) != reference.capture_size - len(reference.request_ids):
        raise ValueError("Explicit context lengths are required for every graph-padding lane")
    workload = DecodeWorkload(len(reference.request_ids),
        tuple(n + 1 for n in reference.context_lens) + padding_context_lens)
    contract = SGLangCostContract(model_config, identity)
    rank0_routes = tuple(r for r in routes if r.rank == 0)
    queries = contract.queries(workload, rank0_routes)
    return contract, workload, rank0_routes, queries, audit


def identity_for_capture(manifest, model, device):
    """Shared admission gate for runtime query planning and profile production."""
    topology, server = manifest["topology"], manifest["server_args"]
    if (manifest["status"] != "complete" or manifest.get("engine") != "sglang"
            or manifest.get("workload_mode") != "static_batch"
            or Path(manifest["model_path"]).name != Path(model.name).name
            or topology != {"tp": topology["tp"], "ep": 1, "pp": 1, "nodes": 1}
            or server.get("tp_size") != topology["tp"]
            or server.get("ep_size", 1) != 1 or server.get("pp_size", 1) != 1
            or server.get("dp_size", 1) != 1 or server.get("enable_dp_attention", False)
            or server.get("moe_dp_size", 1) != 1 or server.get("moe_a2a_backend", "none") != "none"
            or server.get("enable_fused_qk_norm_rope", False)
            or server.get("enable_fused_moe_sum_all_reduce", False)
            or server.get("enable_two_batch_overlap", False) or server.get("enable_single_batch_overlap", False)
            or server.get("enable_eplb", False) or server.get("ep_num_redundant_experts", 0)
            or server.get("speculative_algorithm")):
        raise ValueError("Capture does not match the one-node static runtime contract")
    if not manifest["devices"] or any(device.lower() not in d["device_name"].lower().replace(" ", "")
                                      for d in manifest["devices"]):
        raise ValueError("Requested device does not match captured hardware")
    # Cost producer provenance is independent of the Frontier capture observer.
    return RuntimeIdentity.for_model(model, device=device, tensor_parallel_size=topology["tp"],
                                        runtime_stack_signature=capture_stack_signature(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--pass-name", choices=("routing", "graph"), default="routing")
    parser.add_argument("--padding-context-lens", type=int, nargs="*", default=[])
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.predict and args.profile is None:
        parser.error("--predict requires --profile; missing costs are never fabricated")
    if args.allow_diagnostic and not args.predict:
        parser.error("--allow-diagnostic only applies to --predict")
    from frontier.profiling.common.model_config import ModelConfig
    model = ModelConfig.from_model_name(args.model)
    manifest = json.loads((args.capture_dir / "manifest.json").read_text())
    identity = identity_for_capture(manifest, model, args.device)
    records = [r for r in read_batches(args.capture_dir / f"{args.pass_name}-batches.jsonl")
               if r.batch_id == args.batch_id]
    routes = []
    for rank in range(identity.tensor_parallel_size):
        for route in read_routing(args.capture_dir / f"{args.pass_name}-routing-rank{rank}.jsonl.gz"):
            if route.rank != rank:
                raise ValueError("Routing rank disagrees with source file")
            if route.batch_id == args.batch_id:
                routes.append(route)
    contract, workload, rank0, queries, audit = plan_captured_batch(records, routes,
        model_config=model, identity=identity, padding_context_lens=args.padding_context_lens)
    payload = json.loads(args.profile.read_text()) if args.profile else {
        "schema_version": 1, "identity": asdict(identity), "rows": []}
    table = ExactRuntimeCostTable(payload, identity=identity)
    result = {"schema_version": 1, "batch_id": args.batch_id, "identity": asdict(identity),
        "routing_audit": audit, "profile_coverage": table.audit(queries),
        "queries": [{"query": asdict(q), "query_key": q.key} for q in queries],
        "boundary_note": "Routing-conditioned static decoder estimate; not unprofiled forward, "
                         "serving, or distributed scaling validation. Fixed inputs do not freeze routes."}
    if args.predict:
        prediction = contract.predict(workload, rank0, table, allow_diagnostic=args.allow_diagnostic)
        result["prediction"] = prediction.summary()
        result["native_op_times_per_layer_average"] = dict(prediction.execution.op_times)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"batch_id": args.batch_id, "layers": model.num_layers,
        "queries": len(queries), "missing_profiles": len(result["profile_coverage"]["missing"]),
        "prediction": result.get("prediction"), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
