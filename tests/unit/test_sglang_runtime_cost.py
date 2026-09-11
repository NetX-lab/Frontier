from dataclasses import asdict, replace
import gzip
import json

import numpy as np
import pytest

from frontier.operators.binding import bind_operator_query, build_operator_manifest
from frontier.profiling.common.model_config import ModelConfig
from frontier.runtime_cost.sglang import (
    CostEstimate, DecodeWorkload, ExactRuntimeCostTable, RuntimeIdentity, SGLangCostContract,
)
from frontier.validation.routing import snapshot_routing
from frontier.validation.batch_record import BatchRecord
from frontier.validation.runtime_cost import plan_captured_batch


@pytest.fixture
def case():
    model = ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")
    model.num_layers = 4
    if model.layer_types:
        model.layer_types = model.layer_types[:4]
    identity = RuntimeIdentity.for_model(model, device="test_device", tensor_parallel_size=8,
                                       runtime_stack_signature="synthetic-unit-test-only")
    contract = SGLangCostContract(model, identity)
    workload = DecodeWorkload(2, (1025, 1025, 1, 1))
    routes = tuple(snapshot_routing(
        np.tile(np.arange(10, dtype=np.int32), (4, 1)), np.full((4, 10), .1, dtype=np.float32),
        batch_id="synthetic", rank=0, layer_id=layer, capture_id=1,
        logical_size=2, num_experts=512, top_k=10) for layer in range(model.num_layers))
    return contract, workload, routes


def provider(query):
    # Deliberately distinct layer/op values, never a performance fixture.
    return CostEstimate(query.key, 1. + query.layer_id + len(query.component) / 100,
        "synthetic-unit-test-only", "KERNEL_ONLY",
        "calibrated_collective" if query.is_collective else "isolated_compute")


def test_native_composition_preserves_every_layer_and_charges_fusions_once(case):
    contract, workload, routes = case
    result = contract.predict(workload, routes, provider)
    expected = sum(provider(q).milliseconds for q in result.queries)
    assert result.decoder_ms == pytest.approx(expected)
    assert sum(e.model_time_ms for e in result.layer_executions) == pytest.approx(expected)
    assert len(result.layer_executions) == 4
    assert {q.layer_kind for q in result.queries} == {"gdn", "attention"}
    assert not result.diagnostic
    assert not result.summary()["full_forward_parity_admitted"]
    execution = result.execution
    assert execution.share_expert_tensor_parallel_allreduce_time == 0
    assert execution.add_attn_residual_time == execution.add_ffn_residual_time == 0
    assert all(name.startswith("sglang_") for name in execution.op_times)
    reductions = [q for q in result.queries if q.component == "mlp_tp_allreduce"]
    assert len(reductions) == 4
    assert execution.op_times["sglang_mlp_tp_allreduce"] * 4 == pytest.approx(sum(provider(q).milliseconds for q in reductions))
    # Both shared activation and the separate fused gate survive their common
    # legacy carrier. It is subtracted once, not once per physical name.
    for layer in result.layer_executions:
        assert layer.model_time_ms == pytest.approx(sum(layer.op_times.values()))


def test_residual_ownership_and_physical_padding_are_explicit(case):
    contract, workload, routes = case
    queries = contract.queries(workload, routes)
    norms = [q for q in queries if q.component == "input_layernorm"]
    assert [q.residual_from_layer for q in norms] == [None, 0, 1, 2]
    post = [q for q in queries if q.component == "post_attention_layernorm"]
    assert [q.residual_from_layer for q in post] == [0, 1, 2, 3]
    experts = [q for q in queries if q.component == "moe_experts_quant_gemm_combine"]
    assert all(sum(q.physical_expert_counts) == 40 for q in experts)
    assert all(q.workload.logical_size == 2 and q.workload.physical_size == 4 for q in queries)
    assert all(q.workload.physical_context_lens[-2:] == (1, 1) for q in queries)
    assert all(not q.physical_expert_counts for q in queries if q.component not in {
        "moe_sorting", "moe_experts_quant_gemm_combine"})


@pytest.mark.parametrize("field,value", [("nodes", 2), ("expert_parallel_size", 2),
    ("pipeline_parallel_size", 2), ("data_parallel_size", 2), ("tensor_parallel_size", 1),
    ("tensor_parallel_size", True), ("runtime_stack_signature", ""), ("contract", "other")])
def test_unsupported_runtime_topology_and_identity_fail(case, field, value):
    with pytest.raises(ValueError):
        replace(case[0].identity, **{field: value})


@pytest.mark.parametrize("logical,contexts,mode", [(0, (1,), "FULL"), (2, (1,), "FULL"),
    (1, (0,), "FULL"), (1, (True,), "FULL"), (1, (1,), "NONE")])
def test_workload_requires_explicit_valid_physical_decode_shape(logical, contexts, mode):
    with pytest.raises(ValueError):
        DecodeWorkload(logical, contexts, mode)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "rank", "batch", "capture", "shape"])
def test_routing_admission_fails_closed(case, mutation):
    contract, workload, routes = case
    if mutation == "missing":
        routes = routes[:-1]
    elif mutation == "duplicate":
        routes = (*routes[:-1], routes[0])
    elif mutation == "shape":
        workload = DecodeWorkload(3, (1025, 1025, 1, 1))
    else:
        field, value = {"rank": ("rank", 1), "batch": ("batch_id", "other"),
                        "capture": ("capture_id", 2)}[mutation]
        routes = (*routes[:-1], replace(routes[-1], **{field: value}))
    with pytest.raises(ValueError):
        contract.queries(workload, routes)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_cost_estimates_reject_invalid_numbers(value):
    with pytest.raises(ValueError):
        CostEstimate("a" * 64, value, "unit", "KERNEL_ONLY", "isolated_compute")


def test_profile_family_and_diagnostic_guards(case):
    contract, workload, routes = case
    def diagnostic(q):
        return replace(provider(q), measurement_type="HIP_GRAPH_EVENT", basis="in_situ_diagnostic")
    with pytest.raises(ValueError, match="diagnostic opt-in"):
        contract.predict(workload, routes, diagnostic)
    result = contract.predict(workload, routes, diagnostic, allow_diagnostic=True)
    assert result.diagnostic and result.summary()["measurement_types"] == ["HIP_GRAPH_EVENT"]
    with pytest.raises(ValueError, match="diagnostic"):
        replace(provider(result.queries[0]), measurement_type="HIP_GRAPH_EVENT")
    with pytest.raises(ValueError, match="Eager"):
        contract.predict(workload, routes, lambda q: replace(provider(q), measurement_type="CUDA_EVENT"))
    with pytest.raises(ValueError, match="cannot be interchanged"):
        contract.predict(workload, routes, lambda q: replace(provider(q), basis="isolated_compute"))
    with pytest.raises(ValueError, match="mismatched"):
        contract.predict(workload, routes, lambda q: replace(provider(q), query_key="0" * 64))


def table_payload(contract, queries):
    return json.loads(json.dumps({"schema_version": 1, "identity": asdict(contract.identity),
        "rows": [{"query": asdict(q), "estimate": asdict(provider(q))} for q in queries]}))


def test_exact_profile_roundtrip_and_no_silent_fallback_or_padding_erasure(case):
    contract, workload, routes = case
    queries = contract.queries(workload, routes)
    table = ExactRuntimeCostTable(table_payload(contract, queries), identity=contract.identity)
    assert table.audit(queries)["complete"]
    assert contract.predict(workload, routes, table).decoder_ms > 0
    changed = replace(queries[0], workload=DecodeWorkload(2, (1025, 1025, 2, 2)))
    assert not table.audit([changed])["complete"]
    with pytest.raises(ValueError, match="No exact"):
        table(changed)
    different_layer = replace(queries[0], layer_id=1, residual_from_layer=0)
    assert different_layer.key != queries[0].key
    # Dropping one gate cost does not turn it into a zero estimate.
    payload = table_payload(contract, queries[:-1])
    table = ExactRuntimeCostTable(payload, identity=contract.identity)
    assert len(table.audit(queries)["missing"]) == 1
    with pytest.raises(ValueError, match="No exact"):
        contract.predict(workload, routes, table)


@pytest.mark.parametrize("mutation", ["schema", "identity", "duplicate", "key"])
def test_profile_table_rejects_cross_runtime_or_malformed_rows(case, mutation):
    contract, workload, routes = case
    payload = table_payload(contract, contract.queries(workload, routes)[:1])
    if mutation == "schema":
        payload["schema_version"] = True
    elif mutation == "identity":
        payload["identity"]["runtime_stack_signature"] = "other"
    elif mutation == "duplicate":
        payload["rows"] *= 2
    else:
        payload["rows"][0]["estimate"]["query_key"] = "0" * 64
    with pytest.raises(ValueError):
        ExactRuntimeCostTable(payload, identity=contract.identity)


def test_runtime_operators_have_native_registry_ownership(case):
    contract, workload, routes = case
    for query in contract.queries(workload, routes):
        binding = bind_operator_query(query.operator)
        assert binding.family_id == "sglang_runtime"
        assert not binding.operator.profiling_target
    # Registering an optional runtime family must not change model selection.
    manifest = build_operator_manifest(case[0].model_config)
    assert "sglang_runtime" not in {family.family_id for family in manifest.families()}


def cohort(case):
    contract, _, routes = case
    records = tuple(BatchRecord(batch_id="synthetic", rank=rank, phase="decode",
        request_ids=("a", "b"), query_lens=(1, 1), context_lens=(1024, 1024),
        prefill_mask=(False, False), graph_mode="FULL", capture_size=4, profiled=True,
        decode_input_sha256="a" * 64, step=1) for rank in range(8))
    all_routes = tuple(replace(r, rank=rank) for rank in range(8) for r in routes)
    return records, all_routes


def test_capture_plan_verifies_tp_and_keeps_observations_out_of_queries(case):
    records, routes = cohort(case)
    kwargs = dict(model_config=case[0].model_config, identity=case[0].identity, padding_context_lens=(1, 1))
    _, workload, rank0, queries, audit = plan_captured_batch(records, routes, **kwargs)
    assert workload == case[1] and len(rank0) == 4
    assert not audit["tp_logical_mismatches"]
    changed = tuple(replace(r, forward_gpu_ms=10000, step_wall_ms=20000) for r in records)
    assert [q.key for q in queries] == [q.key for q in plan_captured_batch(changed, routes, **kwargs)[3]]
    changed_routes = (*routes[:-1], replace(routes[-1], logical_weights_sha256="b" * 64))
    with pytest.raises(ValueError, match="TP routing mismatch"):
        plan_captured_batch(records, changed_routes, **kwargs)
    with pytest.raises(ValueError, match="padding"):
        plan_captured_batch(records, routes, **{**kwargs, "padding_context_lens": ()})


def test_model_mutation_cannot_reuse_an_old_runtime_identity(case):
    contract, _, _ = case
    changed = ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")
    with pytest.raises(ValueError, match="identity"):
        SGLangCostContract(changed, contract.identity)


def test_model_fingerprint_normalizes_tensor_and_string_dtypes(case):
    model = case[0].model_config
    class TensorDtype:
        def __str__(self):
            return "torch.bfloat16"
    def identity():
        return RuntimeIdentity.for_model(model, device="test_device", tensor_parallel_size=8,
                                         runtime_stack_signature="synthetic-unit-test-only")
    model._dtype = "bf16"
    expected = identity()
    model._dtype = TensorDtype()
    assert identity() == expected


def test_model_fingerprint_includes_head_dim_override(case):
    model = case[0].model_config
    assert model.get_head_dim() == 256
    previous = case[0].identity
    model._head_dim = 128
    changed = RuntimeIdentity.for_model(model, device=previous.device, tensor_parallel_size=8,
                                       runtime_stack_signature=previous.runtime_stack_signature)
    assert changed != previous


def test_isolated_graph_replay_measurements_remain_explicit(case):
    contract, workload, routes = case
    result = contract.predict(workload, routes,
        lambda q: replace(provider(q), measurement_type="HIP_GRAPH_REPLAY"))
    assert result.summary()["measurement_types"] == ["HIP_GRAPH_REPLAY"]
    assert not result.summary()["full_forward_parity_admitted"]


def test_query_rejects_wrong_boundary_or_residual_ownership(case):
    query = case[0].queries(case[1], case[2])[0]
    with pytest.raises(ValueError, match="boundary"):
        replace(query, component="invented")
    with pytest.raises(ValueError, match="residual"):
        replace(query, residual_from_layer=0)
    with pytest.raises(ValueError, match="Routed"):
        replace(query, component="moe_sorting")
    routes = (*case[2][:-1], replace(case[2][-1], padding_positive_weight_slots=0))
    with pytest.raises(ValueError, match="pruning-aware"):
        case[0].queries(case[1], routes)


def test_runtime_cost_cli_audits_then_composes_only_explicit_profiles(case, tmp_path, monkeypatch):
    from frontier.validation.runtime_cost import main
    model = case[0].model_config
    monkeypatch.setattr(ModelConfig, "from_model_name", lambda name: model)
    manifest = {"status": "complete", "engine": "sglang", "workload_mode": "static_batch",
        "model_path": model.name, "topology": {"tp": 8, "ep": 1, "pp": 1, "nodes": 1},
        "server_args": {"tp_size": 8}, "devices": [{"device_name": "test_device"}],
        "versions": {"sglang_commit": "synthetic-only"}, "environment": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    records, routes = cohort(case)
    (tmp_path / "routing-batches.jsonl").write_text("".join(json.dumps(asdict(r)) + "\n" for r in records))
    for rank in range(8):
        with gzip.open(tmp_path / f"routing-routing-rank{rank}.jsonl.gz", "wt") as stream:
            for route in routes:
                if route.rank == rank:
                    stream.write(json.dumps(asdict(route)) + "\n")
    base = ["runtime-cost", "--capture-dir", str(tmp_path), "--model", model.name,
            "--device", "test_device", "--batch-id", "synthetic", "--padding-context-lens", "1", "1"]
    plan_file = tmp_path / "plan.json"
    monkeypatch.setattr("sys.argv", [*base, "--output", str(plan_file)])
    main()
    plan = json.loads(plan_file.read_text())
    assert not plan["profile_coverage"]["complete"] and "prediction" not in plan
    assert plan["queries"][0]["query"]["workload"]["physical_context_lens"] == [1025, 1025, 1, 1]
    rows = []
    for entry in plan["queries"]:
        collective = entry["query"]["component"] in {"attention_tp_allreduce", "mlp_tp_allreduce"}
        rows.append({"query": entry["query"], "estimate": asdict(CostEstimate(entry["query_key"],
            2., "synthetic-unit-test-only", "KERNEL_ONLY",
            "calibrated_collective" if collective else "isolated_compute"))})
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema_version": 1, "identity": plan["identity"], "rows": rows}))
    prediction_file = tmp_path / "prediction.json"
    monkeypatch.setattr("sys.argv", [*base, "--output", str(prediction_file), "--profile", str(profile), "--predict"])
    main()
    prediction = json.loads(prediction_file.read_text())
    assert prediction["prediction"]["decoder_ms"] == pytest.approx(2 * len(rows))
    assert not prediction["prediction"]["full_forward_parity_admitted"]
    # The explicit contract rejects a changed fused-collective runtime even
    # when the selected checkpoint and rank-0 routing look identical.
    manifest["server_args"]["enable_fused_moe_sum_all_reduce"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="runtime contract"):
        main()
