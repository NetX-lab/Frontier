"""CPU checks for candidate-only Step communication cost sidecars."""

from copy import deepcopy
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from frontier.profiling.collectives.vllm_forward_collectives import DTYPE_BYTES


SCRIPT = Path(__file__).parents[1] / "integration/step5_comm/export_costs.py"
SPEC = importlib.util.spec_from_file_location("step5_comm_cost_export", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC.loader.exec_module(EXPORT)
finally:
    sys.path.pop(0)

CONFIG = {"engine": {"tensor_parallel_size": 8, "data_parallel_size": 4},
          "gpus_per_node": 8, "repeats": 30, "passes": 3, "calls_per_repeat": 5}
AR_COUNTS = {"embedding_all_reduce": 1, "attention_all_reduce": 92,
             "dense_down_proj_all_reduce": 4, "shared_expert_down_proj_all_reduce": 88,
             "routed_expert_all_reduce": 88}


def fixture_record(op="all_gatherv", tokens=(8, 8, 8, 8), rails=tuple(range(8)),
                   useful=None, graph=False, call=None, pass_id=0):
    tp, world = 8, 32
    group = "tp" if op in ("all_reduce", "all_gather") else "dp"
    host = op == "cpu_all_reduce"
    if host:
        rails = (0,)
    groups = ([list(range(start, start + tp)) for start in range(0, world, tp)]
              if group == "tp" else [list(range(rail, world, tp)) for rail in rails])
    ranks = sorted(rank for members in groups for rank in members)
    dtypes, columns = (["float8_e4m3fn", "float32", "int32", "float32"], [4096, 8, 8, 32])
    if op == "all_reduce" or op.startswith("reduce_scatter"):
        dtypes, columns = ["bfloat16"], [4096]
    elif op == "all_gather":
        dtypes, columns = ["bfloat16"], [16112]
    elif host:
        dtypes, columns = ["int32"], [4]
    rows = [6 if host else tokens[rank // tp] if rank in ranks else 0 for rank in range(world)]
    row_bytes = sum(DTYPE_BYTES[d] * c for d, c in zip(dtypes, columns))
    endpoints = []
    for members in groups:
        total = sum(rows[rank] for rank in members) * row_bytes
        for rank in members:
            local = rows[rank] * row_bytes
            endpoints.append((total if op.startswith("reduce_scatter") else local,
                              total if op in ("all_gather", "all_gatherv") else local))
    timings = [[1000 + rank + j for j in range(30)] if rank in ranks else None
               for rank in range(world)]
    return {
        "pass": pass_id, "scenario": "graph_decode_b8" if graph else "eager_uniform_t8",
        "call": call or {"all_reduce": "embedding_all_reduce", "all_gather": "logits_all_gather",
                         "all_gatherv": "moe_dispatch_all_gatherv_fp8",
                         "reduce_scatter": "moe_combine_reduce_scatterv",
                         "reduce_scatterv": "moe_combine_reduce_scatterv",
                         "cpu_all_reduce": "dp_batch_sync_all_reduce"}[op],
        "op": op, "group": group, "group_size": len(groups[0]), "cuda_graph": graph,
        "calls_per_forward": AR_COUNTS.get(call, 1), "tokens_per_dp_rank": list(tokens),
        "useful_tokens_per_dp_rank": list(useful or tokens), "padded_tokens_per_dp_rank": list(tokens),
        "dtypes": dtypes, "columns": columns, "concurrent_dp_rails": len(groups) if group == "dp" else None,
        "participating_ranks": ranks, "group_ranks": groups, "shard_rows": rows,
        "source": "native/source.py:10", "max_input_bytes": max(pair[0] for pair in endpoints),
        "max_output_bytes": max(pair[1] for pair in endpoints), "per_rank_call_us": timings,
        "per_rank_launch_call_us": [None] * world if host else deepcopy(timings),
        "per_rank_queue_held": [None if host or rank not in ranks else [True] * 30 for rank in range(world)],
    }


def source_rows(record, backend=None):
    host = record["op"] == "cpu_all_reduce"
    critical = [max(record["per_rank_call_us"][rank][j] for rank in record["participating_ranks"])
                for j in range(30)]
    return ({"pass": record["pass"], "scenario": record["scenario"], "call": record["call"],
             "timing_domain": "host" if host else "device",
             "median_us": EXPORT.timing_stats(critical)["median_us"]},
            {"scenario": record["scenario"], "call": record["call"],
             "communicator": backend or ("host_gloo" if host else "nccl"),
             "kernels": [] if host else ["native_collective_kernel"]})


def observe(record, backend=None):
    summary, kernel = source_rows(record, backend)
    return EXPORT.observation(record, summary, kernel, CONFIG, 0.01)


@pytest.mark.parametrize("op,input_bytes,output_bytes,devices,api", [
    ("all_reduce", 65536, 65536, 8, "allreduce"),
    ("all_gatherv", 34304, 137216, 4, "allgather"),
    ("all_gather", 257792, 2062336, 8, "allgather"),
    ("reduce_scatterv", 262144, 65536, 4, "reduce_scatter"),
    ("reduce_scatter", 262144, 65536, 4, "reduce_scatter"),
])
def test_scalar_api_uses_endpoint_input_bytes_not_ep_world_or_rs_output(op, input_bytes, output_bytes, devices, api):
    row = observe(fixture_record(op))
    assert row["scalar_api_representable"]
    assert row["api_operation"] == api
    assert row["data_size_bytes"] == input_bytes
    assert row["per_rank_input_bytes"][0] == input_bytes
    assert row["per_rank_output_bytes"][0] == output_bytes
    assert row["num_devices"] == devices
    assert row["comm_domain"] == row["group"].upper()
    assert row["world_size"] == 32
    assert row["payload_maxima_match"]
    assert row["production_consumable"] is False


def test_fp8_includes_router_and_activation_scale_bytes():
    row = observe(fixture_record())
    assert row["row_bytes"] == 4096 + 32 + 32 + 128 == 4288
    assert row["active_dp_rails"] == list(range(8))
    assert row["concurrent_dp_rails"] == 8
    assert row["group_payload_vectors"][0] == {
        "ranks": [0, 8, 16, 24], "input_bytes": [34304] * 4, "output_bytes": [137216] * 4}


def test_bf16_dispatch_has_router_metadata_but_no_fp8_scales():
    record = fixture_record(call="moe_dispatch_all_gatherv_bf16")
    record.update(dtypes=["bfloat16", "float32", "int32"], columns=[4096, 8, 8],
                  max_input_bytes=66048, max_output_bytes=264192)
    row = observe(record)
    assert row["row_bytes"] == 8256
    assert row["data_size_bytes"] == 66048
    assert row["per_rank_output_bytes"][0] == 264192
    assert row["payload_maxima_match"]


def test_into_output_combine_keeps_actual_gathered_input_endpoint():
    record=fixture_record("reduce_scatterv",tokens=(16,1,4,2))
    record["op"]="reduce_scatterv_into_output"
    row=observe(record)
    assert row["per_rank_input_bytes"][0] == 23*8192
    assert row["per_rank_output_bytes"][0] == 16*8192
    assert row["payload_maxima_match"] and not row["scalar_api_representable"]


def test_nnls_affine_preserves_negative_remainder_and_independent_errors():
    cells=[dict(operand_bytes=x,duration_ms=y,split="calibration",quality="PASS")
           for x,y in ((1,-2.),(16,-1.),(256,1.))]
    cells += [dict(operand_bytes=64,duration_ms=.5,split="qualification",quality="PASS")]
    fit=EXPORT.fit_affine(cells,positive_slope=False,supported_range=(1,256),calibration_receipt="cpu_fixture")
    assert fit["design_matrix_rank"] == 2 and fit["distinct_sizes"] == 3
    assert fit["signed_residuals_ms"][0] < 0
    assert fit["raw_cells"][0]["duration_ms"] == -2.
    assert fit["qualification"]["status"] == "UNVERIFIED" and fit["candidate_only"]
    assert len(fit["qualification"]["observations"]) == 1
    with pytest.raises(ValueError,match="anchors"):
        EXPORT.fit_affine(cells[1:],positive_slope=False,supported_range=(16,256),calibration_receipt="cpu_fixture")
    with pytest.raises(ValueError,match="extrapolate"):
        EXPORT.fit_affine(cells,positive_slope=False,supported_range=(1,8192),calibration_receipt="cpu_fixture")
    corrupted=deepcopy(cells); corrupted[0]["quality"]="FAIL"
    with pytest.raises(ValueError,match="excluded"):
        EXPORT.fit_affine(corrupted,positive_slope=False,supported_range=(1,256),calibration_receipt="cpu_fixture")


def test_equal_bytes_preserve_distinct_physical_cells_and_all_passes():
    cells=[dict(operand_bytes=17,duration_ms=t,split="qualification",cell_id=key,source=key)
           for key,t in (("tail",1.),("tail",3.),("tail_swap",8.))]
    values=EXPORT.cell_medians(cells)
    assert len(values) == 2 and [c["duration_ms"] for c in values] == [2.,8.]
    assert [len(c["raw_passes"]) for c in values] == [2,1]


@pytest.mark.parametrize("defect",["fixture","hash","independence","coverage","heldout"])
def test_parent_selection_never_qualifies_fixtures_or_missing_model_evidence(tmp_path,defect):
    fit=EXPORT.fit_affine([dict(operand_bytes=x,duration_ms=x,split=split,quality="PASS")
        for x,split in ((1,"calibration"),(16,"calibration"),(256,"calibration"),(64,"qualification"))],
        positive_slope=True,supported_range=(1,256),calibration_receipt="cpu_schema_negative_test")
    artifact=dict(evidence_kind="native_measurement",candidate_only=True,models=[dict(parameters=dict(protocol=fit))])
    if defect == "fixture": artifact["evidence_kind"]="cpu_fixture"
    if defect == "heldout": fit["qualification"]["observations"]=[]
    candidate=tmp_path/"candidate.json";candidate.write_text(json.dumps(artifact))
    receipt=dict(schema_version="step5.collective_qualification.v1",candidate_sha256=EXPORT.sha256(candidate),
        status="PASS",independent=True,observations=["cpu_schema_negative_test"],
        model_checks=[dict(model_index=0,status="PASS",independent=True,observations=["cpu_schema_negative_test"])])
    if defect == "hash": receipt["candidate_sha256"]="wrong"
    if defect == "independence": receipt["independent"]=False
    if defect == "coverage": receipt["model_checks"]=[]
    path=tmp_path/"receipt.json";path.write_text(json.dumps(receipt));output=tmp_path/"selected.json"
    with pytest.raises(ValueError): EXPORT.qualify_finite_parameters(candidate,path,output)
    assert not output.exists()


def test_parent_selection_preserves_frozen_parameters_and_signed_evidence(tmp_path):
    fit=EXPORT.fit_affine([dict(operand_bytes=x,duration_ms=x,split=split,quality="PASS")
        for x,split in ((1,"calibration"),(16,"calibration"),(256,"calibration"),(64,"qualification"))],
        positive_slope=True,supported_range=(1,256),calibration_receipt="cpu_schema_only")
    artifact=dict(evidence_kind="native_measurement",candidate_only=True,models=[dict(parameters=dict(protocol=fit))])
    candidate=tmp_path/"candidate.json";candidate.write_text(json.dumps(artifact));before=candidate.read_bytes()
    receipt=dict(schema_version="step5.collective_qualification.v1",candidate_sha256=EXPORT.sha256(candidate),
        status="PASS",independent=True,observations=["cpu_schema_only"],
        model_checks=[dict(model_index=0,status="PASS",independent=True,observations=["cpu_schema_only"])])
    path=tmp_path/"receipt.json";path.write_text(json.dumps(receipt))
    selected=EXPORT.qualify_finite_parameters(candidate,path,tmp_path/"selected.json")
    result=selected["models"][0]["parameters"]["protocol"]
    assert candidate.read_bytes() == before and selected["candidate_only"] is False
    assert result["qualification"]["parent_receipt"] == str(path)
    assert all(result[key] == fit[key] for key in ("a_ms","b_ms_per_byte","signed_residuals_ms","raw_cells","range_bytes"))


@pytest.mark.parametrize("network_defect",[None,"partial","zero","nan","inf"])
def test_finite_exporter_executes_joint_htsim_without_blessing_fixture(tmp_path,monkeypatch,network_defect):
    submodule=Path(__file__).parents[2]/"frontier/cc_backend/backends/collective-sim"
    if not (submodule/"sim/datacenter/htsim_ndp").exists():
        pytest.skip("Real HTSIM binary unavailable")
    monkeypatch.syspath_prepend(str(submodule/"python"))
    from collective_sim_core.schema import Scenario,ClusterConfig,ParallelismConfig,CollectiveConfig,TopologyConfig
    from dataclasses import asdict
    scenario=Scenario(ClusterConfig(2,8),ParallelismConfig(tp=8,dp=2),topology=TopologyConfig(kind="rail",servers_per_tor=1),
        collective=CollectiveConfig(kind="plan",tensor_bytes=0,domain_dims=("DP",),placement_order=("TP","DP"),intra_server_combine_rule="sum"))
    frozen=tmp_path/"frozen.json"
    frozen.write_text(json.dumps(dict(device_type="h800",scenario=asdict(scenario),collective_sim_root=str(submodule))))
    root=tmp_path/"run/collectives";root.mkdir(parents=True)
    config=dict(frozen_model_config=str(frozen),frozen_model_config_sha256=EXPORT.sha256(frozen),
                grid=dict(scenario_splits={f"s{r}":split for r,split in ((1,"calibration"),(16,"calibration"),(256,"calibration"),(64,"qualification"))}))
    source=dict(config=config,evidence_kind="cpu_fixture",runtime=dict(native_facts=dict(shared="cpu_fixture")))
    (root/"plan.json").write_text(json.dumps(source))
    records,kernels,local=[],[],[]
    for rows,split in ((1,"calibration"),(16,"calibration"),(256,"calibration"),(64,"qualification")):
        name=f"s{rows}"
        local.append(dict(tensor_name="activation",dtype="bfloat16",columns=4,cuda_graph=False,kind="copy",
            name=name,rows=rows,row_stride=4,scope="independent_preallocated_operand",split=split,**{"pass":0},
            per_rank_queue_held=[[True]*3 for _ in range(16)],per_rank_call_us=[[1+rows*.001]*3 for _ in range(16)]))
        records.append(dict(scenario=name,call="moe_dispatch_all_gatherv_bf16",op="all_gatherv",
            native_context=dict(shared="cpu_fixture"),evidence_kind="cpu_fixture",dtypes=["bfloat16"],columns=[4],
            tokens_per_dp_rank=[rows,rows],group_ranks=[[r,r+8] for r in range(8)],cuda_graph=False,
            buffer_receipt=dict(ordinary=True),participating_ranks=list(range(16)),**{"pass":0},
            per_rank_call_us=[[1000+rows]*3 for _ in range(16)],per_rank_launch_call_us=[[1000+rows]*3 for _ in range(16)],
            per_rank_queue_held=[[True]*3 for _ in range(16)]))
        kernels.append(dict(scenario=name,call=records[-1]["call"],per_rank_kernel_counts=[7]*16,
                             per_rank_events=[["cpu_fixture"]*35 for _ in range(16)],profiled_calls=5))
        records.append({**records[-1],"call":"shared_expert_down_proj_all_reduce","op":"all_reduce",
            "group_ranks":[list(range(8)),list(range(8,16))],
            "per_rank_call_us":[[14.+1.75*rows*8/200e9*1e6]*3 for _ in range(16)]})
        kernels.append({**kernels[-1],"call":records[-1]["call"]})
    for name,values in (("measurements",records),("kernels",kernels),("local_work",local)):
        (root/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in values))
    if network_defect:
        from collective_sim_core import runner
        payload=dict(expected_flows=8,finished_flows=8,makespan_us=1.)
        if network_defect == "partial": payload["finished_flows"]=7
        else: payload["makespan_us"]={"zero":0.,"nan":float("nan"),"inf":float("inf")}[network_defect]
        monkeypatch.setattr(runner,"run_htsim_runner",lambda **kw:payload)
        with pytest.raises(ValueError,match="completed"):
            EXPORT.export_finite_parameters(root.parent,tmp_path/"candidate")
        assert not (tmp_path/"candidate/collective_local.json").exists()
        return
    artifact=EXPORT.export_finite_parameters(root.parent,tmp_path/"candidate")
    assert artifact["evidence_kind"] == "cpu_fixture" and artifact["candidate_only"]
    assert artifact["qualification"]["status"] == "UNVERIFIED"
    fit=next(m["parameters"]["protocol"] for m in artifact["models"] if m["phase"] == "dispatch")
    assert fit["design_matrix_rank"] == 2 and fit["qualification"]["status"] == "UNVERIFIED"
    error=fit["qualification"]["observations"][0]
    assert error["source"]["network_ms"] > 0 and error["full_plan_native_ms"] == pytest.approx(1.064)
    assert len(artifact["counts"]) == 8 and len(artifact["models"]) == 2
    tp=next(m["parameters"]["tp_nvlink"] for m in artifact["models"] if m["phase"] == "shared_tp")
    assert tp["fit"]["design_matrix_rank"] == 2 and tp["nvlink_efficiency"] > 0
    assert all(c["evidence_kind"] == "cpu_fixture" for c in artifact["counts"].values())
    config["frozen_model_config_sha256"]="changed";(root/"plan.json").write_text(json.dumps(source))
    with pytest.raises(ValueError,match="receipt"):
        EXPORT.export_finite_parameters(root.parent,tmp_path/"rejected")


def test_critical_rank_batch_average_repeats_are_ms_not_native_calls():
    row = observe(fixture_record())
    assert row["time_unit"] == "ms"
    assert row["median_ms"] == pytest.approx(1.0455)
    assert row["critical_path_repeats_ms"] == pytest.approx([(1031 + j) / 1000 for j in range(30)])
    assert row["min_ms"] == pytest.approx(1.031)
    assert row["max_ms"] == pytest.approx(1.060)
    assert row["sample_cv"] > 0
    assert row["repeat_count"] == 30
    assert row["calls_per_repeat"] == 5
    assert "not independent native" in row["repeat_semantics"]
    assert row["native_forward_timing_observed"] is False
    assert not any(key.endswith("_us") for key in row)


@pytest.mark.parametrize("op", ["all_gatherv", "reduce_scatterv", "all_reduce"])
def test_uneven_vectors_are_preserved_and_not_scalarized(op):
    row = observe(fixture_record(op, tokens=(8, 1, 4, 2)))
    assert row["scalar_api_representable"] is False
    assert row["data_size_bytes"] is None
    assert row["api_operation"] is not None
    assert row["tokens_per_dp_rank"] == [8, 1, 4, 2]
    assert row["group_payload_vectors"]
    if op == "reduce_scatterv":
        assert set(row["per_rank_input_bytes"].values()) == {15 * 8192}
        assert row["per_rank_output_bytes"][8] == 8192


def test_graph_idle_is_not_scalar_even_with_equal_physical_payloads():
    row = observe(fixture_record(graph=True, useful=(8, 0, 8, 8)))
    assert len(set(row["per_rank_input_bytes"].values())) == 1
    assert row["scalar_api_representable"] is False
    assert row["useful_tokens_per_dp_rank"] == [8, 0, 8, 8]


def test_host_gloo_has_no_production_api():
    row = observe(fixture_record("cpu_all_reduce"))
    assert row["backend"] == "host_gloo"
    assert row["queue_gate"] == "NOT_APPLICABLE"
    assert row["api_operation"] is None
    assert row["comm_domain"] is None
    assert row["data_size_bytes"] is None
    assert row["scalar_api_representable"] is False
    assert row["observation_quality"] == "NOT_APPLICABLE"


def test_three_passes_merge_identical_ar_labels_but_count_native_multiplicity_once():
    observations = [observe(fixture_record("all_reduce", call=label, pass_id=pass_id))
                    for pass_id in range(3) for label in AR_COUNTS]
    curve, = EXPORT.aggregate_curves(observations)
    assert curve["calls_per_forward"] == 273  # Not 819 from adding three passes.
    assert curve["calls_per_forward_by_label"] == AR_COUNTS
    assert curve["call_labels"] == sorted(AR_COUNTS)
    assert curve["observation_count"] == 15
    assert curve["repeat_count"] == 450  # Not 2250 independent native calls.
    assert curve["pass_count"] == 3
    assert curve["passes_complete"]
    assert curve["per_label_pass_counts"] == dict.fromkeys(AR_COUNTS, 3)
    assert [item["pass"] for item in curve["per_pass"]] == [0, 1, 2]
    assert all(len(item["labels"]) == 5 and item["repeat_count"] == 150 for item in curve["per_pass"])
    assert all("sample_cv" in item and "range_over_median" in item for item in curve["per_pass"])
    assert curve["outliers_removed"] == 0


def test_union_of_pass_ids_does_not_hide_a_label_missing_one_pass():
    observations = [observe(fixture_record("all_reduce", call=label, pass_id=pass_id))
                    for pass_id in range(3) for label in AR_COUNTS
                    if not (pass_id == 2 and label == "embedding_all_reduce")]
    curve, = EXPORT.aggregate_curves(observations)
    assert curve["pass_count"] == 3
    assert curve["passes_complete"] is False
    assert curve["per_label_pass_counts"]["embedding_all_reduce"] == 2


@pytest.mark.parametrize("dimension", ["scenario", "cuda_graph", "backend", "kernels", "dtypes", "columns",
                                       "active_dp_rails", "per_rank_input_bytes", "useful_tokens_per_dp_rank"])
def test_ar_merging_never_crosses_physical_or_execution_context(dimension):
    first = observe(fixture_record("all_reduce", call="embedding_all_reduce"))
    other = observe(fixture_record("all_reduce", call="attention_all_reduce"))
    changes = {"scenario": "other_scenario", "cuda_graph": True, "backend": "symmetric_memory",
               "kernels": ["another_kernel"], "dtypes": ["float16"], "columns": [2048],
               "active_dp_rails": [0], "per_rank_input_bytes": {0: 1},
               "useful_tokens_per_dp_rank": [8, 0, 8, 8]}
    other[dimension] = changes[dimension]
    assert len(EXPORT.aggregate_curves([first, other])) == 2


def test_one_rail_and_all_rails_are_separate_cells():
    all_rails = observe(fixture_record())
    one_rail = observe(fixture_record(rails=(0,)))
    assert one_rail["active_dp_rails"] == [0]
    assert one_rail["num_devices"] == 4
    assert len(EXPORT.aggregate_curves([all_rails, one_rail])) == 2


def test_different_non_ar_labels_do_not_merge():
    first = observe(fixture_record())
    other = deepcopy(first)
    other["call"] = "different_dispatch_context"
    assert len(EXPORT.aggregate_curves([first, other])) == 2


def test_queue_failures_are_marked_and_retained_not_admitted_or_outlier_deleted():
    record = fixture_record()
    for rank in record["participating_ranks"]:
        record["per_rank_call_us"][rank] = [1000] * 15 + [2000] * 15
        record["per_rank_queue_held"][rank] = [False] * 15 + [True] * 15
    failed = observe(record)
    assert failed["queue_gate"] == failed["observation_quality"] == "FAIL"
    assert failed["filtered_median_ms"] == 2
    assert failed["median_ms"] == 1.5
    assert failed["filtered_median_shift_ms"] == 0.5
    assert failed["absolute_fractional_median_delta"] == pytest.approx(1 / 3)
    assert len(failed["critical_path_repeats_ms"]) == 30
    curve, = EXPORT.aggregate_curves([failed, observe(fixture_record(pass_id=1))])
    assert curve["quality_fail_count"] == 1
    assert curve["pooled_repeats_include_failed_observations"]
    assert curve["observation_count"] == 2 and curve["repeat_count"] == 60
    assert curve["passes_complete"] is False
    for row in (failed, curve):
        assert row["candidate_only"] is True
        assert row["cost_data_qualified"] is False
        assert row["training_selected"] is False
        assert row["production_consumable"] is False


def test_missing_queue_flags_are_unverified_not_passed():
    record = fixture_record()
    record["per_rank_queue_held"][0] = None
    row = observe(record)
    curve, = EXPORT.aggregate_curves([row])
    assert curve["quality_unverified_count"] == 1
    assert curve["observation_quality"] == "UNVERIFIED"
    assert row["filtered_median_shift_ms"] is None


def test_source_summary_mismatch_is_a_quality_failure_not_replaced():
    record = fixture_record()
    summary, kernel = source_rows(record)
    summary["median_us"] = 999
    row = EXPORT.observation(record, summary, kernel, CONFIG, 0.01)
    assert row["median_ms"] == pytest.approx(1.0455)
    assert row["observation_quality"] == "FAIL"
    assert "source_median_matches" in row["quality_fail_reasons"]


@pytest.mark.parametrize("compressed", [False, True])
def test_local_cli_exports_all_passes_with_streamed_gzip_and_manifest(tmp_path, monkeypatch, compressed, capsys):
    directory = tmp_path / "copied_run"
    collectives = directory / "collectives"
    collectives.mkdir(parents=True)
    records = [fixture_record(pass_id=pass_id) for pass_id in range(3)]
    for rank in records[1]["participating_ranks"]:
        records[1]["per_rank_call_us"][rank] = [1000] * 15 + [2000] * 15
        records[1]["per_rank_queue_held"][rank] = [False] * 15 + [True] * 15
    path = collectives / ("measurements.jsonl.gz" if compressed else "measurements.jsonl")
    opener = gzip.open if compressed else open
    with opener(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (collectives / "plan.json").write_text(json.dumps({"config": CONFIG}))
    (collectives / "kernels.jsonl").write_text(json.dumps(source_rows(records[0])[1]) + "\n")
    summaries = [source_rows(record)[0] for record in records]
    EXPORT.write_csv(collectives / "summary.csv", summaries)
    (directory / "manifest.json").write_text(json.dumps({"job": {"image": "locked@sha256:fixture",
                                                                  "run_id": "fixture", "source_hashes": {}}}))
    source_hash = EXPORT.sha256(path)
    # No full read_bytes of compressed measurements or other source artifacts.
    monkeypatch.setattr(Path, "read_bytes", lambda path: pytest.fail(f"Non-streaming read: {path}"))
    output = tmp_path / "costs"
    assert EXPORT.main([str(directory), "--output-dir", str(output), "--environment-note",
                        "Co-tenancy is parent reported; causal effect unknown."]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"]["observations"] == 3
    assert manifest["counts"]["curves"] == 1
    assert manifest["counts"]["observation_quality"]["FAIL"] == 1
    assert manifest["input_sha256"][str(path.relative_to(directory))] == source_hash
    assert EXPORT.sha256(path) == source_hash
    assert manifest["cost_data_qualified"] is False
    assert "causal effect unknown" in manifest["environment_note"]
    assert "100000" in " ".join(manifest["limits"])
    with (output / "observations.csv").open() as handle:
        observations = list(csv.DictReader(handle))
    assert [int(row["pass"]) for row in observations] == [0, 1, 2]
    assert observations[1]["observation_quality"] == "FAIL"
    assert all(len(json.loads(row["critical_path_repeats_ms"])) == 30 for row in observations)
    with (output / "curves.csv").open() as handle:
        curve, = list(csv.DictReader(handle))
    assert len(json.loads(curve["per_pass"])) == 3
    assert int(curve["num_devices"]) == 4
    assert json.loads(capsys.readouterr().out)["advisory_only"] is True


def test_output_cannot_replace_source_worker_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"job": {"run_id": "source"}}\n')
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="source worker manifest"):
        EXPORT.export_costs(tmp_path, tmp_path)
    assert manifest.read_bytes() == before
