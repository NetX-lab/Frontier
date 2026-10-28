"""Synthetic native traces and G5h directories exercise the CPU-only contract."""
from collections import Counter
import gzip
import json
from pathlib import Path

import pytest

import v4_collective_kernels as kernels
import v4_runtime_census as census
from frontier.profiling.collectives.vllm_dsv4_forward_collectives import DeepseekV4Shape, dsv4_plan
from frontier.profiling.collectives.vllm_forward_collectives import ServingParallelism, group_members

MODEL = {"hidden_size": 16, "num_hidden_layers": 3, "num_experts_per_tok": 2,
         "vocab_size": 128, "hc_mult": 4, "n_routed_experts": 8}
ENGINE = {"tensor_parallel_size": 2, "data_parallel_size": 2, "pipeline_parallel_size": 1,
          "enable_expert_parallel": True, "dtype": "bfloat16", "async_scheduling": True}
GRID = {"graph_tokens_per_lane": [1, 2, 4], "eager_lane_tokens": [1, 8], "logits_rows_per_lane": [1, 2]}
SHAPE = (1, 1, 1)
BLOCK = (32, 1, 1)


def device_event(name, correlation=1, grid=SHAPE, duration=5, category="kernel", timestamp=100):
    return {"ph": "X", "name": name, "cat": category, "ts": timestamp, "dur": duration,
            "pid": 10, "tid": 7, "args": {"correlation": correlation, "stream": 7,
                                          "grid": list(grid), "block": list(BLOCK)}}


def write_native(root, role="decode", lane=0, starts=(100, 1100), mode="FULL", rows=1,
                 real=False, scheduled=None, post=False, prepare=False, unknown=False):
    events = []
    for index, start in enumerate(starts):
        def annotation(name, offset, duration, tid=1):
            events.append({"ph": "X", "name": name, "cat": "user_annotation", "pid": 10,
                           "tid": tid, "ts": start + offset, "dur": duration, "args": {}})
        annotation(f"frontier_native_metadata|{index}|attention_metadata", -20, 10)
        annotation({"FULL": "model_dispatch_run_fullgraph", "PIECEWISE": "model_dispatch_run_pw_graph",
                    "NONE": "model_forward"}[mode], 0, 100)
        annotation("gloo:all_reduce", -40, 10, tid=99)
        if real:
            annotation(f"frontier_native_ownership|{index}|logits", 110, 20)
            tokens = rows if scheduled is None else scheduled
            annotation(f"execute_context_0(0)_generation_1({tokens})", -60, 220)
        work = [("dispatch", kernels.ROW_COUNT_KERNEL, (rows, 2, 1)),
                ("dispatch", "ncclDevKernel_AllGather_RING_LL", SHAPE)]
        if post:
            work.append(("post", "ncclDevKernel_AllGather_RING_LL", (7, 1, 1)))
        if prepare:
            work.append(("prepare", "ncclDevKernel_AllGather_RING_LL", SHAPE))
        if unknown:
            work.append(("dispatch", "unclassified_all_reduce", SHAPE))
        for ordinal, (segment, name, grid) in enumerate(work):
            correlation = index * 100 + ordinal
            timestamp = start + {"prepare": -25, "dispatch": 10, "post": 110}[segment] + ordinal
            events.append({"ph": "X", "name": "cudaGraphLaunch" if mode == "FULL" else "cudaLaunchKernel",
                           "cat": "cuda_runtime", "ts": timestamp, "dur": 1, "pid": 10,
                           "tid": 1, "args": {"correlation": correlation}})
            events.append(device_event(name, correlation, grid, timestamp=timestamp + 200))
            events.append(device_event("Memcpy DtoD", correlation, category="gpu_memcpy", timestamp=timestamp + 205))
    path = root / role / f"dp{lane}" / f"dp{lane}_pp0_tp0_synthetic.pt.trace.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)
    return path


def write_benchmark(root, dp=2, grid=None, duration=5, repeats=2, settings=False, shared=False):
    root.mkdir(parents=True, exist_ok=True)
    engine = {**ENGINE, "data_parallel_size": dp}
    parallel = ServingParallelism.from_engine(engine)
    model_path = root / "model.json"
    model_path.write_text(json.dumps(MODEL))
    config = {"stage": f"synthetic_dp{dp}", "engine": engine, "grid": GRID if grid is None else grid,
              "model_config": "model.json"}
    config.update({"settings": {"calls_per_repeat": repeats}} if settings else {"calls_per_repeat": repeats})
    (root / "plan.json").write_text(json.dumps({"config": config, "runtime": {
        "group_ranks": {group: group_members(parallel, group, 0) for group in ("tp", "ep", "dp")}}}))
    plan = dsv4_plan(DeepseekV4Shape.from_hf_config(MODEL), parallel, config["grid"])
    records = []
    for scenario, call in plan:
        per_rank = []
        for rank in group_members(parallel, "ep", 0):
            name = "ncclDevKernel_shared" if shared else f"ncclDevKernel_{call.name}_{scenario.name}_rank{rank}"
            events = [device_event(name, duration=duration) for _ in range(repeats)]
            events += [device_event("pad_copy_elementwise_kernel", duration=1) for _ in range(repeats)]
            events += [device_event("Memcpy DtoD", category="gpu_memcpy", duration=2) for _ in range(repeats)]
            path = root / "traces" / scenario.name / call.name / f"rank{rank}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"traceEvents": events}))
            per_rank.append({"rank": rank, "trace": f"/worker/not/on/host/{path.name}", "kernels": [name]})
        records.append({"scenario": scenario.name, "call": call.name, "per_rank": per_rank})
    (root / "kernels.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    return root


def make_group(benchmarks, mode="FULL", rows=1, vector=(1, 1), real=False, requests=None, lanes=(0,)):
    group = {"id": 0, "key": {"role": "decode", "mode": mode, "real": real, "own_rows": rows,
                              "lane_vector": list(vector), "requests": requests},
             "forward_count": len(lanes), "distinct_multisets": len(lanes), "multisets": [], "kernel_stats": []}
    forwards, pooled = [], {}
    for lane in lanes:
        expected = kernels._expected(group, lane, benchmarks, ENGINE["tensor_parallel_size"])
        assert not expected["missing"] and not expected["errors"]
        rows_list = []
        for row in expected["kernels"]:
            identity = kernels.row_key(row)
            # These fixtures use distinct names per call; the logits name can
            # therefore identify the post segment without testing the parser.
            segment = "post" if "logits_all_gather" in row["name"] else "dispatch"
            rows_list.append({**row, "segment": segment, "durations_us": [2, 5, 10]})
            pooled.setdefault((segment, *identity), []).extend([2, 5, 10])
        reference = {"lane": lane, "index": 0, "trace_path": f"native_lane{lane}"}
        group["multisets"].append({"forward_count": 1, "kernels": rows_list, "forwards": [reference]})
        forwards.append({**reference, "kernels": rows_list})
    group["kernel_stats"] = [kernels.key_row(key, duration_us=kernels.duration_stats(values, native=True))
                             for key, values in pooled.items()]
    return group, forwards


def make_native(group, forwards):
    return {"trace_dir": "synthetic_native", "tensor_parallel_size": ENGINE["tensor_parallel_size"],
            "duration_note": "Native waits are upper bounds.", "review_list": [], "groups": [group],
            "lanes": [{"role": "decode", "forwards": forwards}]}


@pytest.mark.parametrize("prefix,family", list(kernels.COLLECTIVE_PREFIXES.items()))
def test_classification_requires_a_seen_prefix(prefix, family):
    assert kernels.collective_family(prefix + "kernel_template") == family
    assert kernels.collective_family("another_" + prefix) is None


def test_review_names_do_not_expand_classification():
    names = ["one_shot_new", "All_Reduce", "Two_Shot_new", "ncclDevKernel_seen", "normal_matmul"]
    events = [device_event(name) for name in names] + [device_event("one_shot_new")]
    assert kernels.review_names(events) == Counter({"one_shot_new": 2, "All_Reduce": 1, "Two_Shot_new": 1})


@pytest.mark.parametrize("mode,rows", [(mode, rows) for mode in ("FULL", "PIECEWISE", "NONE") for rows in (1, 2, 8)])
def test_native_rows_segmentation_and_census_multiset(tmp_path, mode, rows):
    path = write_native(tmp_path, mode=mode, rows=rows, real=True, post=True, unknown=True)
    public = list(census.forward_kernel_events("decode", path))
    old_style = census.dispatch_kernel_multisets("decode", path)
    assert len(public) == len(old_style) == 2
    assert old_style[0].kernels[(kernels.ROW_COUNT_KERNEL, (rows, 2, 1), BLOCK)] == 1
    assert len(public[0].kernels_by_segment["dispatch"]) == 3
    assert len(public[0].kernels_by_segment["post"]) == 1
    table = kernels.native_table(tmp_path)
    forward = table["lanes"][0]["forwards"][0]
    assert forward["own_rows"] == forward["scheduled_tokens"] == forward["row_check"]["expected_rows"] == rows
    assert forward["requests"] == 1 and forward["step_class"] == "decode"
    assert {row["segment"] for row in forward["kernels"]} == {"dispatch", "post"}
    assert table["review_list"] == [{"name": "unclassified_all_reduce", "count": 2}]


@pytest.mark.parametrize("mode,scheduled", [("NONE", 2), ("FULL", 3), ("PIECEWISE", 3)])
def test_real_rows_below_scheduled_or_wrong_eager_count_are_errors(tmp_path, mode, scheduled):
    write_native(tmp_path, rows=1, real=True, scheduled=scheduled, mode=mode)
    with pytest.raises(ValueError, match="own rows.*scheduled tokens"):
        kernels.native_table(tmp_path)


def test_graph_rows_check_observed_padding_bucket(tmp_path):
    write_native(tmp_path, rows=4, real=True, scheduled=3)
    table = kernels.native_table(tmp_path)
    assert table["lanes"][0]["forwards"][0]["row_check"]["padding_rows"] == 1


def test_prepare_collective_requires_review(tmp_path):
    write_native(tmp_path, prepare=True)
    with pytest.raises(ValueError, match="collective in prepare.*requires lead review"):
        kernels.native_table(tmp_path)


def test_pairing_native_windows_nearest_mode_and_left_out(tmp_path):
    write_native(tmp_path, lane=0, starts=(100, 1100, 2100, 3100))
    write_native(tmp_path, lane=1, starts=(120, 1120, 2140))
    table = kernels.native_table(tmp_path)
    pairing, = table["pairing"]
    assert (pairing["accepted_pairs"], pairing["accepted_tuples"], pairing["left_out_per_lane"]) == (3, 3, {"0": 1, "1": 0})
    assert pairing["start_difference_us"] == {"median": 20, "max": 40}
    assert sum(group["forward_count"] for group in table["groups"]) == 6
    assert table["lanes"][0]["forwards"][-1]["lane_vector"] is None


def test_pairing_beyond_half_smaller_neighbor_gap_rejected(tmp_path):
    write_native(tmp_path, lane=0, starts=(100, 200, 1200))
    write_native(tmp_path, lane=1, starts=(160, 800, 1800))
    pairing = kernels.native_table(tmp_path)["pairing"][0]
    assert pairing["accepted_pairs"] == 2
    assert pairing["rejected_anchors"][0]["radius_us"] == 50
    assert pairing["rejected_anchors"][0]["difference_us"] == 60


def test_pairing_all_lanes_required(tmp_path):
    for lane, starts in enumerate(((100, 1100, 2100), (110, 1110), (120, 1120, 2120))):
        write_native(tmp_path, lane=lane, starts=starts)
    pairing = kernels.native_table(tmp_path)["pairing"][0]
    assert pairing["accepted_tuples"] == 2 and pairing["accepted_pairs"] == 4
    assert pairing["left_out_per_lane"] == {"0": 1, "1": 0, "2": 1}


def test_pairing_tie_requires_lead_review(tmp_path):
    write_native(tmp_path, lane=0, starts=(1000, 2000))
    write_native(tmp_path, lane=1, starts=(900, 1100, 2100))
    with pytest.raises(ValueError, match="ambiguous nearest pairing"):
        kernels.native_table(tmp_path)


@pytest.mark.parametrize("repeats", [1, 2, 5, 10])
def test_benchmark_repeat_division_and_auxiliary_work(tmp_path, repeats):
    path = tmp_path / "rank0.json"
    events = [device_event("ncclDevKernel_test", duration=index + 1) for index in range(repeats * 2)]
    events += [device_event("copy_pad_elementwise_kernel"), device_event("Memcpy DtoD", category="gpu_memcpy"),
               device_event("Memset", category="gpu_memset")]
    path.write_text(json.dumps({"traceEvents": events}))
    case = kernels.benchmark_case(path, repeats)
    assert case["errors"] == [] and case["kernels"][0]["count"] == 2
    assert case["kernels"][0]["duration_us"] == {"min": 1, "median": (repeats * 2 + 1) / 2, "max": repeats * 2}
    assert {row["category"] for row in case["auxiliary"]} == {"kernel", "gpu_memcpy", "gpu_memset"}


def test_uneven_repeat_division_reports_case_error_without_rounding(tmp_path):
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"traceEvents": [device_event("ncclDevKernel_test")] * 3}))
    case = kernels.benchmark_case(path, 2)
    assert "3 occurrences" in case["errors"][0] and "not divisible" in case["errors"][0]
    assert case["kernels"] == []


@pytest.mark.parametrize("mode", ["FULL", "PIECEWISE"])
def test_expected_graph_calls_and_equal_native_variants_for_each_rank(tmp_path, mode):
    directory = write_benchmark(tmp_path / "benchmark")
    benchmark = kernels.Benchmark.read(directory)
    group, forwards = make_group([benchmark], mode=mode, lanes=(0, 1))
    comparison = kernels.compare_tables(make_native(group, forwards), [directory])
    result, = comparison["groups"]
    assert result["kernels"] == "equal" and result["status"] == "matched"
    for expected in result["expected_by_lane"]:
        assert {call["scenario"] for call in expected["selected_calls"]} == {"graph_t1"}
        assert {call["rank"] for call in expected["selected_calls"]} == {expected["lane"] * 2}
        assert sorted(row["count"] for row in expected["kernels"]) == [1, 1, 3, 3, 3, 3]
    assert all(row["verdict"] == "within" for row in result["durations"])
    assert comparison["cases"][0]["auxiliary"]


def test_expected_mixed_eager_uses_uniform_tp_and_mixed_ep(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark")
    group, forwards = make_group([kernels.Benchmark.read(directory)], mode="NONE", rows=8, vector=(8, 1))
    result = kernels.compare_tables(make_native(group, forwards), [directory])["groups"][0]
    assert result["kernels"] == "equal"
    assert {call["scenario"] for call in result["expected_by_lane"][0]["selected_calls"]
            if call["call"].startswith("moe_")} == {"eager_t8_1"}
    assert {call["scenario"] for call in result["expected_by_lane"][0]["selected_calls"]
            if not call["call"].startswith("moe_")} == {"eager_t8_8"}


def test_real_forward_adds_logits_in_post_and_segment_is_ignored_for_equality(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark")
    group, forwards = make_group([kernels.Benchmark.read(directory)], real=True, requests=2)
    result = kernels.compare_tables(make_native(group, forwards), [directory])["groups"][0]
    assert result["kernels"] == "equal"
    call = result["expected_by_lane"][0]["selected_calls"][-1]
    assert (call["scenario"], call["call"], call["segment"]) == ("logits_b2", "logits_all_gather", "post")
    assert any(row["segment"] == "post" and "logits_all_gather" in row["name"]
               for row in group["multisets"][0]["kernels"])
    for row in group["multisets"][0]["kernels"]:
        row["segment"] = "post"
    assert kernels.compare_tables(make_native(group, forwards), [directory])["groups"][0]["kernels"] == "equal"


def test_multi_directory_matching_dp_ep_and_tp_reuse(tmp_path):
    tp = write_benchmark(tmp_path / "dp2")
    ep_grid = {**GRID, "calls": ["moe_dispatch_all_gatherv", "moe_combine_reduce_scatterv"]}
    ep = write_benchmark(tmp_path / "dp4", dp=4, grid=ep_grid)
    benchmarks = [kernels.Benchmark.read(path) for path in (tp, ep)]
    group, forwards = make_group(benchmarks, mode="NONE", rows=8, vector=(8, 1, 8, 1), lanes=(0, 2))
    result = kernels.compare_tables(make_native(group, forwards), [tp, ep])["groups"][0]
    assert result["kernels"] == "equal"
    for expected in result["expected_by_lane"]:
        for call in expected["selected_calls"]:
            assert call["benchmark"] == str(ep if call["call"].startswith("moe_") else tp)
    unmatched = kernels.compare_tables(make_native(group, forwards), [tp])["groups"][0]
    assert unmatched["status"] == "unmatched"
    assert {row["call"] for row in unmatched["missing"]} == {"moe_dispatch_all_gatherv", "moe_combine_reduce_scatterv"}


def test_equal_different_and_unmatched_are_independent_of_duration(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark")
    group, forwards = make_group([kernels.Benchmark.read(directory)])
    group["multisets"][0]["kernels"][0]["count"] += 1
    result = kernels.compare_tables(make_native(group, forwards), [directory])["groups"][0]
    assert result["kernels"] == "different" and result["differences"][0]["kernels"][0]["native_minus_expected"] == 1
    assert all(row["verdict"] == "within" for row in result["durations"])
    group["key"]["own_rows"] = 999
    result = kernels.compare_tables(make_native(group, forwards), [directory])["groups"][0]
    assert result["status"] == "unmatched" and result["kernels"] is None
    assert {row["scenario"] for row in result["missing"]} == {"graph_t999"}


@pytest.mark.parametrize("duration,verdict", [(1, "below"), (2, "within"), (5, "within"), (10, "within"), (11, "above")])
def test_duration_verdict_bounds_are_inclusive(tmp_path, duration, verdict):
    directory = write_benchmark(tmp_path / "benchmark", duration=duration)
    group, forwards = make_group([kernels.Benchmark.read(directory)])
    result = kernels.compare_tables(make_native(group, forwards), [directory])["groups"][0]
    assert result["kernels"] == "equal"
    assert {row["verdict"] for row in result["durations"]} == {verdict}


def test_case_error_propagates_without_kernel_verdict(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark")
    group, forwards = make_group([kernels.Benchmark.read(directory)])
    path = directory / "traces/graph_t1/embedding_all_reduce/rank0.json"
    document = json.loads(path.read_text())
    document["traceEvents"].append(device_event("ncclDevKernel_uneven"))
    path.write_text(json.dumps(document))
    result = kernels.compare_tables(make_native(group, forwards), [directory])["groups"][0]
    assert result["status"] == "error" and result["kernels"] is None
    assert "not divisible" in result["errors"][0]


def test_recorded_layout_is_checked_against_runner_helper(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark")
    path = directory / "plan.json"
    document = json.loads(path.read_text())
    document["runtime"]["group_ranks"]["dp"] = [0, 1]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="differs from runner layout"):
        kernels.Benchmark.read(directory)


def test_both_settings_layouts_and_local_trace_resolution(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark", settings=True)
    benchmark = kernels.Benchmark.read(directory)
    rank, case = benchmark.case("graph_t1", "embedding_all_reduce", 1)
    assert rank == 2 and case["errors"] == []
    assert Path(case["trace_path"]).is_relative_to(directory)
    assert benchmark.calls_per_repeat == 2


def test_pooled_dispatch_and_post_durations_for_same_identity(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark", shared=True)
    group, forwards = make_group([kernels.Benchmark.read(directory)], real=True, requests=1)
    identity = kernels.row_key(group["multisets"][0]["kernels"][0])
    dispatch_count = group["multisets"][0]["kernels"][0]["count"] - 1
    group["multisets"][0]["kernels"] = [kernels.key_row((segment, *identity), count=count)
                                         for segment, count in (("dispatch", dispatch_count), ("post", 1))]
    forwards[0]["kernels"] = [kernels.key_row((segment, *identity), count=count, durations_us=values)
                              for segment, count, values in (("dispatch", dispatch_count, [2, 5]), ("post", 1, [10, 20]))]
    group["kernel_stats"] = [kernels.key_row((segment, *identity), duration_us=kernels.duration_stats(values, native=True))
                              for segment, values in (("dispatch", [2, 5]), ("post", [10, 20]))]
    result = kernels.compare_tables(make_native(group, forwards), [directory])["groups"][0]
    assert result["kernels"] == "equal"
    assert result["durations"][0]["native_us"]["median"] == 7.5
    assert result["durations"][0]["native_us"]["p90"] == 20


def test_cli_writes_reports_and_verdict_lines(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark")
    group, forwards = make_group([kernels.Benchmark.read(directory)])
    native = tmp_path / "native.json"
    native.write_text(json.dumps(make_native(group, forwards)))
    output, summary = tmp_path / "comparison.json", tmp_path / "comparison.txt"
    assert kernels.main(["compare", "--native", str(native), "--benchmark", str(directory),
                         "--output", str(output), "--summary", str(summary)]) == 0
    assert json.loads(output.read_text())["verdict_counts"] == {"equal": 1}
    assert "VERDICT group 0: kernels=equal" in summary.read_text()
    assert "Auxiliary device work" in summary.read_text() and "Native review list: empty" in summary.read_text()


@pytest.mark.parametrize("field,value", [("grid", [1, 1]), ("block", [32.0, 1, 1])])
def test_collective_launch_shapes_must_be_three_integers(tmp_path, field, value):
    event = device_event("ncclDevKernel_test")
    event["args"][field] = value
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"traceEvents": [event]}))
    with pytest.raises(ValueError, match=f"requires {field} with three ints"):
        kernels.benchmark_case(path, 1)


def test_missing_row_count_kernel_reports_trace_and_forward(tmp_path):
    path = write_native(tmp_path)
    with gzip.open(path, "rt") as handle:
        document = json.load(handle)
    document["traceEvents"] = [event for event in document["traceEvents"]
                               if event["name"] != kernels.ROW_COUNT_KERNEL]
    with gzip.open(path, "wt") as handle:
        json.dump(document, handle)
    with pytest.raises(ValueError, match="forward 0: missing row-count kernel") as error:
        kernels.native_table(tmp_path)
    assert str(path) in str(error.value)


def test_all_rank_auxiliary_work_and_review_names_are_retained(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark")
    group, forwards = make_group([kernels.Benchmark.read(directory)])
    path = directory / "traces/graph_t1/embedding_all_reduce/rank1.json"
    document = json.loads(path.read_text())
    document["traceEvents"] += [device_event("unknown_two_shot")] * 2
    path.write_text(json.dumps(document))
    comparison = kernels.compare_tables(make_native(group, forwards), [directory])
    cases = [case for case in comparison["cases"]
             if case["scenario"] == "graph_t1" and case["call"] == "embedding_all_reduce"]
    assert {case["rank"] for case in cases} == {0, 1, 2, 3}
    assert comparison["benchmark_review_list"] == [{"name": "unknown_two_shot", "count": 2}]
    assert comparison["groups"][0]["kernels"] == "equal"


def test_each_distinct_native_multiset_must_match(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark")
    group, forwards = make_group([kernels.Benchmark.read(directory)], lanes=(0, 1))
    group["multisets"][1]["kernels"][0]["count"] += 1
    result = kernels.compare_tables(make_native(group, forwards), [directory])["groups"][0]
    assert result["kernels"] == "different"
    assert result["differences"][0]["lane"] == 1
    assert result["differences"][0]["native_multiset"] == 1


def test_far_equidistant_candidates_are_rejected_without_claiming_pair_ambiguity(tmp_path):
    write_native(tmp_path, lane=0, starts=(1000, 2000, 3000))
    write_native(tmp_path, lane=1, starts=(-1000, 7000))
    pairing = kernels.native_table(tmp_path)["pairing"][0]
    assert pairing["accepted_pairs"] == 0


def test_planned_call_with_missing_trace_is_an_error_not_an_unmatched_scenario(tmp_path):
    directory = write_benchmark(tmp_path / "benchmark")
    group, forwards = make_group([kernels.Benchmark.read(directory)])
    path = directory / "kernels.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        if (record["scenario"], record["call"]) == ("graph_t1", "embedding_all_reduce"):
            record["per_rank"][0]["trace"] = None
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = kernels.compare_tables(make_native(group, forwards), [directory])["groups"][0]
    assert result["status"] == "error" and result["missing"] == []
    assert "missing kernel evidence" in result["errors"][0]
