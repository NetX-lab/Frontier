"""CPU checks for the pure helpers of the vLLM collective, NIXL READ and EP + NIXL benchmarks."""

import types

import pytest

from frontier.profiling.collectives.ep_kv_read_overlap_benchmark import (
    ep_plan,
    expected_notifications,
    issue_load,
    load_runs,
    overlap_read,
    summarize_overlap,
)
from frontier.profiling.collectives.nixl_kv_read_benchmark import (
    KVRegionLayout,
    ReadCase,
    block_descriptors,
    descriptor_ids,
    read_cases,
    request_block_ids,
    summarize_case,
)
from frontier.profiling.collectives.vllm_collective_benchmark import (
    forward_totals,
    network_all_reduce_rows,
    pass_order,
    summarize,
)
from frontier.profiling.collectives.vllm_forward_collectives import ModelShape, ServingParallelism

KV_CACHE = {
    "regions": [{"name": "main_mla", "count": 3, "block_bytes": 1152},
                {"name": "indexer", "count": 2, "block_bytes": 132}],
    "block_size": 64, "num_gpu_blocks": 10, "resident_bytes_per_block": 3 * 1152 + 2 * 132,
}


def _record(call="c", op="all_reduce", group="tp", graph=False, timings=None, nbytes=1024):
    timings = timings or [[1.0, 2.0, 3.0], [4.0, 1.0, 1.0], None]
    return {"pass": 1, "scenario": "s", "call": call, "op": op, "group": group, "group_size": 8,
            "cuda_graph": graph, "calls_per_forward": 3, "tokens_per_dp_rank": [4, 1],
            "max_input_bytes": nbytes, "max_output_bytes": nbytes, "kernels": ["k"],
            "per_rank_call_us": timings,
            "per_rank_launch_call_us": [r and [10 * v for v in r] for r in timings],
            "per_rank_queue_held": [r and [True, True, False] for r in timings]}


def test_summary_takes_slowest_rank_per_repeat():
    row = summarize(_record())
    assert (row["median_us"], row["min_us"], row["max_us"]) == (3.0, 2.0, 4.0)
    assert row["launch_median_us"] == 30.0
    assert row["queue_held_fraction"] == pytest.approx(2 / 3)
    assert row["tokens_per_dp_rank"] == "4/1"
    cpu = {**_record(op="cpu_all_reduce"), "per_rank_launch_call_us": [None, None, None],
           "per_rank_queue_held": [None, None, None]}
    assert summarize(cpu)["launch_median_us"] is None
    assert summarize(cpu)["queue_held_fraction"] is None
    totals = forward_totals([row, summarize(_record(call="d"))])
    assert totals == [{"pass": 1, "scenario": "s", "cuda_graph": False,
                       "tokens_per_dp_rank": "4/1", "forward_comm_us": 18.0,
                       "c_us": 9.0, "d_us": 9.0}]
    other_pass = forward_totals([row, {**row, "pass": 2}])
    assert [t["pass"] for t in other_pass] == [1, 2]


def test_pass_order_is_a_seeded_permutation():
    items = list(range(20))
    first = pass_order(items, 7, 0)
    assert first == pass_order(items, 7, 0) and sorted(first) == items
    assert first != pass_order(items, 7, 1) and first != items
    assert pass_order(items, 6, 1) == pass_order(items, 7, 0)


def test_network_rows_keep_eager_single_node_tp_all_reduce():
    rows = [summarize(_record(nbytes=4096)),
            summarize(_record(nbytes=2048, timings=[[1.0, 1.0, 1.0]])),
            summarize(_record(graph=True)),
            summarize(_record(op="all_gather")),
            summarize(_record(group="dp"))]
    small, large = network_all_reduce_rows(rows, gpus_per_node=8)
    assert small["size"] == 2048 and small["time_stats.all_reduce.median"] == 1e-3
    assert large["size"] == 4096 and large["time_stats.all_reduce.median"] == 3e-3
    assert small["num_workers"] == small["devices_per_node"] == 8
    assert network_all_reduce_rows(rows, gpus_per_node=4) == []


def test_layout_checks_the_cache_contract():
    layout = KVRegionLayout.from_config(KV_CACHE)
    assert layout.num_regions == 5 and layout.block_bytes == 3720
    with pytest.raises(ValueError, match="cache contract"):
        KVRegionLayout.from_config({**KV_CACHE, "resident_bytes_per_block": 1})


def test_descriptors_are_region_major():
    layout = KVRegionLayout.from_config(KV_CACHE)
    assert descriptor_ids(5, 10, [2, 3]) == [2, 3, 12, 13, 22, 23, 32, 33, 42, 43]
    descs = block_descriptors([0, 100_000, 200_000, 300_000, 400_000], layout, 3)
    assert len(descs) == 50
    assert descs[12] == (100_000 + 2 * 1152, 1152, 3)
    assert descs[41] == (400_000 + 132, 132, 3)


def test_cases_cover_grid_and_respect_capacity():
    layout = KVRegionLayout.from_config(KV_CACHE)
    cases = read_cases([{"prompt_tokens": [65, 128], "active_pairs": [1, 8],
                         "requests_in_flight": [1, 2]},
                        {"prompt_tokens": [320], "active_pairs": [8],
                         "requests_in_flight": [2]}], layout, 8)
    assert len(cases) == 9 and cases[0].blocks_per_request == 2
    assert request_block_ids(cases[1]) == [[0, 1], [2, 3]]
    assert cases[-1].name == "t320_pairs8_inflight2" and cases[-1].blocks_per_request == 5
    with pytest.raises(ValueError, match="exceed"):
        read_cases([{"prompt_tokens": [640], "active_pairs": [1],
                     "requests_in_flight": [2]}], layout, 8)
    with pytest.raises(ValueError, match="active_pairs"):
        read_cases([{"prompt_tokens": [64], "active_pairs": [9],
                     "requests_in_flight": [1]}], layout, 8)
    with pytest.raises(ValueError, match="repeat a case"):
        read_cases([{"prompt_tokens": [64], "active_pairs": [1], "requests_in_flight": [1]}] * 2,
                   layout, 8)


def test_case_summary_uses_slowest_reader():
    case = ReadCase("c", 128, 2, 2, 1)
    record = {"pass": 2, "case": case.name, "prompt_tokens": 128, "blocks_per_request": 2,
              "active_pairs": 2, "requests_in_flight": 1, "descriptors_per_request": 10,
              "bytes_per_reader": 1000,
              "readers": [{"timings": [{"complete_us": 10.0, "prep_us": 1.0}] * 3,
                           "data_verified": True},
                          {"timings": [{"complete_us": 20.0, "prep_us": 2.0}] * 3,
                           "data_verified": True}],
              "notifications_received": 6, "notifications_expected": 6}
    row = summarize_case(record)
    assert row["complete_median_us"] == 20.0 and row["prep_median_us"] == 2.0
    assert row["per_reader_gigabytes_per_s"] == pytest.approx(0.05)
    assert row["aggregate_gigabytes_per_s"] == pytest.approx(0.1)
    assert row["data_verified"] and row["pass"] == 2


def test_overlap_plan_holds_only_ep_calls():
    model = ModelShape.from_hf_config({
        "n_routed_experts": 8, "hidden_size": 16, "num_hidden_layers": 4,
        "first_k_dense_replace": 1, "num_experts_per_tok": 2, "vocab_size": 128})
    parallel = ServingParallelism.from_engine({
        "tensor_parallel_size": 2, "data_parallel_size": 2, "pipeline_parallel_size": 1,
        "enable_expert_parallel": True, "dtype": "bfloat16", "async_scheduling": True})
    plan = ep_plan(model, parallel, {"eager": [], "cuda_graph_batch_sizes": [2, 4]})
    assert [(s.name, c.op) for s, c in plan] == [
        ("graph_decode_b2", "all_gatherv"), ("graph_decode_b2", "reduce_scatterv"),
        ("graph_decode_b4", "all_gatherv"), ("graph_decode_b4", "reduce_scatterv")]


def test_overlap_read_and_load_sizes():
    layout = KVRegionLayout.from_config(KV_CACHE)
    read = overlap_read({"read": {"prompt_tokens": 128, "requests_in_flight": 2}}, layout, 4)
    assert read == ReadCase("t128_pairs4_inflight2", 128, 2, 4, 2)
    assert expected_notifications(read, {"read_warmup": 2, "repeats": 5}) == 2 * (2 + 3 * 5)
    assert load_runs(1000.0, 300.0, 3.0) == 10
    assert load_runs(10.0, 300.0, 3.0) == 1


class _Event:
    """Completes on its second ``query``."""

    def __init__(self):
        self.queries = 0

    def record(self):
        pass

    def query(self):
        self.queries += 1
        return self.queries >= 2


def test_issue_load_reports_whether_polled_work_ended_first():
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=_Event))
    runs = []
    polls = iter([False] * 3 + [True] * 100)
    assert issue_load(torch, lambda: runs.append(1), 5, 2, lambda: next(polls))
    assert len(runs) == 5
    assert not issue_load(torch, lambda: None, 5, 2, lambda: False)


def test_overlap_summary_uses_slowest_d_rank():
    def rank(scale):
        return {"ep_alone_call_us": [10.0 * scale] * 3, "ep_loaded_call_us": [15.0 * scale] * 3,
                "ep_alone_queue_held": [True] * 3, "ep_loaded_queue_held": [True, True, False],
                "ep_loaded_covered": [True] * 3, "ep_load_runs": 7,
                "read_alone": [{"complete_us": 100.0 * scale}] * 3,
                "read_loaded": [{"complete_us": 150.0 * scale}] * 3,
                "read_loaded_covered": [True, False, True], "data_verified": True}
    record = {"pass": 0, "scenario": "graph_decode_b8", "call": "dispatch", "op": "all_gatherv",
              "cuda_graph": True, "tokens_per_dp_rank": [8, 8], "max_input_bytes": 1,
              "max_output_bytes": 2, "read_case": "r", "bytes_per_reader": 2000,
              "per_rank": [rank(1), rank(2)], "notifications_received": 4,
              "notifications_expected": 4}
    row = summarize_overlap(record)
    assert (row["ep_alone_median_us"], row["ep_loaded_median_us"]) == (20.0, 30.0)
    assert row["ep_loaded_over_alone"] == pytest.approx(1.5)
    assert row["read_loaded_over_alone"] == pytest.approx(1.5)
    assert row["read_alone_per_reader_gigabytes_per_s"] == pytest.approx(0.01)
    assert row["ep_loaded_queue_held_fraction"] == pytest.approx(2 / 3)
    assert row["read_loaded_covered_fraction"] == pytest.approx(2 / 3)
    assert row["ep_load_runs"] == 7 and row["data_verified"]
