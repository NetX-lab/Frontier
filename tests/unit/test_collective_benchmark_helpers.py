"""CPU checks for the pure helpers of the vLLM collective and NIXL READ benchmarks."""

import pytest

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
    summarize,
)

KV_CACHE = {
    "regions": [{"name": "main_mla", "count": 3, "block_bytes": 1152},
                {"name": "indexer", "count": 2, "block_bytes": 132}],
    "block_size": 64, "num_gpu_blocks": 10, "resident_bytes_per_block": 3 * 1152 + 2 * 132,
}


def _record(call="c", op="all_reduce", group="tp", graph=False, timings=None, nbytes=1024):
    timings = timings or [[1.0, 2.0, 3.0], [4.0, 1.0, 1.0], None]
    return {"scenario": "s", "call": call, "op": op, "group": group, "group_size": 8,
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
    assert totals == [{"scenario": "s", "cuda_graph": False, "tokens_per_dp_rank": "4/1",
                       "forward_comm_us": 18.0, "c_us": 9.0, "d_us": 9.0}]


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
    cases = read_cases({"prompt_tokens": [65, 128], "active_pairs": [1, 8],
                        "requests_in_flight": [1, 2]}, layout, 8)
    assert len(cases) == 8 and cases[0].blocks_per_request == 2
    assert request_block_ids(cases[1]) == [[0, 1], [2, 3]]
    with pytest.raises(ValueError, match="exceed"):
        read_cases({"prompt_tokens": [640], "active_pairs": [1],
                    "requests_in_flight": [2]}, layout, 8)
    with pytest.raises(ValueError, match="active_pairs"):
        read_cases({"prompt_tokens": [64], "active_pairs": [9],
                    "requests_in_flight": [1]}, layout, 8)


def test_case_summary_uses_slowest_reader():
    case = ReadCase("c", 128, 2, 2, 1)
    record = {"case": case.name, "prompt_tokens": 128, "blocks_per_request": 2,
              "active_pairs": 2, "requests_in_flight": 1, "descriptors_per_request": 10,
              "bytes_per_reader": 1000,
              "readers": [{"timings": [{"complete_us": 10.0, "prep_us": 1.0}] * 3,
                           "data_verified": True},
                          {"timings": [{"complete_us": 20.0, "prep_us": 2.0}] * 3,
                           "data_verified": True}],
              "notifications_received": 6, "notifications_expected": 6}
    row = summarize_case(record)
    assert row["complete_median_us"] == 20.0 and row["prep_median_us"] == 2.0
    assert row["per_reader_gbps"] == pytest.approx(0.05)
    assert row["aggregate_gbps"] == pytest.approx(0.1)
    assert row["data_verified"]
