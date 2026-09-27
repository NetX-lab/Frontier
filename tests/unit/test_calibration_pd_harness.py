"""Inputs and joins the 1P1D ground-truth harness builds.

A request id the connector cannot parse, a wrong connector switch or a wrong
join is only visible after a two-GPU run, so the pieces that decide them are
checked here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.comparison.calibration.e2e_metrics_gap import NORMALIZED_COLUMNS
from tests.comparison.calibration.pd_metrics import engine_request_id, join_pd_rows, write_pd_rows
from tests.comparison.calibration.pd_proxy import proxied_request_id
from tests.comparison.calibration.pd_replay import (
    connector_check,
    instance_devices,
    instance_env,
    kv_transfer_config,
)

# P2pNcclConnector.parse_request_id, vLLM-BS ff328f82a
# vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:514-530.
DECODE_ADDRESS_OF_PREFILL = re.compile(r"___decode_addr_(.*):(\d+)")
PREFILL_ADDRESS_OF_DECODE = re.compile(r"___prefill_addr_(.*):(\d+)___")


@pytest.mark.parametrize("client_request_id", ["g3-p000", "g3-batch-w15", "7-1", "a_b"])
def test_both_instances_parse_the_other_address_from_the_proxied_id(client_request_id):
    proxied = proxied_request_id(client_request_id, "127.0.0.1:21001", "127.0.0.1:22001")
    engine_id = engine_request_id(proxied)

    assert DECODE_ADDRESS_OF_PREFILL.search(engine_id).groups() == ("127.0.0.1", "22001")
    assert PREFILL_ADDRESS_OF_DECODE.search(engine_id).groups() == ("127.0.0.1", "21001")
    assert engine_id == f"cmpl-{proxied}-0"
    assert proxied.endswith(f"_{client_request_id}")


@pytest.mark.parametrize("client_request_id", ["g3:p000", "_g3-p000"])
def test_a_client_id_that_would_move_an_address_match_is_rejected(client_request_id):
    with pytest.raises(ValueError):
        proxied_request_id(client_request_id, "127.0.0.1:21001", "127.0.0.1:22001")


def proxy_record(request_id: str, arrival: float, prefill_status: int = 200) -> dict:
    record = {
        "request_id": request_id,
        "proxied_request_id": proxied_request_id(request_id, "127.0.0.1:21001", "127.0.0.1:22001"),
        "proxy_arrival_time": arrival,
        "prefill_status": prefill_status,
    }
    if prefill_status == 200:
        record["decode_status"] = 200
    return record


def instance_row(request_id: str, arrival: float, ttft_ms: float, completion: float,
                 decode_tokens: int, tpot_ms: float = 0.0) -> dict:
    return {
        "request_id": engine_request_id(
            proxied_request_id(request_id, "127.0.0.1:21001", "127.0.0.1:22001")
        ),
        "arrival_time": arrival,
        "completion_time": completion,
        "ttft": ttft_ms,
        "tpot": tpot_ms,
        "request_e2e_time": (completion - arrival) * 1000.0,
        "request_num_prefill_tokens": 2048,
        "request_num_decode_tokens": decode_tokens,
    }


def synthetic_run() -> tuple[list[dict], list[dict], list[dict]]:
    """a joins; b lacks its decode row; c has two decode rows; e failed at prefill."""

    proxy = [proxy_record("a", 100.0), proxy_record("b", 101.0), proxy_record("c", 102.0),
             proxy_record("e", 103.0, prefill_status=500)]
    prefill = [instance_row("a", 100.25, 250.0, 100.5, 1), instance_row("b", 101.25, 250.0, 101.5, 1),
               instance_row("c", 102.25, 250.0, 102.5, 1)]
    decode = [instance_row("a", 100.75, 125.0, 102.0, 256, tpot_ms=5.0),
              instance_row("c", 102.75, 125.0, 104.0, 256, tpot_ms=5.0),
              instance_row("c", 103.75, 125.0, 105.0, 256, tpot_ms=5.0),
              instance_row("x", 104.0, 125.0, 105.0, 256, tpot_ms=5.0)]
    return proxy, prefill, decode


def test_the_join_applies_the_pd_metric_contract_and_reports_every_other_request():
    rows, gaps = join_pd_rows(*synthetic_run())

    assert rows == [{
        "request_id": "a",
        "arrival_s": 100.0,
        "completion_s": 102.0,
        "ttft_ms": 500.0,             # prefill first token 100.25 + 0.25 s, minus proxy arrival
        "tpot_ms": 5.0,
        "request_e2e_time_ms": 2000.0,
        "request_num_prefill_tokens": 2048,
        "request_num_decode_tokens": 256,
        "decode_ttft_ms": 125.0,
        "handoff_ms": 250.0,          # decode arrival 100.75 - prefill completion 100.5
        "prefill_arrival_s": 100.25,
        "prefill_first_token_s": 100.5,
        "prefill_completion_s": 100.5,
        "decode_arrival_s": 100.75,
        "prefill_num_prefill_tokens": 2048,
        "prefill_num_decode_tokens": 1,
    }]
    assert set(NORMALIZED_COLUMNS) <= set(rows[0])
    assert {kind: ids for kind, ids in gaps.items() if ids} == {
        "missing_decode_rows": ["b"],
        "duplicate_decode_rows": ["c"],
        "proxy_failures": ["e"],
        "unmatched_decode_rows": [instance_row("x", 0.0, 0.0, 0.0, 0)["request_id"]],
    }


def test_a_prefill_with_more_than_one_output_token_is_reported_but_kept():
    proxy, prefill, decode = synthetic_run()
    prefill[0]["request_num_decode_tokens"] = 2

    rows, gaps = join_pd_rows(proxy[:1], prefill[:1], decode[:1])

    assert [row["request_id"] for row in rows] == ["a"]
    assert gaps["prefill_output_token_mismatches"] == ["a"]


def test_the_joined_files_are_written_once(tmp_path: Path):
    proxy, prefill, decode = synthetic_run()
    for path, rows in ((tmp_path / "proxy_requests.jsonl", proxy),
                       (tmp_path / "prefill" / "request_metrics.jsonl", prefill),
                       (tmp_path / "decode" / "request_metrics.jsonl", decode)):
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = write_pd_rows(tmp_path)

    assert (report["joined"], report["complete"]) == (1, False)
    assert [json.loads(line)["request_id"]
            for line in (tmp_path / "pd_request_metrics.jsonl").read_text().splitlines()] == ["a"]
    assert json.loads((tmp_path / "pd_join_report.json").read_text()) == report
    with pytest.raises(FileExistsError):
        write_pd_rows(tmp_path)


INIT_LINE = (
    "\x1b[1;36m(EngineCore_DP0 pid=7)\x1b[0;0m INFO 09-27 12:00:00 [p2p_nccl_engine.py:168] "
    "💯P2pNcclEngine init, rank:0, local_rank:0, http_address:127.0.0.1:8200, "
    "zmq_address:127.0.0.1:22001, proxy_address:, send_type:PUT_ASYNC, "
    "buffer_size_threshold:8000000000.00, nccl_num_channels:16"
)
FALLBACK_LINE = (
    "\x1b[1;36m(EngineCore_DP0 pid=7)\x1b[0;0m WARNING 09-27 12:00:05 [p2p_nccl_engine.py:347] "
    "🔴[PUT]Recv Tensor, Out Of Threshold, 127.0.0.1:22001👈127.0.0.1:21001, data:{}, addr:0"
)


@pytest.mark.parametrize(("lines", "faults", "passed"), [
    ([INIT_LINE, "INFO serving"], {}, True),
    ([INIT_LINE, FALLBACK_LINE, FALLBACK_LINE], {"[PUT]Recv Tensor, Out Of Threshold": 2}, False),
    (["INFO serving"], {}, False),
    ([INIT_LINE.replace("send_type:PUT_ASYNC", "send_type:PUT")], {}, False),
])
def test_the_connector_check_needs_its_put_async_init_line_and_no_fault(tmp_path, lines, faults, passed):
    server_log = tmp_path / "server.log"
    server_log.write_text("\n".join(lines) + "\n")

    check = connector_check(server_log)

    assert (check["faults"], check["passed"]) == (faults, passed)


def test_each_instance_gets_its_role_gpu_and_connector_values_from_the_engine_file():
    engine = {"attention_backend": "FLASHINFER", "kv_transfer": {
        "kv_buffer_size": 8e9, "mem_pool_size_gb": 1, "prefill_kv_port": 21001,
        "decode_kv_port": 22001, "nccl_num_channels": 16,
    }}
    inherited = {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "3,5",
                 "VLLM_FRONTIER_KV_TRANSFER_LOG_PATH": "/old/kv.jsonl"}
    devices = instance_devices(inherited)

    env, set_env = instance_env("clean", engine, Path("/run/decode"), inherited, devices["decode"])

    assert devices == {"prefill": "3", "decode": "5"}
    assert set_env == {
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/decode/dp_placement",
        "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH": "/run/decode/request_metrics.jsonl",
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        "CUDA_VISIBLE_DEVICES": "5",
        "VLLM_FRONTIER_KV_TRANSFER_LOG_PATH": "/run/decode/kv_transfer.jsonl",
    }
    assert env == {"PATH": "/usr/bin", **set_env}
    assert kv_transfer_config(engine, "decode", 8200) == {
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_consumer",
        "kv_buffer_size": 8e9,
        "kv_ip": "127.0.0.1",
        "kv_port": 22001,
        "kv_connector_extra_config": {
            "http_port": "8200", "send_type": "PUT_ASYNC",
            "nccl_num_channels": "16", "mem_pool_size_gb": "1",
        },
    }
    with pytest.raises(SystemExit):
        instance_devices({"CUDA_VISIBLE_DEVICES": "4"})
