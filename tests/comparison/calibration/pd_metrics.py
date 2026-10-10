#!/usr/bin/env python3
"""Join the prefill and decode request metrics of a 1P1D vLLM run into one row per request.

Inputs in ``--run-dir``: ``prefill/request_metrics.jsonl`` and
``decode/request_metrics.jsonl``, the fork's ``VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH``
files with one line per finished engine request, and ``proxy_requests.jsonl``
from pd_proxy.py with one line per client request. Both instances name a request
``cmpl-<proxied id>-0``; the proxy log maps the proxied id to the client
request id of the trace.

Each joined row has the normalized columns of e2e_metrics_gap.py under the PD
metric contract (calibration decision Q13):

``arrival_s``
    proxy arrival.
``completion_s``
    decode-instance completion.
``ttft_ms``
    prefill-instance first-token time - proxy arrival.
``tpot_ms``
    decode-instance (last token - first token) / (N - 1).
``request_e2e_time_ms``
    decode-instance completion - proxy arrival.
``request_num_prefill_tokens``, ``request_num_decode_tokens``
    prompt and output token counts of the decode instance.

It also has the diagnostics ``decode_ttft_ms`` (decode-instance first token -
decode arrival) and ``handoff_ms`` (decode arrival - prefill completion), and the
instance times and token counts they come from.

Clocks: an instance row's ``arrival_time``, ``completion_time`` and ``ttft`` are
``time.time()`` values or intervals taken in that instance's API server process,
and the proxy times are ``time.time()`` values of the proxy process. The three
processes run on one host and read one wall clock; a clock step during the run
would shift the cross-process intervals, and nothing here corrects for it.
``tpot`` is an interval between two ``time.monotonic()`` timestamps of the
decode instance's engine core and is used as it is.

A request is joined when the proxy logged it once with status 200 from both
instances and each instance logged it once. Every other request gets no row
and is listed in ``pd_join_report.json`` under the reason. Exit status: 0 every
request joined, 1 the report lists a request.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

JOIN_GAPS = (
    "proxy_failures",
    "duplicate_proxy_records",
    "duplicate_prefill_rows",
    "duplicate_decode_rows",
    "missing_prefill_rows",
    "missing_decode_rows",
    "unmatched_prefill_rows",
    "unmatched_decode_rows",
    "prefill_output_token_mismatches",
    "prompt_token_mismatches",
)


def engine_request_id(proxied_request_id: str) -> str:
    # The vLLM completions server names the engine request cmpl-<X-Request-Id>-0.
    return f"cmpl-{proxied_request_id}-0"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_request_id(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["request_id"]].append(row)
    return grouped


def pd_row(record: dict, prefill: dict, decode: dict) -> dict:
    arrival = record["proxy_arrival_time"]
    prefill_first_token = prefill["arrival_time"] + prefill["ttft"] / 1000.0
    return {
        "request_id": record["request_id"],
        "arrival_s": arrival,
        "completion_s": decode["completion_time"],
        "ttft_ms": (prefill_first_token - arrival) * 1000.0,
        "tpot_ms": decode["tpot"],
        "request_e2e_time_ms": (decode["completion_time"] - arrival) * 1000.0,
        "request_num_prefill_tokens": decode["request_num_prefill_tokens"],
        "request_num_decode_tokens": decode["request_num_decode_tokens"],
        "decode_ttft_ms": decode["ttft"],
        "handoff_ms": (decode["arrival_time"] - prefill["completion_time"]) * 1000.0,
        "prefill_arrival_s": prefill["arrival_time"],
        "prefill_first_token_s": prefill_first_token,
        "prefill_completion_s": prefill["completion_time"],
        "decode_arrival_s": decode["arrival_time"],
        "prefill_num_prefill_tokens": prefill["request_num_prefill_tokens"],
        "prefill_num_decode_tokens": prefill["request_num_decode_tokens"],
    }


def join_pd_rows(proxy_records: list[dict], prefill_rows: list[dict],
                 decode_rows: list[dict]) -> tuple[list[dict], dict]:
    prefill_by_id, decode_by_id = by_request_id(prefill_rows), by_request_id(decode_rows)
    gaps: dict[str, list[str]] = {kind: [] for kind in JOIN_GAPS}
    proxied_engine_ids = set()
    rows = []
    for client_request_id, records in sorted(by_request_id(proxy_records).items()):
        engine_id = engine_request_id(records[0]["proxied_request_id"])
        proxied_engine_ids.add(engine_id)
        prefill, decode = prefill_by_id.get(engine_id, []), decode_by_id.get(engine_id, [])
        # A failed or repeated proxy request leaves partial instance rows by
        # design; only its proxy record is reported.
        if len(records) > 1:
            reasons = ["duplicate_proxy_records"]
        elif records[0].get("prefill_status") != 200 or records[0].get("decode_status") != 200:
            reasons = ["proxy_failures"]
        else:
            reasons = [kind for failed, kind in (
                (len(prefill) > 1, "duplicate_prefill_rows"),
                (len(decode) > 1, "duplicate_decode_rows"),
                (not prefill, "missing_prefill_rows"),
                (not decode, "missing_decode_rows"),
            ) if failed]
        for kind in reasons:
            gaps[kind].append(client_request_id)
        if reasons:
            continue
        row = pd_row(records[0], prefill[0], decode[0])
        if row["prefill_num_decode_tokens"] != 1:
            gaps["prefill_output_token_mismatches"].append(client_request_id)
        if row["prefill_num_prefill_tokens"] != row["request_num_prefill_tokens"]:
            gaps["prompt_token_mismatches"].append(client_request_id)
        rows.append(row)
    gaps["unmatched_prefill_rows"] = sorted(set(prefill_by_id) - proxied_engine_ids)
    gaps["unmatched_decode_rows"] = sorted(set(decode_by_id) - proxied_engine_ids)
    rows.sort(key=lambda row: row["arrival_s"])
    return rows, gaps


def write_pd_rows(run_dir: Path) -> dict:
    """Write pd_request_metrics.jsonl and pd_join_report.json; return the report."""

    rows, gaps = join_pd_rows(
        read_jsonl(run_dir / "proxy_requests.jsonl"),
        read_jsonl(run_dir / "prefill" / "request_metrics.jsonl"),
        read_jsonl(run_dir / "decode" / "request_metrics.jsonl"),
    )
    report = {"joined": len(rows), "complete": not any(gaps.values()), **gaps}
    # A run's evidence is never rewritten: both files must be new.
    with (run_dir / "pd_request_metrics.jsonl").open("x") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))
    with (run_dir / "pd_join_report.json").open("x") as handle:
        handle.write(json.dumps(report, indent=1) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = write_pd_rows(args.run_dir)
    print("PD_JOIN", json.dumps({"joined": report["joined"], "complete": report["complete"],
                                 **{kind: len(report[kind]) for kind in JOIN_GAPS if report[kind]}}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
