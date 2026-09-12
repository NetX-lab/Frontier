"""Run the Issue26 replay with a Kineto window around the first formal batch.

The client deliberately keeps Frontier instrumentation disabled.  It starts the
vLLM built-in profiler after all warmup replays have drained and stops it when
the first formal request emits its first token.  The remaining formal requests
continue after profiling so the run still produces the complete 100-request
formal set.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp

from issue26_token_id_client import build_token_ids


async def post_profile(session: aiohttp.ClientSession, base_url: str,
                       endpoint: str) -> None:
    async with session.post(base_url.rstrip("/") + endpoint) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(
                f"profile endpoint {endpoint} failed with HTTP "
                f"{response.status}: {body[:500]}"
            )


async def send_request(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    request_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    scheduled_at: float,
    output_path: Path,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    first_token_hook: Callable[[str, int], Awaitable[None]] | None = None,
) -> None:
    delay = scheduled_at - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)

    wall_start_ns = time.time_ns()
    monotonic_start_ns = time.perf_counter_ns()
    dispatch_monotonic_s = time.monotonic()
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    headers = {"X-Request-Id": request_id}

    async with semaphore:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(
                    f"request {request_id} failed with HTTP {response.status}: "
                    f"{body[:500]}"
                )

            first_token_ns: int | None = None
            response_id: str | None = None
            completion_tokens = 0
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                chunk: dict[str, Any] = json.loads(data)
                response_id = response_id or chunk.get("id")
                choices = chunk.get("choices") or []
                has_token = bool(choices and any(
                    choice.get("text") or choice.get("token_ids")
                    for choice in choices
                ))
                if has_token and first_token_ns is None:
                    first_token_ns = time.perf_counter_ns()
                    if first_token_hook is not None:
                        await first_token_hook(request_id, first_token_ns)
                for choice in choices:
                    completion_tokens += len(choice.get("token_ids") or [])

    if first_token_ns is None:
        raise RuntimeError(f"{request_id} produced no streamed token")
    completed_ns = time.perf_counter_ns()
    record = {
        "request_id": request_id,
        "response_id": response_id,
        "request_arrival_wall_time_ns": wall_start_ns,
        "planned_arrival_monotonic_s": scheduled_at,
        "dispatch_monotonic_s": dispatch_monotonic_s,
        "dispatch_lag_ms": (dispatch_monotonic_s - scheduled_at) * 1000.0,
        "client_first_token_time_ns": wall_start_ns +
        (first_token_ns - monotonic_start_ns),
        "client_completion_time_ns": wall_start_ns +
        (completed_ns - monotonic_start_ns),
        "client_ttft_ms": (first_token_ns - monotonic_start_ns) / 1e6,
        "client_e2e_ms": (completed_ns - monotonic_start_ns) / 1e6,
        "prompt_tokens": len(prompt_token_ids),
        "max_tokens": max_tokens,
        "completion_tokens_observed": completion_tokens,
    }
    async with write_lock:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


async def run(args: argparse.Namespace) -> None:
    if args.requests != 100:
        raise ValueError("Issue26 profiler run requires exactly 100 formal requests")
    if args.warmups < 10:
        raise ValueError("warmups must be at least ten")
    if args.qps <= 0 or not math.isfinite(args.qps):
        raise ValueError("qps must be finite and positive")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    offsets = [0.0]
    rng = random.Random(args.seed)
    for _ in range(1, args.requests):
        offsets.append(offsets[-1] + rng.expovariate(args.qps))
    with output_path.with_suffix(".planned_arrivals.csv").open(
            "x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["request_id", "arrived_at", "num_prefill_tokens",
                         "num_decode_tokens"])
        for replay in range(args.warmups + 1):
            prefix = (f"warmup:{args.row}:r{replay}:"
                      if replay < args.warmups else f"{args.row}:")
            writer.writerows((f"{prefix}{index}", offset, args.prefill_tokens,
                              args.decode_tokens)
                             for index, offset in enumerate(offsets))

    url = args.base_url.rstrip("/") + "/v1/completions"
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)
    connector = aiohttp.TCPConnector(force_close=True, limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    profile_stop_lock = asyncio.Lock()
    profile_stop_task: asyncio.Task[None] | None = None
    profile_stop_started = False
    phase_records: list[dict[str, Any]] = []

    async with aiohttp.ClientSession(timeout=timeout,
                                     connector=connector) as session:
        for replay in range(args.warmups):
            phase_start = time.monotonic()
            jobs = [asyncio.create_task(send_request(
                session,
                url=url,
                model=args.model,
                request_id=f"warmup:{args.row}:r{replay}:{index}",
                prompt_token_ids=build_token_ids(
                    args.prefill_tokens, args.seed, replay * args.requests + index),
                max_tokens=args.decode_tokens,
                scheduled_at=phase_start + offset,
                output_path=output_path,
                semaphore=semaphore,
                write_lock=write_lock,
            )) for index, offset in enumerate(offsets)]
            await asyncio.gather(*jobs)
            phase_records.append({
                "phase": f"warmup:{replay}",
                "completed_requests": len(jobs),
                "phase_start_monotonic_s": phase_start,
                "phase_end_monotonic_s": time.monotonic(),
            })
            print(json.dumps(phase_records[-1]), flush=True)

        await post_profile(session, args.base_url, "/start_profile")
        profile_start = time.time_ns()

        async def stop_on_first_token(request_id: str, token_ns: int) -> None:
            nonlocal profile_stop_started, profile_stop_task
            async with profile_stop_lock:
                if profile_stop_started:
                    return
                profile_stop_started = True
                profile_stop_task = asyncio.create_task(
                    post_profile(session, args.base_url, "/stop_profile"))
                phase_records.append({
                    "event": "first_formal_first_token",
                    "request_id": request_id,
                    "token_perf_counter_ns": token_ns,
                    "profile_start_wall_ns": profile_start,
                })
                print(json.dumps(phase_records[-1]), flush=True)

        formal_start = time.monotonic()
        jobs = [asyncio.create_task(send_request(
            session,
            url=url,
            model=args.model,
            request_id=f"{args.row}:{index}",
            prompt_token_ids=build_token_ids(
                args.prefill_tokens,
                args.seed,
                args.warmups * args.requests + index,
            ),
            max_tokens=args.decode_tokens,
            scheduled_at=formal_start + offset,
            output_path=output_path,
            semaphore=semaphore,
            write_lock=write_lock,
            first_token_hook=stop_on_first_token if index == 0 else None,
        )) for index, offset in enumerate(offsets)]
        await asyncio.gather(*jobs)
        if profile_stop_task is None:
            await post_profile(session, args.base_url, "/stop_profile")
        else:
            await profile_stop_task
        phase_records.append({
            "phase": "formal",
            "completed_requests": len(jobs),
            "phase_start_monotonic_s": formal_start,
            "phase_end_monotonic_s": time.monotonic(),
        })
        print(json.dumps(phase_records[-1]), flush=True)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    expected = {f"warmup:{args.row}:r{r}:{i}"
                for r in range(args.warmups) for i in range(args.requests)}
    expected |= {f"{args.row}:{i}" for i in range(args.requests)}
    observed = {record["request_id"] for record in records}
    if observed != expected or len(records) != len(expected):
        raise AssertionError(
            f"request identity mismatch expected={len(expected)} "
            f"observed={len(records)} unique={len(observed)}"
        )
    if any(record["prompt_tokens"] != args.prefill_tokens or
           record["completion_tokens_observed"] != args.decode_tokens
           for record in records):
        raise AssertionError("token counts do not match the Issue26 manifest")
    output_path.with_name("phase_records.json").write_text(
        json.dumps(phase_records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"formal_requests": args.requests,
                      "warmup_replays": args.warmups,
                      "total_rows": len(records),
                      "profile_stop_seen": profile_stop_started}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--row", default="pf4096_dc1024")
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=1024)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--qps", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
