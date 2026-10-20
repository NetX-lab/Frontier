"""CPU-only native streaming and canonical repetition tests using local HTTP."""

import asyncio
import copy
import json
from pathlib import Path
import types

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tests.comparison.calibration import pd_replay
from tests.comparison.calibration.pd_proxy import build_app
from tests.comparison.dp_placement_pp import vllm_replay


TRACE = Path(__file__).resolve().parents[2] / "data/processed_traces/canonical_fixed_6144_1024_req128_qps1"
PARAMS = {"do_remote_prefill": True, "do_remote_decode": False, "transfer_mode": "pull",
          "remote_engine_id": "mock-p", "remote_request_id": "mock-request",
          "remote_host": "192.0.2.1", "remote_port": 5557, "remote_block_ids": [[1]]}


def event(payload):
    return ("data: " + (payload if isinstance(payload, str) else json.dumps(payload)) + "\n\n").encode()


def token_event(tokens, finish=None):
    return {"choices": [{"index": 0, "text": "", "token_ids": tokens, "finish_reason": finish}]}


def usage_event(prompt, output):
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": output,
                                     "total_tokens": prompt + output}}


def row(name="r0", output=3):
    return {"request_id": name, "role": "formal", "arrived_at": 0.,
            "num_prefill_tokens": 4, "num_decode_tokens": output}


def test_fragmented_sse_keeps_only_token_event_endpoints(monkeypatch):
    # A scripted clock and byte chunks cover UTF-8 and CRLF split boundaries,
    # multiline data, comments, usage-only/empty events and delayed DONE.
    clock = iter([1., 2., 3., 4., 5., 6., 7., 8.])
    chunks = [b": ping\r", b"\n\r\ndata: {\"choices\":\r\ndata: []}\r\n\r\n",
              b'data: {"choices": [{"text": "\xc3',
              b'\xa9", "token_ids": [17], "index": 0}]}\n\n',
              event(token_event([18, 19], "length")),
              event(token_event([])),
              event(usage_event(4, 3)), event("[DONE]")]

    class Content:
        async def iter_any(self):
            for chunk in chunks:
                yield chunk

    record = row() | {"validation_errors": []}
    monkeypatch.setattr(vllm_replay, "time", types.SimpleNamespace(monotonic=lambda: next(clock)))
    asyncio.run(vllm_replay.read_token_stream(types.SimpleNamespace(content=Content()), record))
    assert record["output_token_count"] == 3 and record["stream_done"]
    assert record["first_token_monotonic"] == 4.
    assert record["last_token_monotonic"] == 5.
    assert record["validation_errors"] == []


def test_client_first_token_arrives_before_mock_decode_can_finish(tmp_path, monkeypatch):
    async def exercise():
        consumed = asyncio.Event()
        finished = []
        seen = []
        parse = vllm_replay.sse_events

        async def observed(content):
            async for data, stamp in parse(content):
                yield data, stamp
                if data != "[DONE]" and any(choice.get("token_ids") for choice in json.loads(data).get("choices", [])):
                    consumed.set()

        monkeypatch.setattr(vllm_replay, "sse_events", observed)

        async def prefill(request):
            body = await request.json()
            seen.append(("p", body, request.headers["X-Request-Id"]))
            return web.json_response({"kv_transfer_params": PARAMS})

        async def decode(request):
            body = await request.json()
            seen.append(("d", body, request.headers["X-Request-Id"]))
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(event(token_event([1])))
            await asyncio.wait_for(consumed.wait(), timeout=5)
            finished.append(vllm_replay.time.monotonic())
            await response.write(event(token_event([2, 3], "length")))
            await response.write(event(usage_event(4, 3)) + event("[DONE]"))
            return response

        p, d = web.Application(), web.Application()
        p.router.add_post("/v1/completions", prefill)
        d.router.add_post("/v1/completions", decode)
        async with TestServer(p) as ps, TestServer(d) as ds:
            proxy = build_app(str(ps.make_url("")).rstrip("/"), str(ds.make_url("")).rstrip("/"),
                              "", "", tmp_path / "proxy.jsonl", vllm_version="0.28.0")
            async with TestServer(proxy) as front:
                _, records = await vllm_replay.replay([row()], front.port, 1024, 0., vllm_version="0.28.0")
        return records, finished, seen

    records, finished, seen = asyncio.run(exercise())
    record = records[0]
    assert record["status"] == "completed"
    assert record["first_token_monotonic"] < finished[0] <= record["last_token_monotonic"]
    assert record["ttft_ms"] < record["e2e_ms"]
    assert record["tpot_ms"] == pytest.approx((record["e2e_ms"] - record["ttft_ms"]) / 2)
    assert seen[0][1]["stream"] is False and seen[0][1]["max_tokens"] == 1
    assert seen[1][1]["stream"] is True and seen[1][1]["return_token_ids"] is True
    assert seen[1][1]["kv_transfer_params"] == PARAMS
    assert seen[0][2] == seen[1][2] == "r0"


@pytest.mark.parametrize("fault", ["wrong_count", "wrong_usage", "eof", "partial_event", "stream_error",
                                  "bad_json", "http", "no_tokens", "text_only"])
def test_stream_failures_are_isolated_and_every_planned_row_survives(fault):
    async def exercise():
        async def complete(request):
            bad = request.headers["X-Request-Id"] == "bad"
            if bad and fault == "http":
                return web.json_response({"error": "injected"}, status=503)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            payload = token_event([1, 2] if bad and fault == "wrong_count" else [1, 2, 3], "length")
            if bad and fault in ("no_tokens", "text_only"):
                payload = {"choices": [{"index": 0, "text": "not a token timestamp", "finish_reason": "length"}]}
            await response.write(event(payload))
            if bad and fault == "partial_event":
                await response.write(b'data: {"choices":')
                return response
            if bad and fault == "bad_json":
                await response.write(event("{invalid"))
            if bad and fault == "stream_error":
                await response.write(event({"error": {"message": "injected native failure"}}))
            await response.write(event(usage_event(4, 2 if bad and fault == "wrong_usage" else 3)))
            if not (bad and fault == "eof"):
                await response.write(event("[DONE]"))
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", complete)
        async with TestServer(app) as server:
            return await vllm_replay.replay([row("bad"), row("good")], server.port, 1024, 0.,
                                           vllm_version="0.28.0")

    _, records = asyncio.run(exercise())
    assert [record["request_id"] for record in records] == ["bad", "good"]
    assert records[0]["status"] == "failed"
    assert records[0].get("error") or records[0]["validation_errors"]
    assert records[1]["status"] == "completed"
    if fault in ("no_tokens", "text_only"):
        assert records[0]["first_token_monotonic"] is None
        assert records[0]["last_token_monotonic"] is None


def metrics(completed, busy=0, dp_size=1):
    return "\n".join(
        f'vllm:{name}{{model_name="dp_pp_case",engine="{lane}"}} {value}'
        for lane in range(dp_size)
        for name, value in (("num_requests_running", busy), ("num_requests_waiting", 0),
                            ("kv_cache_usage_perc", busy), ("request_success_total", completed / dp_size)))


def test_canonical_loader_preserves_frozen_rows_and_manifest():
    manifest, formal, warmup = pd_replay.canonical_workload(TRACE)
    assert manifest["repetitions"]["repeat_indices"] == [1, 2, 3]
    assert [entry["arrived_at"] for entry in formal] == list(range(128))
    assert [entry["arrived_at"] for entry in warmup] == list(range(16))
    assert sum(entry["num_prefill_tokens"] + entry["num_decode_tokens"] for entry in formal) == 917504
    assert len({entry["request_id"] for entry in formal + warmup}) == 144


def test_pd_main_dispatches_canonical_three_repeat_client_without_native_ownership(tmp_path, monkeypatch):
    from tests.unit.test_calibration_pd_harness import native_replay_inputs, nixl_engine

    argv, output = native_replay_inputs(tmp_path, nixl_engine())
    argv[argv.index("--trace-dir") + 1] = str(TRACE)
    argv += ["--prefill-url", "http://192.0.2.1:8100", "--decode-url", "http://192.0.2.2:8200"]
    calls = []

    def spawn(command, **kwargs):
        assert command[1].endswith("/pd_proxy.py")
        calls.append(command)
        return object()

    async def repetitions(workload, port, vocab, lead, urls, dp, timeout, poll, *, checkpoint, records):
        assert len(workload[1]) == 128 and len(workload[2]) == 16
        assert workload[0]["repetitions"]["repeat_indices"] == [1, 2, 3]
        assert urls == {"prefill": "http://192.0.2.1:8100", "decode": "http://192.0.2.2:8200"}
        assert (timeout, poll) == (180., 1.)
        checkpoint({"fixture": True}, records)
        return {"client_workload_verified": True}, records

    monkeypatch.setattr(pd_replay.subprocess, "Popen", spawn)
    monkeypatch.setattr(pd_replay, "wait_until_ready", lambda *args, **kwargs: 0.)
    monkeypatch.setattr(pd_replay, "stop_server", lambda *args: {"signal": "SIGINT", "returncode": 0})
    monkeypatch.setattr(pd_replay, "replay_repetitions", repetitions)
    assert pd_replay.main(argv) == 0
    assert len(calls) == 1
    summary = json.loads((output / "replay_summary.json").read_text())
    assert summary["canonical_repetitions"]["client_workload_verified"]
    assert json.loads((output / "canonical_repetitions.json").read_text()) == {"fixture": True}
    argv[argv.index("--mode") + 1] = "light"
    with pytest.raises(ValueError, match="clean mode"):
        pd_replay.main(argv)
    assert len(calls) == 1


@pytest.mark.parametrize("fault", [None, "idle", "validator", "checkpoint", "request"])
def test_three_repetitions_use_existing_replay_and_observed_idle(tmp_path, monkeypatch, fault):
    async def exercise():
        # Explicit CPU fixture: immutable canonical rows copied in memory with
        # 1/10000 arrival spacing. Token lengths/counts and identities stay exact.
        workload = copy.deepcopy(pd_replay.canonical_workload(TRACE))
        for template in workload[1] + workload[2]:
            template["arrived_at"] /= 10000
        completed = {"prefill": 0, "decode": 0}
        seen = []
        boundaries = []
        metrics_reads = []

        def endpoint(role):
            async def complete(request):
                body = await request.json()
                identity = request.headers["X-Request-Id"]
                seen.append((role, identity, body))
                completed[role] += 1
                if role == "prefill":
                    return web.json_response({"kv_transfer_params": PARAMS})
                count = body["max_tokens"]
                if fault == "request" and identity.endswith("formal-r000-repeat01"):
                    count -= 1
                return web.Response(body=event(token_event([1] * count, "length"))
                                    + event(usage_event(len(body["prompt"]), count)) + event("[DONE]"),
                                    content_type="text/event-stream")
            return complete

        def scrape(role):
            async def get(request):
                metrics_reads.append((role, completed[role]))
                if fault == "idle":
                    return web.Response(text="missing gauges")
                return web.Response(text=metrics(completed[role]))
            return get

        p, d = web.Application(), web.Application()
        for role, app in (("prefill", p), ("decode", d)):
            app.router.add_post("/v1/completions", endpoint(role))
            app.router.add_get("/metrics", scrape(role))
        async with TestServer(p) as ps, TestServer(d) as ds:
            urls = {"prefill": str(ps.make_url("")).rstrip("/"),
                    "decode": str(ds.make_url("")).rstrip("/")}
            proxy = build_app(urls["prefill"], urls["decode"], "", "", tmp_path / "proxy.jsonl",
                              vllm_version="0.28.0")
            async with TestServer(proxy) as front:
                def checkpoint(result, records):
                    if fault == "checkpoint":
                        raise OSError("Injected auxiliary report error")
                    boundaries[:] = result["idle_boundaries"]

                if fault == "validator":
                    def reject(records):
                        raise ValueError("Injected auxiliary aggregation error")
                    monkeypatch.setattr(pd_replay, "formal_metrics", reject)
                result, records = await pd_replay.replay_repetitions(
                    workload, front.port, 154880, 0., urls, 1, .05 if fault == "idle" else 5.,
                    .001, checkpoint=checkpoint)
        return result, records, seen, completed, boundaries, metrics_reads

    result, records, seen, completed, boundaries, metrics_reads = asyncio.run(exercise())
    assert len(records) == 3 * (16 + 128)
    assert completed == {"prefill": 432, "decode": 432}
    assert len({record["request_id"] for record in records}) == 432
    prompts = {}
    for role, identity, body in seen:
        template, repeat = identity.rsplit("-repeat", 1)
        assert repeat in ("01", "02", "03")
        assert len(body["prompt"]) == 6144
        assert prompts.setdefault(template, body["prompt"]) == body["prompt"]
        assert body["max_tokens"] == (1 if role == "prefill" else 1024)
    for repeat in (1, 2, 3):
        selected = [record for record in records if record["repeat"] == repeat]
        assert len([record for record in selected if record["role"] == "warmup"]) == 16
        formal = [record for record in selected if record["role"] == "formal"]
        assert len(formal) == 128
        assert [record["arrived_at"] for record in formal] == [index / 10000 for index in range(128)]
        metric = result["repetitions"][repeat - 1]["metrics"]
        if fault != "validator" and not (fault == "request" and repeat == 1):
            assert metric["total_tokens"] == 917504
            assert metric["first_submission_monotonic"] == min(record["dispatch_monotonic"] for record in formal)
            assert metric["last_token_monotonic"] == max(record["last_token_monotonic"] for record in formal)
            assert metric["total_thpt_tokens_s"] == pytest.approx(917504 / metric["window_s"])
            assert metric["ttft_ms"]["mean"] > 0
            assert metric["tpot_ms"]["mean"] == 0  # One mock delta containing all 1024 tokens.
    assert len(result["idle_boundaries"]) == 7
    assert result["reference_admission"] == "UNVERIFIED"
    assert result["client_workload_verified"] is (fault is None)
    assert result["status"] == ("CLIENT_WORKLOAD_VERIFIED" if fault is None else "REFERENCE_UNSTABLE")
    if fault is None:
        assert all(boundary["verified"] for boundary in boundaries)
        assert metrics_reads == [(role, count) for count in (0, 16, 144, 160, 288, 304, 432)
                                 for role in ("prefill", "decode")]
    if fault == "request":
        assert sum(record["status"] == "failed" for record in records) == 1
        assert result["repetitions"][0]["metrics"] is None
    if fault in ("validator", "checkpoint"):
        assert result["observation_errors"]


def test_idle_metrics_requires_all_lanes_and_zero_kv():
    assert pd_replay.idle_metrics(metrics(5, dp_size=4), 4)["idle"]
    assert not pd_replay.idle_metrics(metrics(5, busy=1, dp_size=4), 4)["idle"]
    with pytest.raises(ValueError, match="Missing"):
        pd_replay.idle_metrics(metrics(5), 4)


def test_cancelled_replay_keeps_all_planned_records_and_marks_unfinished():
    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()
        records = [row("in-flight"), row("not-yet-due") | {"arrived_at": 10000.}]

        async def complete(request):
            started.set()
            await release.wait()
            return web.Response(text="")

        app = web.Application()
        app.router.add_post("/v1/completions", complete)
        async with TestServer(app) as server:
            task = asyncio.create_task(vllm_replay.replay(records, server.port, 1024, 0.,
                                                        vllm_version="0.28.0", records=records))
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
        assert len(records) == 2
        assert all(record["status"] == "unfinished" for record in records)
        assert records[0]["dispatch_monotonic"] is not None
        assert records[1]["dispatch_monotonic"] is None
    asyncio.run(exercise())


def test_idle_observation_waits_for_fresh_completion_and_release():
    async def exercise():
        samples = iter([metrics(1), metrics(2, busy=1), metrics(2)])
        seen = []

        async def scrape(request):
            text = next(samples)
            seen.append(text)
            return web.Response(text=text)

        app = web.Application()
        app.router.add_get("/metrics", scrape)
        async with TestServer(app) as server:
            url = str(server.make_url("")).rstrip("/")
            # Each round reads both roles, so use identical paired snapshots.
            paired = [metrics(1)] * 2 + [metrics(2, busy=1)] * 2 + [metrics(2)] * 2
            samples = iter(paired)
            result = await pd_replay.observe_idle({role: url for role in ("prefill", "decode")},
                                                  1, 5, .0001,
                                                  completed_after={"prefill": 2, "decode": 2})
        assert result["verified"] and len(result["observations"]) == 3
    asyncio.run(exercise())
