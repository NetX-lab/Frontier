"""CPU prediction of synchronous, FCFS chunked pilot request states.

Source contract: vLLM 0.28 scheduler.py schedule (running before waiting,
long_prefill_token_threshold, token budget), _update_after_schedule, and
update_from_output. Prefix caching, speculative tokens, asynchronous scheduling,
preemption and external KV transfer are disabled by this profiling runner.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from frontier.profiling.attention.dsa_plan import (
    DsaCase, DsaEngineSpec, DsaMeasurementLimits, DP_FULL, DP_NONE, DP_PIECEWISE,
    PHASE_DECODE, PHASE_DUMMY, PHASE_FIRST_TAIL, PHASE_PREFILL,
)


def requests_match_case(case: Any, requests: list[dict[str, int]]) -> bool:
    """Match the submission definition, including heterogeneous continuing rows."""
    if not requests or len(requests) != case.batch_size:
        return False
    lengths = case.prompt_lens or (case.prompt_len,) * case.batch_size
    if Counter(r["prompt_len"] for r in requests) != Counter(lengths):
        return False
    prefills = [r for r in requests if r["num_computed_tokens"] < r["prompt_len"]]
    if case.phase == PHASE_PREFILL:
        return len(prefills) == case.batch_size
    if any(r["num_scheduled_tokens"] != 1 for r in requests):
        return False
    if case.phase == PHASE_DECODE:
        return not prefills
    if case.phase == PHASE_FIRST_TAIL:
        return (len(prefills) == case.tail_count
                and all(r["prompt_len"] == case.prompt_len
                        and r["num_computed_tokens"] + 1 == r["prompt_len"] for r in prefills))
    return False


@dataclass
class _Request:
    prompt: int
    max_tokens: int
    computed: int = 0
    outputs: int = 0

    @property
    def finished(self) -> bool:
        return self.outputs >= self.max_tokens


class _Schedule:
    def __init__(self, engine: DsaEngineSpec):
        self.engine = engine
        self.running: list[_Request] = []
        self.waiting: list[_Request] = []
        self.steps = 0
        self.all_requests: list[_Request] = []

    def add(self, lengths: tuple[int, ...], max_tokens: int) -> list[_Request]:
        requests = [_Request(length, max_tokens) for length in lengths]
        self.waiting.extend(requests)
        self.all_requests.extend(requests)
        return requests

    def step(self) -> list[dict[str, int]]:
        self.running = [r for r in self.running if not r.finished]
        budget = self.engine.chunk_budget
        scheduled: list[tuple[_Request, int]] = []
        new: list[tuple[_Request, int]] = []
        for r in self.running:
            q = min(r.prompt + r.outputs - r.computed, budget,
                    self.engine.max_model_len - r.computed - 1)
            if self.engine.prefill_query_len:
                q = min(q, self.engine.prefill_query_len)
            if q > 0:
                scheduled.append((r, q))
                budget -= q
        while self.waiting and budget > 0 and len(self.running) < self.engine.max_num_seqs:
            r = self.waiting.pop(0)
            q = min(r.prompt, budget)
            if self.engine.prefill_query_len:
                q = min(q, self.engine.prefill_query_len)
            new.append((r, q))
            self.running.append(r)
            budget -= q
        # SchedulerOutput lists new requests before cached requests.
        rows = [{"prompt_len": r.prompt, "num_computed_tokens": r.computed,
                 "num_scheduled_tokens": q} for r, q in [*new, *scheduled]]
        if not rows:
            raise ValueError("infeasible pilot: requests stop advancing before the measured window (max_context_len or max_tokens)")
        for r, q in [*new, *scheduled]:
            r.computed += q
            if r.computed >= r.prompt:
                r.outputs += 1
        self.steps += 1
        return rows


def predict_case_requests(case: DsaCase, engine: DsaEngineSpec
                          ) -> tuple[list[list[dict[str, int]]], int, list[int]]:
    """Return measured request vectors, total engine steps, and output lifetimes."""
    if case.phase == PHASE_DUMMY:
        return ([[] for _ in range(case.measured)], case.warmup + case.measured, [])
    schedule = _Schedule(engine)
    states: list[list[dict[str, int]]] = []
    matches = 0

    def observe(rows):
        nonlocal matches
        if requests_match_case(case, rows):
            matches += 1
            if matches > case.warmup and len(states) < case.measured:
                states.append(rows)

    if case.phase == PHASE_FIRST_TAIL:
        continuing = schedule.add(case.request_prompt_lens[:-case.tail_count], case.max_tokens)
        while any(r.outputs == 0 for r in continuing):
            schedule.step()
        for _ in range(case.warmup + case.measured):
            tails = schedule.add(case.request_prompt_lens[-case.tail_count:], 1)
            while any(not r.finished for r in tails):
                observe(schedule.step())
    elif case.phase == PHASE_PREFILL:
        per_repeat = (case.request_prompt_lens[0] + engine.chunk_budget - 1) // engine.chunk_budget
        for repeat in range(case.warmup + case.measured):
            added = schedule.add(case.request_prompt_lens, case.max_tokens)
            while any(not r.finished for r in added):
                rows = schedule.step()
                if requests_match_case(case, rows) and repeat >= case.warmup:
                    states.append(rows)
        if len(states) != case.measured * per_repeat:
            raise ValueError("infeasible pilot: incomplete measured prefill window")
        return states, schedule.steps, [r.outputs for r in schedule.all_requests]
    else:
        schedule.add(case.request_prompt_lens, case.max_tokens)
        while len(states) < case.measured:
            observe(schedule.step())
    if len(states) != case.measured:
        raise ValueError(f"infeasible pilot: recorded {len(states)} of {case.measured} predicted steps")
    return states, schedule.steps, [r.outputs for r in schedule.all_requests]


def predict_dispatch(case: DsaCase, requests: list[dict[str, int]], limits: DsaMeasurementLimits
                     ) -> dict[str, Any]:
    """No-LoRA, non-varlen V2 graph manager plus dp_utils redispatch rules."""
    tokens = 1 if case.phase == PHASE_DUMMY else sum(r["num_scheduled_tokens"] for r in requests)
    num_reqs = 1 if case.phase == PHASE_DUMMY else len(requests)
    uniform = 1 if case.phase == PHASE_DUMMY or all(
        r["num_scheduled_tokens"] == 1 and r["num_computed_tokens"] >= r["prompt_len"]
        for r in requests) else None

    def local(width, token_count):
        capture = next((n for n in limits.cudagraph_capture_sizes if n >= width), None)
        if capture is None or limits.cudagraph_mode == DP_NONE:
            return {"cg_mode": DP_NONE, "num_tokens": width, "num_reqs": num_reqs}
        full = (limits.cudagraph_mode == "FULL_AND_PIECEWISE" and token_count == 1
                and capture <= limits.max_num_seqs)
        return {"cg_mode": DP_FULL if full else DP_PIECEWISE, "num_tokens": capture,
                "num_reqs": capture if full else None}

    result = local(tokens, uniform)
    peer = case.dp_peer
    if peer is not None:
        if DP_NONE in (result["cg_mode"], peer.cg_mode):
            result = {"cg_mode": DP_NONE, "num_tokens": tokens, "num_reqs": num_reqs}
            across = [tokens, peer.num_tokens]
        else:
            synced_uniform = uniform if uniform == peer.uniform_token_count else None
            result = local(max(tokens, peer.num_tokens), synced_uniform)
            across = [result["num_tokens"]] * 2
    else:
        across = None
    return {**result, "num_tokens_across_dp": across}


def predict_case(case: DsaCase, engine: DsaEngineSpec, limits: DsaMeasurementLimits
                 ) -> list[dict[str, Any]]:
    states, _, _ = predict_case_requests(case, engine)
    return [{"kind": "step", "engine_index": case.engine_index, "case_id": case.case_id,
             "phase": case.phase, "case_step": i, "requests": rows,
             "cudagraph_dispatch": predict_dispatch(case, rows, limits)}
            for i, rows in enumerate(states)]
