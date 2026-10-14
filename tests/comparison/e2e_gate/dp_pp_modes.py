"""DP x PP modes over time of a two-lane DP x PP cell (decisions T43-MODEGATE-RULE, T43-PAIRGATE).

The two DP lanes (vLLM engines, Frontier attention-DP lanes) of these cells move between four
modes, each with its own per-token period:

- in_phase: each lane holds one batch and runs it in every other forward; the lanes' batch
  forwards start together.
- anti_phase: each lane holds one batch; one lane's batch forward pairs with the other's dummy.
- split: one lane holds two microbatches and runs a batch in every forward.
- double_split: both lanes hold two microbatches; no forward is a dummy.

A lane's batch events are the vLLM engine iterations that schedule tokens (engine_iteration with
num_scheduled_tokens > 0, at their end stamp) or the Frontier stage-0 batches (stage_start_ts),
over the formal requests' window (vLLM: first formal dispatch to last formal response; Frontier:
the batches holding a formal request). An event holds two batches when the lane's previous event
is in the preceding engine step (vLLM) or its previous batch has not left the last stage
(Frontier). The time from the first formal arrival is cut into windows of window_s seconds. A
window is double_split when both lanes hold two batches in at least SPLIT_SHARE of their events
in it, split when one lane does, else in_phase or anti_phase by the majority phase of its lane-1
events: a lane-1 event is in phase when its phase within lane 0's surrounding event interval is
below IN_PHASE_EDGE or above 1 - IN_PHASE_EDGE. A window without a lane-1 event inside lane 0's
events (one lane idle) has no mode.

A request's TTFT belongs to the mode of its arrival window, and its TPOT to the mode that covers
DECODE_MODE_SHARE of the windows from its first token to its completion.
"""

from __future__ import annotations

import bisect
import json
from collections import Counter
from pathlib import Path

SPLIT_SHARE = 0.5
IN_PHASE_EDGE = 0.25
DECODE_MODE_SHARE = 0.7

Lanes = dict[int, list[tuple[float, bool]]]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def vllm_formal_dispatch(run_dir: Path) -> dict[str, float]:
    """Client dispatch time (monotonic s, the clock of the engines' logs) of each formal request."""
    return {r["request_id"]: r["dispatch_monotonic"]
            for r in read_jsonl(run_dir / "run" / "client_requests.jsonl") if r["role"] == "formal"}


def vllm_lanes(run_dir: Path) -> Lanes:
    """Batch events of each DP engine of a vLLM run (`<run_dir>/run/dp_placement/*.jsonl`)."""
    formal = [r for r in read_jsonl(run_dir / "run" / "client_requests.jsonl") if r["role"] == "formal"]
    start = min(r["dispatch_monotonic"] for r in formal)
    end = max(r["response_monotonic"] for r in formal)
    lanes: Lanes = {}
    for log in sorted((run_dir / "run" / "dp_placement").glob("dp_placement_*.jsonl")):
        steps = [(r["engine"], r["wave"], r["step"], r["monotonic"]) for r in read_jsonl(log)
                 if r["kind"] == "engine_iteration" and r["num_scheduled_tokens"] > 0]
        previous = None
        for engine, wave, step, time in steps:
            if start <= time <= end:
                lanes.setdefault(engine, []).append((time, previous == (wave, step - 1)))
            previous = (wave, step)
    return lanes


def frontier_lanes(metrics_dir: Path, formal_frontier_ids: set[str]) -> Lanes:
    """Batch events of each attention-DP lane of a Frontier run (its stage batch ledger)."""
    records = [r for r in read_jsonl(metrics_dir / "frontier_stage_batch_ledger.jsonl")
               if r["execution_scope"] == "ATTN_DP_LANE"]
    last_stage = max(r["stage_id"] for r in records)
    batch_end = {(r["replica_local_id"], r["batch_id"]): r["stage_end_ts"] for r in records
                 if r["stage_id"] == last_stage}
    lanes: Lanes = {}
    previous_end: dict[int, float] = {}
    for r in sorted((r for r in records if r["stage_id"] == 0), key=lambda r: r["stage_start_ts"]):
        lane = r["replica_local_id"]
        if formal_frontier_ids & set(map(str, r["request_ids"])):
            lanes.setdefault(lane, []).append((r["stage_start_ts"], r["stage_start_ts"] < previous_end.get(lane, 0.0)))
        previous_end[lane] = batch_end[(lane, r["batch_id"])]
    return lanes


def window_modes(lanes: Lanes, start: float, window_s: float) -> dict[int, str]:
    """Mode of every window index that has one."""
    if sorted(lanes) != [0, 1]:
        raise ValueError(f"expected DP lanes 0 and 1, got {sorted(lanes)}")
    two_batch: dict[int, list[list[int]]] = {}
    for lane, events in lanes.items():
        for time, holds_two in events:
            counts = two_batch.setdefault(int((time - start) // window_s), [[0, 0], [0, 0]])[lane]
            counts[0] += holds_two
            counts[1] += 1
    lane0 = [time for time, _ in lanes[0]]
    phases: dict[int, Counter] = {}
    for time, _ in lanes[1]:
        index = bisect.bisect_right(lane0, time)
        if 0 < index < len(lane0):
            phase = (time - lane0[index - 1]) / (lane0[index] - lane0[index - 1])
            in_phase = phase < IN_PHASE_EDGE or phase > 1 - IN_PHASE_EDGE
            phases.setdefault(int((time - start) // window_s), Counter())["in_phase" if in_phase else "anti_phase"] += 1
    modes = {}
    for window, counts in two_batch.items():
        split = [events > 0 and two / events >= SPLIT_SHARE for two, events in counts]
        if all(split):
            modes[window] = "double_split"
        elif any(split):
            modes[window] = "split"
        elif window in phases:
            modes[window] = phases[window].most_common(1)[0][0]
    return modes


def request_modes(arrival: float, first_token: float, completion: float, start: float, window_s: float,
                  modes: dict[int, str]) -> tuple[str | None, str | None]:
    """The modes a request's TTFT and TPOT belong to (times on the clock of the modes)."""
    decode = [modes.get(window) for window in range(int((first_token - start) // window_s),
                                                    int((completion - start) // window_s) + 1)]
    top = Counter(mode for mode in decode if mode).most_common(1)
    tpot_mode = top[0][0] if top and top[0][1] >= DECODE_MODE_SHARE * len(decode) else None
    return modes.get(int((arrival - start) // window_s)), tpot_mode
