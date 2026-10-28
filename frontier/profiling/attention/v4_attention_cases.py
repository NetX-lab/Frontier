"""Build executable C1 attention windows and preserve infeasible targets."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import json
from typing import Any, Iterable

from frontier.profiling.attention.dsa_case_prediction import predict_case, predict_case_requests
from frontier.profiling.attention.dsa_plan import (
    DsaCase, DsaEngineSpec, DsaMeasurementLimits, build_pilot_cases,
    DP_FULL, DP_NONE, DP_PIECEWISE, PHASE_DECODE, PHASE_DUMMY, PHASE_FIRST_TAIL, PHASE_PREFILL,
)
from frontier.profiling.attention.v4_attention_windows import (
    AttentionWindow, C1, assign_windows, required_windows, profile_targets,
)

PLACEMENT_ITERATIONS = 32


def _cached_candidates(target: int) -> range:
    return range(max(C1["cached_low"], target - C1["context_tolerance"]),
                 min(C1["cached_high"], target + C1["context_tolerance"]) + 1)


def _write_phase(cached: int) -> tuple[bool, ...]:
    return tuple((cached + 1) // ratio > cached // ratio for ratio in C1["ratios"])


def _phase_candidates(cached: int, target: int) -> list[int]:
    return [candidate for candidate in _cached_candidates(target)
            if _write_phase(candidate) == _write_phase(cached)]


def _align_context_sum(window: AttentionWindow, contexts: tuple[int, ...],
                       chunk_budget: int) -> tuple[int, ...]:
    """Avoid an impossible last FCFS prefill step, preserving each write phase.

    Before the final construction step, prompt backlog fills every token budget.
    That final step must schedule at least one token for every own request. Thus
    its total cached sum cannot leave a remainder strictly between zero and own.
    """
    remainder = sum(contexts) % chunk_budget
    if remainder == 0 or remainder >= window.own:
        return contexts
    choices = {0: (0, ())}
    for cached, target in zip(contexts, profile_targets(window.profile, window.own)):
        candidates = _phase_candidates(cached, target)
        following = {}
        for delta, (cost, previous) in choices.items():
            for candidate in candidates:
                new_delta = delta + candidate - cached
                entry = (cost + abs(candidate - target), (*previous, candidate))
                if new_delta not in following or entry < following[new_delta]:
                    following[new_delta] = entry
        choices = following
    valid = [entry for delta, entry in choices.items()
             if (sum(contexts) + delta) % chunk_budget == 0
             or (sum(contexts) + delta) % chunk_budget >= window.own]
    if not valid:
        raise ValueError("selected write phases cannot fit the last FCFS construction step within the profile tolerance")
    return min(valid)[1]


def _contexts(window: AttentionWindow, chunk_budget: int) -> tuple[int, ...]:
    tails = window.tails
    needed = (window.mix[0] - tails, window.mix[1] - tails)
    targets = profile_targets(window.profile, window.own - tails)
    choices = {(0, 0): (0, ())}
    for target in targets:
        candidates = _cached_candidates(target)
        following = {}
        for counts, (cost, contexts) in choices.items():
            for cached in candidates:
                writes = _write_phase(cached)
                total = tuple(n + w for n, w in zip(counts, writes))
                if any(n > bound for n, bound in zip(total, needed)):
                    continue
                entry = (cost + abs(cached - target), (*contexts, cached))
                if total not in following or entry < following[total]:
                    following[total] = entry
        choices = following
    if needed not in choices:
        raise ValueError(f"mix {window.mix} has no placement within context targets +/- {C1['context_tolerance']}")
    contexts = choices[needed][1]
    return contexts if tails else _align_context_sum(window, contexts, chunk_budget)


def _context_adjustments(window: AttentionWindow, contexts: tuple[int, ...]
                         ) -> Iterable[tuple[int, ...]]:
    """Try the nearest phase-preserving adjustment if FCFS inversion cycles."""
    yield contexts
    targets = profile_targets(window.profile, len(contexts))
    alternatives = []
    for index, (cached, target) in enumerate(zip(contexts, targets)):
        for candidate in _phase_candidates(cached, target):
            if candidate == cached:
                continue
            adjusted = contexts[:index] + (candidate,) + contexts[index + 1:]
            cost = sum(abs(c - t) for c, t in zip(adjusted, targets))
            alternatives.append((cost, adjusted))
    yield from (adjusted for _, adjusted in sorted(alternatives))


def _place(window: AttentionWindow, contexts: tuple[int, ...], engine: DsaEngineSpec,
           case_id: str) -> dict[str, Any]:
    """Invert FCFS construction with the same prediction used for admission.

    For tails, invert the continuing requests' first pure decode step first.
    The tail chunks then advance them once before the measured first-tail step.
    This lets the inverse update observe high contexts before a bad guess would
    stop those requests during the tail construction. The full case prediction
    still has to reach the requested contexts before the row is returned.
    """
    tail_prompt = C1["prefill_query"]
    phase = PHASE_FIRST_TAIL if window.tails else PHASE_DECODE
    for adjusted in _context_adjustments(window, contexts):
        shift = math.ceil(sum(adjusted) / engine.chunk_budget) + 2
        lengths = tuple(c - shift for c in adjusted) + (tail_prompt,) * window.tails
        desired = adjusted + (tail_prompt - 1,) * window.tails
        seen = set()
        for _ in range(PLACEMENT_ITERATIONS):
            if lengths in seen:
                break
            seen.add(lengths)
            case = DsaCase(case_id, engine.engine_index, phase, window.own,
                           tail_prompt if window.tails else lengths[0], engine.max_model_len,
                           0, 1, 0, None, lengths, max(window.tails, 1))
            if window.tails and adjusted:
                construction = replace(case, phase=PHASE_DECODE, batch_size=len(adjusted),
                                       prompt_lens=lengths[:-window.tails])
                states, _, _ = predict_case_requests(construction, engine)
                tail_chunk_steps = math.ceil(tail_prompt / engine.prefill_query_len) - 1
                actual = tuple(r["num_computed_tokens"] + tail_chunk_steps for r in states[0])
                actual += (tail_prompt - 1,) * window.tails
            else:
                states, _, _ = predict_case_requests(case, engine)
                actual = tuple(r["num_computed_tokens"] for r in states[0])
            if actual == desired:
                states, _, _ = predict_case_requests(case, engine)
                if tuple(r["num_computed_tokens"] for r in states[0]) == desired:
                    return {"case_id": case_id, "engine_index": engine.engine_index, "phase": phase,
                            "batch_size": window.own, "prompt_len": case.prompt_len,
                            "prompt_lens": list(lengths), "warmup": 0, "measured": 1,
                            "dp_peer": {"num_tokens": window.width, "cg_mode": window.mode,
                                        "uniform_token_count": None if window.mode == DP_PIECEWISE else 1,
                                        "max_query_len": 1},
                            **({"tail_count": window.tails} if window.tails else {})}
                break
            lengths = tuple(length + target - observed for length, target, observed
                            in zip(lengths, desired, actual))
    raise ValueError("FCFS placement did not converge within the native context profile and write phases")


def build_attention_cases(limits: DsaMeasurementLimits) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return admitted pilot cases and the complete amended C1 target manifest.

    One measured step per window avoids advancing away from its requested mix.
    Prediction must assign the measured step back to the requested window.
    """
    base_budget, = limits.chunk_budgets
    q = C1["prefill_query"] - 1
    multi_budget = max(tails for _, tails in C1["multi_tail"]) * C1["prefill_query"]
    engines = [DsaEngineSpec(i, budget, limits.max_num_seqs, limits.max_context_len,
                             limits.block_size, query)
               for i, budget, query in ((0, base_budget, 0), (1, base_budget, q),
                                       (2, max(base_budget, multi_budget), q))]
    pilot_engines = [{"engine_index": e.engine_index, "prefill_query_len": e.prefill_query_len,
                     **({"chunk_budget": e.chunk_budget} if e.chunk_budget != base_budget else {})}
                    for e in engines]
    cases, targets = [], []
    submissions = {}
    for index, window in enumerate(required_windows()):
        engine = engines[2 if window.tails > 1 else 1 if window.tails else 0]
        case_id = f"g1-e{engine.engine_index}-{index:04d}-{window.category.lower()}"
        entry = window.to_json()
        try:
            if window.category == PHASE_DUMMY:
                row = {"case_id": case_id, "engine_index": 0, "phase": PHASE_DUMMY,
                       "batch_size": 1, "prompt_len": 0, "warmup": 0, "measured": 1,
                       "dp_peer": {"num_tokens": C1["prefill_query"] if window.mode == DP_NONE else window.width,
                                   "cg_mode": window.mode,
                                   "uniform_token_count": 1 if window.mode == DP_FULL else None,
                                   "max_query_len": C1["prefill_query"] if window.mode == DP_NONE else 1}}
            elif window.category == PHASE_PREFILL:
                row = {"case_id": case_id, "engine_index": 0, "phase": PHASE_PREFILL,
                       "batch_size": 1, "prompt_len": C1["prefill_query"], "dp_peer": None,
                       "warmup": 0, "measured": 1}
            else:
                row = _place(window, _contexts(window, engine.chunk_budget), engine, case_id)
            # The pilot loader is the only admission path, also used by consumers.
            _, admitted = build_pilot_cases(limits, {"engines": pilot_engines, "cases": [row]})
            steps = predict_case(admitted[0], engine, limits)
            if any(window not in assign_windows(step) for step in steps):
                raise ValueError("constructed prediction does not reach its requested window")
            submission = json.dumps({key: value for key, value in row.items() if key != "case_id"},
                                    sort_keys=True)
            if submission in submissions:
                entry.update(status="predicted", case_id=submissions[submission],
                             shared_case=True)
            else:
                submissions[submission] = case_id
                entry.update(status="predicted", case_id=case_id)
                cases.append(row)
        except ValueError as error:
            entry.update(status="infeasible", reason=str(error))
        targets.append(entry)
    return {"engines": pilot_engines, "cases": cases}, {
        "schema_version": "v4_attention_grid.v1", "context_tolerance": C1["context_tolerance"],
        "grid_decision": C1["decision"],
        "grid": {key: value for key, value in C1.items() if key != "max_writes"},
        "max_writes": {str(k): v for k, v in C1["max_writes"].items()},
        "dummy_aliases": [{"mode": DP_FULL, "width": 1, "roles": ["DECODE", "PREFILL"],
                           "measured_once": True}],
        "limits": asdict(limits), "windows": targets,
        "complete": all(t["status"] == "predicted" for t in targets),
    }
