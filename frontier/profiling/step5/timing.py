"""CUDA-event windows for already-warmed native attention replays."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import ceil, fsum, isfinite
from numbers import Real


_SCOPED_LAYER = "step5_layer_scoped"


@dataclass(frozen=True)
class ReplayTimings:
    """Independent clean-window cohorts and boundary diagnostics, in milliseconds."""

    single_replay_times: tuple[float, ...]
    amortized_times: tuple[float, ...]
    window_durations_ms: tuple[float, ...]
    window_replays: tuple[int, ...]
    reference_times: tuple[float, ...]
    reference_window_durations_ms: tuple[float, ...]
    reference_window_replays: tuple[int, ...]
    last_output: object


@dataclass(frozen=True)
class ScopedReplay:
    """A native replay and a reader for its independently owned scope events."""

    run: Callable[[], object]
    times: Callable[[], Mapping[str, float]]


@dataclass(frozen=True)
class ScopedTimings:
    """Matched operator/root samples in milliseconds, with separate outer windows."""

    amortized_times: dict[str, tuple[float, ...]]
    per_replay_times: dict[str, tuple[float, ...]]
    window_durations_ms: tuple[float, ...]
    last_output: object


def _event_duration_ms(start, end) -> float:
    duration_ms = start.elapsed_time(end)
    if not isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("Replay window duration must be positive and finite")
    return duration_ms


def _window_duration_ms(start, end) -> float:
    end.synchronize()
    return _event_duration_ms(start, end)


def _prime_events(events) -> None:
    # CUDA event handles are allocated lazily on their first record.
    for event in events:
        event.record()
    events[-1].synchronize()


def collect_replay_timings(
    torch,
    run: Callable[[], object],
    prepare: Callable[[], object],
    *,
    sample_count: int,
    min_window_ms: float,
) -> ReplayTimings:
    """Measure boundary diagnostics and independent candidate/reference windows.

    The caller owns native warmup, capture and the safety of consecutive replays.
    Event handles are primed before preparation and any measured replay.
    A single prepared cohort uses shared per-iteration boundary events and
    synchronizes only after its last replay. Preparation then runs once before
    each window, never between its replays. The final single-replay sample also
    serves as the initial repeat-count pilot. Short windows increase the count
    from their measured duration and are not retained as amortized samples.
    Each candidate window and its independently executed reference use the same
    replay count and distinct primed outer events. If either is short, discard
    both and increase their common count. This is minimum-duration treatment,
    never an agreement or stability retry. Boundary samples are diagnostics only.
    """
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    if (
        not isinstance(min_window_ms, (int, float))
        or isinstance(min_window_ms, bool)
        or not isfinite(min_window_ms)
        or min_window_ms <= 0
    ):
        raise ValueError("min_window_ms must be positive and finite")

    last_output = None

    def measure_window(replays: int) -> float:
        nonlocal last_output
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        _prime_events((start, end))
        prepare()
        start.record()
        for _ in range(replays):
            last_output = run()
        end.record()
        return _window_duration_ms(start, end)

    boundaries = tuple(torch.cuda.Event(enable_timing=True) for _ in range(sample_count + 1))
    _prime_events(boundaries)
    prepare()
    boundaries[0].record()
    for index in range(sample_count):
        last_output = run()
        boundaries[index + 1].record()
    boundaries[-1].synchronize()
    singles = tuple(_event_duration_ms(start, end) for start, end in zip(boundaries, boundaries[1:]))
    replays = max(1, ceil(min_window_ms / singles[-1]))
    durations = []
    reference_durations = []
    counts = []
    while len(durations) < sample_count:
        duration_ms = measure_window(replays)
        reference_ms = measure_window(replays)
        shortest_ms = min(duration_ms, reference_ms)
        if shortest_ms < min_window_ms:
            replays = max(replays + 1, ceil(replays * (min_window_ms / shortest_ms)))
            continue
        durations.append(duration_ms)
        reference_durations.append(reference_ms)
        counts.append(replays)

    return ReplayTimings(
        single_replay_times=singles,
        amortized_times=tuple(duration / count for duration, count in zip(durations, counts)),
        window_durations_ms=tuple(durations),
        window_replays=tuple(counts),
        reference_times=tuple(duration / count for duration, count in zip(reference_durations, counts)),
        reference_window_durations_ms=tuple(reference_durations),
        reference_window_replays=tuple(counts),
        last_output=last_output,
    )


def collect_scoped_timings(
    torch,
    make_replay: Callable[[], ScopedReplay],
    prepare: Callable[[], object],
    *,
    window_replays: tuple[int, ...],
    operator_names: tuple[str, ...],
) -> ScopedTimings:
    """Collect complete cohorts without overwriting unread invocation events.

    The factory runs exactly max(window_replays) times before any measured
    window. It owns capture and warmup, and each instance must own independent
    events. Each times() mapping contains exactly operator_names and the
    step5_layer_scoped root enclosing those exclusive spans from that invocation.
    Raw samples are ordered by window, then by instance within the window.
    """
    if len(window_replays) < 2 or any(type(count) is not int or count < 1 for count in window_replays):
        raise ValueError("window_replays must contain at least two positive integer counts")
    if (
        not operator_names
        or any(not isinstance(name, str) or not name for name in operator_names)
        or len(set(operator_names)) != len(operator_names)
        or _SCOPED_LAYER in operator_names
    ):
        raise ValueError("operator_names must contain distinct nonempty names excluding step5_layer_scoped")

    instances = tuple(make_replay() for _ in range(max(window_replays)))
    if len({id(instance) for instance in instances}) != len(instances):
        raise ValueError("make_replay must return distinct ScopedReplay instances")
    names = (*operator_names, _SCOPED_LAYER)
    expected_names = set(names)
    amortized = {name: [] for name in names}
    per_replay = {name: [] for name in names}
    durations = []
    last_output = None
    for window, count in enumerate(window_replays):
        cohort = instances[:count]
        prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for instance in cohort:
            last_output = instance.run()
        end.record()
        durations.append(_window_duration_ms(start, end))

        samples = []
        for replay_index, instance in enumerate(cohort):
            sample = dict(instance.times())
            if set(sample) != expected_names:
                raise ValueError(
                    f"Replay scope names differ: missing={expected_names - set(sample)}, "
                    f"unexpected={set(sample) - expected_names}"
                )
            for name in names:
                value = sample[name]
                is_root = name == _SCOPED_LAYER
                if (
                    not isinstance(value, Real)
                    or isinstance(value, bool)
                    or not isfinite(value)
                    or (value <= 0 if is_root else value < 0)
                ):
                    bound = "positive" if is_root else "nonnegative"
                    raise ValueError(f"{name} duration must be finite and {bound}")
            operator_sum = fsum(sample[name] for name in operator_names)
            if operator_sum > sample[_SCOPED_LAYER]:
                raise ValueError(
                    f"Exclusive operator sum {operator_sum} ms exceeds its same-execution "
                    f"root {sample[_SCOPED_LAYER]} ms in window {window}, replay {replay_index}"
                )
            samples.append(sample)
        for name in names:
            values = tuple(sample[name] for sample in samples)
            per_replay[name].extend(values)
            amortized[name].append(fsum(values) / count)

    return ScopedTimings(
        amortized_times={name: tuple(values) for name, values in amortized.items()},
        per_replay_times={name: tuple(values) for name, values in per_replay.items()},
        window_durations_ms=tuple(durations),
        last_output=last_output,
    )
