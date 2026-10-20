"""CPU CUDA-clock contracts, not measurements of native attention performance."""

from fractions import Fraction
from statistics import mean, median
from types import SimpleNamespace

import pytest

from frontier.profiling.step5 import timing
from frontier.profiling.step5.timing import ReplayTimings, collect_replay_timings


class FakeCudaClock:
    def __init__(self, *, replay_ms="0.01", prepare_ms="0", window_replay_ms=(), replay_durations_ms=()):
        self.time_ms = Fraction(0)
        self.replay_ms = Fraction(replay_ms)
        self.prepare_ms = Fraction(prepare_ms)
        self.window_replay_ms = tuple(Fraction(value) for value in window_replay_ms)
        self.replay_durations_ms = tuple(Fraction(value) for value in replay_durations_ms)
        self.calls = []
        self.events = []
        self.windows = []
        self.active_replays = None
        self.run_count = 0
        self.completed_event_index = -1
        self.torch = SimpleNamespace(cuda=SimpleNamespace(Event=self.event))

    def event(self, *, enable_timing):
        assert enable_timing is True
        event = FakeEvent(self, len(self.events))
        self.events.append(event)
        self.calls.append(("event", event.index))
        return event

    def prepare(self):
        assert self.active_replays is None, "Preparation must be outside the window"
        self.calls.append(("prepare",))
        self.time_ms += self.prepare_ms

    def run(self):
        assert self.active_replays is not None, "No unmeasured warmup or capture is allowed"
        self.calls.append(("run",))
        self.active_replays += 1
        index = len(self.windows)
        duration = self.window_replay_ms[index] if index < len(self.window_replay_ms) else self.replay_ms
        if self.run_count < len(self.replay_durations_ms):
            duration = self.replay_durations_ms[self.run_count]
        self.time_ms += duration
        self.run_count += 1


class FakeEvent:
    def __init__(self, clock, index):
        self.clock = clock
        self.index = index
        self.timestamp = None
        self.synchronized = False

    def record(self):
        clock = self.clock
        clock.calls.append(("record", self.index))
        self.timestamp = clock.time_ms
        if clock.active_replays is None:
            clock.active_replays = 0

    def synchronize(self):
        assert self.index == len(self.clock.events) - 1, "Only the final boundary should synchronize"
        assert self.clock.active_replays is not None
        self.synchronized = True
        self.clock.calls.append(("synchronize", self.index))
        self.clock.windows.append(self.clock.active_replays)
        self.clock.active_replays = None
        self.clock.completed_event_index = self.index

    def elapsed_time(self, end):
        assert end.index == self.index + 1
        assert end.index <= self.clock.completed_event_index, "Read elapsed time only after synchronization"
        self.clock.calls.append(("elapsed_time", self.index, end.index))
        return float(end.timestamp - self.timestamp)


class AsyncCudaClock:
    """Independent host/device clocks with deterministic asynchronous enqueueing."""

    def __init__(self):
        self.host_ms = Fraction(0)
        self.device_ms = Fraction(0)
        self.completed_ms = Fraction(0)
        self.calls = []
        self.events = []
        self.outputs = []
        self.state = 0
        self.torch = SimpleNamespace(cuda=SimpleNamespace(Event=self.event))

    def event(self, *, enable_timing):
        assert enable_timing is True
        event = AsyncEvent(self, len(self.events))
        self.events.append(event)
        self.calls.append(("event", event.index))
        return event

    def prepare(self):
        self.calls.append(("prepare",))
        self.host_ms += Fraction("1")
        self.state = 0

    def run(self):
        self.host_ms += Fraction("0.006")
        self.device_ms = max(self.device_ms, self.host_ms) + Fraction("0.05")
        self.state += 1
        output = SimpleNamespace(invocation=len(self.outputs) + 1, state=self.state)
        self.outputs.append(output)
        self.calls.append(("run", output.invocation))
        return output


class AsyncEvent:
    def __init__(self, clock, index):
        self.clock = clock
        self.index = index
        self.timestamp = None

    def record(self):
        clock = self.clock
        clock.host_ms += Fraction("0.0001")
        clock.device_ms = max(clock.device_ms, clock.host_ms)
        self.timestamp = clock.device_ms
        clock.calls.append(("record", self.index))

    def synchronize(self):
        clock = self.clock
        clock.host_ms = max(clock.host_ms, self.timestamp)
        clock.completed_ms = max(clock.completed_ms, self.timestamp)
        clock.calls.append(("synchronize", self.index))

    def elapsed_time(self, end):
        assert end.timestamp <= self.clock.completed_ms
        self.clock.calls.append(("elapsed_time", self.index, end.index))
        return float(end.timestamp - self.timestamp)


def test_async_queued_singletons_agree_with_amortized_windows():
    clock = AsyncCudaClock()
    result = collect_replay_timings(clock.torch, clock.run, clock.prepare, sample_count=5, min_window_ms=0.5)
    single_median = median(result.single_replay_times)
    amortized_mean = mean(result.amortized_times)
    relative_gap = abs(single_median - amortized_mean) / amortized_mean

    assert relative_gap <= 0.05, (
        f"single_median_ms={single_median:.6f}, amortized_mean_ms={amortized_mean:.6f}, "
        f"relative_gap={relative_gap:.6%}"
    )
    assert result.single_replay_times == pytest.approx((0.056, 0.05, 0.05, 0.05, 0.05))
    assert result.amortized_times == pytest.approx((0.0506,) * 5)


def test_singleton_boundaries_are_queued_before_first_sync_and_final_window_output_is_retained():
    clock = AsyncCudaClock()
    result = collect_replay_timings(clock.torch, clock.run, clock.prepare, sample_count=5, min_window_ms=0.5)
    first_sync = next(index for index, call in enumerate(clock.calls) if call[0] == "synchronize")
    queued = clock.calls[:first_sync]

    assert [call for call in queued if call[0] == "record"] == [("record", index) for index in range(6)]
    assert [call for call in queued if call[0] == "event"] == [("event", index) for index in range(6)]
    assert sum(call[0] == "run" for call in queued) == 5
    assert sum(call[0] == "prepare" for call in queued) == 1
    assert clock.calls[first_sync] == ("synchronize", 5)
    assert clock.calls[first_sync + 1:first_sync + 6] == [("elapsed_time", i, i + 1) for i in range(5)]
    assert sum(call[0] == "synchronize" for call in clock.calls) == 6
    assert result.last_output is clock.outputs[-1]
    assert result.last_output.invocation == 5 + sum(result.window_replays)
    assert result.last_output.state == clock.state == result.window_replays[-1]


def collect(clock, *, sample_count=3, min_window_ms=0.5):
    return collect_replay_timings(
        clock.torch, clock.run, clock.prepare,
        sample_count=sample_count, min_window_ms=min_window_ms,
    )


def assert_window_calls(clock, *, single_samples_read=None):
    """One boundary cohort precedes windows using exactly one event pair each."""
    sample_count = clock.windows[0]
    expected = [("prepare",)]
    expected.extend(("event", index) for index in range(sample_count + 1))
    expected.append(("record", 0))
    for index in range(sample_count):
        expected.extend([("run",), ("record", index + 1)])
    expected.append(("synchronize", sample_count))
    read_count = sample_count if single_samples_read is None else single_samples_read
    expected.extend(("elapsed_time", index, index + 1) for index in range(read_count))
    for index, replays in enumerate(clock.windows[1:]):
        start = sample_count + 1 + 2 * index
        end = start + 1
        expected.extend([("prepare",), ("event", start), ("event", end), ("record", start)])
        expected.extend([("run",)] * replays)
        expected.extend([("record", end), ("synchronize", end), ("elapsed_time", start, end)])
    assert clock.calls == expected
    assert len(clock.events) == sample_count + 1 + 2 * (len(clock.windows) - 1)


def test_fifty_ten_microsecond_replays_reach_half_millisecond():
    clock = FakeCudaClock()
    result = collect(clock)

    assert isinstance(result, ReplayTimings)
    assert result.single_replay_times == (0.01,) * 3
    assert result.amortized_times == (0.01,) * 3
    assert result.window_durations_ms == (0.5,) * 3
    assert result.window_replays == (50,) * 3
    assert clock.windows == [3, 50, 50, 50]
    assert_window_calls(clock)


def test_clean_last_output_is_the_last_measured_invocation_reference():
    clock = FakeCudaClock()
    outputs = []

    def run():
        clock.run()
        output = object()
        outputs.append(output)
        return output

    result = collect_replay_timings(clock.torch, run, clock.prepare, sample_count=2, min_window_ms=0.5)

    assert len(outputs) == sum(clock.windows) == 102
    assert result.last_output is outputs[-1]
    assert_window_calls(clock)


@pytest.mark.parametrize("replay_ms,target", [("0.02", 0.7), ("0.125", 0.6), ("0.75", 0.5)])
def test_repeat_count_comes_from_observed_duration_and_supplied_target(replay_ms, target):
    clock = FakeCudaClock(replay_ms=replay_ms)
    result = collect(clock, sample_count=2, min_window_ms=target)

    for duration, count, amortized in zip(
        result.window_durations_ms, result.window_replays, result.amortized_times,
    ):
        assert duration >= target
        assert amortized == duration / count
        assert amortized == pytest.approx(float(replay_ms))
    if float(replay_ms) >= target:
        assert result.window_replays == (1, 1)
    assert_window_calls(clock)


def test_preparation_cost_is_excluded_from_singles_and_amortized_windows():
    clock = FakeCudaClock(prepare_ms="1000")
    result = collect(clock, sample_count=2)

    assert result.single_replay_times == (0.01, 0.01)
    assert result.window_durations_ms == (0.5, 0.5)
    assert result.amortized_times == (0.01, 0.01)
    assert clock.time_ms == Fraction("3001.02")
    assert_window_calls(clock)


def test_each_short_window_increases_replays_until_it_meets_the_target():
    clock = FakeCudaClock(
        replay_ms="0.01", window_replay_ms=("0.01", "0.005", "0.005", "0.0025"),
    )
    result = collect(clock, sample_count=2)

    assert clock.windows == [2, 50, 100, 100, 200]
    assert result.window_replays == (100, 200)
    assert result.window_durations_ms == (0.5, 2.0)
    assert result.amortized_times == (0.005, 0.01)
    assert result.single_replay_times == (0.01, 0.01)
    assert_window_calls(clock)


def test_noise_between_single_and_amortized_samples_is_not_corrected():
    clock = FakeCudaClock(
        replay_ms="0.03", replay_durations_ms=("0.04", "0.01", "0.02"),
        window_replay_ms=("0.03", "0.03", "0.04"),
    )
    result = collect(clock)

    assert result.single_replay_times == (0.04, 0.01, 0.02)
    assert result.window_replays == (25, 25, 25)
    assert result.window_durations_ms == (0.75, 1.0, 0.75)
    assert result.amortized_times == (0.03, 0.04, 0.03)
    assert_window_calls(clock)


def test_parent_warmup_and_capture_costs_do_not_enter_collected_samples():
    clock = FakeCudaClock()
    parent_calls = []

    def parent_phase(name, duration):
        assert not clock.events
        parent_calls.append(name)
        clock.time_ms += Fraction(duration)

    parent_phase("warmup", "2000")
    parent_phase("capture", "5000")
    result = collect(clock, sample_count=2)

    assert parent_calls == ["warmup", "capture"]
    assert result.single_replay_times == result.amortized_times == (0.01, 0.01)
    assert clock.time_ms == Fraction("7001.02")
    assert_window_calls(clock)


@pytest.mark.parametrize("sample_count", [0, 1, -1, 2.5, 2.0, True, None, "2"])
def test_invalid_sample_count_fails_before_preparation_or_events(sample_count):
    clock = FakeCudaClock()
    with pytest.raises(ValueError, match="sample_count"):
        collect(clock, sample_count=sample_count)
    assert not clock.calls


@pytest.mark.parametrize("target", [0, -0.1, float("nan"), float("inf"), -float("inf"), True, None, "0.5"])
def test_invalid_target_fails_before_preparation_or_events(target):
    clock = FakeCudaClock()
    with pytest.raises(ValueError, match="min_window_ms"):
        collect(clock, min_window_ms=target)
    assert not clock.calls


@pytest.mark.parametrize("duration", [0.0, -0.01, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("interval", ["single", "first_window", "last_window"])
def test_invalid_single_or_window_duration_is_rejected(monkeypatch, duration, interval):
    clock = FakeCudaClock()
    elapsed_time = FakeEvent.elapsed_time
    start_index = {"single": 0, "first_window": 3, "last_window": 5}[interval]

    def measured(self, end):
        actual = elapsed_time(self, end)
        return duration if self.index == start_index else actual

    monkeypatch.setattr(FakeEvent, "elapsed_time", measured)
    with pytest.raises(ValueError, match="duration must be positive and finite"):
        collect(clock, sample_count=2)
    assert len(clock.windows) == {"single": 1, "first_window": 2, "last_window": 3}[interval]
    assert_window_calls(clock, single_samples_read=1 if interval == "single" else 2)


class FakeScopedCudaClock(FakeCudaClock):
    def __init__(self, sample=None):
        super().__init__(prepare_ms="1000")
        self.sample = sample or self.uneven_sample
        self.instances = []
        self.pending = set()
        self.invocations = []
        self.materialized = []
        self.outside_root_ms = Fraction("0.004")

    @staticmethod
    def uneven_sample(instance_id, invocation_id):
        qkv = Fraction(instance_id + 1, 100)
        attention = Fraction(invocation_id, 200)
        return {
            "qkv": float(qkv), "attention": float(attention),
            "step5_layer_scoped": float(qkv + attention + Fraction("0.01")),
        }

    def make_replay(self):
        assert not self.events, "Every factory call must precede all timed windows"
        instance_id = len(self.instances)
        self.calls.extend([("capture", instance_id), ("warmup", instance_id)])
        self.time_ms += Fraction("3000")
        values = {}
        invocation = None

        def run():
            nonlocal invocation
            assert self.active_replays is not None
            assert instance_id not in self.pending, "Unread invocation event handles were overwritten"
            self.active_replays += 1
            invocation = (len(self.windows), instance_id, len(self.invocations) + 1)
            self.calls.append(("scoped_run", *invocation))
            self.invocations.append(invocation)
            values.clear()
            values.update(self.sample(instance_id, invocation[-1]))
            self.time_ms += Fraction(str(values["step5_layer_scoped"])) + self.outside_root_ms
            self.pending.add(instance_id)

        def times():
            assert self.active_replays is None
            assert self.events[-1].synchronized, "Scope events must be read after outer synchronization"
            assert instance_id in self.pending
            self.calls.append(("times", *invocation))
            self.materialized.append(invocation)
            self.pending.remove(instance_id)
            return values

        replay = timing.ScopedReplay(run=run, times=times)
        self.instances.append(replay)
        return replay

    def prepare(self):
        assert not self.pending, "Materialize the complete window before preparing the next one"
        super().prepare()


def collect_scoped(clock, *, window_replays=(1, 3, 2), operator_names=("qkv", "attention"), make_replay=None):
    return timing.collect_scoped_timings(
        clock.torch, make_replay or clock.make_replay, clock.prepare,
        window_replays=window_replays, operator_names=operator_names,
    )


def assert_scoped_window_calls(clock, window_replays):
    expected = []
    for instance in range(max(window_replays)):
        expected.extend([("capture", instance), ("warmup", instance)])
    invocation_id = 0
    for window, count in enumerate(window_replays):
        start, end = 2 * window, 2 * window + 1
        expected.extend([("prepare",), ("event", start), ("event", end), ("record", start)])
        invocations = []
        for instance in range(count):
            invocation_id += 1
            invocation = (window, instance, invocation_id)
            invocations.append(invocation)
            expected.append(("scoped_run", *invocation))
        expected.extend([("record", end), ("synchronize", end), ("elapsed_time", start, end)])
        expected.extend(("times", *invocation) for invocation in invocations)
    assert clock.calls == expected
    assert len(clock.events) == 2 * len(window_replays)


def test_scoped_cohorts_sum_every_distinct_instance_and_retain_invocation_samples():
    clock = FakeScopedCudaClock()
    result = collect_scoped(clock)

    assert len(clock.instances) == len({id(instance) for instance in clock.instances}) == 3
    assert clock.materialized == clock.invocations
    assert not clock.pending
    assert clock.windows == [1, 3, 2]
    assert result.per_replay_times == {
        "qkv": (0.01, 0.01, 0.02, 0.03, 0.01, 0.02),
        "attention": (0.005, 0.01, 0.015, 0.02, 0.025, 0.03),
        "step5_layer_scoped": (0.025, 0.03, 0.045, 0.06, 0.045, 0.06),
    }
    assert result.amortized_times["qkv"] == pytest.approx((0.01, 0.02, 0.015))
    assert result.amortized_times["attention"] == pytest.approx((0.005, 0.015, 0.0275))
    assert result.amortized_times["step5_layer_scoped"] == pytest.approx((0.025, 0.045, 0.0525))
    assert result.window_durations_ms == pytest.approx((0.029, 0.147, 0.113))
    assert clock.time_ms == Fraction("12000.289")
    for qkv, attention, root in zip(*result.amortized_times.values()):
        assert (qkv + attention) / root <= 1
    assert_scoped_window_calls(clock, (1, 3, 2))


def test_scoped_last_output_is_the_last_measured_instance_invocation_reference():
    clock = FakeScopedCudaClock()
    outputs = []

    def make_replay():
        replay = clock.make_replay()

        def run():
            replay.run()
            output = object()
            outputs.append((clock.invocations[-1], output))
            return output

        return timing.ScopedReplay(run=run, times=replay.times)

    result = collect_scoped(clock, make_replay=make_replay)

    assert len(outputs) == 6
    assert outputs[-1][0] == (2, 1, 6)
    assert result.last_output is outputs[-1][1]
    assert_scoped_window_calls(clock, (1, 3, 2))


def test_fixed_instrumentation_cost_is_visible_against_clean_timing_not_scaled_away():
    clean = collect(FakeCudaClock(replay_ms="0.032"), sample_count=2)
    clock = FakeScopedCudaClock(sample=lambda *_: {
        "qkv": 0.01, "attention": 0.02, "step5_layer_scoped": 0.036,
    })
    scoped = collect_scoped(clock, window_replays=clean.window_replays)

    assert clean.window_replays == (16, 16)
    assert clean.amortized_times == (0.032, 0.032)
    assert scoped.amortized_times["step5_layer_scoped"] == pytest.approx((0.036, 0.036))
    for index, duration in enumerate(scoped.window_durations_ms):
        outer_per_replay = duration / clean.window_replays[index]
        scoped_root = scoped.amortized_times["step5_layer_scoped"][index]
        assert outer_per_replay == pytest.approx(0.04)
        assert outer_per_replay - clean.amortized_times[index] == pytest.approx(0.008)
        assert scoped_root - clean.amortized_times[index] == pytest.approx(0.004)
        operator_sum = scoped.amortized_times["qkv"][index] + scoped.amortized_times["attention"][index]
        assert operator_sum / scoped_root == pytest.approx(5 / 6)
    assert len(scoped.per_replay_times["qkv"]) == 32
    assert_scoped_window_calls(clock, (16, 16))


@pytest.mark.parametrize("counts", [(), (1,), (0, 1), (-1, 1), (True, 1), (2.0, 1), ("2", 1), (1, None)])
def test_invalid_cohort_counts_fail_before_factory_or_events(counts):
    clock = FakeScopedCudaClock()
    with pytest.raises(ValueError, match="window_replays"):
        collect_scoped(clock, window_replays=counts)
    assert not clock.calls


@pytest.mark.parametrize("names", [(), ("",), (1,), ("qkv", "qkv"), ("step5_layer_scoped",)])
def test_invalid_operator_names_fail_before_factory_or_events(names):
    clock = FakeScopedCudaClock()
    with pytest.raises(ValueError, match="operator_names"):
        collect_scoped(clock, operator_names=names)
    assert not clock.calls


def edited_scoped_factory(clock, edit):
    def make_replay():
        replay = clock.make_replay()

        def times():
            sample = dict(replay.times())
            edit(sample)
            return sample

        return timing.ScopedReplay(run=replay.run, times=times)

    return make_replay


@pytest.mark.parametrize("change", ["missing_root", "missing_operator", "unexpected_operator"])
def test_scope_names_must_match_the_requested_operators_and_root(change):
    clock = FakeScopedCudaClock()

    def edit(sample):
        if change == "missing_root":
            del sample["step5_layer_scoped"]
        elif change == "missing_operator":
            del sample["attention"]
        else:
            sample["unrequested"] = 0.01

    with pytest.raises(ValueError, match="scope names"):
        collect_scoped(clock, make_replay=edited_scoped_factory(clock, edit))


@pytest.mark.parametrize("duration", [0, -0.1, float("nan"), float("inf"), -float("inf"), None, "0.1", True])
def test_scoped_root_must_be_finite_and_positive(duration):
    clock = FakeScopedCudaClock()
    make_replay = edited_scoped_factory(clock, lambda sample: sample.update(step5_layer_scoped=duration))
    with pytest.raises(ValueError, match="step5_layer_scoped.*positive"):
        collect_scoped(clock, make_replay=make_replay)


@pytest.mark.parametrize("duration", [-0.1, float("nan"), float("inf"), -float("inf"), None, "0.1", True])
def test_operator_duration_must_be_finite_and_nonnegative(duration):
    clock = FakeScopedCudaClock()
    make_replay = edited_scoped_factory(clock, lambda sample: sample.update(qkv=duration))
    with pytest.raises(ValueError, match="qkv.*nonnegative"):
        collect_scoped(clock, make_replay=make_replay)


def test_zero_operator_duration_is_retained():
    clock = FakeScopedCudaClock()
    make_replay = edited_scoped_factory(clock, lambda sample: sample.update(qkv=0.0))
    result = collect_scoped(clock, make_replay=make_replay)
    assert result.per_replay_times["qkv"] == (0.0,) * 6
    assert result.amortized_times["qkv"] == (0.0,) * 3


def test_exclusive_operator_sum_is_checked_against_each_matching_root_not_window_average():
    clock = FakeScopedCudaClock()

    def edit(sample):
        if len(clock.materialized) == 1:
            sample.update(qkv=0.03, attention=0.02, step5_layer_scoped=0.04)
        else:
            sample.update(qkv=0.01, attention=0.01, step5_layer_scoped=1.0)

    with pytest.raises(ValueError, match="Exclusive operator sum.*same-execution.*window 0.*replay 0"):
        collect_scoped(clock, window_replays=(2, 2), make_replay=edited_scoped_factory(clock, edit))
    assert len(clock.materialized) == 1


def test_repeated_factory_instance_is_rejected_before_measured_windows():
    clock = FakeScopedCudaClock()
    replay = clock.make_replay()
    factory_calls = []

    def make_replay():
        factory_calls.append(1)
        return replay

    with pytest.raises(ValueError, match="distinct.*ScopedReplay"):
        collect_scoped(clock, make_replay=make_replay)
    assert len(factory_calls) == 3
    assert not clock.events and not clock.windows


@pytest.mark.parametrize("duration", [0.0, -0.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_scoped_outer_window_duration_is_rejected(monkeypatch, duration):
    clock = FakeScopedCudaClock()
    elapsed_time = FakeEvent.elapsed_time

    def measured(self, end):
        elapsed_time(self, end)
        return duration

    monkeypatch.setattr(FakeEvent, "elapsed_time", measured)
    with pytest.raises(ValueError, match="duration must be positive and finite"):
        collect_scoped(clock)
