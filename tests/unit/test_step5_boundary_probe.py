"""CPU protocol/packaging contracts; no native GPU or platform operation."""

import copy
from contextlib import nullcontext
from fractions import Fraction
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.unit.test_step5_attention_timing import AsyncCudaClock, AsyncEvent


SCRIPTS = Path(__file__).resolve().parents[1] / "integration/step5_profiling"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import boundary_probe
    import boundary_probe_job

    return boundary_probe, boundary_probe_job


class BoundaryClock(AsyncCudaClock):
    def __init__(self, event_cost):
        super().__init__()
        self.event_cost = Fraction(event_cost)

    def event(self, *, enable_timing):
        assert enable_timing
        event = BoundaryEvent(self, len(self.events))
        self.events.append(event)
        self.calls.append(("event", event.index))
        return event


class BoundaryEvent(AsyncEvent):
    primed = False

    def record(self):
        super().record()
        self.primed = True
        self.clock.device_ms += self.clock.event_cost


def settings(probe):
    return json.loads(probe.CONFIG_PATH.read_text())


def kernels():
    item = {"kernel_names": ["native_bf16_gemm", "native_swa"], "kernel_count": 135,
            "memory_copy_names": [], "memory_copy_count": 0}
    return {"A": copy.deepcopy(item), "B": copy.deepcopy(item)}


def execute(probe, measurement, event_cost="0"):
    clock = BoundaryClock(event_cost)
    events = probe.events_for(clock.torch, measurement["replays"])
    observed, validations, recorded = [], [], []

    def run():
        assert all(event.primed for event in events)
        return clock.run()

    def observe():
        observed.append(len(clock.outputs))
        return {"fields": {"clocks.current.sm": "1980"}}

    rows = probe.collect(run, clock.prepare, events, measurement, observe_clock=observe,
                         validate=lambda value: validations.append(value), record=recorded.append)
    return rows, clock, observed, validations, recorded


@pytest.mark.parametrize("replays", [2, 5, 15, 30, 50])
@pytest.mark.parametrize("windows", [2, 4, 6, 10, 12])
@pytest.mark.parametrize("event_cost", ["0", "0.002"])
def test_fifty_protocol_settings_preserve_every_window_and_exact_calls(modules, replays, windows, event_cost):
    probe, _ = modules
    measurement = {"replays": replays, "windows_per_cohort": windows, "min_window_ms": .5}
    rows, clock, observed, validations, recorded = execute(probe, measurement, event_cost)
    assert len(rows) == 2 * windows and rows == recorded
    assert len(clock.events) == replays + 3
    assert len(clock.outputs) == 2 * windows * replays
    assert len(observed) == 4 * windows and len(validations) == 2 * windows
    assert len({row["replay_callable_id"] for row in rows}) == 1
    assert [row["order"] for row in rows] == [order for _ in range(windows // 2) for order in ("AB", "AB", "BA", "BA")]
    for row in rows:
        assert row["replays"] == replays
        assert len(row["inner_durations_ms"]) == (replays if row["cohort"] == "B" else 0)
        assert row["minimum_window_met"] == (row["outer_duration_ms"] >= .5)
    calls = clock.calls
    first_prepare = next(i for i, call in enumerate(calls) if call[0] == "prepare")
    assert calls[:first_prepare] == ([('event', i) for i in range(replays + 3)]
                                     + [('record', i) for i in range(replays + 3)]
                                     + [('synchronize', replays + 2)])
    expected_records = 2 * windows * (replays + 3 + 2) + windows * (replays + 1)
    assert sum(call[0] == "record" for call in calls) == expected_records
    assert sum(call[0] == "prepare" for call in calls) == len(rows)
    assert sum(call[0] == "synchronize" for call in calls) == 2 * len(rows)
    assert validations[-1] is clock.outputs[-1]


def test_predeclared_mechanism_detects_boundary_cost_not_just_old_failure(modules):
    probe, _ = modules
    config = settings(probe)
    control, *_ = execute(probe, config["measurement"], "0")
    affected, *_ = execute(probe, config["measurement"], "0.002")
    zero = probe.evaluate(control, kernels(), config)
    biased = probe.evaluate(affected, kernels(), config)
    assert not zero["H1_compatible_with_prior_gap_in_both_orders"]
    assert biased["H1_compatible_with_prior_gap_in_both_orders"]
    assert all(1.8 < value < 2.3 for value in biased["outer_delta_us_by_order"].values())
    assert not biased["H2_order_reversal_signature"]
    assert not biased["H3_kernel_identity_difference"]
    assert biased["H2_clock_observations"]["A"]["duration_pearson_r"] is None
    assert not biased["admitted"]


def test_order_and_kernel_alternatives_have_separate_falsifiable_signals(modules):
    probe, _ = modules
    config = settings(probe)
    rows, *_ = execute(probe, config["measurement"])
    for row in rows:
        row["outer_per_replay_us"] = 34 + row["position"] * 2
    result = probe.evaluate(rows, kernels(), config)
    assert result["H2_order_reversal_signature"]
    assert not result["H1_compatible_with_prior_gap_in_both_orders"]
    differing = kernels()
    differing["B"]["kernel_names"].append("different_native_work")
    result = probe.evaluate(rows, differing, config)
    assert result["H3_kernel_identity_difference"]
    assert not result["protocol_checks"]["same_correlated_kernel_work"]


def test_clock_correlations_are_reported_per_cohort_without_correcting_times(modules):
    probe, _ = modules
    config = settings(probe)
    rows, *_ = execute(probe, config["measurement"])
    for index, row in enumerate(rows):
        row["clock_before"] = {"fields": {"clocks.current.sm": str(1000 + index * 10)}}
        row["outer_per_replay_us"] = 50 - index * .1
    original = copy.deepcopy(rows)
    result = probe.evaluate(rows, kernels(), config)
    assert result["H2_clock_observations"]["A"]["duration_pearson_r"] == pytest.approx(-1)
    assert result["H2_clock_observations"]["B"]["duration_pearson_r"] == pytest.approx(-1)
    assert rows == original


@pytest.mark.parametrize("windows", [0, 1, 3, 11, True])
def test_unbalanced_protocol_is_rejected(modules, windows):
    probe, _ = modules
    with pytest.raises(ValueError, match="balanced"):
        probe.schedule(windows)


def test_short_windows_are_retained_without_n_adaptation_or_hypothesis_pass(modules):
    probe, _ = modules
    config = settings(probe)
    config["measurement"]["min_window_ms"] = 10
    rows, clock, *_ = execute(probe, config["measurement"], "0.002")
    result = probe.evaluate(rows, kernels(), config)
    assert len(rows) == 20 and len(clock.outputs) == 300
    assert not any(row["minimum_window_met"] for row in rows)
    assert not result["protocol_checks"]["minimum_windows"]
    assert not result["H1_compatible_with_prior_gap_in_both_orders"]


def test_cpu_dry_run_binds_existing_cell_and_never_imports_native_wrapper(modules, capsys, monkeypatch):
    probe, _ = modules
    monkeypatch.setitem(sys.modules, "frontier.profiling.step5.vllm_wrapper", None)
    probe.main(["--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["cell"] == {"global_layer_id": 0, "query_lens": [1], "context_lens": [6144],
                                "phase": "decode", "graph_mode": "full", "has_attention_metadata": True,
                                "block_table_columns": None, "dummy_descriptor": None,
                                "physical_query_rows": 1}
    assert (result["measured_replays"], result["trace_replays"], result["event_handles"], result["native_graphs"]) == (300, 30, 18, 1)
    assert result["total_graph_replays"] == 331


def test_description_reuses_launcher_with_six_hundred_second_cap(modules, tmp_path):
    probe, job = modules
    manifest = job.description(tmp_path / "stage")
    assert manifest["timeout_seconds"] == 600 and manifest["max_gpu_hours"] == pytest.approx(1 / 6)
    assert (manifest["gpus"], manifest["gpu_tag"], manifest["charged_group"]) == (1, "H200", "step_main")
    assert "--timeout-seconds 600" in manifest["submission_command"]
    assert shlex.split(manifest["submission_command"])[-1] == manifest["worker_command"]
    assert "--verify" in manifest["worker_command"]
    assert "linear_op.main" not in manifest["worker_command"]
    assert not manifest["admitted"] and not (tmp_path / "stage").exists()
    subprocess.run(["bash", "-n"], input=manifest["worker_command"], text=True, check=True)


def test_staging_refuses_uncommitted_source_before_creating_directory(modules, tmp_path, monkeypatch):
    _, job = modules
    monkeypatch.setattr(job, "source_state", lambda: ("CPU_FIXTURE_NOT_A_COMMIT", False))
    with pytest.raises(ValueError, match="clean source"):
        job.stage_experiment(tmp_path / "stage", dependency_site=tmp_path)
    assert not (tmp_path / "stage").exists()


def test_clock_observation_uses_only_read_only_query_and_retains_raw_values(modules, monkeypatch):
    probe, _ = modules
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "now, GPU-fixture, H200, 1980, 1980, 3200, P0, 100, 40\n", "")

    monkeypatch.setattr(probe.subprocess, "run", run)
    observed = probe.observe_clocks()
    assert observed["fields"]["clocks.current.sm"] == "1980"
    assert observed["stdout"] and observed["returncode"] == 0
    assert calls == [["nvidia-smi", "--id=0", "--query-gpu=" + ",".join(probe.CLOCK_FIELDS), "--format=csv,noheader,nounits"]]


def test_native_wiring_owns_exactly_one_graph_and_reuses_it_for_both_traces(modules, tmp_path, monkeypatch):
    probe, _ = modules
    import frontier.profiling.step5.device_activity as activity
    import frontier.profiling.step5.replay as replay
    import frontier.profiling.step5.vllm_wrapper as native
    import frontier.profiling.utils.record_function_tracer as tracer_module

    clock = BoundaryClock("0.002")
    calls = {"captures": 0, "forwards": 0, "validations": 0, "trace_replays": [], "closed": False}

    class Tensor:
        def __init__(self, size=1):
            self.size = size

        def clone(self):
            return Tensor(self.size)

        def data_ptr(self):
            return id(self)

    torch = clock.torch
    torch.no_grad = nullcontext
    torch.manual_seed = lambda seed: calls.update(seed=seed)
    torch.randn = lambda shape, **kwargs: Tensor(shape[0])
    torch.bfloat16 = "CPU_FIXTURE_BF16"
    torch.cuda.synchronize = lambda: None
    torch.profiler = SimpleNamespace(record_function=lambda name: nullcontext())

    class Wrapper:
        def __init__(self, **kwargs):
            self.torch, self.native = torch, SimpleNamespace()
            self.layer = SimpleNamespace(attn=SimpleNamespace(impl=SimpleNamespace()))
            self.runtime_signature = "CPU_FIXTURE_NOT_NATIVE_EVIDENCE"
            self.state, self.empty_state, self.value = [Tensor()], 0, 0
            (kwargs['trace_dir']).mkdir(parents=True)
            calls['wrapper'] = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls['closed'] = True

        def _set_current(self, native):
            return nullcontext()

        def _restore(self, value):
            self.value = value
            clock.prepare()

        def _snapshot(self):
            return self.value

        def _metadata(self, cell):
            return Tensor(), Tensor(), None, cell.graph_mode

        def _forward(self, hidden, prepared, **kwargs):
            calls['forwards'] += 1
            self.value = hidden.size + 1
            return Tensor(hidden.size)

        def _validate_output_state(self, actual, expected, state):
            assert actual.size == expected.size and self.value == state
            calls['validations'] += 1

    def capture(torch, native, layer, forward, graph_mode):
        calls['captures'] += 1
        output = forward()

        def run():
            clock.run()
            calls['wrapper'].value = output.size + 1
            return output

        return run

    class Tracer:
        def __init__(self, output):
            self.trace_path = None

        def __enter__(self):
            self.begin = len(clock.outputs)
            return self

        def __exit__(self, *args):
            calls['trace_replays'].append(len(clock.outputs) - self.begin)
            Path(self.trace_path).write_text('{}')

    def parsed(path, operators, iteration_prefix):
        assert operators == ('native_replay',) and iteration_prefix == 'boundary_probe/'
        return copy.deepcopy(kernels()['A'])

    monkeypatch.setattr(native, 'VllmStep5AttentionWrapper', Wrapper)
    monkeypatch.setattr(replay, 'capture_native_replay', capture)
    monkeypatch.setattr(tracer_module, 'RecordFunctionTracer', Tracer)
    monkeypatch.setattr(activity, 'parse_device_activity', parsed)
    monkeypatch.setattr(probe, 'observe_clocks', lambda: {'fields': {'clocks.current.sm': '1980'}})
    probe.native_run(settings(probe), 'CPU_FIXTURE_NOT_A_MODEL_PATH', tmp_path)
    result = json.loads((tmp_path / 'results.json').read_text())
    assert calls['captures'] == 1 and calls['forwards'] == 6 and calls['validations'] == 23
    assert calls['trace_replays'] == [15, 15] and len(clock.outputs) == 331
    assert calls['closed'] and calls['seed'] == 0
    assert len(result['windows']) == 20 and result['evaluation']['H1_compatible_with_prior_gap_in_both_orders']
    assert len((tmp_path / 'windows.jsonl').read_text().splitlines()) == 20
