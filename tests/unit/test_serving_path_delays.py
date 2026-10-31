"""Serving front-end delays through real online DES runs and flat configs."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from frontier.config import MetricsConfig, SimulationConfig, global_vars
from frontier.config.flat_dataclass import create_flat_dataclass
from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig

HANDOFF = "analytical_kv_cache_transfer_config_request_handoff_delay_ms"


@pytest.fixture(autouse=True)
def reset_globals():
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


def test_delay_defaults_are_zero(tmp_path):
    config = SimulationConfig(metrics_config=MetricsConfig(output_dir=str(tmp_path / "metrics")))
    assert config.request_ingress_delay_ms == 0.0
    assert config.kv_cache_transfer_config.request_handoff_delay_ms == 0.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["request_ingress_delay_ms", "request_handoff_delay_ms"])
def test_invalid_delays_rejected(field, value):
    constructor = SimulationConfig if field == "request_ingress_delay_ms" else AnalyticalKVCacheTransferConfig
    with pytest.raises(ValueError, match=field):
        constructor(**{field: value})


@pytest.mark.parametrize("architecture", ["co-location", "pd-disaggregation", "pd-af-disaggregation"])
def test_ingress_rejected_in_offline_batch_mode(architecture):
    with pytest.raises(ValueError, match="request_ingress_delay_ms applies to request arrival events"):
        SimulationConfig(sys_arch=architecture, enable_parallel_clusters=False,
                         request_ingress_delay_ms=1.0)


def test_flat_flags_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "frontier.main", "--simulation_mode", "online", "--request_ingress_delay_ms", "125",
        "--" + HANDOFF, "250", "--metrics_config_output_dir", str(tmp_path / "metrics"),
    ])
    config = SimulationConfig.create_from_cli_args()
    assert config.request_ingress_delay_ms == 125.0
    assert config.kv_cache_transfer_config.request_handoff_delay_ms == 250.0
    assert config.__flat_config__.request_ingress_delay_ms == 125.0
    assert getattr(config.__flat_config__, HANDOFF) == 250.0


@pytest.mark.parametrize("connector", ["none", "p2p_nccl", "nixl"])
def test_handoff_delay_independent_of_connector(connector):
    config = AnalyticalKVCacheTransferConfig(kv_connector=connector, request_handoff_delay_ms=250.0)
    assert config.request_handoff_delay_ms == 250.0


def test_parallel_arrivals_use_delayed_factory():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from frontier.simulator import Simulator
    from frontier.types import ClusterType

    requests = [SimpleNamespace(arrived_at=0.5), SimpleNamespace(arrived_at=1.0)]
    events = []
    simulator = object.__new__(Simulator)
    simulator._config = SimpleNamespace(simulation_mode="online", request_ingress_delay_ms=125.0,
                                        is_disaggregated_mode=lambda: True)
    simulator._request_generator = SimpleNamespace(generate=lambda: requests)
    simulator._metric_store = SimpleNamespace(register_total_requests=Mock())
    simulator._cluster_simulators = {ClusterType.PREFILL: SimpleNamespace(add_event=events.append)}
    simulator._init_parallel_events()
    assert [event.time for event in events] == [0.625, 1.125]
    assert [request.arrived_at for request in requests] == [0.5, 1.0]
    assert all(event.get_target_cluster() == ClusterType.PREFILL for event in events)


def run_case(root, architecture, fields):
    """One isolated process per run preserves entity IDs for byte comparisons."""
    from frontier.events import RequestArrivalEvent
    from frontier.simulator import Simulator
    from frontier.types import ClusterType
    from frontier.utils.random import set_seeds

    root = Path(root)
    root.mkdir(parents=True)
    trace = root / "trace.csv"
    trace.write_text("arrived_at,num_prefill_tokens,num_decode_tokens\n0.5,8,3\n")
    flat = create_flat_dataclass(SimulationConfig)(
        simulation_mode="online", sys_arch=architecture, enable_parallel_clusters=False,
        request_generator_config_type="trace_replay", trace_request_generator_config_trace_file=str(trace),
        replica_config_model_name="meta-llama/Llama-2-7b-hf", replica_scheduler_config_type="vllm_v1",
        cc_backend_config_type="analytical", decode_cuda_graph_mode="none",
        random_forrest_execution_time_predictor_config_enable_dummy_mode=True,
        random_forrest_execution_time_predictor_config_dummy_execution_time_ms=125.0,
        metrics_config_output_dir=str(root / "metrics"), metrics_config_cache_dir=str(root / "cache"),
        metrics_config_run_id="serving_delay", metrics_config_store_plots=False,
        metrics_config_enable_metrics_ground_truth_trace=True,
        # Llama2's 8-token payload = 4,194,304 bytes. This bandwidth gives
        # 33,554.432 bytes/ms, hence exactly 125 ms wire time, with zero latency.
        analytical_kv_cache_transfer_config_network_bandwidth_gbps=0.268435456,
        analytical_kv_cache_transfer_config_network_latency_ms=0.0,
        **({"cluster_config_prefill_cluster_num_replicas": 1,
            "cluster_config_decode_cluster_num_replicas": 1} if architecture == "pd-disaggregation"
           else {"cluster_config_num_replicas": 1}),
        **fields,
    )
    config = flat.reconstruct_original_dataclass()
    config.__flat_config__ = flat
    set_seeds(config.seed)
    simulator = Simulator(config)
    events = [event for _, event in simulator._event_queue if isinstance(event, RequestArrivalEvent)]
    assert len(events) == 1
    arrival_time = events[0].time
    simulator.run()
    request, = simulator._all_requests
    role = ClusterType.PREFILL if architecture == "pd-disaggregation" else ClusterType.MONOLITHIC
    facts = dict(arrival_event_time=arrival_time, arrived_at=request.arrived_at,
                 queue_entry_time=request._cluster_arrival_times[role][0], ttft=request.ttft,
                 prefill_end=request.prefill_completed_at,
                 transfer_start=request.kv_cache_transfer_start_time,
                 transfer_end=request.kv_cache_transfer_end_time,
                 decode_first_token=request.decode_first_token_completed_at,
                 metrics_dir=config.metrics_config.output_dir)
    (root / "facts.json").write_text(json.dumps(facts))


def online_run(root, architecture="pd-disaggregation", **fields):
    command = [sys.executable, str(Path(__file__).resolve()), str(root), architecture, json.dumps(fields)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((root / "facts.json").read_text())


@pytest.mark.parametrize("architecture", ["co-location", "pd-disaggregation"])
def test_online_ingress_delays_queue_entry_and_ttft(tmp_path, architecture):
    baseline = online_run(tmp_path / "baseline", architecture)
    delayed = online_run(tmp_path / "delayed", architecture, request_ingress_delay_ms=125.0)
    assert delayed["arrival_event_time"] == delayed["arrived_at"] + 0.125
    assert delayed["arrived_at"] == baseline["arrived_at"] == 0.5
    assert delayed["queue_entry_time"] == delayed["arrival_event_time"]
    assert delayed["ttft"] - baseline["ttft"] == 0.125
    rows = [json.loads(line) for line in (Path(delayed["metrics_dir"]) / "metrics_ground_truth.jsonl").read_text().splitlines()]
    arrival, = [row for row in rows if row["event_type"] == "request_arrival"]
    assert arrival["time_s"] == 0.625
    assert arrival["arrived_at"] == 0.5


def test_pdd_online_handoff_shifts_transfer_and_decode(tmp_path):
    baseline = online_run(tmp_path / "baseline")
    delayed = online_run(tmp_path / "delayed", request_ingress_delay_ms=125.0, **{HANDOFF: 250.0})
    # One request, PP1/DP1, no graphs/CPU costs or preemption; binary-rational
    # costs and delays prevent queueing, step alignment and rounding ambiguity.
    assert delayed["prefill_end"] - baseline["prefill_end"] == 0.125
    for key in ("transfer_start", "transfer_end", "decode_first_token"):
        assert delayed[key] - baseline[key] == 0.375
    assert delayed["transfer_start"] - delayed["prefill_end"] == 0.25
    assert baseline["transfer_end"] - baseline["transfer_start"] == 0.125
    assert delayed["transfer_end"] - delayed["transfer_start"] == 0.125


@pytest.mark.parametrize("architecture", ["pd-disaggregation", "co-location"])
def test_explicit_zero_delays_preserve_output_bytes(tmp_path, architecture):
    baseline = online_run(tmp_path / "baseline", architecture)
    explicit = online_run(tmp_path / "explicit", architecture, request_ingress_delay_ms=0.0, **{HANDOFF: 0.0})
    for name in ("request_metrics.csv", "system_metrics.json"):
        assert (Path(baseline["metrics_dir"]) / name).read_bytes() == (Path(explicit["metrics_dir"]) / name).read_bytes()


if __name__ == "__main__":
    run_case(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
