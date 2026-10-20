"""Measured collective lookup, coverage, pooling and opt-in configuration."""

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from frontier.cc_backend import (
    CCBackendFactory,
    MeasuredCCBackend,
    MeasuredCCBackendConfig,
    MeasuredCCOutOfRangeError,
)
from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
from frontier.config import AnalyticalKVCacheTransferConfig, ClusterConfig
from frontier.config.cluster_role_config import _get_cc_backend_configs
from frontier.entities.cluster import Cluster
from frontier.types import CCBackendType, ClusterType

ROOT = Path(__file__).resolve().parents[2]


def _row(size=100, time_us=10.0):
    return dict(
        scenario="eager_uniform_t8_r1", call="embedding_all_reduce",
        op="all_reduce", group="tp", group_size=8,
        max_input_bytes=size, median_us=time_us, cuda_graph=False,
        launch_median_us=9999.0,
    )


def _backend(tmp_path, rows):
    path = tmp_path / "summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return MeasuredCCBackend(
        MeasuredCCBackendConfig(summary_csv_path=str(path)),
        ClusterType.PREFILL, "h800", "h800_dgx", 32,
    )


@pytest.mark.parametrize("size,expected", [(100, .01), (200, .02), (400, .04)])
def test_lookup_and_log_log_interpolation(tmp_path, size, expected):
    backend = _backend(tmp_path, [_row(), _row(400, 40.0)])
    assert backend.predict_comm_cost(
        "allreduce", data_size_bytes=size, num_devices=8, comm_domain="ATTN_TP",
        cluster_type=ClusterType.DECODE,
    ) == pytest.approx(expected)
    assert not hasattr(backend, "estimate_intra_server_allreduce_launch_overhead_ms")


@pytest.mark.parametrize("size", [-1, 0, 99, 401])
def test_outside_range_is_typed_without_extrapolation(tmp_path, size):
    backend = _backend(tmp_path, [_row(), _row(400, 40.0)])
    with pytest.raises(MeasuredCCOutOfRangeError) as caught:
        backend.predict_allreduce(size, 8, comm_domain="ATTN_TP")
    assert caught.value.curve_key == ("all_reduce", "tp", 8)
    assert caught.value.requested_bytes == size
    assert caught.value.covered_range == (100, 400)


@pytest.mark.parametrize("operation,domain,size", [
    ("allreduce", "ATTN_TP", 16), ("allreduce", "MOE_TP", 8),
    ("allgather", "ATTN_TP", 8), ("all_to_all", "EP", 32),
    ("broadcast", "ATTN_TP", 8), ("allreduce", "DP", 4),
    ("allreduce", None, 8),
])
def test_unknown_curve_is_typed(tmp_path, operation, domain, size):
    backend = _backend(tmp_path, [_row()])
    with pytest.raises(MeasuredCCOutOfRangeError) as caught:
        backend.predict_comm_cost(
            operation, data_size_bytes=100, num_devices=size, comm_domain=domain,
        )
    assert caught.value.covered_range is None
    with pytest.raises(MeasuredCCOutOfRangeError):
        backend.predict_send_recv(100, comm_domain="PP")


@pytest.mark.parametrize("operation,domain,op,group,devices", [
    ("allreduce", "ATTN_TP", "all_reduce", "tp", 8),
    ("allgather", "ATTN_TP", "all_gather", "tp", 8),
    ("reduce_scatter", "ATTN_TP", "reduce_scatter", "tp", 8),
    ("allgather", "EP", "all_gatherv", "ep", 32),
    ("allgather", "MOE_EP", "all_gatherv", "ep", 32),
    ("reduce_scatter", "EP", "reduce_scatterv", "ep", 32),
    ("reduce_scatter", "MOE_EP", "reduce_scatterv", "ep", 32),
])
def test_domain_mapping(tmp_path, operation, domain, op, group, devices):
    row = _row()
    row.update(op=op, group=group, group_size=devices)
    backend = _backend(tmp_path, [row])
    assert backend.predict_comm_cost(
        operation, data_size_bytes=100, num_devices=devices, comm_domain=domain,
    ) == .01
    assert backend.describe_curves()[0]["key"] == (op, group, devices)


def test_pool_graph_eager_calls_passes_and_ignore_host_launch_time(tmp_path):
    rows = [_row(), _row(100, 20), _row(100, 60), _row(100, 80)]
    rows[1].update(cuda_graph=True)
    rows[2].update(call="dense_layer_all_reduce")
    rows[3].update(call="dense_layer_all_reduce", cuda_graph=True)
    for i, row in enumerate(rows):
        row["pass"] = i % 2
    backend = _backend(tmp_path, rows)
    assert backend.predict_allreduce(100, 8, comm_domain="ATTN_TP") == .04
    description = backend.describe_curves()[0]
    assert description["point_count"] == 1
    assert description["calls"] == ("dense_layer_all_reduce", "embedding_all_reduce")
    assert description["source_files"] == (str(tmp_path / "summary.csv"),)
    assert description["time_range_us"] == (40, 40)
    assert description["loo_absolute_relative_error"]["median"] is None


def test_uneven_and_cpu_holdouts_and_logits_do_not_fit_the_curves(tmp_path):
    rows = [_row(), _row(100, 1000), _row(100, 2000), _row(100, 3000)]
    rows[1]["scenario"] = "eager_single_active_t64_r1"
    rows[2].update(group="dp", op="cpu_all_reduce", call="dp_batch_sync_all_reduce")
    rows[3].update(op="all_gather", call="logits_all_gather")
    backend = _backend(tmp_path, rows)
    assert len(backend.describe_curves()) == 1
    assert backend.predict_allreduce(100, 8, comm_domain="ATTN_TP") == .01


def test_uneven_dispatch_uses_caller_supplied_largest_rank_bytes(tmp_path):
    rows = [_row(), _row(400, 40)]
    for row in rows:
        row.update(op="all_gatherv", group="ep", group_size=32)
    backend = _backend(tmp_path, rows)
    rank_bytes = [100, 100, 100, 400]
    assert backend.predict_allgather(max(rank_bytes), 32, comm_domain="EP") == .04
    # K2 owns row geometry and must use the same typed error before a uniform
    # reduce-scatter lookup when unequal SP rows select the per-root NCCL kernel.
    error = MeasuredCCOutOfRangeError(
        ("reduce_scatterv", "ep", 32), sum(rank_bytes), (100, 400),
        reason="unequal SP rows select an unmeasured combine kernel",
    )
    assert "unmeasured combine kernel" in str(error)


def test_logit_option_b_uses_hidden_state_curve(tmp_path):
    rows = [_row(), _row(400, 40), _row(200, 2000)]
    for row in rows:
        row.update(op="all_gather", call="moe_layer_input_all_gather")
    rows[-1]["call"] = "logits_all_gather"
    backend = _backend(tmp_path, rows)
    assert backend.describe_curves()[0]["point_count"] == 2
    assert backend.predict_allgather(200, 8, comm_domain="ATTN_TP") == pytest.approx(.02)


def test_nonmonotone_points_are_logged_not_smoothed(tmp_path, monkeypatch):
    from frontier.cc_backend.backends.measured_cc_backend import logger

    warnings = []
    monkeypatch.setattr(logger, "warning", lambda *args: warnings.append(args))
    backend = _backend(tmp_path, [_row(), _row(200, 5), _row(400, 40)])
    assert backend.predict_allreduce(200, 8, comm_domain="ATTN_TP") == .005
    assert "Non-monotone measured collective" in warnings[0][0]
    loo = backend.describe_curves()[0]["loo_absolute_relative_error"]
    assert loo == pytest.approx({"point_count": 1, "median": 3.0, "p90": 3.0, "max": 3.0})


def test_conflicting_call_meanings_are_rejected(tmp_path):
    rows = [_row(), _row(200, 20)]
    rows[1]["op"] = "all_gather"
    with pytest.raises(ValueError, match="ambiguous call"):
        _backend(tmp_path, rows)


@pytest.mark.parametrize("change", [
    {"op": "all_gatherv"}, {"group_size": 1}, {"max_input_bytes": 0},
    {"median_us": 0}, {"median_us": "nan"}, {"median_us": "inf"},
])
def test_invalid_measured_rows_are_rejected(tmp_path, change):
    row = _row()
    row.update(change)
    with pytest.raises(ValueError):
        _backend(tmp_path, [row])


def test_missing_columns_and_no_curves_fail(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        _backend(tmp_path, [{"median_us": 1}])
    row = _row()
    row["group"] = "dp"
    with pytest.raises(ValueError, match="no device collective curves"):
        _backend(tmp_path, [row])
    with pytest.raises(FileNotFoundError):
        MeasuredCCBackend(
            MeasuredCCBackendConfig(summary_csv_path=str(tmp_path / "missing.csv")),
            ClusterType.DECODE, "h800", "h800_dgx", 32,
        )


@pytest.mark.parametrize("config_class", _get_cc_backend_configs()[1:])
def test_cluster_selection_uses_config_type(tmp_path, monkeypatch, config_class):
    seen = {}
    sentinel = object()

    def create(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(CCBackendFactory, "create", create)
    # Replica materialization is covered by the existing layout tests. This
    # isolates selection, including the internal aiconfigurator configuration.
    cluster = object.__new__(Cluster)
    cluster._config = SimpleNamespace(cc_backend_config=config_class(), replica_config=None)
    cluster._cluster_type = ClusterType.DECODE
    assert cluster._initialize_cc_backend() is sentinel
    assert seen["backend_type"] == config_class.get_type()


@pytest.mark.parametrize("role", ["prefill", "decode", "decode_attn", "decode_ffn"])
def test_role_selection_preserves_measured_path_and_defaults(role):
    base = MeasuredCCBackendConfig(summary_csv_path="/curves/summary.csv")
    config = object.__new__(ClusterConfig)
    config.cc_backend_config = base
    assert config._create_cc_backend_config_for_cluster(role) == base
    setattr(config, f"{role}_cc_backend_config_type", "measured")
    assert config._create_cc_backend_config_for_cluster(role) == base
    config.cc_backend_config = AnalyticalCCBackendConfig()
    assert isinstance(config._create_cc_backend_config_for_cluster(role), MeasuredCCBackendConfig)


def test_config_cli_factory_and_path_template(tmp_path):
    args = [
        "--sys_arch", "pd-disaggregation", "--no-enable_parallel_clusters",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", "1",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--vllm_v1_scheduler_config_num_blocks", "128",
        "--cc_backend_config_type", "measured",
        "--measured_cc_backend_config_summary_csv_path", "{profiling_data_dir}/{NETWORK_DEVICE}/summary.csv",
        "--cluster_config_prefill_cc_backend_config_type", "measured",
        "--cluster_config_decode_cc_backend_config_type", "measured",
        "--metrics_config_output_dir", str(tmp_path / "metrics"),
        "--no-metrics_config_write_metrics",
    ]
    code = """
import json
from frontier.config import SimulationConfig
c = SimulationConfig.create_from_cli_args().cluster_config
print(json.dumps([c.cc_backend_config.get_name(), c.cc_backend_config.summary_csv_path,
    c._create_cc_backend_config_for_cluster('prefill').get_name(),
    c._create_cc_backend_config_for_cluster('decode').get_name()]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, *args], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == [
        "measured", "{profiling_data_dir}/{NETWORK_DEVICE}/summary.csv", "measured", "measured",
    ]
    backend = CCBackendFactory.create_from_str(
        "measured", MeasuredCCBackendConfig(), ClusterType.DECODE, "h800", "h800_dgx", 32,
    )
    assert isinstance(backend, MeasuredCCBackend)
    assert CCBackendType.MEASURED.value == 6
    assert CCBackendFactory.is_registered(CCBackendType.MEASURED)
    assert "measured" in CCBackendFactory.get_available_backends()


def test_checked_in_curves_match_producer_medians_and_ranges():
    backend = MeasuredCCBackend(
        MeasuredCCBackendConfig(), ClusterType.PREFILL, "h800", "h800_dgx", 32,
    )
    rows = list(csv.DictReader((ROOT / "data/profiling/network/h800_dgx/summary.csv").open()))
    assert len(rows) == 971
    expected = {
        ("all_reduce", "tp", 8): (14, (12288, 100663296)),
        ("all_gather", "tp", 8): (12, (12288, 25165824)),
        ("reduce_scatter", "tp", 8): (11, (98304, 100663296)),
        ("all_gatherv", "ep", 32): (11, (12352, 12648448)),
        ("reduce_scatterv", "ep", 32): (11, (393216, 402653184)),
    }
    assert {c["key"]: (c["point_count"], c["byte_range"]) for c in backend.describe_curves()} == expected
    assert all("logits_all_gather" not in c["calls"] for c in backend.describe_curves())
    from statistics import median
    times = [float(r["median_us"]) for r in rows if r["op"] == "all_gatherv"
             and "single_active" not in r["scenario"] and int(r["max_input_bytes"]) == 12352]
    assert backend.predict_allgather(12352, 32, comm_domain="EP") == median(times) / 1000
    # Maximum measured prefill rows; a larger dispatch cannot silently extrapolate.
    assert math.isfinite(backend.predict_allgather(12648448, 32, comm_domain="EP"))
    with pytest.raises(MeasuredCCOutOfRangeError):
        backend.predict_allgather(12648448 + 12352, 32, comm_domain="EP")
    # R=512 logits remain inside the hidden-state byte range under option B.
    assert math.isfinite(backend.predict_allgather(19824640, 8, comm_domain="ATTN_TP"))


def test_glm_run_uses_k5_nixl_values_without_changing_global_defaults(tmp_path):
    path = ROOT / "tests/integration/run_glm53_structural_pd.py"
    spec = importlib.util.spec_from_file_location("glm53_pd_run", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    args = driver.command("offline", tmp_path)
    assert args[args.index("--analytical_kv_cache_transfer_config_network_bandwidth_gbps") + 1] == "126.2"
    assert args[args.index("--analytical_kv_cache_transfer_config_network_latency_ms") + 1] == "0.0647"
    default = AnalyticalKVCacheTransferConfig()
    assert default.network_bandwidth_gbps == 100
    assert default.network_latency_ms == 0.1
    rows = list(csv.DictReader((ROOT / "data/profiling/network/h800_dgx/nixl_summary.csv").open()))
    assert len(rows) == 186
    errors = []
    for row in rows:
        if int(row["active_pairs"]) != 8:
            continue
        inflight = int(row["requests_in_flight"])
        measured_ms = float(row["complete_median_us"]) / inflight / 1000
        predicted_ms = .0647 + int(row["bytes_per_reader"]) / inflight / (126.2 * 1e9 / 8 / 1000)
        errors.append(predicted_ms / measured_ms - 1)
    assert min(errors) >= -.04
    assert max(errors) <= .024
