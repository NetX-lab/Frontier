"""Kernel-gap pricing of kernel-only steps (ticket 33 decisions T33-G7..G12)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

import frontier.config  # noqa: F401  (imports the operator registry in dependency order)
from frontier.attention.families import get_attention_family
from frontier.entities import StageExecutionTime
from frontier.entities.time_components import AttentionTime
from frontier.execution_time_predictor.kernel_gap import (
    KernelCountTraining,
    KernelGapPricing,
    kernel_count_column,
    load_kernel_gap_ms,
    read_kernel_gap_ms,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.operators.families import COMM_FAMILY
from frontier.scheduler.utils.expert_parallel import get_ep_phase_times_ms
from frontier.types import ClusterType, MeasurementType
from tests.unit.test_stage_finalized_contract import _layer
from tests.unit.test_stage_reporting_contract import _isolated_simulation_globals, _schedule, _store  # noqa: F401

GAP_MS = {"eager": 0.002, "cuda_graph": 0.0005}


def _write_gap_table(path, rows):
    pd.DataFrame(rows, columns=["execution_mode", "kernel_gap_us"]).to_csv(path, index=False)
    return str(path)


def _replica(gdn_layers=0, spec_decode=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(get_num_gdn_layers=lambda: gdn_layers),
        speculative_decoding_config=SimpleNamespace(enabled=spec_decode),
    )


def test_kernel_count_column_pairs_each_operator_time_column():
    assert kernel_count_column("time_stats.mlp_up_proj.median") == "time_stats.mlp_up_proj.kernel_count"
    assert kernel_count_column("time_stats.mlp_up_proj.kernel_count") is None
    assert kernel_count_column("num_tokens") is None


def test_gap_table_reads_one_row_per_execution_mode(tmp_path):
    path = _write_gap_table(tmp_path / "kernel_gap.csv", [("eager", 2.0), ("cuda_graph", 0.5)])
    assert read_kernel_gap_ms(path) == pytest.approx(GAP_MS)

    for rows in ([("eager", 2.0)], [("eager", 2.0), ("cuda_graph", -0.5)],
                 [("eager", 2.0), ("eager", 2.1), ("cuda_graph", 0.5)]):
        with pytest.raises(ValueError, match="needs one finite, non-negative kernel_gap_us row"):
            read_kernel_gap_ms(_write_gap_table(tmp_path / "bad.csv", rows))


def test_gap_is_active_only_with_the_table_outside_dummy_mode(tmp_path):
    path = _write_gap_table(tmp_path / "kernel_gap.csv", [("eager", 2.0), ("cuda_graph", 0.5)])
    measured = SimpleNamespace(enable_dummy_mode=False)

    assert load_kernel_gap_ms(measured, path, _replica(), sys_arch="co-location") == pytest.approx(GAP_MS)
    assert load_kernel_gap_ms(SimpleNamespace(enable_dummy_mode=True), path, _replica(), sys_arch="co-location") is None
    assert load_kernel_gap_ms(measured, str(tmp_path / "absent.csv"), _replica(), sys_arch="co-location") is None
    for replica, sys_arch, unsupported in (
        (_replica(), "pd-af-disaggregation", "PD-AF"),
        (_replica(gdn_layers=2), "co-location", "GDN layers"),
        (_replica(spec_decode=True), "pd-disaggregation", "speculative decoding"),
    ):
        with pytest.raises(ValueError, match=f"which {unsupported} does not support"):
            load_kernel_gap_ms(measured, path, replica, sys_arch=sys_arch)


class _CountModel:
    def __init__(self, exact_lookup=None, predicted=7.0):
        self._frontier_exact_lookup = exact_lookup or {}
        self.predicted = predicted
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        return [self.predicted]


class _Pricing(KernelCountTraining, KernelGapPricing):
    _measurement_family_name = staticmethod(SklearnExecutionTimePredictor._measurement_family_name)
    _is_event_measurement_type = staticmethod(SklearnExecutionTimePredictor._is_event_measurement_type)

    def __init__(self, gap_ms=GAP_MS, active=MeasurementType.KERNEL_ONLY, step=MeasurementType.CUDA_EVENT):
        self._kernel_gap_ms = gap_ms
        self._kernel_gap_input_file = "kernel_gap.csv"
        self._active_measurement_type = active
        self._step_measurement_type = step
        self._runtime_cache = defaultdict(lambda: defaultdict(dict))


def _operator_rows(**columns):
    rows = {"num_tokens": [8, 16, 32], "time_stats.mlp_up_proj.median": [1.0, 2.0, None]}
    rows.update(columns)
    return pd.DataFrame(rows)


def test_kernel_only_time_model_gets_a_count_model_from_its_measured_rows():
    trained = []

    def train(name, rows, features, target):
        trained.append((name, rows["num_tokens"].tolist(), features, target))
        return SimpleNamespace()

    model = SimpleNamespace()
    rows = _operator_rows(**{"time_stats.mlp_up_proj.kernel_count": [3, 3, None]})
    assert _Pricing()._paired_with_kernel_count_model(
        model, "linear_mlp_up_proj", rows, ["num_tokens"], "time_stats.mlp_up_proj.median", train,
    ) is model

    assert trained == [("linear_mlp_up_proj_kernel_count", [8, 16], ["num_tokens"],
                        "time_stats.mlp_up_proj.kernel_count")]
    assert model._frontier_kernel_count_model._frontier_operator_name == "mlp_up_proj"


@pytest.mark.parametrize("pricing", [_Pricing(gap_ms=None), _Pricing(active=MeasurementType.CUDA_EVENT)])
def test_count_models_are_not_trained_without_a_kernel_only_gap(pricing):
    model = SimpleNamespace()
    pricing._paired_with_kernel_count_model(
        model, "linear_mlp_up_proj", _operator_rows(), ["num_tokens"], "time_stats.mlp_up_proj.median",
        lambda *args: pytest.fail("trained a count model"),
    )
    assert not hasattr(model, "_frontier_kernel_count_model")


@pytest.mark.parametrize("counts", [None, [3, None, None]])
def test_kernel_only_rows_without_counts_fail_fast(counts):
    rows = _operator_rows() if counts is None else _operator_rows(**{"time_stats.mlp_up_proj.kernel_count": counts})
    with pytest.raises(ValueError, match="needs time_stats.mlp_up_proj.kernel_count"):
        _Pricing()._paired_with_kernel_count_model(
            SimpleNamespace(), "linear_mlp_up_proj", rows, ["num_tokens"], "time_stats.mlp_up_proj.median",
            lambda *args: SimpleNamespace(),
        )


def _priced_model(operator, count_model):
    count_model._frontier_operator_name = operator
    return SimpleNamespace(_frontier_kernel_count_model=count_model)


def test_counts_come_from_measured_rows_then_from_cached_predictions():
    pricing = _Pricing()
    count_model = _CountModel(exact_lookup={(8.0,): 3.0}, predicted=4.6)
    model = _priced_model("mlp_up_proj", count_model)

    pricing._record_kernel_count("linear_mlp_up_proj", model, (8.0,), ["num_tokens"])
    with pricing._counting_kernels() as counts:
        pricing._record_kernel_count("linear_mlp_up_proj", model, (8.0,), ["num_tokens"])
        assert counts.operators == {"mlp_up_proj": 3.0}
        for _ in range(2):
            pricing._record_kernel_count("linear_mlp_up_proj", model, (12.0,), ["num_tokens"])
        # A repeated lookup of one operator replaces its count.
        assert counts.operators == {"mlp_up_proj": 4.6}
        pricing._record_kernel_count("linear_attn_rope", SimpleNamespace(), (8.0,), ["num_tokens"])
    assert count_model.calls == 1
    assert pricing._runtime_cache["kernel_only"]["linear_mlp_up_proj_kernel_count"] == {(12.0,): 4.6}
    assert counts.operators == {"mlp_up_proj": 4.6}
    assert pricing._kernel_count_scopes == ()


class _LayerPricing(_Pricing):
    """Prices one layer from an attention lookup and an MLP lookup."""

    predict_attention_layer_time = SklearnExecutionTimePredictor.predict_attention_layer_time

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_model = _priced_model("attn_prefill", _CountModel(exact_lookup={(8.0,): 2.0}))
        self.mlp_model = _priced_model("mlp_up_proj", _CountModel(exact_lookup={(8.0,): 3.0}))

    def _predict_attention_layer_time(self, batch, layer_id, cluster_type):
        self._record_kernel_count("attn", self.attention_model, (8.0,), ["num_tokens"])
        return AttentionTime(attention_prefill_execution_time=2.0)

    def predict_layer(self, layer):
        self.attention = self.predict_attention_layer_time(None, 7, ClusterType.MONOLITHIC)
        self._record_kernel_count("mlp", self.mlp_model, (8.0,), ["num_tokens"])
        return layer


@pytest.mark.parametrize("step,gap_ms", [
    (MeasurementType.CUDA_EVENT, GAP_MS["eager"]),
    (MeasurementType.DEVICE_EVENT, GAP_MS["eager"]),
    (MeasurementType.KERNEL_ONLY, GAP_MS["cuda_graph"]),
])
def test_kernel_only_layers_pay_the_step_mode_gap_before_each_kernel(step, gap_ms):
    pricing = _LayerPricing(step=step)
    layer = _layer(7, op_times={"attn_prefill": 2.0, "mlp_up_proj": 3.0})
    before = layer.model_time_ms

    assert pricing._predict_layer_with_kernel_gap(pricing.predict_layer, layer) is layer

    assert pricing.attention.kernel_count == 2.0
    assert layer.get_single_layer_attention_scope_time() == pytest.approx(2.0 + 2 * gap_ms)
    assert layer.kernel_gap_time == pytest.approx(5 * gap_ms)
    assert layer.model_time_ms - before == pytest.approx(5 * gap_ms)
    assert pricing._kernel_count_scopes == ()


@pytest.mark.parametrize("pricing", [_LayerPricing(gap_ms=None), _LayerPricing(active=MeasurementType.CUDA_EVENT)])
def test_layers_without_a_kernel_only_gap_keep_their_time(pricing):
    layer = _layer(7, op_times={"attn_prefill": 2.0, "mlp_up_proj": 3.0})
    before = layer.model_time_ms

    pricing._predict_layer_with_kernel_gap(pricing.predict_layer, layer)

    assert not layer.has_kernel_gap
    assert layer.model_time_ms == before


def test_dense_layer_gap_counts_each_priced_tensor_parallel_allreduce():
    layer = _layer(7, op_times={
        "attn_prefill": 2.0, "mlp_up_proj": 3.0,
        "attn_tensor_parallel_allreduce": 5.0, "mlp_tensor_parallel_allreduce": 6.0,
    })
    scope, before = layer.get_single_layer_attention_scope_time(), layer.model_time_ms

    layer.add_kernel_gap(0.01, 3.0, {"mlp_up_proj": 1.0, "mlp_act": 1.0, "mlp_down_proj": 2.0})

    assert layer.get_single_layer_attention_scope_time() - scope == pytest.approx(0.04)
    assert layer.kernel_gap_time == pytest.approx(0.09)
    assert layer.model_time_ms - before == pytest.approx(0.09)


def _moe_layer(**op_times):
    return _layer(7, is_moe=True, op_times={
        "attn_prefill": 3.0, "moe_gating_linear": 1.0, "moe_grouped_gemm": 5.0,
        "expert_parallel_alltoall_dispatch": 2.0, "expert_parallel_alltoall_combine": 4.0,
        "add_ffn_residual": 1.0, **op_times,
    })


def test_moe_gap_joins_the_lane_phase_before_each_collective():
    layer = _moe_layer()
    phases_before = get_ep_phase_times_ms(layer, cluster_type=None, batch_id=0, layer_id=7, ep_id=0)

    layer.add_kernel_gap(0.01, 4.0, {"moe_gating_linear": 2.0, "moe_grouped_gemm": 3.0, "add_ffn_residual": 1.0})

    phases = get_ep_phase_times_ms(layer, cluster_type=None, batch_id=0, layer_id=7, ep_id=0)
    # Dispatch's kernel joins pre-dispatch and combine's joins routed compute.
    assert [after - before for after, before in zip(phases, phases_before)] == pytest.approx(
        [0.03, 0.0, 0.04, 0.0, 0.01]
    )
    assert layer.get_single_layer_moe_dispatch_time() == 2.0
    assert layer.get_single_layer_moe_combine_time() == 4.0
    assert layer.moe_phase_operator_times("routed_compute")[0] == ("COMPUTE", "moe_grouped_gemm", 5.0)
    assert layer.kernel_gap_time == pytest.approx(0.12)


def test_moe_gap_rejects_an_operator_outside_every_ep_phase():
    with pytest.raises(ValueError, match=r"No EP phase of a MoE layer runs operators \['mlp_up_proj'\]"):
        _moe_layer().add_kernel_gap(0.01, 0.0, {"mlp_up_proj": 1.0})


def test_stage_ledger_reports_the_gap_as_its_own_component(tmp_path):
    store, _ = _store(tmp_path, enable_op_level_tracing=False, store_frontier_stage_batch_ledger=True)
    op_times = {
        **{operator.name: 0.0 for operator in get_attention_family("dense_attention").e2e_trace_ops()},
        "attn_prefill": 2.0, "mlp_up_proj": 3.0,
    }
    plain, priced = _layer(7, op_times=op_times), _layer(7, op_times=op_times)
    priced.add_kernel_gap(0.25, 2.0, {"mlp_up_proj": 2.0})

    assert "kernel_gap_time" not in store._build_frontier_stage_batch_component_ledger(plain)
    ledger = store._build_frontier_stage_batch_component_ledger(StageExecutionTime((priced,)))
    assert ledger["kernel_gap_time"] == 1.0
    assert sum(ledger.values()) == pytest.approx(6.0)
    batch = _schedule(store, StageExecutionTime((priced,)))
    store.on_batch_stage_end(batch_stage=batch, time=1.25, replica_id=0, stage_id=0,
                             cluster_type=ClusterType.MONOLITHIC)
    assert store._frontier_stage_batch_ledger_rows[0]["execution_time"]["total_time_ms"] == 6.0


def test_every_collective_launches_one_kernel_by_default():
    assert {operator.kernel_count for operator in COMM_FAMILY.operators} == {1}
    for kernel_count in (0, 1.0, True):
        with pytest.raises(ValueError, match="kernel_count must be a positive int"):
            replace(COMM_FAMILY.operators[0], kernel_count=kernel_count)
