"""Grouped-GEMM features from moe_align_block_size's padded token count (decisions T33-C2-F, T43-PAD).

The padding uses the BLOCK_SIZE_M ranges of vLLM's kernel config that the MoE profiler records.
"""

from __future__ import annotations

import json
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from frontier.execution_time_predictor.moe_dataset_training import (
    BLOCK_SIZE_M_COLUMN,
    BLOCK_SIZE_M_RANGES_COLUMN,
    NUM_TOKENS_POST_PADDED_COLUMN,
    block_size_m_ranges,
    select_moe_operator_features,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.moe_ep_workload import EPLaneWorkload, split_global_expert_tokens_into_lanes
from frontier.moe_load_imbalance import (
    MOE_GROUPED_GEMM_PADDED_FEATURES,
    MOE_LOAD_IMBALANCE_FEATURES,
    block_size_m_for_tokens,
    num_tokens_post_padded,
)
from frontier.types import ClusterType, MeasurementType


def test_padded_count_rounds_each_expert_up_to_the_block() -> None:
    assert num_tokens_post_padded([128] * 128, 64) == 16384
    assert num_tokens_post_padded([130] * 128, 64) == 24576
    assert num_tokens_post_padded([0, 1, 16, 17], 16) == 0 + 16 + 16 + 32


def test_block_size_comes_from_the_kernel_config_range_of_the_step() -> None:
    # vLLM's default config for 128 local experts: 16 up to 128 tokens, 64 above.
    default_ranges = ((1, 128, 16), (129, 256, 64))
    # A tuned config with keys 64 (16) and 256 (64): the nearest key changes after 160 tokens.
    tuned_ranges = ((1, 160, 16), (161, 256, 64))

    assert block_size_m_for_tokens(default_ranges, 1) == 16
    assert block_size_m_for_tokens(default_ranges, 128) == 16
    assert block_size_m_for_tokens(default_ranges, 129) == 64
    assert block_size_m_for_tokens(tuned_ranges, 150) == 16
    assert block_size_m_for_tokens(tuned_ranges, 161) == 64
    assert block_size_m_for_tokens(tuned_ranges, 256) == 64
    with pytest.raises(ValueError, match="above the largest profiled MoE row"):
        block_size_m_for_tokens(tuned_ranges, 257)


def _moe_rows(**columns: list[Any]) -> pd.DataFrame:
    rows = {"num_tokens": [8, 8, 16], "time_stats.moe_grouped_gemm.median": [1.0, 1.1, 2.0]}
    rows.update(columns)
    return pd.DataFrame(rows)


def _load_columns() -> dict[str, list[float]]:
    return {name: [1.0, 2.0, 3.0] for name in MOE_LOAD_IMBALANCE_FEATURES}


# Kernel-config ranges of the 8-, 8- and 16-token rows: 16 up to 12 tokens, 64 above.
_ROW_RANGES = [json.dumps([[1, 8, 16]])] * 2 + [json.dumps([[1, 12, 16], [13, 16, 64]])]


def test_padded_table_trains_grouped_gemm_on_routed_and_padded_tokens() -> None:
    op_df = _moe_rows(
        **_load_columns(),
        **{
            NUM_TOKENS_POST_PADDED_COLUMN: [256, 272, 512],
            BLOCK_SIZE_M_COLUMN: [16, 16, 64],
            BLOCK_SIZE_M_RANGES_COLUMN: _ROW_RANGES,
        },
    )

    training_df, features = select_moe_operator_features("moe_grouped_gemm", op_df)

    assert features == list(MOE_GROUPED_GEMM_PADDED_FEATURES)
    assert training_df["num_tokens_post_padded"].tolist() == [256, 272, 512]
    assert block_size_m_ranges(training_df) == ((1, 12, 16), (13, 16, 64))
    shuffling_df, shuffling_features = select_moe_operator_features("moe_shuffling", op_df)
    assert shuffling_df is op_df
    assert shuffling_features == list(MOE_LOAD_IMBALANCE_FEATURES)


def test_tables_without_the_padded_column_keep_the_previous_features() -> None:
    load_df = _moe_rows(**_load_columns())
    assert select_moe_operator_features("moe_grouped_gemm", load_df) == (load_df, list(MOE_LOAD_IMBALANCE_FEATURES))

    plain_df = _moe_rows()
    assert select_moe_operator_features("moe_grouped_gemm", plain_df) == (plain_df, ["num_tokens"])
    assert select_moe_operator_features("moe_gating_linear", load_df) == (load_df, ["num_tokens"])

    partial_df = _moe_rows(total_routed_tokens=[16, 16, 32])
    with pytest.raises(ValueError, match="Partial load imbalance features"):
        select_moe_operator_features("moe_grouped_gemm", partial_df)
    assert select_moe_operator_features("moe_shuffling", partial_df) == (partial_df, ["num_tokens"])


def test_padded_column_must_cover_every_row() -> None:
    op_df = _moe_rows(**{NUM_TOKENS_POST_PADDED_COLUMN: [256, None, 512], BLOCK_SIZE_M_COLUMN: [16, None, 64]})

    with pytest.raises(ValueError, match="1 of 3 MoE rows lack"):
        select_moe_operator_features("moe_grouped_gemm", op_df)


def test_a_padded_table_without_block_size_ranges_is_rejected() -> None:
    op_df = _moe_rows(**{NUM_TOKENS_POST_PADDED_COLUMN: [256, 272, 512], BLOCK_SIZE_M_COLUMN: [16, 16, 64]})

    with pytest.raises(ValueError, match="re-profile the MoE table"):
        block_size_m_ranges(op_df)
    with pytest.raises(ValueError, match="1 of 3 MoE rows lack"):
        block_size_m_ranges(op_df.assign(**{BLOCK_SIZE_M_RANGES_COLUMN: [None, *_ROW_RANGES[1:]]}))


def test_a_row_block_size_that_disagrees_with_the_ranges_is_rejected() -> None:
    op_df = _moe_rows(
        **{
            NUM_TOKENS_POST_PADDED_COLUMN: [256, 256, 512],
            BLOCK_SIZE_M_COLUMN: [16, 64, 64],
            BLOCK_SIZE_M_RANGES_COLUMN: _ROW_RANGES,
        }
    )

    with pytest.raises(ValueError, match=r"\(8, 64\)\] disagree"):
        block_size_m_ranges(op_df)


class _ConcreteMoEPredictor(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise AssertionError("not used")


class _RecordingPaddedModel:
    def __init__(self, block_size_m_ranges: tuple[tuple[int, int, int], ...]) -> None:
        self.n_features_in_ = 2
        self._frontier_feature_names = list(MOE_GROUPED_GEMM_PADDED_FEATURES)
        self._frontier_block_size_m_ranges = block_size_m_ranges
        self.seen: list[list[float]] = []

    def predict(self, features: Any) -> list[float]:
        assert tuple(features.columns) == MOE_GROUPED_GEMM_PADDED_FEATURES
        self.seen.append(features.iloc[0].tolist())
        return [0.5]


def _padded_predictor(
    exact_lookup: dict[tuple[float, ...], float],
    block_size_m_ranges: tuple[tuple[int, int, int], ...] = ((1, 8, 16), (9, 16, 64)),
) -> tuple[_ConcreteMoEPredictor, _RecordingPaddedModel]:
    predictor = _ConcreteMoEPredictor.__new__(_ConcreteMoEPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.KERNEL_ONLY
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._supports_operation = lambda _operation: True
    model = _RecordingPaddedModel(block_size_m_ranges)
    predictor._predictions = {
        "moe_grouped_gemm": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": list(MOE_GROUPED_GEMM_PADDED_FEATURES),
            "_exact_lookup": exact_lookup,
        }
    }
    return predictor, model


def _lane(local_token_counts: tuple[int, ...]) -> EPLaneWorkload:
    return EPLaneWorkload(
        ep_id=0,
        moe_expert_parallel_size=1,
        total_expert_num=len(local_token_counts),
        owned_expert_ids=tuple(range(len(local_token_counts))),
        local_token_counts=local_token_counts,
        global_token_counts=local_token_counts,
        routed_token_count=sum(local_token_counts),
        router_topk=2,
    )


def _batch(effective_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_prefill_tokens=0,
        get_effective_total_tokens_rounded=lambda _cluster_type: effective_tokens,
    )


def test_runtime_pads_each_local_expert_with_the_block_size_of_the_step() -> None:
    predictor, model = _padded_predictor({})

    # 10 pre-routing tokens -> range 9..16 -> BLOCK_SIZE_M 64.
    assert predictor._get_grouped_gemm_time(_lane((17, 3, 0, 0)), batch=_batch(10)) == 0.5
    # 8 pre-routing tokens -> range 1..8 -> BLOCK_SIZE_M 16.
    assert predictor._get_grouped_gemm_time(_lane((13, 3, 0, 0)), batch=_batch(8)) == 0.5

    assert model.seen == [[20.0, 128.0], [16.0, 32.0]]


def test_a_step_between_profiled_rows_pads_with_the_kernel_config_block_size() -> None:
    # bf16, 128 local experts, top-k 2, no tuned config: vLLM selects 16 up to 128 tokens.
    # The 64- and 256-token profiled rows record 16 and 64; the 100-token step uses 16,
    # not the 256-token row's 64 (8,192 padded rows).
    predictor, model = _padded_predictor({}, block_size_m_ranges=((1, 128, 16), (129, 256, 64)))

    predictor._get_grouped_gemm_time(_lane((2,) * 72 + (1,) * 56), batch=_batch(100))

    assert model.seen == [[200.0, 2048.0]]


def test_runtime_padded_features_hit_the_profiled_row_exactly() -> None:
    predictor, model = _padded_predictor({(16.0, 32.0): 0.25})

    assert predictor._get_grouped_gemm_time(_lane((13, 3, 0, 0)), batch=_batch(8)) == 0.25
    assert model.seen == []


def test_each_ep_rank_pads_every_expert_of_the_domain() -> None:
    predictor, model = _padded_predictor({})
    lanes = split_global_expert_tokens_into_lanes(
        {0: 17, 1: 3, 2: 5, 3: 0},
        total_expert_num=4,
        moe_expert_parallel_size=2,
        router_topk=2,
    )

    # 10 pre-routing tokens -> BLOCK_SIZE_M 64; experts 0, 1 and 2 each pad to one block.
    for lane in lanes:
        assert predictor._get_grouped_gemm_time(lane, batch=_batch(10)) == 0.5

    assert model.seen == [[20.0, 192.0], [5.0, 192.0]]


def _position_source_rows(model, *, quant_dtype="float8_e4m3fn", context="standalone_legacy"):
    from frontier.config.precision_type import PrecisionType
    from frontier.operators.binding import bind_operator_query
    from frontier.operators.families import get_moe_profiling_operators
    from frontier.profiling.moe.inputs import MoESourceFrame, MoERoutingProfileInput, MoERoutingProfileCell
    from frontier.entities import Batch, Request
    from frontier.entities.batch import DummyForwardBatch
    from frontier.config import global_vars
    from frontier.scheduler.utils.forward_start_agreement import bind_forward_start
    from frontier.moe_routing_policy import build_wave_routing_source
    from tests.unit.test_moe_position_routing import materialize
    native = dict(model_architecture_profile="step5", routing_runtime_path="position_cyclic",
                  quant_dtype=quant_dtype, defer_input_quant=False)
    contracts, facts = {}, {}
    for spec in get_moe_profiling_operators(native):
        contract = model.get_model_architecture_profile().resolve_layer_contract(
            model, layer_id=3, operator_name=spec.name, attention_tp_size=8,
            ffn_tp_size=8, moe_tp_size=1, expert_parallel_size=16)
        contracts[spec.name] = contract.typed_metadata_identity()
        if spec.ep_agnostic:
            name = spec.name + ("__prefill_hot" if spec.precision_name() == "moe_gating" and context == "prefill_hot" else "")
            facts[name] = dict(spec=spec, binding=bind_operator_query(spec.name), layer_contract=contract,
                model_name=name, native_context=native, frame_scope="source_local", gating_runtime_context=context,
                precision=PrecisionType.FP32 if spec.precision_name() == "moe_gating" else PrecisionType.FP8,
                feature_columns=("num_tokens", "live_routing_token_count"), row_metadata={"moe_layer_ids": "3-87"})
    rows = []
    for live in (0, 4):
        batch = Batch(0, [Request(0, 6144, 1024)], [live], is_moe=True, num_context_tokens=[6128]) if live else DummyForwardBatch(0, 0, 0)
        batches = {0: batch, 1: DummyForwardBatch(0, 0, 0)}
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
            patch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", [8])
            bind_forward_start(list(batches.values()), group_id=0, request_capacity=128)
            source = build_wave_routing_source(batches, policy_id="position_cyclic", cluster_type=ClusterType.DECODE,
                gathered_physical_rows=16, model_config=model)
        lane = source.source_lanes[0]
        frame = MoESourceFrame(0, tuple(i.length for i in lane.intervals), tuple(i.start for i in lane.intervals), lane.physical_rows, context)
        assert materialize(source).live_routing_token_count == live
        routing = MoERoutingProfileInput("live" if live else "empty",
            (frame, MoESourceFrame(1, (), (), source.source_lanes[1].physical_rows, context)), source.gathered_physical_rows)
        cell = MoERoutingProfileCell(routing, "source_local", 0)
        row = dict(num_tokens=frame.physical_rows, live_routing_token_count=frame.live_rows,
            masked_routing_token_count=frame.physical_rows-frame.live_rows, moe_input_id=routing.moe_input_id,
            moe_source_lane_id=cell.source_lane_id, moe_ep_rank=cell.ep_id,
            routing_input=json.dumps(cell.to_json()["routing_input"]),
            num_tensor_parallel_workers=1, expert_parallel_size=16, moe_frame_scope="source_local",
            gating_runtime_context=context, moe_layer_ids="3-87", model_architecture_profile="step5",
            routing_runtime_path="position_cyclic", profiling_precision="BF16", measurement_type="CUDA_EVENT",
            runtime_stack_signature="cpu_fixture_unadmitted", evidence_kind="cpu_fixture",
            quant_signature=model.get_quant_signature(),
            typed_operator_contracts=json.dumps({name: value for name, value in contracts.items() if name != "moe_grouped_gemm"}))
        for fact in facts.values():
            name = fact["spec"].name
            row.update({f"time_stats.{name}.median": 1. + live, f"time_stats.{name}.kernel_count": 3})
        rows.append(row)
    return pd.DataFrame(rows), facts


def test_moe_loader_derives_family_once_without_skipping_unselected_rows(tmp_path):
    from unittest.mock import patch
    from frontier.attention import step5
    from frontier.config.model_config import BaseModelConfig
    from frontier.execution_time_predictor.profiling_dataframe_loaders import ProfilingDataFrameLoaders

    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    rows, facts = _position_source_rows(model)
    rows = pd.concat([rows, rows.iloc[:1].assign(moe_layer_ids="3")], ignore_index=True)
    path = tmp_path / "moe.csv"
    rows.to_csv(path, index=False)
    fact = facts["moe_input_quantize"]
    replica = SimpleNamespace(model_config=model)
    loader = ProfilingDataFrameLoaders()
    expected = pd.read_csv(path, float_precision="round_trip")
    with patch.object(step5, "validate_step5_config", wraps=step5.validate_step5_config) as derive:
        for _ in range(2):
            derive.reset_mock()
            loaded = loader._load_moe_df(str(path), replica, tensor_parallel_size=1,
                layer_contract=fact["layer_contract"], operator_name="moe_input_quantize", operator_facts=fact)
            pd.testing.assert_frame_equal(loaded, expected.iloc[:2].assign(
                source_profiling_precision="BF16", profiling_precision="FP8"))
            assert derive.call_count == 1
        typed = json.loads(rows.loc[2, "typed_operator_contracts"])
        typed["moe_gating_linear"]["tensor_parallel_mode"] = "invalid"
        rows.loc[2, "typed_operator_contracts"] = json.dumps(typed)
        rows.to_csv(path, index=False)
        derive.reset_mock()
        with pytest.raises(ValueError, match="tensor_parallel_mode"):
            loader._load_moe_df(str(path), replica, tensor_parallel_size=1,
                layer_contract=fact["layer_contract"], operator_name="moe_input_quantize", operator_facts=fact)
        assert derive.call_count == 1


def test_selected_sources_reach_real_preflights_loader_and_both_trainers_before_old_helpers(tmp_path, monkeypatch):
    from frontier.config.model_config import BaseModelConfig
    from frontier.execution_time_predictor.moe_dataset_training import MoeDatasetTraining
    from frontier.execution_time_predictor.step5 import step5_operator_training_rows
    from tests.unit.test_step5_measured_training import _manager
    from tests.unit.test_step5_measured_ffn import _config, _tables
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    replica = SimpleNamespace(model_config=model, model_name="stepfun-ai/Step-5-Preview", device="h800",
        attn_tensor_parallel_size=8, moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
        moe_routing_distribution_type="position_cyclic", speculative_decoding_config=None)
    linear, moe = _tables(tmp_path, model, expert_parallel_size=16)
    linear_rows = pd.read_csv(linear)
    linear_rows["measurement_type"] = "CUDA_EVENT"
    linear_rows.to_csv(linear, index=False)
    rows, facts = _position_source_rows(model)
    rows.to_csv(moe, index=False)
    manager = _manager(tmp_path / "selected", MeasurementType.CUDA_EVENT)
    manager._cluster_configs = {ClusterType.MONOLITHIC: SimpleNamespace(replica_config=replica)}
    def forbidden(*args, **kwargs):
        pytest.fail("unbound legacy MoE helper reached")
    monkeypatch.setattr(manager, "_get_moe_op_tp_key", forbidden)
    monkeypatch.setattr(manager, "_is_moe_op_ep_agnostic", forbidden)
    manager._validate_moe_dataset_contract(str(moe), replica, list(facts), ClusterType.MONOLITHIC, operator_facts=facts)
    prepare = facts["moe_input_quantize"]
    loaded = manager._load_moe_df(str(moe), replica, tensor_parallel_size=1, expert_parallel_size=None,
        layer_contract=prepare["layer_contract"], operator_name="moe_input_quantize", operator_facts=prepare)
    assert loaded["profiling_precision"].tolist() == ["FP8", "FP8"]
    assert loaded["source_profiling_precision"].tolist() == ["BF16", "BF16"]
    models = manager._train_ffn_models_for_cluster(ClusterType.MONOLITHIC, replica, _config(),
        str(linear), str(moe), True, set(), operator_facts=facts)
    for name in facts:
        fitted = models[name]
        assert fitted._frontier_exact_lookup[(8., 0.)] == 1.
        assert fitted._frontier_kernel_count_model._frontier_exact_lookup[(8., 0.)] == 3
        assert set(fitted._frontier_exact_lookup) == set(fitted._frontier_kernel_count_model._frontier_exact_lookup)
    warm = _manager(tmp_path / "selected", MeasurementType.CUDA_EVENT)
    warm._cluster_configs = manager._cluster_configs
    monkeypatch.setattr(warm, "_get_moe_op_tp_key", forbidden)
    monkeypatch.setattr(warm, "_is_moe_op_ep_agnostic", forbidden)
    monkeypatch.setattr(warm, "_create_estimator_and_params", forbidden)
    reloaded = warm._train_ffn_models_for_cluster(ClusterType.MONOLITHIC, replica, _config(),
        str(linear), str(moe), True, set(), operator_facts=facts)
    for name in facts:
        assert reloaded[name]._frontier_model_hash == models[name]._frontier_model_hash
        assert reloaded[name]._frontier_kernel_count_model._frontier_measurement_identity == reloaded[name]._frontier_measurement_identity
    rebound_facts = {name: {**item, "native_context": {**item["native_context"], "backend": "cpu_fixture_other"}} for name, item in facts.items()}
    rebound = manager._train_ffn_models_for_cluster(ClusterType.MONOLITHIC, replica, _config(), str(linear), str(moe), True, set(), operator_facts=rebound_facts)
    assert rebound["moe_gating_linear"]._frontier_model_hash != models["moe_gating_linear"]._frontier_model_hash
    independent = object.__new__(_ConcreteMoEPredictor)
    independent._model_config, independent._replica_config = model, replica
    independent._moe_input_file = str(moe)
    independent._active_measurement_type = MeasurementType.CUDA_EVENT
    for method in ("_get_moe_op_tp_key", "_is_moe_op_ep_agnostic"):
        monkeypatch.setattr(independent, method, forbidden)
    from frontier.config.precision_type import PrecisionType
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import ProfilingMetadata
    monkeypatch.setattr(independent, "_get_profiling_metadata", lambda *args: ProfilingMetadata(
        profiling_precision=PrecisionType.BF16, model_arch="generic", model_architecture_profile="step5",
        quant_signature=model.get_quant_signature(), measurement_type=MeasurementType.CUDA_EVENT))
    monkeypatch.setattr(independent, "_validate_active_measurement_type", lambda *args: None)
    monkeypatch.setattr(independent, "_register_profiling_metadata_for_ops", lambda *args: None)
    monkeypatch.setattr(independent, "_store_model_in_cache", lambda *args: None)
    calls = []
    def fit(name, frame, columns, target):
        from frontier.execution_time_predictor.prediction_model_identity import _build_exact_feature_lookup
        calls.append((name, tuple(columns), frame["profiling_precision"].unique().tolist()))
        return SimpleNamespace(_frontier_model_hash=name, _frontier_exact_lookup=_build_exact_feature_lookup(frame, columns, target))
    monkeypatch.setattr(independent, "_train_model", fit)
    MoeDatasetTraining._train_moe_models(independent, operator_facts=facts)
    assert ("moe_input_quantize", ("num_tokens", "live_routing_token_count"), ["FP8"]) in calls
    assert ("moe_input_quantize_kernel_count", ("num_tokens", "live_routing_token_count"), ["FP8"]) in calls
    hot, hot_facts = _position_source_rows(model, context="prefill_hot")
    _, bound = step5_operator_training_rows("moe_gating_linear__prefill_hot", hot,
        dict(operator_facts=hot_facts["moe_gating_linear__prefill_hot"], input_file=str(moe)), MeasurementType.CUDA_EVENT)[0]
    assert bound["measurement_identity"]["gating_runtime_context"] == "prefill_hot"
    bf16, bf16_facts = _position_source_rows(model, quant_dtype=None)
    assert "moe_input_quantize" not in bf16_facts


@pytest.mark.parametrize("damage", ["zero", "opposite", "typed", "missing", "physical", "context"])
def test_selected_source_filter_rejects_unmeasured_or_wrong_physical_rows(damage):
    from frontier.config.model_config import BaseModelConfig
    from frontier.moe_routing_runtime import filter_moe_operator_rows
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    rows, facts = _position_source_rows(model)
    if damage == "zero": rows.loc[0, "time_stats.moe_input_quantize.median"] = 0
    if damage == "opposite": rows["time_stats.moe_grouped_gemm.median"] = 0
    if damage == "typed": rows["typed_operator_contracts"] = "{}"
    if damage == "missing": rows = rows.drop(columns="time_stats.moe_input_quantize.kernel_count")
    if damage == "physical": rows.loc[0, "masked_routing_token_count"] = 7
    if damage == "context": rows["gating_runtime_context"] = "prefill_hot"
    with pytest.raises(ValueError):
        filter_moe_operator_rows(rows, operator_facts=facts["moe_gating_linear"],
            requested_runtime_path="position_cyclic", source_name="explicit_cpu_source_fixture")


def test_independent_selected_actual_fit_cache_binds_native_context(tmp_path, monkeypatch):
    from frontier.config.model_config import BaseModelConfig
    from sklearn.ensemble import RandomForestRegressor
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    rows, facts = _position_source_rows(model)
    rows["model_arch"] = model.model_arch
    path = tmp_path / "moe.csv"
    rows.to_csv(path, index=False)
    predictor = object.__new__(_ConcreteMoEPredictor)
    predictor._model_config, predictor._moe_input_file = model, str(path)
    predictor._replica_config = SimpleNamespace(model_config=model, moe_tensor_parallel_size=1, moe_expert_parallel_size=16)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._config = SimpleNamespace(no_cache=False, k_fold_cv_splits=2, num_training_job_threads=1)
    predictor._cache_dir, predictor._kernel_gap_ms = str(tmp_path), None
    monkeypatch.setattr(predictor, "to_dict", lambda: {})
    monkeypatch.setattr(predictor, "_get_estimator", lambda: RandomForestRegressor(n_estimators=2, random_state=0))
    monkeypatch.setattr(predictor, "_get_grid_search_params", lambda: {})
    first = predictor._train_moe_models(operator_facts=facts)
    monkeypatch.setattr(predictor, "_get_estimator", lambda: pytest.fail("expected actual cache hit"))
    cached = predictor._train_moe_models(operator_facts=facts)
    for name in facts:
        assert cached[name]._frontier_model_hash == first[name]._frontier_model_hash
        assert cached[name]._frontier_kernel_count_model._frontier_exact_lookup[(8., 0.)] == 3.
    first_hash = first["moe_gating_linear"]._frontier_model_hash
    for item in facts.values():
        item["native_context"] = {**item["native_context"], "backend": "cpu_fixture_changed_binding"}
    monkeypatch.setattr(predictor, "_get_estimator", lambda: RandomForestRegressor(n_estimators=2, random_state=0))
    rebound = predictor._train_moe_models(operator_facts=facts)
    assert rebound["moe_gating_linear"]._frontier_model_hash != first_hash


def test_selected_output_inventory_uses_declared_cells_without_structural_zero_fill():
    from frontier.config.model_config import BaseModelConfig
    from frontier.profiling.moe.main import _validate_canonical_moe_result_df
    rows, facts = _position_source_rows(BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview"))
    rows["moe_input_id"], rows["moe_source_lane_id"], rows["moe_ep_rank"] = ["empty", "live"], 0, None
    expected = {(name, "3-87", "source_local", 0, None) for name in ("empty", "live")}
    _validate_canonical_moe_result_df(rows, model="cpu_fixture", operator_facts=facts, expected_cells=expected)
    for broken in (rows.iloc[:1], pd.concat([rows, rows.iloc[:1]])):
        with pytest.raises(ValueError, match="physical cells"):
            _validate_canonical_moe_result_df(broken, model="cpu_fixture", operator_facts=facts, expected_cells=expected)
