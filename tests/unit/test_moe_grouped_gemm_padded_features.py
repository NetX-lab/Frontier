"""Grouped-GEMM features from moe_align_block_size's padded token count (decisions T33-C2-F, T33-C2-B)."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from frontier.execution_time_predictor.moe_dataset_training import (
    BLOCK_SIZE_M_COLUMN,
    NUM_TOKENS_POST_PADDED_COLUMN,
    profiled_block_sizes,
    select_moe_operator_features,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.moe_ep_workload import EPLaneWorkload
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


def test_block_size_comes_from_the_smallest_profiled_row_at_or_above_the_step() -> None:
    profiled = ((8, 16), (16, 16), (24, 64), (32, 64))

    assert block_size_m_for_tokens(profiled, 1) == 16
    assert block_size_m_for_tokens(profiled, 16) == 16
    assert block_size_m_for_tokens(profiled, 17) == 64
    assert block_size_m_for_tokens(profiled, 32) == 64
    with pytest.raises(ValueError, match="above the largest profiled MoE row"):
        block_size_m_for_tokens(profiled, 33)


def _moe_rows(**columns: list[Any]) -> pd.DataFrame:
    rows = {"num_tokens": [8, 8, 16], "time_stats.moe_grouped_gemm.median": [1.0, 1.1, 2.0]}
    rows.update(columns)
    return pd.DataFrame(rows)


def _load_columns() -> dict[str, list[float]]:
    return {name: [1.0, 2.0, 3.0] for name in MOE_LOAD_IMBALANCE_FEATURES}


def test_padded_table_trains_grouped_gemm_on_routed_and_padded_tokens() -> None:
    op_df = _moe_rows(
        **_load_columns(),
        **{NUM_TOKENS_POST_PADDED_COLUMN: [256, 272, 512], BLOCK_SIZE_M_COLUMN: [16, 16, 64]},
    )

    training_df, features = select_moe_operator_features("moe_grouped_gemm", op_df)

    assert features == list(MOE_GROUPED_GEMM_PADDED_FEATURES)
    assert training_df["num_tokens_post_padded"].tolist() == [256, 272, 512]
    assert profiled_block_sizes(training_df) == ((8, 16), (16, 64))
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


def test_one_num_tokens_with_two_block_sizes_is_rejected() -> None:
    op_df = _moe_rows(**{NUM_TOKENS_POST_PADDED_COLUMN: [256, 256, 512], BLOCK_SIZE_M_COLUMN: [16, 64, 64]})

    with pytest.raises(ValueError, match="record different"):
        profiled_block_sizes(op_df)


class _ConcreteMoEPredictor(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise AssertionError("not used")


class _RecordingPaddedModel:
    def __init__(self) -> None:
        self.n_features_in_ = 2
        self._frontier_feature_names = list(MOE_GROUPED_GEMM_PADDED_FEATURES)
        self._frontier_block_size_m = ((8, 16), (16, 64))
        self.seen: list[list[float]] = []

    def predict(self, features: Any) -> list[float]:
        assert tuple(features.columns) == MOE_GROUPED_GEMM_PADDED_FEATURES
        self.seen.append(features.iloc[0].tolist())
        return [0.5]


def _padded_predictor(exact_lookup: dict[tuple[float, ...], float]) -> tuple[_ConcreteMoEPredictor, _RecordingPaddedModel]:
    predictor = _ConcreteMoEPredictor.__new__(_ConcreteMoEPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.KERNEL_ONLY
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._supports_operation = lambda _operation: True
    model = _RecordingPaddedModel()
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
        routed_token_count=sum(local_token_counts),
        router_topk=2,
    )


def _batch(effective_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_prefill_tokens=0,
        get_effective_total_tokens_rounded=lambda _cluster_type: effective_tokens,
    )


def test_runtime_pads_each_local_expert_with_the_ceiling_row_block_size() -> None:
    predictor, model = _padded_predictor({})

    # 10 pre-routing tokens -> profiled row 16 -> BLOCK_SIZE_M 64.
    assert predictor._get_grouped_gemm_time(_lane((17, 3, 0, 0)), batch=_batch(10)) == 0.5
    # 8 pre-routing tokens -> profiled row 8 -> BLOCK_SIZE_M 16.
    assert predictor._get_grouped_gemm_time(_lane((13, 3, 0, 0)), batch=_batch(8)) == 0.5

    assert model.seen == [[20.0, 128.0], [16.0, 32.0]]


def test_runtime_padded_features_hit_the_profiled_row_exactly() -> None:
    predictor, model = _padded_predictor({(16.0, 32.0): 0.25})

    assert predictor._get_grouped_gemm_time(_lane((13, 3, 0, 0)), batch=_batch(8)) == 0.25
    assert model.seen == []
