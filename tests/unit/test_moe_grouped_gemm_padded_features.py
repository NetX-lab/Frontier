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
