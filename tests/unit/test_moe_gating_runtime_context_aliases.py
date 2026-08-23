from __future__ import annotations

import warnings
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import frontier.moe_gating_runtime as gating_runtime


def test_canonical_context_metadata_uses_new_public_values_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        direct_metadata = gating_runtime.get_moe_gating_runtime_context_metadata(
            "direct"
        )
        warmed_metadata = gating_runtime.get_moe_gating_runtime_context_metadata(
            "prefill_warmed"
        )

    assert caught == []
    assert direct_metadata == {
        "gating_runtime_context": "direct",
        "gating_runtime_context_impl": "none",
    }
    assert warmed_metadata == {
        "gating_runtime_context": "prefill_warmed",
        "gating_runtime_context_impl": "ffn_like_prefix_20x",
    }


@pytest.mark.parametrize(
    ("legacy_value", "canonical_value"),
    [
        ("standalone_legacy", "direct"),
        ("prefill_hot", "prefill_warmed"),
    ],
)
def test_legacy_context_values_normalize_with_visible_removal_warning(
    legacy_value: str,
    canonical_value: str,
) -> None:
    with pytest.warns(
        FutureWarning,
        match=rf"{legacy_value}.*will be removed in a future release.*{canonical_value}",
    ):
        normalized = gating_runtime.normalize_moe_gating_runtime_context(
            legacy_value
        )

    assert normalized == canonical_value


def test_supported_context_values_include_canonical_and_legacy_cli_inputs() -> None:
    assert set(gating_runtime.get_supported_moe_gating_runtime_context_values()) == {
        "direct",
        "prefill_warmed",
        "standalone_legacy",
        "prefill_hot",
    }


def test_canonical_filter_consumes_legacy_csv_rows_and_returns_canonical_label() -> None:
    legacy_df = pd.DataFrame(
        {
            "gating_runtime_context": ["standalone_legacy", "prefill_hot"],
            "gating_runtime_context_impl": ["none", "ffn_like_prefix_20x"],
            "num_tokens": [1, 2],
        }
    )

    with pytest.warns(FutureWarning) as caught:
        filtered_df = gating_runtime.filter_moe_gating_rows_by_runtime_context(
            legacy_df,
            requested_context="prefill_warmed",
            source_name="legacy.csv",
        )

    assert {str(item.message) for item in caught}
    assert filtered_df["gating_runtime_context"].tolist() == ["prefill_warmed"]
    assert filtered_df["num_tokens"].tolist() == [2]


def test_prefill_warmed_row_probe_accepts_legacy_label_only_for_current_impl() -> None:
    legacy_current_df = pd.DataFrame(
        {
            "gating_runtime_context": ["prefill_hot"],
            "gating_runtime_context_impl": ["ffn_like_prefix_20x"],
        }
    )
    legacy_obsolete_df = pd.DataFrame(
        {
            "gating_runtime_context": ["prefill_hot"],
            "gating_runtime_context_impl": ["ffn_like_prefix_6x"],
        }
    )

    with pytest.warns(FutureWarning):
        assert gating_runtime.has_prefill_warmed_moe_gating_rows(legacy_current_df)
    with pytest.warns(FutureWarning):
        assert not gating_runtime.has_prefill_warmed_moe_gating_rows(
            legacy_obsolete_df
        )


def test_prediction_model_names_emit_canonical_suffix_and_parse_legacy_suffix() -> None:
    canonical_model_name = gating_runtime.get_moe_gating_prediction_model_name(
        "moe_family_gate",
        requested_context="prefill_warmed",
    )

    assert canonical_model_name == "moe_family_gate__prefill_warmed"
    assert (
        gating_runtime.get_moe_gating_base_model_name(canonical_model_name)
        == "moe_family_gate"
    )
    assert (
        gating_runtime.get_moe_gating_prediction_model_context(canonical_model_name)
        == "prefill_warmed"
    )

    with pytest.warns(
        FutureWarning,
        match=r"__prefill_hot.*will be removed in a future release.*__prefill_warmed",
    ):
        assert (
            gating_runtime.get_moe_gating_base_model_name(
                "moe_family_gate__prefill_hot"
            )
            == "moe_family_gate"
        )
    with pytest.warns(FutureWarning):
        assert (
            gating_runtime.get_moe_gating_prediction_model_context(
                "moe_family_gate__prefill_hot"
            )
            == "prefill_warmed"
        )


@pytest.mark.parametrize("invalid_value", ["", "hot", "unknown_context"])
def test_unknown_context_values_remain_fail_fast(invalid_value: str) -> None:
    with pytest.raises(ValueError, match="Unsupported gating_runtime_context"):
        gating_runtime.normalize_moe_gating_runtime_context(invalid_value)


def test_predictor_metadata_probe_surfaces_unknown_csv_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import frontier.execution_time_predictor.sklearn_moe_execution_time_predictor as predictor_module

    class _TestPredictor(
        predictor_module.SklearnMoEExecutionTimePredictor
    ):
        def _get_estimator(self):
            return None

        def _get_grid_search_params(self):
            return {}

    predictor = object.__new__(_TestPredictor)
    predictor._moe_input_file = "unknown-context.csv"
    predictor._model_config = SimpleNamespace(model_arch="qwen3_moe")
    predictor._register_profiling_metadata_from_file = lambda *_: None
    monkeypatch.setattr(
        predictor_module.pd,
        "read_csv",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "gating_runtime_context": ["unknown_context"],
                "gating_runtime_context_impl": ["none"],
            }
        ),
    )

    with pytest.raises(ValueError, match="unknown_context"):
        predictor._register_additional_profiling_metadata_from_files()


def test_profiling_cli_normalizes_legacy_input_and_keeps_canonical_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from frontier.profiling.moe.main import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frontier.profiling.moe.main",
            "--device",
            "h200",
            "--gating_runtime_context",
            "prefill_hot",
        ],
    )

    with pytest.warns(FutureWarning):
        args, _ = parse_args()

    assert args.gating_runtime_context == "prefill_warmed"


def test_training_cli_accepts_canonical_context_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from frontier.training.cli import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frontier.training.cli",
            "moe",
            "--dataset_path",
            "legacy.csv",
            "--measurement_type",
            "CUDA_EVENT",
            "--gating_runtime_context",
            "direct",
        ],
    )

    args = parse_args()

    assert args.gating_runtime_context == "direct"


def test_training_cli_normalizes_legacy_context_with_visible_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from frontier.training.cli import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frontier.training.cli",
            "moe",
            "--dataset_path",
            "legacy.csv",
            "--measurement_type",
            "CUDA_EVENT",
            "--gating_runtime_context",
            "standalone_legacy",
        ],
    )

    with pytest.warns(FutureWarning):
        args = parse_args()

    assert args.gating_runtime_context == "direct"


def test_moe_consumers_do_not_encode_prediction_model_suffixes() -> None:
    consumer_paths = (
        Path("frontier/training/moe_trainer.py"),
        Path(
            "frontier/execution_time_predictor/"
            "sklearn_moe_execution_time_predictor.py"
        ),
        Path(
            "frontier/execution_time_predictor/"
            "shared_prediction_model_manager.py"
        ),
    )

    for path in consumer_paths:
        source = path.read_text(encoding="utf-8")
        assert "__prefill_hot" not in source
        assert "__prefill_warmed" not in source
