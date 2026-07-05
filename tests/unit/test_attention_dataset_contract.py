from __future__ import annotations

from pathlib import Path

import pytest

from frontier.execution_time_predictor.attention_dataset_contract import (
    enforce_mixed_attention_input_contract,
)


def test_mixed_attention_contract_recommends_true_mixed_merge_not_combined_supplement(
    tmp_path: Path,
) -> None:
    attention_file = tmp_path / "attention.csv"
    (tmp_path / "attention_true_mixed.csv").write_text("header\n", encoding="utf-8")
    (tmp_path / "attention_combined.csv").write_text("header\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        enforce_mixed_attention_input_contract(
            str(attention_file),
            available_columns=("is_decode",),
        )

    message = str(exc_info.value)
    assert "attention_true_mixed.csv" in message
    assert "attention_kernel_only.csv" in message
    assert "Do not use attention_combined*.csv as the true-mixed supplement source" in message
    assert "recommended: " not in message
