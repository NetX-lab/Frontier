"""Independent attention passes retain matching kernel costs and stream offsets."""
import importlib.util
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
spec = importlib.util.spec_from_file_location("v4_attention_multipass_table", ANALYSIS / "v4_attention_multipass_table.py")
table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table)


def published_row(case="measured"):
    return {"case_id": case, "attention_variant": "csa", "plans_tile_schedule": "False",
            "operator_scopes": '["scope"]', "time_stats.scope.median": "1.0",
            "time_stats.scope.count": "4", "gap_stats.scope.median": "-0.5",
            "layer_window_stats.median": "2.0", "gap_stats.absent.median": ""}


def measured_row():
    return {"case_id": "measured", "attention_variant": "csa", "plans_tile_schedule": False,
            "operator_scopes": '["scope"]', "time_stats.scope.median": 1.5,
            "time_stats.scope.count": 4, "gap_stats.scope.median": -0.75,
            "layer_window_stats.median": 3.0}


def test_independent_pass_keeps_kernel_gap_and_window_together():
    original, measured = published_row(), measured_row()
    [actual] = table.independent_pass_rows([original], {"identity": measured})
    assert actual == {**original, "time_stats.scope.median": "1.5",
                      "gap_stats.scope.median": "-0.75", "layer_window_stats.median": "3.0"}
    assert original["time_stats.scope.median"] == "1.0"


def test_unshared_rows_are_unchanged():
    row = published_row("unshared")
    assert table.independent_pass_rows([row], {"identity": measured_row()}) == [row]


def test_incomplete_timing_schema_is_rejected():
    measured = measured_row()
    del measured["gap_stats.scope.median"]
    with pytest.raises(ValueError, match="Timing statistics differ"):
        table.independent_pass_rows([published_row()], {"identity": measured})
