"""CPU promotion invariants for the accepted H200 operator measurements."""

import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from frontier.execution_time_predictor.kernel_gap import read_dsv4_kernel_gap_ms
from frontier.profiling.common.runtime_identity import file_sha256
from frontier.profiling.kernel_gap.schema import DSV4_EXECUTION_MODES

ROOT = Path(__file__).resolve().parents[2]
PROMOTION_DIRECTORY = (ROOT / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/"
                       "h3o_operator_costs/h200_tables")
TABLE_DIRECTORY = ROOT / "data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash"
EXPECTED_GAPS_US = {
    "cuda_graph": 0.49773414267747085,
    "cuda_graph_piecewise": 0.6249537992538311,
    "eager": 0.5446049595929223,
    "eager_preparation": 1.2208671118771937,
}


@pytest.fixture
def promotion():
    spec = importlib.util.spec_from_file_location("dsv4_h200_promotion", PROMOTION_DIRECTORY / "promote_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_promotion_default_output_root_is_the_repository(promotion):
    assert promotion.ROOT == ROOT
    assert promotion.MODEL_DIRECTORY == TABLE_DIRECTORY.relative_to(ROOT / "data/profiling")


def test_union_keeps_serialized_measurements_and_absent_columns(tmp_path, promotion):
    first, second, output = (tmp_path / name for name in ("first.csv", "second.csv", "union.csv"))
    first.write_text('coordinate,cost,operator_only\n1,0.0075835,yes\n')
    second.write_text('coordinate,cost,shared_only\n768,0.029472,yes\n')
    assert promotion.union_csv([first, second], output) == 2
    expected = ('coordinate,cost,operator_only,shared_only\n'
                '1,0.0075835,yes,\n768,0.029472,,yes\n')
    assert output.read_text() == expected
    before = output.read_bytes()
    promotion.union_csv([first, second], output)
    assert output.read_bytes() == before


def test_revision_reader_does_not_import_the_job_launcher(tmp_path, promotion):
    launcher = tmp_path / "launcher.py"
    launcher.write_text('raise RuntimeError("must not execute")\nMODEL_CONFIG_REVISION = "pinned"\n')
    assert promotion.recorded_model_revision(launcher) == "pinned"
    launcher.write_text('OTHER_REVISION = "irrelevant"\n')
    with pytest.raises(ValueError, match="does not record MODEL_CONFIG_REVISION"):
        promotion.recorded_model_revision(launcher)


@pytest.mark.parametrize("filename,count", [
    ("attention_graph_replay_kernel_only.csv", 4945),
    ("linear_op_kernel_only.csv", 41), ("moe_kernel_only.csv", 33), ("kernel_gap.csv", 4),
])
def test_output_hashes_and_row_counts_equal_manifest(filename, count, promotion):
    manifest = json.loads((PROMOTION_DIRECTORY / "manifest.json").read_text())
    path = TABLE_DIRECTORY / filename
    entry, = (entry for entry in manifest["outputs"] if Path(entry["path"]).name == filename)
    assert file_sha256(path) == entry["sha256"]
    assert promotion.row_count(path) == entry["row_count"] == count


def test_only_kernel_only_compute_tables_are_promoted():
    manifest = json.loads((PROMOTION_DIRECTORY / "manifest.json").read_text())
    assert not any("CUDA_EVENT" in entry["path"] for entry in manifest["inputs"])
    for filename in ("attention_graph_replay_kernel_only.csv", "linear_op_kernel_only.csv", "moe_kernel_only.csv"):
        with (TABLE_DIRECTORY / filename).open(newline="") as handle:
            assert {row["measurement_type"] for row in csv.DictReader(handle)} == {"KERNEL_ONLY"}


def test_all_four_gap_values_round_trip_every_digit():
    path = TABLE_DIRECTORY / "kernel_gap.csv"
    gaps = read_dsv4_kernel_gap_ms(str(path), 8)
    assert tuple(gaps) == DSV4_EXECUTION_MODES
    assert gaps == {mode: value * 1e-3 for mode, value in EXPECTED_GAPS_US.items()}
    with path.open(newline="") as handle:
        values = {row["execution_mode"]: float(row["kernel_gap_us"]) for row in csv.DictReader(handle)}
    assert values == EXPECTED_GAPS_US


@pytest.mark.parametrize("change", ["missing", "duplicate", "negative", "fractional_tp"])
def test_v4_gap_reader_rejects_unqualified_rows(tmp_path, change):
    table = pd.read_csv(TABLE_DIRECTORY / "kernel_gap.csv", float_precision="round_trip")
    if change == "missing":
        table = table.iloc[:-1]
    elif change == "duplicate":
        table = pd.concat([table, table.iloc[:1]])
    elif change == "negative":
        table.loc[0, "kernel_gap_us"] = -1.0
    else:
        table["num_tensor_parallel_workers"] = 8.5
    path = tmp_path / "kernel_gap.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="needs one finite, non-negative"):
        read_dsv4_kernel_gap_ms(str(path), 8)
