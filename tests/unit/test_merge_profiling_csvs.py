"""Merge rules of `tests/comparison/calibration/merge_profiling_csvs.py`."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tests.comparison.calibration.merge_profiling_csvs import main


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish_run(run_dir: Path, sources: dict[str, list[dict]], status: int = 0) -> None:
    """Lay out a run the way `run_profiling_plan.py` publishes one."""

    for source, rows in sources.items():
        write_csv(run_dir / source, rows)
    run_dir.joinpath("manifest.json").write_text(json.dumps({"invocations": [{
        "csvs": [
            {"path": source, "sha256": hashlib.sha256((run_dir / source).read_bytes()).hexdigest()}
            for source in sources
        ],
    }]}))
    run_dir.joinpath("COMPLETE").write_text(f"status={status}\n")


STANDARD = [
    {"batch_size": "1", "kv_cache_size": "0", "time_stats.attn_prefill.median": "0.05"},
    {"batch_size": "2", "kv_cache_size": "64", "time_stats.attn_prefill.median": "0.07"},
]
TRUE_MIXED = [
    {"batch_size": "1", "is_true_mixed_batch": "True", "time_stats.attn_prefill.median": "0.09"},
]


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_a_table_takes_every_source_row_under_the_union_of_columns(tmp_path):
    run = tmp_path / "run"
    publish_run(run, {"out/a/standard.csv": STANDARD, "out/a/true_mixed.csv": TRUE_MIXED})
    output = tmp_path / "supplement"

    assert main([
        "--run-dir", str(run), "--output-dir", str(output),
        "--table", "attention.csv", "out/a/standard.csv", "out/a/true_mixed.csv",
    ]) == 0

    columns, rows = read_rows(output / "attention.csv")
    assert columns == [
        "batch_size", "kv_cache_size", "time_stats.attn_prefill.median", "is_true_mixed_batch",
    ]
    assert [row["time_stats.attn_prefill.median"] for row in rows] == ["0.05", "0.07", "0.09"]
    assert rows[0]["is_true_mixed_batch"] == "" and rows[2]["kv_cache_size"] == ""
    receipt = json.loads((output / "merge_receipt.json").read_text())["tables"]["attention.csv"]
    assert [source["output_rows"] for source in receipt["sources"]] == [[0, 2], [2, 3]]
    assert receipt["sha256"] == hashlib.sha256((output / "attention.csv").read_bytes()).hexdigest()
    assert (output / "COMPLETE").read_text() == "status=0\n"
    assert not (tmp_path / "supplement.partial").exists()


@pytest.mark.parametrize("second, message", [
    ([{**STANDARD[0], "time_stats.attn_prefill.median": "0.06"}], "repeats the identity"),
    ([{"batch_size": "4", "kv_cache_size": "0", "time_stats.attn_prefill.median": ""}],
     "has no measurement"),
])
def test_a_contradicting_or_unmeasured_row_rejects_the_merge(tmp_path, second, message):
    run = tmp_path / "run"
    publish_run(run, {"out/first.csv": STANDARD, "out/second.csv": second})

    with pytest.raises(ValueError, match=message):
        main([
            "--run-dir", str(run), "--output-dir", str(tmp_path / "supplement"),
            "--table", "attention.csv", "out/first.csv", "out/second.csv",
        ])
    assert not (tmp_path / "supplement.partial").exists()


def test_only_the_files_a_passing_run_published_are_merged(tmp_path):
    changed = tmp_path / "changed"
    publish_run(changed, {"out/first.csv": STANDARD})
    write_csv(changed / "out/first.csv", STANDARD[:1])
    failed = tmp_path / "failed"
    publish_run(failed, {"out/first.csv": STANDARD}, status=4)

    with pytest.raises(ValueError, match="not the file the run published"):
        main(["--run-dir", str(changed), "--output-dir", str(tmp_path / "a"),
              "--table", "attention.csv", "out/first.csv"])
    with pytest.raises(ValueError, match="not a passing run"):
        main(["--run-dir", str(failed), "--output-dir", str(tmp_path / "b"),
              "--table", "attention.csv", "out/first.csv"])


def test_an_existing_output_is_never_written_over(tmp_path):
    run = tmp_path / "run"
    publish_run(run, {"out/first.csv": STANDARD})
    output = tmp_path / "supplement"
    output.mkdir()

    assert main([
        "--run-dir", str(run), "--output-dir", str(output),
        "--table", "attention.csv", "out/first.csv",
    ]) == 2
    assert list(output.iterdir()) == []
