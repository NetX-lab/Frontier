"""Independent signed-median examples for the post-GPU package comparison."""
import csv
import json

from frontier.profiling.attention.v4_attention_windows import repeat_windows
from v4_attention_package_difference import compare_packages


def write_table(path, samples):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "attention_variant", "plans_tile_schedule",
                                                   "operator_scopes", "time_stats.scope.median", "gap_stats.scope.median"])
        writer.writeheader()
        for case, cost, gap in samples:
            writer.writerow(dict(case_id=case, attention_variant="csa", plans_tile_schedule=True,
                                 operator_scopes=json.dumps({"scope": ["scope"]}),
                                 **{"time_stats.scope.median": cost, "gap_stats.scope.median": gap}))


def test_medians_relative_signed_gaps_and_nonrepeat_exclusion(tmp_path):
    window = repeat_windows()[0].to_json()
    base_grid = {"windows": [dict(window, case_id="base", status="predicted")]}
    supplement_grid = {"windows": [dict(window, case_id="repeat", status="predicted")]}
    base, supplement = tmp_path / "base.csv", tmp_path / "supplement.csv"
    write_table(base, [("base", 1, -4), ("base", 3, -2), ("other", 100, 100)])
    write_table(supplement, [("repeat", 2, -6), ("repeat", 4, -3), ("new", 100, 100)])
    report = compare_packages(base, supplement, base_grid, supplement_grid)
    assert report["compared"] == 2 and report["missing"] == 0
    time, = (r for r in report["rows"] if r["target"] == "time")
    gap, = (r for r in report["rows"] if r["target"] == "gap")
    assert (time["base_median"], time["supplement_median"], time["relative_difference"]) == (2, 3, 0.5)
    assert (gap["base_median"], gap["supplement_median"], gap["relative_difference"]) == (-3, -4.5, 0.5)
    assert time["base_samples"] == time["supplement_samples"] == 2


def test_zero_gap_denominator_and_missing_side_are_explicit(tmp_path):
    window = repeat_windows()[0].to_json()
    grid = {"windows": [dict(window, case_id="a", status="predicted")]}
    base, supplement = tmp_path / "base.csv", tmp_path / "supplement.csv"
    write_table(base, [("a", 2, 0)])
    write_table(supplement, [("a", 3, 1)])
    report = compare_packages(base, supplement, grid, grid)
    gap, = (r for r in report["rows"] if r["target"] == "gap")
    assert gap["relative_difference"] is None and gap["absolute_difference"] == 1
    write_table(supplement, [])
    report = compare_packages(base, supplement, grid, grid)
    assert report["missing"] == 2
    assert all(r["supplement_median"] is None and r["relative_difference"] is None for r in report["rows"])
