"""Unit tests for tools/validation/compare_plots.py's real-vs-sim agreement statistics.

Covers the geometric-mean relative-error helper, the geometric-standard-deviation (consistency)
helper, the hand-rolled Pearson correlation helper, and a functional smoke test of the report
builders (build_report / _data_table_html / write_html_report) against small synthetic
AggregatedResult/SimResult fixtures -- enough to prove the new Δ column, the three summary rows,
and the "(r=...)" subplot-title annotations actually render, not just that the module imports
cleanly.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from tools.validation.compare_plots import (
    _data_table_html,
    _fmt_delta_pct,
    _fmt_gsd,
    _fmt_r,
    _geo_mean_relative_error_pct,
    _gsd,
    _pearson_r,
    _relative_error_pct,
    build_report,
    write_html_report,
)
from tools.validation.metrics_extractor import SimResult
from tools.validation.real_log_aggregator import AggregatedResult, MetricStats


def _stats(mean: float, n: int = 1) -> MetricStats:
    return MetricStats(mean=mean, std=0.0, min=mean, max=mean, n=n)


def _agg_result(concurrency: int, **field_means: float) -> AggregatedResult:
    """Minimal AggregatedResult for one concurrency level: field_means maps a BenchmarkResult
    field name (e.g. "request_throughput_req_s") to its (single-repetition) mean.

    `raw`/`canonical` are never touched by compare_plots.py -- it only reads `.concurrency` and
    `.get(field)` -- so placeholders are fine here rather than building full BenchmarkResult
    records just to satisfy the dataclass's required fields.
    """
    return AggregatedResult(
        concurrency=concurrency,
        num_prompts=100,
        n_runs=1,
        stats={field: _stats(mean) for field, mean in field_means.items()},
        raw=[],
        canonical=None,  # type: ignore[arg-type]
    )


def _sim_result(concurrency: int, **fields: float) -> SimResult:
    return SimResult(run_id=f"sim-{concurrency}", concurrency=concurrency, num_prompts=100, **fields)


# Two concurrency levels, every _METRICS field populated on both sides with known, non-degenerate
# values -- sim consistently a bit above real at one point and a bit below at the other so the
# per-cell Δ, the geometric-mean aggregate, and the correlation all have something real to compute.
_REAL_8 = dict(
    request_throughput_req_s=10.0,
    input_token_throughput_tok_s=100.0,
    output_token_throughput_tok_s=50.0,
    ttft_mean_ms=200.0,
    ttft_p99_ms=400.0,
    tpot_mean_ms=20.0,
    e2e_mean_ms=500.0,
    e2e_p99_ms=900.0,
    achieved_concurrency=8.0,
)
_SIM_8 = dict(
    request_throughput_req_s=12.0,
    input_token_throughput_tok_s=90.0,
    output_token_throughput_tok_s=55.0,
    ttft_mean_ms=220.0,
    ttft_p99_ms=380.0,
    tpot_mean_ms=22.0,
    e2e_mean_ms=480.0,
    e2e_p99_ms=850.0,
    achieved_concurrency=7.5,
)
_REAL_16 = dict(
    request_throughput_req_s=20.0,
    input_token_throughput_tok_s=200.0,
    output_token_throughput_tok_s=100.0,
    ttft_mean_ms=250.0,
    ttft_p99_ms=450.0,
    tpot_mean_ms=25.0,
    e2e_mean_ms=600.0,
    e2e_p99_ms=1000.0,
    achieved_concurrency=16.0,
)
_SIM_16 = dict(
    request_throughput_req_s=18.0,
    input_token_throughput_tok_s=190.0,
    output_token_throughput_tok_s=105.0,
    ttft_mean_ms=260.0,
    ttft_p99_ms=430.0,
    tpot_mean_ms=24.0,
    e2e_mean_ms=590.0,
    e2e_p99_ms=980.0,
    achieved_concurrency=15.0,
)


def _synthetic_fixtures():
    real = [_agg_result(8, **_REAL_8), _agg_result(16, **_REAL_16)]
    sim = [_sim_result(8, **_SIM_8), _sim_result(16, **_SIM_16)]
    return real, sim


# --- Geometric-mean relative error ------------------------------------------------------------


def test_geo_mean_relative_error_matches_hand_computed_value():
    # real=[10, 20], sim=[12, 18] -> per-point ratios 1.2 and 0.9.
    # Geometric mean of the ratios = sqrt(1.2 * 0.9) = sqrt(1.08) ~= 1.0392304845.
    expected_gm = math.sqrt(1.2 * 0.9)
    expected_pct = (expected_gm - 1) * 100  # ~= +3.92%

    result = _geo_mean_relative_error_pct([10, 20], [12, 18])

    assert result is not None
    assert result == pytest.approx(expected_pct, rel=1e-9)
    assert result == pytest.approx(3.9230484541326228, rel=1e-9)


def test_geo_mean_relative_error_differs_from_naive_arithmetic_mean():
    # Sanity check that this is genuinely the geometric mean and not a relabeled arithmetic mean
    # of the signed percentages: arithmetic mean of [+20%, -10%] is +5%, which does not match.
    result = _geo_mean_relative_error_pct([10, 20], [12, 18])
    arithmetic_mean_pct = ((12 / 10 - 1) * 100 + (18 / 20 - 1) * 100) / 2
    assert arithmetic_mean_pct == pytest.approx(5.0)
    assert result != pytest.approx(arithmetic_mean_pct)


def test_geo_mean_relative_error_skips_missing_and_nonpositive_points():
    # Only the (20, 18) pair is valid: None, zero, and negative real/sim values are all excluded.
    result = _geo_mean_relative_error_pct([None, 0.0, -5.0, 20.0], [1.0, 1.0, 1.0, 18.0])
    assert result == pytest.approx((18 / 20 - 1) * 100)


def test_geo_mean_relative_error_all_invalid_is_none():
    assert _geo_mean_relative_error_pct([None, 0.0, -1.0], [None, 1.0, 1.0]) is None
    assert _geo_mean_relative_error_pct([], []) is None


def test_relative_error_pct_signed_and_missing_cases():
    assert _relative_error_pct(10.0, 12.0) == pytest.approx(20.0)
    assert _relative_error_pct(10.0, 9.0) == pytest.approx(-10.0)
    assert _relative_error_pct(None, 12.0) is None
    assert _relative_error_pct(10.0, None) is None
    assert _relative_error_pct(0.0, 12.0) is None
    assert _relative_error_pct(10.0, -5.0) is None


# --- Geometric standard deviation (consistency of the multiplicative error) --------------------


def test_gsd_identical_ratios_is_one():
    # Every point is off by the exact same factor (2x) -- a consistent bias, so GSD is 1.0 even
    # though the geometric-mean error itself is a hefty +100%.
    result = _gsd([1.0, 5.0, 10.0, 100.0], [2.0, 10.0, 20.0, 200.0])
    assert result == pytest.approx(1.0, abs=1e-9)


def test_gsd_perfect_predictions_is_one():
    # Sim == Real everywhere -> ratio is 1.0 everywhere -> still a consistent (zero) bias.
    result = _gsd([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    assert result == pytest.approx(1.0, abs=1e-9)


def test_gsd_varying_ratios_is_above_one_and_matches_hand_computed_value():
    # real=[10, 20], sim=[12, 18] -> ratios 1.2 and 0.9 (the same pair used for the geometric-mean
    # tests above), computed here from the ln(ratio)-sample-stdev definition directly rather than
    # by importing the module's own helper, so this doesn't just check "the function agrees with
    # itself".
    log_ratios = [math.log(1.2), math.log(0.9)]
    mean_log = sum(log_ratios) / 2
    sample_variance = sum((x - mean_log) ** 2 for x in log_ratios) / (2 - 1)  # ddof=1, n=2
    expected_gsd = math.exp(math.sqrt(sample_variance))

    result = _gsd([10.0, 20.0], [12.0, 18.0])

    assert result is not None
    assert result > 1.0
    assert result == pytest.approx(expected_gsd, rel=1e-9)


def test_gsd_skips_missing_and_nonpositive_points():
    # Only the (10, 12) and (20, 18) pairs are valid; None/zero/negative real values are excluded
    # the same way _geo_mean_relative_error_pct excludes them -- same expected value as the
    # varying-ratios case above since the surviving pairs are identical.
    result = _gsd([None, 0.0, -5.0, 10.0, 20.0], [1.0, 1.0, 1.0, 12.0, 18.0])
    expected = _gsd([10.0, 20.0], [12.0, 18.0])
    assert result == pytest.approx(expected, rel=1e-9)


def test_gsd_insufficient_data_is_none():
    # A sample standard deviation needs at least 2 qualifying points to be defined at all.
    assert _gsd([], []) is None
    assert _gsd([10.0], [12.0]) is None
    # Only one pair survives positivity-filtering, still below the n>=2 floor.
    assert _gsd([None, 20.0], [1.0, 18.0]) is None
    assert _gsd([0.0, -5.0], [1.0, 1.0]) is None


# --- Pearson correlation -----------------------------------------------------------------------


def test_pearson_r_perfect_positive_correlation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert _pearson_r(xs, ys) == pytest.approx(1.0)


def test_pearson_r_perfect_negative_correlation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [10.0, 8.0, 6.0, 4.0, 2.0]
    assert _pearson_r(xs, ys) == pytest.approx(-1.0)


def test_pearson_r_insufficient_data_is_none():
    assert _pearson_r([], []) is None
    assert _pearson_r([1.0], [2.0]) is None
    # Only one pair survives None-filtering, still below the n>=2 floor.
    assert _pearson_r([1.0, None], [2.0, 3.0]) is None


def test_pearson_r_zero_variance_is_none():
    # A constant series has no defined direction of relationship.
    assert _pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert _pearson_r([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None


def test_pearson_r_ignores_unpaired_none_points():
    xs = [1.0, None, 3.0, 4.0]
    ys = [2.0, 4.0, None, 8.0]
    # Only (1, 2) and (4, 8) pair up on both sides -- perfectly correlated.
    assert _pearson_r(xs, ys) == pytest.approx(1.0)


# --- Formatting helpers ------------------------------------------------------------------------


def test_fmt_delta_pct_signs_and_na():
    assert _fmt_delta_pct(12.34) == "+12.3%"
    assert _fmt_delta_pct(-8.5) == "-8.5%"
    assert _fmt_delta_pct(None) == "n/a"


def test_fmt_r_rounds_and_na():
    assert _fmt_r(0.9421) == "0.94"
    assert _fmt_r(-1.0) == "-1.00"
    assert _fmt_r(None) == "n/a"


def test_fmt_gsd_rounds_and_na():
    assert _fmt_gsd(1.0) == "1.00×"
    assert _fmt_gsd(1.2263) == "1.23×"
    assert _fmt_gsd(None) == "n/a"


# --- build_report / _data_table_html / write_html_report smoke tests ---------------------------


_TITLE_STATS_RE = re.compile(r"\(r=(-?\d+\.\d{2}|n/a), GSD=(\d+\.\d{2}×|n/a)\)")


def test_build_report_annotates_subplot_titles_with_correlation_and_gsd():
    real, sim = _synthetic_fixtures()

    fig = build_report(real, sim)

    titles = [ann.text for ann in fig.layout.annotations]
    assert len(titles) == 9  # one per _METRICS entry
    # Every metric has 2 well-defined, non-degenerate points here, so every title should carry
    # a numeric r and GSD rather than falling back to "n/a".
    assert all(_TITLE_STATS_RE.search(t) for t in titles)
    assert any("(r=" in t and "n/a" not in t for t in titles)


def test_build_report_handles_all_missing_gracefully():
    # No overlap at all between real and sim concurrencies -- every series is empty/None, so
    # every correlation/GSD must fall back to "n/a" rather than raising.
    real = [_agg_result(8, **_REAL_8)]
    sim = [_sim_result(32, **_SIM_16)]

    fig = build_report(real, sim)

    titles = [ann.text for ann in fig.layout.annotations]
    assert all(t.endswith("(r=n/a, GSD=n/a)") for t in titles)


def test_data_table_html_covers_all_metrics_with_real_sim_delta_columns():
    real, sim = _synthetic_fixtures()

    html = _data_table_html(real, sim)

    assert html.count("<th>Real</th>") == 9
    assert html.count("<th>Sim</th>") == 9
    assert html.count("<th>Δ</th>") == 9
    assert "Geometric mean error" in html
    assert "GSD" in html
    assert "Correlation (r)" in html
    # At least one signed per-cell/summary percentage is present.
    assert re.search(r"[+-]\d+\.\d%", html)
    # At least one GSD spread-factor value is present.
    assert re.search(r"\d+\.\d{2}×", html)
    # The table is wrapped for horizontal scrolling now that it's much wider.
    assert "overflow-x:auto" in html


def test_data_table_html_missing_concurrency_still_uses_colspan_fallback():
    real = [_agg_result(8, **_REAL_8), _agg_result(16, **_REAL_16)]
    sim = [_sim_result(8, **_SIM_8)]  # 16 missing entirely on the sim side

    html = _data_table_html(real, sim)

    assert "missing data for this concurrency level" in html
    assert "colspan='27'" in html  # 3 columns * 9 metrics


def test_write_html_report_renders_new_elements(tmp_path: Path):
    real, sim = _synthetic_fixtures()
    output_path = tmp_path / "report.html"

    write_html_report(real, sim, output_path, title="Synthetic smoke test")

    html = output_path.read_text()
    assert "Geometric mean error" in html
    assert "GSD" in html
    assert "Correlation (r)" in html
    assert _TITLE_STATS_RE.search(html)
    assert "Δ" in html
    assert re.search(r"[+-]\d+\.\d%", html)
    assert re.search(r"\d+\.\d{2}×", html)
