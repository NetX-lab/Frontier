"""Renders real-vs-simulated comparison plots across a concurrency sweep.

Purely visual comparison by design -- no pass/fail thresholds or scored verdicts.
Each chart is a single metric across concurrency levels, with two series (Real,
Simulated) so the viewer can judge similarity by eye.

The real side is one or more repeated benchmark runs (see real_log_aggregator), so it's
drawn as mean ± std across repetitions rather than a single point -- a lone real run is
noisy, and comparing a simulation against one noisy sample overstates how far off (or how
close) the simulation actually is. The simulated side stays a single line: Frontier's
simulation is deterministic given its inputs, so there's nothing to average.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import List, Optional, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.validation.metrics_extractor import SimResult
from tools.validation.real_log_aggregator import AggregatedResult, MetricStats, load_and_aggregate

# Validated adjacent categorical pair (blue/orange) -- see dataviz skill references/palette.md.
_COLOR_REAL = "#2a78d6"
_COLOR_SIM = "#eb6834"

# (chart title, real field, sim field, y-axis label)
_METRICS = [
    ("Request throughput", "request_throughput_req_s", "request_throughput_req_s", "req/s"),
    ("Input token throughput", "input_token_throughput_tok_s", "input_token_throughput_tok_s", "tok/s"),
    ("Output token throughput", "output_token_throughput_tok_s", "output_token_throughput_tok_s", "tok/s"),
    ("Mean TTFT", "ttft_mean_ms", "ttft_mean_ms", "ms"),
    ("P99 TTFT", "ttft_p99_ms", "ttft_p99_ms", "ms"),
    ("Mean TPOT", "tpot_mean_ms", "tpot_mean_ms", "ms"),
    ("Mean E2E latency", "e2e_mean_ms", "e2e_mean_ms", "ms"),
    ("P99 E2E latency", "e2e_p99_ms", "e2e_p99_ms", "ms"),
    ("Achieved concurrency", "achieved_concurrency", "achieved_concurrency", "in-flight requests"),
]


def _sim_series(records: Sequence[SimResult], concurrencies: List[int], field: str) -> List[Optional[float]]:
    by_conc = {r.concurrency: getattr(r, field, None) for r in records}
    return [by_conc.get(c) for c in concurrencies]


def _real_series_with_error(
    records: Sequence[AggregatedResult], concurrencies: List[int], field: str
) -> "tuple[List[Optional[float]], List[float]]":
    """Mean and std-dev (as a symmetric error-bar half-width) per concurrency level.

    A concurrency level with no repetition reporting this metric becomes a gap (None) rather
    than a fabricated zero -- matches how missing sim points are already handled.
    """
    by_conc = {r.concurrency: r.get(field) for r in records}
    means: List[Optional[float]] = []
    errors: List[float] = []
    for c in concurrencies:
        stats = by_conc.get(c, MetricStats())
        if stats.n == 0:
            means.append(None)
            errors.append(0.0)
        else:
            means.append(stats.mean)
            errors.append(stats.std or 0.0)
    return means, errors


# --- Real-vs-sim agreement statistics --------------------------------------------------------
#
# Two numbers summarize how well a metric's whole concurrency sweep agrees, on top of the
# eyeball comparison the plot already gives: a geometric-mean relative error (how far off, on
# average, and in which direction) and a Pearson correlation (whether the two series move
# together in shape, independent of any constant offset/scale between them). Both operate on
# per-concurrency (real_mean, sim_value) pairs built the same way the plotted series already are
# (_real_series_with_error / _sim_series), so "what counts as a data point here" never drifts
# from what's actually drawn on the chart.


def _relative_error_pct(real_value: Optional[float], sim_value: Optional[float]) -> Optional[float]:
    """Signed relative error of one sim value vs. its real counterpart, as a percentage.

    +12.3 means sim overpredicts real by 12.3%; -8.5 means sim underpredicts by 8.5%. None
    ("n/a" at render time) when either side is missing, or when either is zero/negative -- a
    ratio against a non-positive baseline isn't a meaningful percentage, and the log used by the
    geometric-mean aggregate below is undefined there too, so both stay consistent about which
    points count.
    """
    if real_value is None or sim_value is None or real_value <= 0 or sim_value <= 0:
        return None
    return (sim_value / real_value - 1) * 100


def _geo_mean_relative_error_pct(
    real_values: Sequence[Optional[float]], sim_values: Sequence[Optional[float]]
) -> Optional[float]:
    """Geometric-mean relative error across paired (real, sim) values, as a signed percentage.

    Why geometric mean of the ratio and not an arithmetic mean of the per-point signed percentages
    above: relative error is inherently multiplicative, and the linear percentage scale it's
    usually expressed on is *asymmetric* around "no error". Sim underpredicting real by half is
    -50%; sim overpredicting real by 2x is +100% -- even though both are the "same size" miss
    (a factor of 2) just in opposite directions. A plain arithmetic mean of signed percentages
    therefore weights every overprediction more heavily than an equal-magnitude underprediction
    (which is floored at -100%), which systematically biases the average toward looking like
    "sim underestimates" even for errors that are symmetric multiplicatively.

    Averaging in log space fixes this: ln(ratio) is exactly as far from zero for a 2x
    overprediction as for a 2x underprediction (+0.693 vs. -0.693), so exponentiating the mean
    log-ratio back out gives an aggregate that treats over- and under-prediction symmetrically --
    the geometric mean of the ratios, reported the same way as the per-point signed percentage
    (GM - 1) * 100 so the two read on the same scale.

    Only strictly-positive, present-on-both-sides pairs contribute (same rule as
    _relative_error_pct, applied point-by-point here); returns None ("n/a") if no pair qualifies.
    """
    ratios = [
        sim / real
        for real, sim in zip(real_values, sim_values)
        if real is not None and sim is not None and real > 0 and sim > 0
    ]
    if not ratios:
        return None
    log_mean = statistics.mean(math.log(ratio) for ratio in ratios)
    return (math.exp(log_mean) - 1) * 100


def _gsd(
    real_values: Sequence[Optional[float]], sim_values: Sequence[Optional[float]]
) -> Optional[float]:
    """Geometric standard deviation of the sim/real ratio across paired values.

    Complements _geo_mean_relative_error_pct: that answers "how far off is the simulator, on
    average, and in which direction" (a single multiplicative bias). GSD answers a different
    question -- "how *consistent* is that multiplicative error across concurrency levels" --
    which neither the geometric mean nor the correlation below can tell you. Correlation only
    checks that Real and Sim move together in a straight line; it's blind to a constant
    multiplicative offset between them (Sim = 3.5 * Real at every point still gives r=1.0, a
    perfect *trend* match with a badly wrong *magnitude*). GSD is exp(sample-stdev(ln(ratio))):
    it comes out at exactly 1.0 when every point shares the same ratio -- however far that
    shared ratio is from 1 (a consistent bias, which the geometric-mean-error metric above
    already captures) -- and grows above 1.0 as the ratio itself varies from point to point, an
    operating-point-dependent error that a single averaged figure can hide entirely.

    Same positivity rule as _geo_mean_relative_error_pct (only strictly-positive, paired real/sim
    values contribute -- ln is undefined at/below zero: an explicit filter here, not a silent
    NaN/inf) and the same sample-statistics convention used throughout this module
    (statistics.stdev's default ddof=1, i.e. the n-1 sample standard deviation). A sample stdev
    needs at least 2 qualifying pairs to be defined at all; returns None ("n/a" at render time)
    below that, same as _pearson_r's handling of insufficient data.
    """
    ratios = [
        sim / real
        for real, sim in zip(real_values, sim_values)
        if real is not None and sim is not None and real > 0 and sim > 0
    ]
    if len(ratios) < 2:
        return None
    log_ratios = [math.log(ratio) for ratio in ratios]
    return math.exp(statistics.stdev(log_ratios))


def _pearson_r(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    """Pearson correlation coefficient between two paired series, hand-rolled.

    Deliberately not statistics.correlation: it's Python 3.10+ only, and this codebase already
    hand-rolls its small stats needs with statistics.mean/statistics.stdev rather than reach for
    version-gated stdlib additions or an extra dependency (see real_log_aggregator.MetricStats,
    which does the same for mean/stdev). Sample covariance and sample stdev both use the same
    (n-1) denominator, which cancels out of the ratio, so this is equivalent to the textbook
    Pearson r computed on population statistics.

    Unlike the geometric-mean error above, no positivity requirement -- correlation cares about
    whether the two series move together in shape, not about the sign or scale of their values.
    Needs at least 2 paired points and non-zero variance in *both* series (a single point has
    no spread to correlate against, and a constant series has no defined direction of
    relationship); otherwise returns None ("n/a" at render time) rather than raising or dividing
    by zero.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    x_std = statistics.stdev(x_vals)
    y_std = statistics.stdev(y_vals)
    if x_std == 0 or y_std == 0:
        return None
    x_mean = statistics.mean(x_vals)
    y_mean = statistics.mean(y_vals)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / (len(pairs) - 1)
    return covariance / (x_std * y_std)


def build_report(
    real: Sequence[AggregatedResult], sim: Sequence[SimResult], title: str = "Real vs Simulated"
) -> go.Figure:
    """One subplot per metric, x-axis = concurrency, two series (Real with error bars, Simulated) each."""
    concurrencies = sorted({r.concurrency for r in real} | {s.concurrency for s in sim})
    n = len(_METRICS)
    cols = 2
    rows = (n + cols - 1) // cols

    # Each metric's real/sim series is computed once, up front, so the exact same pairing feeds
    # both the correlation/GSD shown in the subplot title (make_subplots needs all titles before
    # any trace can be added to a specific cell) and the traces plotted into that cell below.
    series_by_metric = []
    subplot_titles = []
    for name, real_field, sim_field, _unit in _METRICS:
        real_y, real_err = _real_series_with_error(real, concurrencies, real_field)
        sim_y = _sim_series(sim, concurrencies, sim_field)
        r = _pearson_r(real_y, sim_y)
        gsd_value = _gsd(real_y, sim_y)
        series_by_metric.append((real_y, real_err, sim_y))
        subplot_titles.append(
            f"{name} (r={_fmt_r(r)}, GSD={_fmt_gsd(gsd_value)})"
        )

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for i, (_name, _real_field, _sim_field, unit) in enumerate(_METRICS):
        row, col = i // cols + 1, i % cols + 1
        real_y, real_err, sim_y = series_by_metric[i]

        fig.add_trace(
            go.Scatter(
                x=concurrencies, y=real_y, name="Real", legendgroup="real",
                showlegend=(i == 0), mode="lines+markers",
                line=dict(color=_COLOR_REAL, width=2), marker=dict(size=8),
                error_y=dict(type="data", array=real_err, visible=True, color=_COLOR_REAL, thickness=1.5, width=4),
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=concurrencies, y=sim_y, name="Simulated", legendgroup="sim",
                showlegend=(i == 0), mode="lines+markers",
                line=dict(color=_COLOR_SIM, width=2, dash="dot"), marker=dict(size=8, symbol="diamond"),
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text="concurrency", type="log", row=row, col=col)
        fig.update_yaxes(title_text=unit, row=row, col=col)

    fig.update_layout(
        title=title,
        height=320 * rows,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _fmt(value: Optional[float], spec: str = "{:.2f}") -> str:
    """Format a plain (sim-side) metric value, or a placeholder when it's missing."""
    return spec.format(value) if value is not None else "n/a"


def _fmt_agg(stats: MetricStats, spec: str = "{:.2f}") -> str:
    """Format a real-side aggregated metric: bare value at n=1, "mean ± std (n=N)" at n>1.

    vLLM's bench_serving output also omits E2E latency and achieved-concurrency entirely unless
    --percentile-metrics includes "e2el", so n==0 (no repetition reported this field) is
    routinely hit too -- unlike sglang, where every field here is always populated.
    """
    if stats.n == 0:
        return "n/a"
    if stats.n == 1:
        return spec.format(stats.mean)
    return f"{spec.format(stats.mean)} ± {spec.format(stats.std)} (n={stats.n})"


def _fmt_delta_pct(value: Optional[float]) -> str:
    """Format a signed relative-error percentage (per-point Δ or the geometric-mean aggregate),
    e.g. "+12.3%" / "-8.5%", or "n/a" when it couldn't be computed (see _relative_error_pct /
    _geo_mean_relative_error_pct for why)."""
    return f"{value:+.1f}%" if value is not None else "n/a"


def _fmt_r(value: Optional[float]) -> str:
    """Format a Pearson correlation coefficient to 2 decimals, or "n/a" (see _pearson_r)."""
    return f"{value:.2f}" if value is not None else "n/a"


def _fmt_gsd(value: Optional[float]) -> str:
    """Format a geometric standard deviation as a multiplicative spread factor, e.g. "1.15×"
    (1.0x = perfectly consistent error across concurrency levels), or "n/a" (see _gsd)."""
    return f"{value:.2f}×" if value is not None else "n/a"


# One decimal is enough resolution for millisecond latencies; everything else (throughputs,
# achieved concurrency) gets two -- matches the precision the previous hand-written table used
# for the 4 metrics it covered, now applied uniformly across all of _METRICS.
_MS_FIELDS = {"ttft_mean_ms", "ttft_p99_ms", "tpot_mean_ms", "e2e_mean_ms", "e2e_p99_ms"}


def _table_spec(real_field: str) -> str:
    return "{:.1f}" if real_field in _MS_FIELDS else "{:.2f}"


def _data_table_html(real: Sequence[AggregatedResult], sim: Sequence[SimResult]) -> str:
    """Plain accessibility-fallback table -- reference data, not a verdict.

    Covers every metric in _METRICS (so the table and the plot never drift apart), three
    sub-columns each: Real | Sim | Δ, where Δ is the signed per-point relative error from
    _relative_error_pct. Three summary rows follow the per-concurrency rows: a per-metric
    geometric-mean relative error (average multiplicative bias), a per-metric geometric standard
    deviation (consistency of that bias across concurrency levels -- see _gsd for why this is a
    distinct question from both the error and the correlation below), and a per-metric
    real-vs-sim Pearson correlation (whether the two series share a trend, independent of any
    constant offset/scale between them). The error and correlation aggregates are also annotated
    onto the plot's subplot titles; all three are given here as an at-a-glance table for whoever's
    scanning the HTML fallback (or reading it after the JS-rendered chart has been stripped, e.g.
    in a text-only diff of the report).
    """
    concurrencies = sorted({r.concurrency for r in real} | {s.concurrency for s in sim})
    real_by_c = {r.concurrency: r for r in real}
    sim_by_c = {s.concurrency: s for s in sim}
    n_metrics = len(_METRICS)

    rows_html = []
    for c in concurrencies:
        r, s = real_by_c.get(c), sim_by_c.get(c)
        if r is None or s is None:
            rows_html.append(
                f"<tr><td>{c}</td>"
                f"<td colspan='{3 * n_metrics}'>missing data for this concurrency level</td></tr>"
            )
            continue
        cells = [f"<td>{c}</td>"]
        for _name, real_field, sim_field, _unit in _METRICS:
            spec = _table_spec(real_field)
            real_stats = r.get(real_field)
            real_value = real_stats.mean if real_stats.n > 0 else None
            sim_value = getattr(s, sim_field, None)
            delta = _relative_error_pct(real_value, sim_value)
            cells.append(f"<td>{_fmt_agg(real_stats, spec)}</td>")
            cells.append(f"<td>{_fmt(sim_value, spec)}</td>")
            cells.append(f"<td>{_fmt_delta_pct(delta)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    # Summary rows: one aggregate figure per metric, spanning that metric's Real|Sim|Δ group so
    # it reads as a single value under the 3-wide column band rather than three empty-looking
    # cells. Computed over the same per-concurrency (real_mean, sim_value) pairing as the rows
    # above (via _real_series_with_error / _sim_series), not just the concurrencies where a full
    # row happened to render -- a metric can be n/a in the table's colspan-fallback row (missing
    # from one side entirely) yet still have a usable point at another concurrency level.
    geo_cells = ["<td><strong>Geometric mean error</strong></td>"]
    gsd_cells = ["<td><strong>GSD</strong></td>"]
    corr_cells = ["<td><strong>Correlation (r)</strong></td>"]
    for _name, real_field, sim_field, _unit in _METRICS:
        real_y, _real_err = _real_series_with_error(real, concurrencies, real_field)
        sim_y = _sim_series(sim, concurrencies, sim_field)
        geo_mean_pct = _geo_mean_relative_error_pct(real_y, sim_y)
        gsd_value = _gsd(real_y, sim_y)
        r_value = _pearson_r(real_y, sim_y)
        geo_cells.append(
            f"<td colspan='3' style='text-align:center'>{_fmt_delta_pct(geo_mean_pct)}</td>"
        )
        gsd_cells.append(f"<td colspan='3' style='text-align:center'>{_fmt_gsd(gsd_value)}</td>")
        corr_cells.append(f"<td colspan='3' style='text-align:center'>{_fmt_r(r_value)}</td>")
    rows_html.append("<tr>" + "".join(geo_cells) + "</tr>")
    rows_html.append("<tr>" + "".join(gsd_cells) + "</tr>")
    rows_html.append("<tr>" + "".join(corr_cells) + "</tr>")

    header_groups = "".join(f"<th colspan='3'>{name} ({unit})</th>" for name, _rf, _sf, unit in _METRICS)
    header_subcols = "<th>Real</th><th>Sim</th><th>Δ</th>" * n_metrics

    return (
        "<p style='font-family:sans-serif;font-size:13px;color:#555'>Real columns show "
        "mean ± std across repeated runs of the same benchmark config (n shown when &gt;1; "
        "a bare value means only one repetition covered that field). Δ is the signed relative "
        "error of Sim vs. Real at that point ("
        "<code>(sim/real - 1) * 100</code>, e.g. +12.3% means sim overpredicts real by 12.3%); "
        "it's \"n/a\" wherever either side is missing or non-positive. The \"Geometric mean "
        "error\" summary row is the per-metric aggregate of that same signed error across all "
        "concurrency levels -- a geometric (not arithmetic) mean in the underlying ratio, so "
        "over- and under-prediction are weighted symmetrically. The \"GSD\" (geometric standard "
        "deviation) row measures how <em>consistent</em> that multiplicative error is across "
        "concurrency levels, as a spread factor (1.00× = the sim/real ratio is identical at "
        "every concurrency level, however far from 1 that ratio is; larger values mean the "
        "error itself varies from point to point) -- a simulator can have a small average error "
        "and a large GSD (accurate on average, but only by cancellation) or vice versa. The "
        "\"Correlation (r)\" row is the Pearson correlation between the real and simulated "
        "series across concurrency levels for that metric, independent of any constant offset "
        "or scale between them -- note this means high correlation alone does not imply "
        "agreement (e.g. Sim = 3.5 × Real at every point still gives r=1.00); use it alongside "
        "the error and GSD rows above, not in place of them.</p>"
        "<div style='overflow-x:auto'>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse;font-family:sans-serif;"
        "font-size:13px;white-space:nowrap'>"
        f"<thead><tr><th rowspan='2'>concurrency</th>{header_groups}</tr>"
        f"<tr>{header_subcols}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></div>"
    )


def write_html_report(
    real: Sequence[AggregatedResult],
    sim: Sequence[SimResult],
    output_path: Path,
    title: str,
    subtitle: Optional[str] = None,
) -> None:
    fig = build_report(real, sim, title)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    table_html = _data_table_html(real, sim)
    subtitle_html = f"<p style='color:#555'>{subtitle}</p>" if subtitle else ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
        f"<body style='font-family:sans-serif;max-width:1400px;margin:2rem auto'>"
        f"<h1>{title}</h1>"
        f"{subtitle_html}"
        f"{fig_html}"
        f"<h2>Data</h2>{table_html}"
        f"</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-dir", required=True,
        help="Real benchmark_results/run_<label>/ directory, or a directory containing several "
        "repeated run_<label>/ subdirectories (same config, different reps) -- see real_log_aggregator",
    )
    parser.add_argument("--sim-json", required=True, help="JSON list of SimResult records (one per concurrency level, from metrics_extractor --json)")
    parser.add_argument("--title", default="Real vs Simulated")
    parser.add_argument("-o", "--output", default="comparison_report.html")
    args = parser.parse_args()

    agg_run = load_and_aggregate(args.real_dir)

    sim_payload = json.loads(Path(args.sim_json).read_text())
    sim_records = [SimResult(**s) for s in sim_payload]

    subtitle = f"Real side: mean ± std across {agg_run.n_runs} repeated benchmark run(s)" if agg_run.n_runs > 1 else None
    write_html_report(agg_run.results, sim_records, Path(args.output), args.title, subtitle=subtitle)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
