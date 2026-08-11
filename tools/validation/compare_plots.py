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


def build_report(
    real: Sequence[AggregatedResult], sim: Sequence[SimResult], title: str = "Real vs Simulated"
) -> go.Figure:
    """One subplot per metric, x-axis = concurrency, two series (Real with error bars, Simulated) each."""
    concurrencies = sorted({r.concurrency for r in real} | {s.concurrency for s in sim})
    n = len(_METRICS)
    cols = 2
    rows = (n + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[m[0] for m in _METRICS])

    for i, (name, real_field, sim_field, unit) in enumerate(_METRICS):
        row, col = i // cols + 1, i % cols + 1
        real_y, real_err = _real_series_with_error(real, concurrencies, real_field)
        sim_y = _sim_series(sim, concurrencies, sim_field)

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


def _data_table_html(real: Sequence[AggregatedResult], sim: Sequence[SimResult]) -> str:
    """Plain accessibility-fallback table -- reference data, not a verdict."""
    concurrencies = sorted({r.concurrency for r in real} | {s.concurrency for s in sim})
    real_by_c = {r.concurrency: r for r in real}
    sim_by_c = {s.concurrency: s for s in sim}

    rows_html = []
    for c in concurrencies:
        r, s = real_by_c.get(c), sim_by_c.get(c)
        rows_html.append(
            "<tr>"
            f"<td>{c}</td>"
            f"<td>{_fmt_agg(r.get('request_throughput_req_s'))}</td><td>{_fmt(s.request_throughput_req_s)}</td>"
            f"<td>{_fmt_agg(r.get('ttft_mean_ms'), '{:.1f}')}</td><td>{_fmt(s.ttft_mean_ms, '{:.1f}')}</td>"
            f"<td>{_fmt_agg(r.get('tpot_mean_ms'), '{:.1f}')}</td><td>{_fmt(s.tpot_mean_ms, '{:.1f}')}</td>"
            f"<td>{_fmt_agg(r.get('e2e_mean_ms'), '{:.1f}')}</td><td>{_fmt(s.e2e_mean_ms, '{:.1f}')}</td>"
            "</tr>"
            if r and s
            else f"<tr><td>{c}</td><td colspan='8'>missing data for this concurrency level</td></tr>"
        )

    return (
        "<p style='font-family:sans-serif;font-size:13px;color:#555'>Real columns show "
        "mean ± std across repeated runs of the same benchmark config (n shown when &gt;1; "
        "a bare value means only one repetition covered that field).</p>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse;font-family:sans-serif;font-size:13px'>"
        "<thead><tr><th>concurrency</th>"
        "<th>req/s (real)</th><th>req/s (sim)</th>"
        "<th>TTFT mean ms (real)</th><th>TTFT mean ms (sim)</th>"
        "<th>TPOT mean ms (real)</th><th>TPOT mean ms (sim)</th>"
        "<th>E2E mean ms (real)</th><th>E2E mean ms (sim)</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
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
