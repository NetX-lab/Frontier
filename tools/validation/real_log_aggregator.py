"""Aggregates repeated real-benchmark runs (same config, different reps) into per-concurrency
mean/std/min/max stats.

A single sglang/vLLM bench_serving sweep is noisy -- network jitter, scheduling luck, and
thermal/contention effects on a shared cluster all move the numbers around run to run. Plotting
one such run as a single point per concurrency level (what real_log_parser.BenchmarkRun gives
you) overstates how precisely Frontier's simulation should be expected to match it.

This module sits on top of real_log_parser: it discovers a directory of repeated
run_<label>/ subdirectories (same benchmark config, launched N times), parses each with
real_log_parser.load_run(), and aggregates them into one AggregatedResult per concurrency level
-- mean/std/min/max/n per metric, computed only over the repetitions that actually reported that
metric (a rep that crashed partway through, or is simply missing bench_output.txt, just
contributes less data instead of blowing up the whole aggregation).

Nothing is discarded: every contributing repetition's full BenchmarkResult is kept on
AggregatedResult.raw, so the aggregation is purely an additional view for plotting/reporting,
not a replacement for the underlying data.

AggregatedRun.benchmark mimics real_log_parser.BenchmarkRun.benchmark (a plain list of
per-concurrency BenchmarkResult, sorted by concurrency) using one representative "canonical"
repetition per concurrency level. frontier_cli_translator.build_sweep only needs workload shape
(concurrency, num_prompts, prefill/decode token lengths) to translate to a Frontier CLI
invocation -- which is identical across repetitions of "the same run and arguments" by
construction -- so it accepts an AggregatedRun exactly where it previously took a BenchmarkRun,
with no code changes on that side.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from tools.validation.real_log_parser import (
    BenchmarkResult,
    BenchmarkRun,
    RunConfig,
    load_run,
    parse_bench_output,
)

# Numeric BenchmarkResult fields worth aggregating -- the ones plotted/tabulated downstream,
# plus a few extras kept for completeness in --json dumps.
_AGGREGATE_FIELDS = [
    "successful_requests",
    "failed_requests",
    "benchmark_duration_s",
    "total_input_tokens",
    "total_generated_tokens",
    "request_throughput_req_s",
    "input_token_throughput_tok_s",
    "output_token_throughput_tok_s",
    "total_token_throughput_tok_s",
    "peak_output_token_throughput_tok_s",
    "peak_concurrent_requests",
    "achieved_concurrency",
    "e2e_mean_ms",
    "e2e_median_ms",
    "e2e_p90_ms",
    "e2e_p99_ms",
    "ttft_mean_ms",
    "ttft_median_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "tpot_median_ms",
    "tpot_p99_ms",
    "itl_mean_ms",
    "itl_median_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "itl_max_ms",
]


@dataclass
class MetricStats:
    """mean/std/min/max over the repetitions that reported a non-None value for one metric."""

    mean: Optional[float] = None
    std: Optional[float] = None  # sample stdev (n-1); 0.0 when n==1; None when n==0
    min: Optional[float] = None
    max: Optional[float] = None
    n: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AggregatedResult:
    """Real-side metrics at one concurrency level, aggregated across repeated runs."""

    concurrency: int
    num_prompts: int
    n_runs: int  # repetitions that had ANY data (result block or not) at this concurrency
    stats: Dict[str, MetricStats]
    raw: List[BenchmarkResult]  # every contributing repetition's full record -- nothing dropped
    canonical: BenchmarkResult  # first raw record with usable workload-shape fields

    def get(self, field_name: str) -> MetricStats:
        return self.stats.get(field_name, MetricStats())

    def to_dict(self) -> Dict:
        return {
            "concurrency": self.concurrency,
            "num_prompts": self.num_prompts,
            "n_runs": self.n_runs,
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
            "raw": [r.to_dict() for r in self.raw],
        }


@dataclass
class AggregatedRun:
    """A group of repeated real runs, aggregated -- the multi-rep counterpart to BenchmarkRun."""

    results: List[AggregatedResult]
    config: Optional[RunConfig]
    source: Path
    n_runs: int  # total repetition directories successfully loaded

    @property
    def benchmark(self) -> List[BenchmarkResult]:
        """Canonical per-concurrency records, sorted by concurrency -- drop-in compatible with
        BenchmarkRun.benchmark for frontier_cli_translator.build_sweep, which only needs
        workload shape, not the aggregated stats."""
        return [r.canonical for r in self.results]


def discover_run_dirs(path: Union[str, Path]) -> List[Path]:
    """Find run directories under `path`.

    If `path` itself contains bench_output.txt, it's a single run (returned as a 1-element
    list). Otherwise, its immediate subdirectories that contain bench_output.txt are treated as
    repetitions of the same run.
    """
    path = Path(path)
    if (path / "bench_output.txt").exists():
        return [path]
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_dir() and (p / "bench_output.txt").exists())


def aggregate_runs(runs: List[BenchmarkRun], phase: str = "benchmark") -> List[AggregatedResult]:
    """Group `runs`' records by concurrency and compute per-metric mean/std/min/max/n.

    A concurrency level with no repetition reporting usable workload-shape fields (prefill/decode
    token lengths, num_prompts -- needed to translate to a Frontier CLI invocation) is dropped
    from the result, since there's nothing to build a comparable simulation point from.
    """
    by_concurrency: Dict[int, List[BenchmarkResult]] = {}
    for run in runs:
        for r in run.by_phase(phase):
            by_concurrency.setdefault(r.concurrency, []).append(r)

    aggregated: List[AggregatedResult] = []
    for concurrency in sorted(by_concurrency):
        records = by_concurrency[concurrency]
        canonical = next(
            (
                r
                for r in records
                if r.random_input_len is not None
                and r.random_output_len is not None
                and r.num_prompts
            ),
            None,
        )
        if canonical is None:
            print(
                f"WARNING: concurrency={concurrency} has no repetition with usable "
                "prefill/decode-length data; skipping this concurrency level entirely.",
                file=sys.stderr,
            )
            continue

        stats: Dict[str, MetricStats] = {}
        for field_name in _AGGREGATE_FIELDS:
            values = [
                v for v in (getattr(r, field_name) for r in records) if v is not None
            ]
            if not values:
                stats[field_name] = MetricStats(n=0)
                continue
            stats[field_name] = MetricStats(
                mean=statistics.mean(values),
                std=statistics.stdev(values) if len(values) > 1 else 0.0,
                min=min(values),
                max=max(values),
                n=len(values),
            )

        aggregated.append(
            AggregatedResult(
                concurrency=concurrency,
                num_prompts=canonical.num_prompts,
                n_runs=len(records),
                stats=stats,
                raw=records,
                canonical=canonical,
            )
        )

    return aggregated


def load_and_aggregate(path: Union[str, Path], phase: str = "benchmark") -> AggregatedRun:
    """Load and aggregate real data from any of: a bare bench_output.txt file, a single
    run_<label>/ directory, or a directory containing several run_<label>/ repetitions.
    """
    path = Path(path)

    if path.is_file():
        run = BenchmarkRun(results=parse_bench_output(path.read_text()), source=path)
        return AggregatedRun(
            results=aggregate_runs([run], phase=phase), config=None, source=path, n_runs=1
        )

    run_dirs = discover_run_dirs(path)
    if not run_dirs:
        raise ValueError(
            f"No bench_output.txt found at {path} or in its immediate subdirectories. Pass "
            "either a single run_<label>/ directory or a directory containing several of them."
        )

    runs: List[BenchmarkRun] = []
    for run_dir in run_dirs:
        try:
            runs.append(load_run(run_dir))
        except Exception as exc:  # noqa: BLE001 -- one bad rep shouldn't sink the whole group
            print(f"WARNING: skipping {run_dir} ({exc})", file=sys.stderr)
    if not runs:
        raise ValueError(f"None of the {len(run_dirs)} candidate run dirs under {path} could be parsed.")

    config = next((r.config for r in runs if r.config is not None), None)
    return AggregatedRun(
        results=aggregate_runs(runs, phase=phase), config=config, source=path, n_runs=len(runs)
    )


def _fmt(stats: MetricStats, spec: str = "{:.2f}") -> str:
    if stats.n == 0:
        return "n/a"
    if stats.n == 1:
        return spec.format(stats.mean)
    return f"{spec.format(stats.mean)}±{spec.format(stats.std)}(n={stats.n})"


def _print_summary(results: List[AggregatedResult]) -> None:
    header = f"{'conc':>5} {'reps':>4} {'req/s':>18} {'ttft_mean':>18} {'tpot_mean':>18} {'e2e_mean':>18}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.concurrency:>5} {r.n_runs:>4} "
            f"{_fmt(r.get('request_throughput_req_s')):>18} "
            f"{_fmt(r.get('ttft_mean_ms')):>18} "
            f"{_fmt(r.get('tpot_mean_ms')):>18} "
            f"{_fmt(r.get('e2e_mean_ms')):>18}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate a group of repeated sglang/vLLM bench_serving sweeps into "
        "per-concurrency mean/std/min/max stats."
    )
    parser.add_argument(
        "path",
        help="A run_<label>/ directory, or a directory containing several of them "
        "(repeated runs of the same benchmark config)",
    )
    parser.add_argument("--phase", choices=["benchmark", "warmup"], default="benchmark")
    parser.add_argument(
        "--json", action="store_true", help="Emit full aggregated+raw records as JSON instead of a summary table"
    )
    args = parser.parse_args()

    agg_run = load_and_aggregate(args.path, phase=args.phase)

    if args.json:
        payload = {
            "source": str(agg_run.source),
            "n_runs": agg_run.n_runs,
            "config": asdict(agg_run.config) if agg_run.config else None,
            "results": [r.to_dict() for r in agg_run.results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"# {agg_run.n_runs} repetition(s) found under {agg_run.source}")
        _print_summary(agg_run.results)


if __name__ == "__main__":
    main()
