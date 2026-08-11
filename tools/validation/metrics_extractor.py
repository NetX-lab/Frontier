"""Extracts normalized serving metrics from a completed Frontier online-serving run.

Uses the same field names/units as real_log_parser.BenchmarkResult where they overlap, so the
two sides can be compared directly without a translation step at plot/report time.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from frontier.utils.output_paths import sanitize_output_component, validate_run_id


@dataclass
class SimResult:
    """Frontier-side counterpart to real_log_parser.BenchmarkResult."""

    run_id: str
    concurrency: int  # the real concurrency level this sim point was built to represent
    num_prompts: int
    calibrated_qps: Optional[float] = None  # only set when the sim used poisson (legacy) mode

    successful_requests: Optional[int] = None
    benchmark_duration_s: Optional[float] = None
    request_throughput_req_s: Optional[float] = None
    input_token_throughput_tok_s: Optional[float] = None
    output_token_throughput_tok_s: Optional[float] = None
    total_token_throughput_tok_s: Optional[float] = None
    achieved_concurrency: Optional[float] = None  # Little's law: req/s * mean(e2e_time_s)

    e2e_mean_ms: Optional[float] = None
    e2e_median_ms: Optional[float] = None
    e2e_p90_ms: Optional[float] = None
    e2e_p99_ms: Optional[float] = None

    ttft_mean_ms: Optional[float] = None
    ttft_median_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None

    tpot_mean_ms: Optional[float] = None
    tpot_median_ms: Optional[float] = None
    tpot_p99_ms: Optional[float] = None

    def to_dict(self):
        return asdict(self)


def find_run_dir(output_root: str, model_name: str, run_id: str, workload_type: str = "online_serving") -> Path:
    """Rebuild the canonical <output_root>/<model>/<workload_type>/<run_id>/ path Frontier writes to."""
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root / sanitize_output_component(model_name, "model_name") / sanitize_output_component(
        workload_type, "workload_type"
    ) / validate_run_id(run_id)


def extract_sim_result(
    run_dir: Path,
    *,
    run_id: str,
    concurrency: int,
    num_prompts: int,
    calibrated_qps: Optional[float] = None,
) -> SimResult:
    system_metrics = json.loads((Path(run_dir) / "system_metrics.json").read_text())

    throughput = system_metrics.get("throughput_metrics", {})
    ttft = system_metrics.get("ttft_statistics", {})
    tpot = system_metrics.get("tpot_statistics", {})
    e2e = system_metrics.get("request_e2e_time_statistics", {})
    meta = system_metrics.get("simulation_metadata", {})

    tokens_per_second = throughput.get("tokens_per_second")
    decode_tokens_per_second = throughput.get("decode_tokens_per_second")
    input_tok_s = (
        tokens_per_second - decode_tokens_per_second
        if tokens_per_second is not None and decode_tokens_per_second is not None
        else None
    )

    req_s = throughput.get("requests_per_second")
    e2e_mean_ms = e2e.get("mean")
    achieved_concurrency = req_s * e2e_mean_ms / 1000.0 if req_s is not None and e2e_mean_ms is not None else None

    return SimResult(
        run_id=run_id,
        concurrency=concurrency,
        num_prompts=num_prompts,
        calibrated_qps=calibrated_qps,
        successful_requests=meta.get("completed_requests"),
        benchmark_duration_s=throughput.get("total_duration_seconds"),
        request_throughput_req_s=req_s,
        input_token_throughput_tok_s=input_tok_s,
        output_token_throughput_tok_s=decode_tokens_per_second,
        total_token_throughput_tok_s=tokens_per_second,
        achieved_concurrency=achieved_concurrency,
        e2e_mean_ms=e2e.get("mean"),
        e2e_median_ms=e2e.get("median"),
        e2e_p90_ms=e2e.get("p90"),
        e2e_p99_ms=e2e.get("p99"),
        ttft_mean_ms=ttft.get("mean"),
        ttft_median_ms=ttft.get("median"),
        ttft_p99_ms=ttft.get("p99"),
        tpot_mean_ms=tpot.get("mean"),
        tpot_median_ms=tpot.get("median"),
        tpot_p99_ms=tpot.get("p99"),
    )


def _print_summary(results) -> None:
    header = (
        f"{'run_id':<28} {'conc':>5} {'req/s':>7} {'in_tok/s':>10} "
        f"{'out_tok/s':>10} {'ttft_mean':>10} {'tpot_mean':>10} {'e2e_mean':>10} {'e2e_p99':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.run_id:<28} {r.concurrency:>5} "
            f"{r.request_throughput_req_s or 0:>7.2f} {r.input_token_throughput_tok_s or 0:>10.1f} "
            f"{r.output_token_throughput_tok_s or 0:>10.1f} {r.ttft_mean_ms or 0:>10.1f} "
            f"{r.tpot_mean_ms or 0:>10.1f} {r.e2e_mean_ms or 0:>10.1f} {r.e2e_p99_ms or 0:>10.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract normalized metrics from a Frontier online-serving run.")
    parser.add_argument("run_dir", help="Path to the run's output directory (contains system_metrics.json)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--calibrated-qps", type=float, default=None, help="Only meaningful for poisson-mode sim runs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = extract_sim_result(
        Path(args.run_dir),
        run_id=args.run_id,
        concurrency=args.concurrency,
        num_prompts=args.num_prompts,
        calibrated_qps=args.calibrated_qps,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_summary([result])


if __name__ == "__main__":
    main()
