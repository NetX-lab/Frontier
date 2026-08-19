"""Parses sglang/vLLM bench_serving sweep output into per-concurrency-level records.

Both backends' sweep launchers emit the same envelope (Warm-up/Benchmark headers, a
Namespace(...) args dump, and a "Serving Benchmark Result" block of "Key: value" lines) --
just with different field sets. Rather than branching on backend, this parser treats the
result-block field map as a superset across both, so unsupported fields on either side just
come back None. Notable gaps for vLLM's bench_serving (as of the vllm_amd-standalone images
in use here): no achieved-concurrency ("Concurrency:") figure and no end-to-end latency
percentiles at all, since those require passing --percentile-metrics with "e2el" explicitly.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

_HEADER_RE = re.compile(
    r"^(Warm-up|Benchmark)\s*\|\s*concurrency=(\d+)\s*\|\s*prompts=(\d+)\s*$",
    re.MULTILINE,
)
# Newer sweep-launcher format (seen from a standalone single-node sglang/vLLM launcher):
# "### SGLang benchmark | concurrency=32 ###" / "### vLLM benchmark | concurrency=32 ###".
# No separate warm-up phase (these launchers are typically run with skip_warmup=true) and no
# prompt count in the header itself -- every match is a "benchmark" phase point, and
# num_prompts is recovered from the benchmark_args Namespace dump instead (see
# _ARG_PATTERNS["num_prompts"]). Tried only when _HEADER_RE finds nothing, so a file already
# matching the older format is never reinterpreted.
_HEADER_RE_NO_PHASE = re.compile(
    r"^#+\s*\S+\s+benchmark\s*\|\s*concurrency=(\d+)\s*#+\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_RESULT_BLOCK_RE = re.compile(
    r"={10,}\s*Serving Benchmark Result\s*={10,}\n(.*?)\n={10,}",
    re.DOTALL,
)
_KV_LINE_RE = re.compile(r"^([A-Za-z][^:]*):\s+(.+?)\s*$")

_ARG_PATTERNS = {
    "backend": re.compile(r"backend='([^']*)'"),
    "model": re.compile(r"model='([^']*)'"),
    "host": re.compile(r"host='([^']*)'"),
    "port": re.compile(r"port=(\d+)"),
    "random_input_len": re.compile(r"random_input_len=(\d+)"),
    "random_output_len": re.compile(r"random_output_len=(\d+)"),
    "random_range_ratio": re.compile(r"random_range_ratio=([\d.]+)"),
    "request_rate": re.compile(r"request_rate=(inf|[\d.]+)"),
    # vLLM's `vllm bench serve` Namespace additionally carries the sweep launcher's own
    # --input-len/--output-len flags. \b excludes the many other "*_output_len=" fields
    # (custom_output_len, sonnet_output_len, random_output_len, ...) that share the suffix.
    "input_len": re.compile(r"\binput_len=(\d+)"),
    "output_len": re.compile(r"\boutput_len=(\d+)"),
    # Only needed as a num_prompts source for _HEADER_RE_NO_PHASE captures, which carry no
    # prompt count of their own -- both engines' Namespace dumps include this field directly.
    "num_prompts": re.compile(r"\bnum_prompts=(\d+)"),
}
_ARG_INT_FIELDS = {
    "port", "random_input_len", "random_output_len", "input_len", "output_len", "num_prompts",
}
_ARG_FLOAT_FIELDS = {"random_range_ratio", "request_rate"}

# raw "Serving Benchmark Result" key -> (BenchmarkResult field name, value type)
_RESULT_FIELD_MAP = {
    "Backend": ("backend_reported", str),
    "Traffic request rate": ("request_rate_reported", float),
    "Max request concurrency": ("concurrency_reported", int),
    # vLLM's `vllm bench serve` reports the same figure under a longer key.
    "Maximum request concurrency": ("concurrency_reported", int),
    "Successful requests": ("successful_requests", int),
    "Failed requests": ("failed_requests", int),
    "Benchmark duration (s)": ("benchmark_duration_s", float),
    "Total input tokens": ("total_input_tokens", int),
    "Total generated tokens": ("total_generated_tokens", int),
    "Total generated tokens (retokenized)": ("total_generated_tokens_retokenized", int),
    "Request throughput (req/s)": ("request_throughput_req_s", float),
    "Input token throughput (tok/s)": ("input_token_throughput_tok_s", float),
    "Output token throughput (tok/s)": ("output_token_throughput_tok_s", float),
    "Peak output token throughput (tok/s)": ("peak_output_token_throughput_tok_s", float),
    "Peak concurrent requests": ("peak_concurrent_requests", int),
    "Total token throughput (tok/s)": ("total_token_throughput_tok_s", float),
    # Newer vLLM bench_serving capitalizes "Token"; exact-string lookup below needs both.
    "Total Token throughput (tok/s)": ("total_token_throughput_tok_s", float),
    "Concurrency": ("achieved_concurrency", float),
    "Mean E2E Latency (ms)": ("e2e_mean_ms", float),
    "Median E2E Latency (ms)": ("e2e_median_ms", float),
    "P90 E2E Latency (ms)": ("e2e_p90_ms", float),
    "P99 E2E Latency (ms)": ("e2e_p99_ms", float),
    "Mean TTFT (ms)": ("ttft_mean_ms", float),
    "Median TTFT (ms)": ("ttft_median_ms", float),
    "P99 TTFT (ms)": ("ttft_p99_ms", float),
    "Mean TPOT (ms)": ("tpot_mean_ms", float),
    "Median TPOT (ms)": ("tpot_median_ms", float),
    "P99 TPOT (ms)": ("tpot_p99_ms", float),
    "Mean ITL (ms)": ("itl_mean_ms", float),
    "Median ITL (ms)": ("itl_median_ms", float),
    "P95 ITL (ms)": ("itl_p95_ms", float),
    "P99 ITL (ms)": ("itl_p99_ms", float),
    "Max ITL (ms)": ("itl_max_ms", float),
}


@dataclass
class BenchmarkResult:
    """One sglang/vLLM bench_serving result block, tagged with its sweep phase/concurrency."""

    phase: str  # "warmup" | "benchmark"
    concurrency: int  # requested max-concurrency, from the sweep header
    num_prompts: int

    backend: Optional[str] = None
    model: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    random_input_len: Optional[int] = None
    random_output_len: Optional[int] = None
    random_range_ratio: Optional[float] = None
    request_rate: Optional[float] = None

    backend_reported: Optional[str] = None
    request_rate_reported: Optional[float] = None
    concurrency_reported: Optional[int] = None
    successful_requests: Optional[int] = None
    failed_requests: Optional[int] = None  # vLLM only; sglang's block has no equivalent line
    benchmark_duration_s: Optional[float] = None
    total_input_tokens: Optional[int] = None
    total_generated_tokens: Optional[int] = None
    total_generated_tokens_retokenized: Optional[int] = None
    request_throughput_req_s: Optional[float] = None
    input_token_throughput_tok_s: Optional[float] = None
    output_token_throughput_tok_s: Optional[float] = None
    peak_output_token_throughput_tok_s: Optional[float] = None
    peak_concurrent_requests: Optional[int] = None
    total_token_throughput_tok_s: Optional[float] = None
    achieved_concurrency: Optional[float] = None

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

    itl_mean_ms: Optional[float] = None
    itl_median_ms: Optional[float] = None
    itl_p95_ms: Optional[float] = None
    itl_p99_ms: Optional[float] = None
    itl_max_ms: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RunConfig:
    """Parsed config.txt written alongside bench_output.txt by the sweep launch script."""

    mode: Optional[str] = None
    node: Optional[str] = None
    prefill_nodes: Optional[str] = None
    decode_nodes: Optional[str] = None
    proxy_node: Optional[str] = None
    bench_ip: Optional[str] = None
    bench_port: Optional[int] = None
    bench_container: Optional[str] = None
    model_path: Optional[str] = None  # sglang sweep launcher's key: local filesystem path
    model: Optional[str] = None  # vLLM sweep launcher's key: HF repo id (e.g. "openai/gpt-oss-120b")
    served_model_name: Optional[str] = None
    input_len: Optional[int] = None
    output_len: Optional[int] = None
    concurrency: Optional[str] = None
    request_rate: Optional[str] = None
    warmup_mult: Optional[int] = None
    bench_mult: Optional[int] = None
    skip_warmup: Optional[bool] = None
    mori: Optional[bool] = None
    name: Optional[str] = None
    timestamp: Optional[str] = None
    raw: Dict[str, str] = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    """A full sweep: parsed results plus (if available) the run's config.txt."""

    results: List[BenchmarkResult]
    config: Optional[RunConfig] = None
    source: Optional[Path] = None

    def by_phase(self, phase: str) -> List[BenchmarkResult]:
        return [r for r in self.results if r.phase == phase]

    @property
    def benchmark(self) -> List[BenchmarkResult]:
        """The non-warmup sweep points -- the actual ground truth to validate against."""
        return sorted(self.by_phase("benchmark"), key=lambda r: r.concurrency)

    @property
    def warmup(self) -> List[BenchmarkResult]:
        return sorted(self.by_phase("warmup"), key=lambda r: r.concurrency)


def _coerce(value: str, kind: type):
    if kind is int:
        return int(float(value))
    if kind is float:
        return float(value)
    return value


def _parse_benchmark_args(args_text: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for name, pattern in _ARG_PATTERNS.items():
        m = pattern.search(args_text)
        if not m:
            continue
        raw = m.group(1)
        if name in _ARG_INT_FIELDS:
            parsed[name] = int(raw)
        elif name in _ARG_FLOAT_FIELDS:
            parsed[name] = float(raw)
        else:
            parsed[name] = raw
    return parsed


def _parse_result_block(block_text: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for line in block_text.splitlines():
        m = _KV_LINE_RE.match(line.strip())
        if not m:
            continue
        raw_key, raw_val = m.group(1).strip(), m.group(2).strip()
        mapping = _RESULT_FIELD_MAP.get(raw_key)
        if mapping is None:
            continue
        field_name, kind = mapping
        try:
            parsed[field_name] = _coerce(raw_val, kind)
        except ValueError:
            parsed[field_name] = raw_val
    return parsed


def parse_bench_output(text: str) -> List[BenchmarkResult]:
    """Parse a bench_output.txt into one BenchmarkResult per (phase, concurrency) point."""
    headers = list(_HEADER_RE.finditer(text))
    phased_headers = True
    if not headers:
        headers = list(_HEADER_RE_NO_PHASE.finditer(text))
        phased_headers = False
    results: List[BenchmarkResult] = []

    for i, header in enumerate(headers):
        if phased_headers:
            phase_label, concurrency, header_num_prompts = header.groups()
            phase = "warmup" if phase_label == "Warm-up" else "benchmark"
        else:
            (concurrency,) = header.groups()
            phase = "benchmark"
            header_num_prompts = None
        window_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        window = text[header.end():window_end]

        # vLLM emits a bare `Namespace(...)` line; sglang prefixes it `benchmark_args=`.
        args_match = re.search(r"(?:benchmark_args=)?Namespace\(.*\)", window)
        args_fields = _parse_benchmark_args(args_match.group(0)) if args_match else {}

        block_match = _RESULT_BLOCK_RE.search(window)
        result_fields = _parse_result_block(block_match.group(1)) if block_match else {}

        # vLLM sweeps pass the effective prefill/decode lengths as top-level --input-len/
        # --output-len; random_input_len/random_output_len are left at their argparse
        # defaults and don't reflect what was actually sampled. Prefer input_len/output_len
        # when present (vLLM), falling back to random_input_len/random_output_len (sglang).
        effective_input_len = args_fields.get("input_len", args_fields.get("random_input_len"))
        effective_output_len = args_fields.get("output_len", args_fields.get("random_output_len"))

        # vLLM's Result block has no "Input token throughput (tok/s)" line at all; derive it
        # the same way sglang computes it, so the two sides stay directly comparable.
        if result_fields.get("input_token_throughput_tok_s") is None:
            total_in = result_fields.get("total_input_tokens")
            duration = result_fields.get("benchmark_duration_s")
            if total_in is not None and duration:
                result_fields["input_token_throughput_tok_s"] = total_in / duration

        # _HEADER_RE_NO_PHASE carries no prompt count; recover it from the Namespace dump
        # instead (both engines report num_prompts there), falling back to the result block's
        # own successful-request count as a last resort. Fail loudly rather than fabricate a
        # count if truly nothing in this window says how many prompts were sent.
        num_prompts = header_num_prompts
        if num_prompts is None:
            num_prompts = args_fields.get("num_prompts")
        if num_prompts is None:
            num_prompts = result_fields.get("successful_requests")
        if num_prompts is None:
            raise ValueError(
                f"Could not determine num_prompts for concurrency={concurrency}: "
                "missing from the header, the benchmark_args Namespace, and the result block."
            )

        results.append(
            BenchmarkResult(
                phase=phase,
                concurrency=int(concurrency),
                num_prompts=int(num_prompts),
                backend=args_fields.get("backend"),
                model=args_fields.get("model"),
                host=args_fields.get("host"),
                port=args_fields.get("port"),
                random_input_len=effective_input_len,
                random_output_len=effective_output_len,
                random_range_ratio=args_fields.get("random_range_ratio"),
                request_rate=args_fields.get("request_rate"),
                **result_fields,
            )
        )

    return results


def parse_config_txt(text: str) -> RunConfig:
    raw: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        raw[key.strip()] = value.strip()

    def _int(v: Optional[str]) -> Optional[int]:
        return int(v) if v else None

    def _bool(v: Optional[str]) -> Optional[bool]:
        return v.lower() == "true" if v else None

    return RunConfig(
        mode=raw.get("mode") or None,
        node=raw.get("node") or None,
        prefill_nodes=raw.get("prefill_nodes") or None,
        decode_nodes=raw.get("decode_nodes") or None,
        proxy_node=raw.get("proxy_node") or None,
        bench_ip=raw.get("bench_ip") or None,
        bench_port=_int(raw.get("bench_port")),
        bench_container=raw.get("bench_container") or None,
        model_path=raw.get("model_path") or None,
        model=raw.get("model") or None,
        served_model_name=raw.get("served_model_name") or None,
        input_len=_int(raw.get("input_len")),
        output_len=_int(raw.get("output_len")),
        concurrency=raw.get("concurrency") or None,
        request_rate=raw.get("request_rate") or None,
        warmup_mult=_int(raw.get("warmup_mult")),
        bench_mult=_int(raw.get("bench_mult")),
        skip_warmup=_bool(raw.get("skip_warmup")),
        mori=_bool(raw.get("mori")),
        name=raw.get("name") or None,
        timestamp=raw.get("timestamp") or None,
        raw=raw,
    )


def engine_label(config: Optional[RunConfig]) -> str:
    """Best-effort "sglang"/"vLLM" label from config.txt's bench_container, for report titles."""
    container = (config.bench_container if config else None) or ""
    container = container.lower()
    if "sglang" in container:
        return "sglang"
    if "vllm" in container:
        return "vLLM"
    return "unknown engine"


def load_run(run_dir: Union[str, Path]) -> BenchmarkRun:
    """Load a benchmark_results/run_<label>/ directory produced by the sweep launcher."""
    run_dir = Path(run_dir)
    bench_output_path = run_dir / "bench_output.txt"
    config_path = run_dir / "config.txt"

    results = parse_bench_output(bench_output_path.read_text())
    config = parse_config_txt(config_path.read_text()) if config_path.exists() else None
    return BenchmarkRun(results=results, config=config, source=run_dir)


def _print_summary(rows: List[BenchmarkResult]) -> None:
    header = (
        f"{'phase':<10} {'conc':>5} {'prompts':>8} {'req/s':>7} {'in_tok/s':>10} "
        f"{'out_tok/s':>10} {'ttft_mean':>10} {'tpot_mean':>10} {'e2e_mean':>10} {'e2e_p99':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.phase:<10} {r.concurrency:>5} {r.num_prompts:>8} "
            f"{r.request_throughput_req_s or 0:>7.2f} {r.input_token_throughput_tok_s or 0:>10.1f} "
            f"{r.output_token_throughput_tok_s or 0:>10.1f} {r.ttft_mean_ms or 0:>10.1f} "
            f"{r.tpot_mean_ms or 0:>10.1f} {r.e2e_mean_ms or 0:>10.1f} {r.e2e_p99_ms or 0:>10.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse an sglang/vLLM bench_serving sweep log.")
    parser.add_argument("path", help="Run directory (containing bench_output.txt) or a bare log file")
    parser.add_argument("--phase", choices=["benchmark", "warmup", "all"], default="benchmark")
    parser.add_argument("--json", action="store_true", help="Print full records as JSON instead of a summary table")
    args = parser.parse_args()

    path = Path(args.path)
    run = load_run(path) if path.is_dir() else BenchmarkRun(results=parse_bench_output(path.read_text()), source=path)

    rows = run.benchmark if args.phase == "benchmark" else run.warmup if args.phase == "warmup" else run.results

    if args.json:
        payload = {"config": asdict(run.config) if run.config else None, "results": [r.to_dict() for r in rows]}
        print(json.dumps(payload, indent=2))
    else:
        _print_summary(rows)


if __name__ == "__main__":
    main()
