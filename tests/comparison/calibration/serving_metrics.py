"""Read vLLM's Prometheus /metrics page and compare native P/D transfer series between scrapes.

prometheus_client exposes a counter as `<name>_total` and a histogram as `<name>_count`,
`<name>_sum` and `<name>_bucket`. Every series carries an `engine` label, the DP rank inside
the serving pod, so per-lane values stay separate and totals sum over lanes. The NIXL series
come from `nixl/stats.py`; the prompt-token split by source from `v1/metrics/stats.py`
(`local_compute + local_cache_hit + external_kv_transfer = total`).
"""

from __future__ import annotations

import re
import urllib.request

NIXL_FAILURE_COUNTERS = (
    "vllm:nixl_num_failed_transfers",
    "vllm:nixl_num_failed_notifications",
    # Counted on P: requests whose held blocks reached the lease end before D's read notification.
    "vllm:nixl_num_kv_expired_reqs",
    "vllm:nixl_num_notifications_after_expiry",
)
NIXL_TRANSFER_HISTOGRAMS = (
    "vllm:nixl_bytes_transferred",
    "vllm:nixl_num_descriptors",
    "vllm:nixl_xfer_time_seconds",
    "vllm:nixl_post_time_seconds",
)
REQUEST_HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prefill_kv_computed_tokens",
)
PROMPT_TOKEN_SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")
KV_CACHE_USAGE = "vllm:kv_cache_usage_perc"
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+-?\d+)?$")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
ESCAPE = re.compile(r"\\(.)")


def parse_prometheus(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Return every sample of a text exposition as (name, labels, value)."""
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.match(line)
        if match is None:
            raise ValueError(f"Unparsable Prometheus sample: {line!r}")
        name, body, value = match.groups()
        labels = {key: ESCAPE.sub(lambda m: "\n" if m.group(1) == "n" else m.group(1), raw)
                  for key, raw in LABEL.findall(body or "")}
        samples.append((name, labels, float(value)))
    return samples


def scrape(base_url: str, timeout_s: float) -> dict:
    """Read `<base_url>/metrics` directly (no proxy); report a failure instead of raising."""
    url = base_url.rstrip("/") + "/metrics"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout_s) as response:
            text = response.read().decode()
        return {"url": url, "text": text, "samples": parse_prometheus(text), "error": None}
    except Exception as error:
        return {"url": url, "text": None, "samples": None, "error": f"{type(error).__name__}: {error}"}


def engine_values(samples, name: str, **labels: str) -> dict[str, float]:
    """Map each `engine` label value to the summed value of the matching samples."""
    values: dict[str, float] = {}
    for sample_name, sample_labels, value in samples:
        if sample_name == name and all(sample_labels.get(k) == v for k, v in labels.items()):
            engine = sample_labels.get("engine", "")
            values[engine] = values.get(engine, 0.0) + value
    return values


def transfer_delta(before, after) -> dict:
    """Changes of the NIXL, prompt-source and request series from `before` to `after`.

    A series absent from both scrapes reports `present: False`, so a zero change is never
    read from a series the server did not expose. KV cache usage is the gauge at `after`.
    """

    def change(name: str, **labels: str) -> dict:
        start, end = engine_values(before, name, **labels), engine_values(after, name, **labels)
        per_engine = {engine: end.get(engine, 0.0) - start.get(engine, 0.0)
                      for engine in sorted(set(start) | set(end))}
        return {"present": bool(per_engine), "per_engine": per_engine,
                "total": sum(per_engine.values())}

    def histogram(name: str) -> dict:
        return {"count": change(name + "_count"), "sum": change(name + "_sum")}

    return {
        "nixl_failures": {name: change(name + "_total") for name in NIXL_FAILURE_COUNTERS},
        "nixl_transfers": {name: histogram(name) for name in NIXL_TRANSFER_HISTOGRAMS},
        "prompt_tokens_by_source": {
            source: change("vllm:prompt_tokens_by_source_total", source=source)
            for source in PROMPT_TOKEN_SOURCES},
        "prompt_tokens": change("vllm:prompt_tokens_total"),
        "requests": {name: histogram(name) for name in REQUEST_HISTOGRAMS},
        "kv_cache_usage_perc": engine_values(after, KV_CACHE_USAGE),
    }


def expected_pull_prompt_sources(num_prompt_tokens: int, block_size: int) -> dict[str, int]:
    """Prompt split on D when P leaves `block_size` trailing tokens to D (Step-5 sets
    `kv_cache_prefix_recompute_tokens = block_size`) and transfers whole blocks only."""
    transferred = num_prompt_tokens - block_size
    transferred -= transferred % block_size
    return {"local_compute": num_prompt_tokens - transferred, "local_cache_hit": 0,
            "external_kv_transfer": transferred}


def transfer_checks(prefill: dict, decode: dict, expected_sources: dict[str, int]) -> dict:
    """Judge one wave's `transfer_delta` of both roles: D read bytes and descriptors through
    NIXL, no failure series changed on either role, and D's prompt split matches."""
    checks = {}
    reads = {name: decode["nixl_transfers"][name]["sum"]["total"]
             for name in ("vllm:nixl_bytes_transferred", "vllm:nixl_num_descriptors")}
    failures = {role: {name: (series["total"] if series["present"] else None)
                       for name, series in delta["nixl_failures"].items()}
                for role, delta in (("prefill", prefill), ("decode", decode))}
    checks["nixl_counters"] = {
        "passed": all(value > 0 for value in reads.values())
                  and all(value == 0 for role in failures.values() for value in role.values()),
        "decode_reads": reads, "failure_changes": failures}
    sources = {source: decode["prompt_tokens_by_source"][source]["total"]
               for source in PROMPT_TOKEN_SOURCES}
    checks["decode_prompt_sources"] = {
        "passed": sources == {k: float(v) for k, v in expected_sources.items()},
        "observed": sources, "expected": expected_sources,
        "per_engine": {source: decode["prompt_tokens_by_source"][source]["per_engine"]
                       for source in PROMPT_TOKEN_SOURCES}}
    return checks
