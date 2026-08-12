# Real Benchmark Data Quality Findings

Two separate issues found while inspecting `tools/inference_bench/` captures — both affect
whether a validation comparison is actually apples-to-apples, independent of anything on the
simulator side.

## 1. Open-loop vs. closed-loop: the discriminator, and what it implies

Both `config.txt` and the `Namespace(...)` args line in `bench_output.txt` carry a
`request_rate` field for every run:

- **Closed-loop** → `request_rate=inf`. The client gates request generation with a concurrency
  semaphore (`max_concurrency`) — a new request only goes out when a slot frees up.
- **Open-loop** → `request_rate=<finite number>`. Requests are injected on a timer at that
  target rate regardless of how many are still in flight. `max_concurrency` (typically 512) is
  still recorded but is a safety ceiling, not the controlling mechanism — don't read
  `concurrency` alone as evidence of which mode was used.

The finite value *is* the Poisson rate directly, no separate field to hunt for. vLLM makes this
explicit — `Burstiness factor: 1.0 (Poisson process)`, with a `burstiness=1.0` field (values
away from 1.0 would model burstier/more-uniform arrivals; every capture we have uses 1.0).
SGLang doesn't print "Poisson" or expose a burstiness knob at all, but a finite `request_rate`
is documented sglang behavior for Poisson arrivals at that mean rate.

**Real systems get genuinely overloaded at plausible-looking rates.** One qwen3/sglang capture
at "pct90" (supposedly 90% of some reference peak): `request_rate=0.738` injected,
`request_throughput_req_s=0.07` achieved — a 10x gap. Confirmed via E2E latency this is real
queueing backlog, not a benchmark-duration artifact: TTFT stayed reasonable (~1.6s, measured from
when a request actually starts) while mean E2E latency was **536-543 seconds** — the gap is
almost entirely time spent queued, not service time. Simulating this workload means feeding
Frontier the *injected* rate (`real.request_rate`), not the achieved throughput — see the open
gap in [VALIDATION_TOOL.md](VALIDATION_TOOL.md#known-gap-open-loop-poisson-rate-logs-arent-runnable-yet).

## 2. gpt-oss stops generating far short of the requested output length

Checked `total_generated_tokens` against the naive expectation (`1024 × num_prompts`, since
every capture's `config.txt` sets `output_len=1024`) across every closed-loop capture:

| model / backend | ratio (actual / expected) | avg tokens/request |
|---|---|---|
| deepseek (sglang, vLLM) | 1.0 | 1024 |
| qwen3 (sglang) | 1.0 | 1024 |
| gpt-oss-120b (sglang) | **1.0** | 1024 |
| gpt-oss-20b (sglang) | **0.53** | ~540-547 |
| gpt-oss-120b (vLLM) | **0.06** | ~57-62 |
| gpt-oss-20b (vLLM) | **0.06** | ~57-71 |

**Confirmed this is not request failure**: `successful_requests == num_prompts` exactly, every
single time, across every one of these. Every request "succeeds" — the model is stopping
generation on its own, well before 1024 tokens, via an actual EOS/stop token.

### Why: `ignore_eos`, checked precisely, doesn't cleanly explain it

vLLM: `ignore_eos=False` uniformly across every vLLM capture, including deepseek — which reaches
full length anyway. So it's not a flag difference between models on the same backend; it's
gpt-oss specifically emitting a real stop token on this synthetic random-prompt task, at a rate
`ignore_eos=False` (the default, EOS respected) allows through. **Fix for future vLLM captures**:
pass `--ignore-eos` explicitly.

SGLang: murkier. The exact flag name differs across captures collected at different times
(`disable_ignore_eos` for deepseek/gpt-oss-120b vs. `ignore_eos` for qwen3/gpt-oss-20b — almost
certainly different client versions, not a deliberate config choice), and — the finding that
rules out "just the flag" as a complete explanation — **qwen3 and gpt-oss-20b share the
identical flag and value** (`ignore_eos=False`) yet only gpt-oss-20b is broken. If the flag alone
explained it, qwen3 would show the same problem. One real hypothesis, unconfirmed: gpt-oss's
"harmony" chat template may terminate via a **stop string** (e.g. `<|return|>`) rather than the
model's literal EOS token, in which case `ignore_eos`/`disable_ignore_eos` (which specifically
govern EOS-token behavior) wouldn't suppress it — an explicit `--stop` override might also be
needed. Worth checking the actual chat template dispatch before assuming one flag flip fixes it.

### Why this matters for validation, not just as a curiosity

`frontier_cli_translator.py` currently simulates every request generating exactly
`--fixed_request_length_generator_config_decode_tokens` = the *nominal* target length (1024),
unconditionally. For gpt-oss/vLLM, real decode length was ~60-70 tokens — **15-18x shorter**
than what gets simulated. Any existing gpt-oss/vLLM comparison report is comparing Frontier's
1024-token simulated decode against a real benchmark that only ever did ~60-70 tokens — a large,
uncorrected workload mismatch that will make Frontier look far slower/higher-latency than the
real system for entirely the wrong reason. gpt-oss-20b/sglang has the same problem at smaller
magnitude (~53% instead of 100%). gpt-oss-120b/sglang and everything deepseek/qwen3 are
unaffected — those are safe to compare as-is.

**Not yet fixed.** Two independent paths forward, not mutually exclusive:
1. Re-collect the affected real captures with `--ignore-eos` (vLLM) / the SGLang equivalent (and
   possibly an explicit `--stop` override) forced, so future captures generate the intended
   1024 tokens.
2. For the *existing* captures (can't retroactively change history), fix the translator to use
   the *actual* average output length (`total_generated_tokens / successful_requests`, derived
   per real record) instead of the nominal `random_output_len` when building the simulated
   request — makes the comparison apples-to-apples against what actually happened, at the cost
   of no longer testing the *intended* 1024-token workload.
