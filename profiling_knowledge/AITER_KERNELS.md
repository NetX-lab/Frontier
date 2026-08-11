# AITER Kernels: What Exists, Where, and the Gap for Dense/GQA Models

`aiter` is AMD's own optimized kernel library. Real sglang deployments on MI355X use it
directly — every sglang launch command captured in `tools/inference_bench/` explicitly passes
`--moe-runner-backend aiter`. Frontier's own profiling, by contrast, has been using
`TORCH_SDPA` — a portable, correctness-first reference backend, explicitly documented (in both
[MI355X_ROCM_COOKBOOK.md](MI355X_ROCM_COOKBOOK.md) and `frontier_cli_translator.py`'s own
docstring) as **not peak-tuned**. That gap is a real, plausible contributor to any large
real-vs-simulated mismatch you see in a comparison report.

## What exists today, and exactly where

Frontier's own `frontier.profiling.attention.main --attention_backend` choices are
`{FLASHINFER, NO_OP, TORCH_SDPA, FLASHINFER_MLA}` — **no AITER option** — on every checkout we
have direct control over: `/home/dn/FrontierBase`, `/home/dn/driventes-frontier`, and
`server1` (`amd-mi355x-1`, both `~/frontier_work/drivenetsfrontier` and
`~/frontier_work/Frontier`).

`server3` (`amd-mi355x-3`) and `server8` (`amd-mi355x-8`) are different: both have
`~/frontier-work/Frontier/frontier/profiling/attention/backends/aiter_mla_attention_wrapper.py`,
and their `AttentionBackend` enum includes a real `AITER` value:

```python
class AttentionBackend(Enum):
    FLASHINFER = "FLASHINFER"
    NO_OP = "NO_OP"
    TORCH_SDPA = "TORCH_SDPA"
    TORCH_SDPA_MLA = "TORCH_SDPA_MLA"
    AITER = "AITER"
```

`get_attention_wrapper()` dispatches `AttentionBackend.AITER` straight to
`AiterMlaAttentionWrapper` — **there is no generic AITER path, "AITER" in this branch means
"AITER for MLA," full stop.**

## What the MLA wrapper actually does

This is real, careful, already-validated work, not a stub. Reading its own docstring:

- It calls AMD's actual production `aiter` MLA kernels — the same ones real sglang serving
  uses via `--attention-backend aiter` — instead of the portable
  `TorchSdpaMlaAttentionWrapper` reference implementation.
- It was built by reading real sglang's own `aiter_backend.py` from the exact
  `lmsysorg/sglang:v0.5.11-rocm700-mi35x` image used for the real DeepSeek-R1 benchmarks it's
  meant to match — including catching and correcting a wrong initial assumption about
  decode-side query absorption (`q` arrives at `forward_decode` already absorbed into
  rank-space; `aiter`'s `mla_decode_fwd` does not do that absorption itself).
- Prefill dispatches through a **persistent fp8 kernel**
  (`mla_fp8_prefill_attn` → `aiter.mla_prefill_ps_asm_fwd`), not the generic
  `aiter.mla_prefill_fwd` (which asserts a shape constraint this model's real dimensions violate:
  `qk_head_dim=192 != kv_lora_rank+qk_rope_head_dim=576`).

## The gap: nothing exists for dense/GQA models

gpt-oss (and Qwen3-30B-A3B) use plain GQA attention — no latent compression, no rank-space
absorption, none of the MLA-specific math this wrapper is built around. It cannot be pointed at
these models. **No dense/GQA AITER wrapper exists on any checkout we've found.**

Building one would mean repeating the same kind of work that produced the MLA wrapper, but for
sglang's *dense* attention dispatch path instead:

1. Read real sglang's `aiter_backend.py` for its non-MLA (GQA) forward-path dispatch — find
   which actual `aiter` kernel(s) it calls for prefill and decode on a GQA model.
2. Implement a new wrapper against Frontier's `BaseAttentionWrapper` interface, following the
   `aiter_mla_attention_wrapper.py` pattern (and cross-check against the correctness-oracle
   `TorchSdpaAttentionWrapper` the way the MLA one was validated).
3. Register it as a new backend value (or extend `AttentionBackend.AITER`'s dispatch to branch
   on attention family, dense vs. MLA, rather than hardcoding MLA).

This is real, unverified-until-run engineering work — not a config flag we forgot to flip.

## How to use what already exists (DeepSeek/MLA, today)

If you're validating DeepSeek-R1 specifically, this is ready right now on `server3`/`server8`:

```bash
python3 -m frontier.profiling.attention.main \
  ... \
  --attention_backend AITER \
  ...
```

It is **not** on `server1`, and not in either `FrontierBase` or `driventes-frontier` — sync
`aiter_mla_attention_wrapper.py` (and the updated `backends/__init__.py`) there first if you want
to run it from one of those checkouts instead. See
[INFRASTRUCTURE_MAP.md](INFRASTRUCTURE_MAP.md) for the full checkout inventory and what's missing
where.

**Packaged as a runnable script**: [`scripts/profile_deepseek_aiter_mla.sh`](scripts/profile_deepseek_aiter_mla.sh)
wraps the command above with a preflight check that fails fast with this same
guidance if `AttentionBackend.AITER`/`aiter_mla_attention_wrapper.py` isn't
present on the checkout it's run from (`--skip-aiter-preflight` bypasses it if
you've verified otherwise), plus a `--attention-backend TORCH_SDPA_MLA` mode
for the portable reference backend on checkouts that don't have AITER at all.
Unlike `profile_true_mixed_batch.sh`, its default grid is sanity-check scale,
not a validated production sweep — there's no single "final AITER command"
recorded from this session to reproduce exactly; widen it deliberately for
your real workload shape.
