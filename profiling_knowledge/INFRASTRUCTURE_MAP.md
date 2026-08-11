# Infrastructure Map: Servers, Checkouts, and What's Missing Where

There are more Frontier checkouts in play than expected, in different states, on different
machines — this exists to stop re-discovering that the hard way.

## Servers

| Name | IP | Frontier checkout(s) |
|---|---|---|
| server1 / `amd-mi355x-1` | 172.30.160.204 | `~/frontier_work/drivenetsfrontier`, **and separately** `~/frontier_work/Frontier` |
| server3 / `amd-mi355x-3` | 172.30.160.119 | `~/frontier-work/Frontier` |
| server8 / `amd-mi355x-8` | 172.30.160.126 | `~/frontier-work/Frontier` |

Plus two local checkouts on this workstation: `/home/dn/FrontierBase` (this one) and
`/home/dn/driventes-frontier`.

**Note the inconsistent naming**: `frontier_work` (underscore, server1) vs. `frontier-work`
(hyphen, server3/server8) are *different path spellings on different hosts*, easy to typo when
switching between them.

## What's missing where (checked directly, not assumed)

- **`tools/validation/`** (the whole validation pipeline — `real_log_parser.py`,
  `real_log_aggregator.py`, `frontier_cli_translator.py`, `metrics_extractor.py`,
  `compare_plots.py`, `run_validation.py`): only ever existed in `/home/dn/FrontierBase`, built
  up over this session. `/home/dn/driventes-frontier` had an *empty* `tools/validation/`
  (only `__pycache__`) until it was copied over — that checkout was sitting at commit
  `d613032 Merge remote-tracking branch 'upstream/main'`, predating all of this work.
- **The AITER MLA wrapper** (`frontier/profiling/attention/backends/aiter_mla_attention_wrapper.py`):
  only on `server3` and `server8`. Not on `server1`, not in either local checkout. See
  [AITER_KERNELS.md](AITER_KERNELS.md).
- **gpt-oss model configs** (`data/config/models/openai__gpt-oss-{120b,20b}.json`): were missing
  entirely on `server3`'s checkout the first time profiling was attempted there — copied over
  from `/home/dn/FrontierBase`. Worth checking on any *new* checkout before assuming a profiling
  command will even get past model-config resolution.
- **gpt-oss true-mixed-batch profiling data** (`attention_combined_block{1,16,32}.csv`,
  see [GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md)): generated
  on `server1`, copied to both `/home/dn/FrontierBase` and `/home/dn/driventes-frontier`. Not
  pushed to `server3`/`server8` or the other `server1` checkout (`~/frontier_work/Frontier`).
  Regenerating it on any checkout is now just `scripts/profile_true_mixed_batch.sh` (~15s per
  model per block_size) rather than the manual sweep — no need to re-SCP if you can just re-run it.

**Takeaway: assume drift, verify before running.** Before running anything on a checkout you
haven't touched recently, check the specific files you need actually exist there — don't assume
a fix, a doc, or a data file that exists on one checkout has propagated to the others. None of
this is currently synced automatically.

## SCP patterns used this session (for repeating)

Pushing something *to* a remote host (e.g. a model config the remote checkout is missing):

```bash
scp /home/dn/FrontierBase/data/config/models/openai__gpt-oss-120b.json \
    dn@<host>:~/<frontier-checkout>/data/config/models/
```

Pulling generated data *back* from a remote host (e.g. freshly profiled CSVs):

```bash
scp dn@<host>:~/<frontier-checkout>/data/profiling/compute/mi355x/openai/gpt-oss-20b/attention*block*.csv \
    /home/dn/FrontierBase/data/profiling/compute/mi355x/openai/gpt-oss-20b/
```

Always verify with a checksum after a copy that's about to be relied on for anything:

```bash
md5sum <local-file> ; ssh dn@<host> md5sum <remote-file>
```

## The `nvidia-smi` GPU-discovery gotcha (every server, every checkout)

`frontier/profiling/attention/main.py`'s `_get_available_gpus()` checks
`CUDA_VISIBLE_DEVICES` first, and only falls back to shelling out to `nvidia-smi` if that's
unset — with **no ROCm equivalent wired in at all**. On a fresh shell (or a fresh `tmux`/`nohup`
session that doesn't inherit an interactively-exported var), this fails outright:

```
RuntimeError: Unable to discover GPUs with nvidia-smi because nvidia-smi was not found.
```

Same root cause as [MI355X_ROCM_COOKBOOK.md](MI355X_ROCM_COOKBOOK.md) gotcha #2, confirmed to
still apply on every checkout touched this session. Fix, every time, before running anything
that spawns a new shell/session:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # ROCm's native equivalent, set for safety
```

Despite the CUDA-specific naming, this is just a plain env var the Python code reads as a
string — it requires no actual NVIDIA hardware or drivers.

`scripts/profile_true_mixed_batch.sh` and `scripts/profile_deepseek_aiter_mla.sh` both export
these automatically (derived from their `NUM_GPUS`/`GPU_IDS` config) before invoking
`attention.main` — a non-issue when using them, still a trap for any new hand-typed command.

## Long-running jobs: use `tmux`/`nohup`, and know the output-flush risk

A plain foreground shell command dies the moment an SSH session drops. `tmux` (session survives
disconnects) or `nohup ... & disown` (detaches from the controlling terminal) are both fine —
but neither one gives you incremental checkpointing from the *profiling tool itself*. If the
underlying `python3` process is killed (by a `timeout`, an OOM, a crashed tmux server, whatever)
before it reaches its own final `to_csv(...)` call, **you get nothing**, no matter how the
process was supervised. See
[GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md#the-86-hour-lesson-profiling-has-no-checkpointing-size-the-grid-deliberately)
for the 86-hour version of this lesson. `scripts/profile_true_mixed_batch.sh`'s default grid is
sized to the fix that came out of that lesson (~15s per model per block_size) specifically to
keep this risk low; `scripts/profile_deepseek_aiter_mla.sh`'s default grid is sanity-check scale
for the same reason — widen either deliberately, and reach for `tmux`/`nohup` if you do.
