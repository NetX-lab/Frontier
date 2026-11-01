# Calibration Pitfalls

Each rule is a lesson from a past calibration failure, written as the behavior
that avoids it. Read the section of the phase you are in. Observations tied
to a vLLM version, probe, or published study apply only when the selected
engine binding verifies the same behavior.

## Ground-Truth Collection

- Keep each GPU job's launcher process alive until the worker reaches a
  terminal status; a launcher that exits can stop its worker.
- Run jobs that bind static ports (PD KV ports, DP worker ports) one at a time.
- The recorded vLLM 0.10.2 DP2 x PP2 case needed eager mode because
  `FULL_DECODE_ONLY` deadlocked. Recheck the selected build before applying
  that workaround; it is not a restriction on every engine or version.
- Warm every prompt-length class of the formal trace, and warm the compile
  caches with an unmeasured pass, before the formal window; the first step at
  a new token count otherwise carries a one-time stall that looks like model
  error.
- Let the warmups drain before the first formal arrival. When warmups and
  formal requests overlap, TTFT includes their queueing and later batches can
  depend on the warmup state.
- Collect the clean run and every run that feeds a table or analysis in one job
  on the same node group and topology. Device lots differ in KV block count and eager speed, and the
  first clean run on a node can still run slow. A paired host-speed check
  supports probe admission; it does not establish reference sufficiency or
  authorize discarding otherwise valid slow clean runs (contract.md,
  Calibration Objective).
- The existing vLLM DP x PP replay uses a heartbeat; after an idle period the client can
  dispatch late by tens of milliseconds, and the DP lane mode then follows
  client jitter.
- Inspect the selected PD connector's KV lifetime when sizing load. The
  existing vLLM P2P connector retains received KV per request on the decode
  GPU; another connector can have different capacity and release behavior.
- When engine startup fails, find the cause in the server log and change the
  engine file before resubmitting; an identical resubmission fails the same
  way.
- Treat a benchmark import failure on a worker as an environment error, not a
  bandwidth result.

## Admission and Tables

- Use the selected probe's validated admission method; a passing run check
  alone does not establish host-speed comparability or low instrumentation cost.
  The existing vLLM producer uses the ratio and rules in
  `tool-boundaries.md`, Existing vLLM Collection Profile.
- In that profile's DP x PP study, admission uses the isolated first-prefill
  step; the per-token lock-mode ratio compares modes, not host speed.
- Profile on the dense default grid. A random-forest predictor jumps between
  sparse grid points, so a formal batch farther from a profiled point than the
  reference spacing is a coverage gap to fill before gating.
- Filter an attention profiling grid against device memory on the CPU first;
  one oversized input stops the whole plan.
- Deduplicate repeated inputs before a merge; the merge rejects them.
- Leave a one-row CPU-overhead table out of training; cross-validation needs
  more rows.
- Exclude the engine's first step from idle-gap rows; it is startup, not idle.
- Price from a median only when the samples behind it are dense and unimodal.
- Re-measure every hardware-dependent quantity on a new device generation:
  operator tables, `kernel_gap.csv`, collective and KV fits, KV block counts,
  and load levels. A table relabeled with another device name is not a
  measurement.

## Frontier Runs

- Pass the current generation's settings file with an explicit argument. A
  default settings file can silently carry older tables; two runs on the same
  wrong file are still byte-identical, so byte identity cannot catch it.
- Take observed KV capacity from the cell's own clean run and verify its
  mapping to Frontier blocks; another engine may use a different cache layout.
- Run gate labels in non-dummy mode; dummy-mode runs check structure only.
- Keep `--op-trace` and other diagnostic runs under their own labels.

## Reading the Gate

- Use request-mode pairs only for a case that validates the `dp_pp_pairs`
  model. The recorded vLLM study had changing lane phases; do not assume that
  every engine with DP and PP shares those modes. Keep ordinary clean-run
  comparisons alongside any conditioned result.
- A cell whose throughput is arrival-bound passes throughput for any model;
  state that the throughput evidence is weak.
- Report absolute values beside every relative error.
- Separate the stack under test from the calibration branch: a reading pins the
  Frontier commit that produced it.

## Analysis and Correction

- Close the error budget before choosing a fix; a change made before the
  sources are separated can move one source and hide another.
- Record measurement cost separately from measured behavior. The vLLM profile
  has serializing `op_timing` scopes and synchronizing `kernel_timing` probes;
  inspect the effects of the selected engine's actual instrumentation.
- Check the selected engine's host/device overlap and synchronization before
  attributing time to a host term.
- Compare every padded width with the reference implementation's: the CUDA Graph capture size, the
  token count padded across a DP cohort, and grouped-GEMM rows padded over the
  whole EP domain. Padding differences were a recurring device-price error.
- Fix a ground-truth defect in the ground truth by decision, as a new build identity
  and run generation. An intentional fork behavior is not such a defect.
- After any change to a table or setting, rerun Frontier and republish the
  readings; swapping a CSV under an existing reading leaves the reading stale.
