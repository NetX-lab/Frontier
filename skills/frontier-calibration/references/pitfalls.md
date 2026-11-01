# Calibration Pitfalls

Each rule is a lesson from a past calibration failure, written as the behavior
that avoids it. Read the section of the phase you are in.

## Ground-Truth Collection

- Keep each GPU job's launcher process alive until the worker reaches a
  terminal status; a launcher that exits can stop its worker.
- Run jobs that bind static ports (PD KV ports, DP worker ports) one at a time.
- Use eager mode for DP2 x PP2 serving on vLLM 0.10.2; `FULL_DECODE_ONLY`
  decode graphs deadlock in that layout.
- Warm every prompt-length class of the formal trace, and warm the compile
  caches with an unmeasured pass, before the formal window; the first step at
  a new token count otherwise carries a one-time stall that looks like model
  error.
- Let the warmups drain before the first formal arrival. When warmups and
  formal requests overlap, TTFT measures queueing and the DP x PP mode follows
  the warmup batches.
- Collect the clean run and every run that feeds a table or analysis in one job
  on one node. Device lots differ in KV block count and eager speed, and the
  first clean run on a node can still run slow. A paired host-speed check
  supports probe admission; it does not establish reference sufficiency or
  authorize discarding otherwise valid slow clean runs (contract.md,
  Calibration Objective).
- Give DP x PP replays a heartbeat; after an idle period the client can
  dispatch late by tens of milliseconds, and the DP lane mode then follows
  client jitter.
- Size a PD cell's load below the decode instance's KV capacity: the P2P
  connector holds received KV per request on the decode GPU.
- When engine startup fails, find the cause in the server log and change the
  engine file before resubmitting; an identical resubmission fails the same
  way.
- Treat a benchmark import failure on a worker as an environment error, not a
  bandwidth result.

## Admission and Tables

- Admit a CPU-probe run only when its admission ratio lies in 0.97-1.03
  against its paired clean run; a passing run check alone does not show that
  the probe ran at the reference host speed.
- For DP x PP cells, admit probe runs by the isolated first-prefill step; the
  per-token lock-mode ratio compares modes, not host speed.
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
- Take a cell's KV block count from that cell's own clean run, never from
  another node or cell.
- Run gate labels in non-dummy mode; dummy-mode runs check structure only.
- Keep `--op-trace` and other diagnostic runs under their own labels.

## Reading the Gate

- Read a DP x PP cell by request-mode pairs. The two DP lanes lock into
  in-phase or anti-phase periods that flip within and between runs, so a single
  run or a whole-cell mean misrepresents the reference.
- A cell whose throughput is arrival-bound passes throughput for any model;
  state that the throughput evidence is weak.
- Report absolute values beside every relative error.
- Separate the stack under test from the calibration branch: a reading pins the
  Frontier commit that produced it.

## Analysis and Correction

- Close the error budget before choosing a fix; a change made before the
  sources are separated can move one source and hide another.
- Record measurement cost separately from measured behavior: `op_timing`
  serializes scopes and `kernel_timing` syncs the device.
- Check how Frontier orders host preparation against the DP synchronization;
  vLLM overlaps preparation with the previous device work.
- Compare every padded width with vLLM's: the CUDA Graph capture size, the
  token count padded across a DP cohort, and grouped-GEMM rows padded over the
  whole EP domain. Padding differences were a recurring device-price error.
- Fix a ground-truth defect in the ground truth by decision, as a new commit
  and run generation.
- After any change to a table or setting, rerun Frontier and republish the
  readings; swapping a CSV under an existing reading leaves the reading stale.
