# Artifact bytes repair progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record source binding and local verification preparation. |

- completed: read repository AGENTS.md and parent requirements/plan/current
  progress plus the controlling scope amendment; clean branch at `2cfbb344`.
- completed: local shared clone from the cost-fields submodule, detached at
  `3dac53c544e546574cb6e8078cd14053d0916113`. No peer writes or network access.
- completed: environment is `/data/ycfeng/envs/frontier-py310/bin/python`,
  Python 3.10.6, pytest 9.1.1, numpy 2.2.6, pandas 2.3.3, sklearn 1.7.2.
- in-progress: derive the source-bound parent setup in fresh scratch and collect
  unchanged-source baseline wave/test evidence before editing code.
- pending: implementation, original-hook instrumentation observation, candidate
  wave/tests, accounting, code commit and compact completion archive.

The request names `setup.py`; the actual parent file is `current_setup.py`
(discovered read-only). Its paired `run.py`/`binding.json` identifies the exact
original setup and unchanged emission body; reuse those rather than inventing a
fixture or weakening the source assertion. This is a filename clarification,
not a changed acceptance scenario.

- observed environment failures (before source edits): unprivileged systemd-run
  requires interactive authentication; the already documented sudo scope works.
  Fresh scratch lacks the relative model JSON required by the existing loader,
  causing the wave setup and 27 unit fixture setups to fail before pricing.
  Resolution: copy only the unchanged Step5 model JSON into fresh scratch `data/`
  and retain all original failure logs. Do not patch the loader or source checks.

- completed: baseline affected four-file suite is 78 PASS in 127.12s, zero
  FAIL/ERROR/SKIP (`baseline_03.xml`). No source/test edits yet.
- diagnostic harness failure: baseline wave business checks passed (208 event
  bytes and full ledger identical; 64 wrong-key negatives; original error prefix),
  but its final fixture metadata assertion compared against an overwritten local:
  the unchanged emission body assigns global `before` to pending trace bytes.
  Original files have no new mtime. Rename only the wrapper observation variable
  and retain explicit before/after metadata in the next fresh scratch run; this
  is not evidence of original fixture changes or of source failure.

- completed: baseline04 source-bound layer60 loop is PASS: 16 EP, 80 current
  text reads/wave, identical payload SHA `3d745054...5ca83`, full 208-event trace
  and ledger equality, and all 262 original fixture file names/sizes/mtimes
  unchanged. Warm median is 1.057538s (1.168213/1.048832/1.057538s).
  The optional whole-artifact parse counter used an unsupported first-300-byte
  schema-position assumption; remove that assumption for the final paired loop.
  Total JSON call counts and both read-method counts were unaffected.

- completed: the final paired-loop baseline05 uses the corrected artifact-length
  observation and unchanged production source/setup/body. Final evidence is under
  `/data/ycfeng/tmp/d29_artifact_bytes_repair/verified03/`.
- completed: replace backend cached representation/read/parser arguments (10
  gross) and workload representation/read/parser arguments/docstring (14 gross).
  Native key and every remaining source/check line are unchanged.
- completed: old backend read_text hook records zero reads after the byte-read
  change, failing its two-read assertion in 1.49s (`old_hook.log/xml`). This is
  instrumentation evidence, not a baseline correctness failure. Adapt the hook
  to read_bytes and two-line ArtifactBytes.decode marker; all assertions remain.
- completed: candidate layer60 wave PASS, 208 exact event bytes/costs/proofs/keys
  and full ledger equal to baseline/reference. Each wave retains 16 core + 64
  backend current reads; cold artifact parses remain two and warm remain zero.
  Baseline/candidate cold pricing 1.759869/1.123448s; warm medians
  1.037333/0.325204s (3.1898x, observed 68.65% reduction). Candidate warm samples
  0.421348/0.310525/0.325204s; concurrent pytest load is not removed from timing.
- completed: candidate four-file suite 78 PASS/124.657s, no FAIL/ERROR/SKIP.
  All baseline IDs/statuses match after the existing test name's text->bytes
  suffix normalization. Runtime replacement/parse/buffer/context error seams and
  estimator/row-tuple/measurement-identity replacement assertions remain unedited.
- completed: all 20 assertion ASTs in the backend file are unchanged. Exact source
  transformation verification confirms all other source bytes, numerical/proof
  checks and the native prediction key are unchanged. Failed read/decode/parse
  paths still precede the existing successful-state assignments.
- completed: Git step +17/-15=32 gross (source24/test8), net+2. Per-path
  gross10/14/8 satisfies12/14/10; retained84/197/235 satisfies91/323/235.
  Only the three allocated code/test paths changed. Source/record diff is ready
  for root's independent review; `code.diff` SHA `51a6171e...1b8ea`.
- completed: verified code committed as
  `2f2d9339d5bce600fbb0a3f78aebbcf6f71be696`, exactly the three allocated paths.
  All three Git-committed SHA values equal the measured candidate values;
  `receipt.json` records that binding and the exact commit numstat.
- completed: independent repair archive prepared with exact commands, Git gross/
  net/retained counts, review-ready diff, selected raw evidence hashes and limits.
  Records commit is the commit containing this archive (resolve from summary.md).
- pending/root-owned: independent review, final owner/global union, regressions,
  final source freeze and full workflow/native/G3/E2E. No local task failure or
  active job remains. Parent requirements/plan/progress and other worktrees remain
  unmodified; original fixture/cache and submodule3dac pin are preserved.
