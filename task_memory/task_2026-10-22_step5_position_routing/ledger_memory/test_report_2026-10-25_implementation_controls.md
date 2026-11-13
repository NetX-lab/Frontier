# EP Lane Ledger Implementation Controls

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 | Record component controls, raw failures, exact local accounting and the unallocated snapshot successor. |

## Outcome

**Partial component delivery; complete acceptance BLOCKED.** Code commit
`6c2f179c9117256bb7086441b0c11a1459f48674` removes real-store historical EP row
retention and actual-validator whole-EP-file decoding. Successful sorted JSONL
values/order and the semantic row body are preserved. Existing performance
save_state cannot serialize an open writer; its minimum successor requires a
primary2-gross path-cap transfer. Four complete integration constructor tests also
fail because this worktree has no resolvable collective-sim checkout.

Task ID: `D29-EP-LANE-LEDGER-IMPLEMENTATION`.
Session: `ses_e91675a86ffegKCQdHRPQFOypp`.
Worktree: `/data/ycfeng/Frontier/.worktrees/step5-d29-ledger-stream`.
Branch: `fix/d29-ledger-stream`.
Allocation: `0ca540251ae1c2bdbe436a2884d48adf29ed955a`.
Production baseline: `b02772872d0c99ceb5ebb31a8e041926f12f7676`.
Cumulative baseline: `1aee0af11cd58691e64603b99eca0ac459a5b350`.

## Environment and exact focused commands

Python: `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6/GCC11.3.0.
Every executed focused command configures systemd `MemoryMax=30G` and uses an
outer60-second timeout. Each scope starts successfully. Memory controller peak
and effective limit were not measured; the auxiliary unified-cgroup reporting
attempt failed on this host's v1 memory layout. The later static-only Python
process reports max RSS17,532KiB; that is not a simulation/RSS bound. No uncapped
fallback was used.

Each pytest command used this exact form, with suffix/working directory/test
arguments/PYTHONPATH from the table below. Redirection preserved the full scope
and pytest output. The first failed launcher is separately retained.

```bash
scratch=/data/ycfeng/tmp/d29_ledger_stream_<suffix>
mkdir -p "$scratch"
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH="<table value>" PYTHONDONTWRITEBYTECODE=1 TMPDIR="$scratch" XDG_CACHE_HOME="$scratch/cache" MPLCONFIGDIR="$scratch/mpl" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest <table test arguments> -q -p no:cacheprovider <table -k argument> --basetemp="$scratch/pytest" --junitxml="$scratch/junit.xml" > "$scratch/raw.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$scratch/exit.txt"
cat "$scratch/raw.log"
exit "$rc"
```

`R` below means the working directory, not a literal environment entry.
`U=R/tests/unit`. `P=/data/ycfeng/Frontier/.worktrees/step5-preview`.
`S=P/frontier/cc_backend/backends/collective-sim/python`.
`I=tests/integration/test_step5_measured_main.py`.
`E=tests/unit/test_stage_reporting_contract.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_ep_trace.py tests/unit/test_typed_ep_trace_contract.py`.

| suffix | Working directory | PYTHONPATH | Tests/filter | Raw exit and result |
| --- | --- | --- | --- | --- |
| red2 | current worktree | R | I; `-k 'releases_rows or (requires_actual_dummy and not damage)'` |1;4FAIL/36PASS/3SKIP/21deselected;3.24s |
| baseline | P | R | I E |0;87PASS/6SKIP;54.48s |
| green_controls | current worktree | R | I; `-k 'releases_rows or requires_actual_dummy'` |1;1FAIL/39PASS/3SKIP/21deselected;3.15s |
| green_integration | current worktree | R | I |1;6FAIL/59PASS/6SKIP;52.76s |
| green_ep | current worktree | R | E |0;30PASS;2.68s |
| baseline_import | P | R | I; `-k corrected_shapes_fit` |1;2FAIL/61deselected;1.85s |
| green_final | current worktree | R:U | I |1;4FAIL/62PASS/6SKIP;52.10s |
| green_pinned | current worktree | R:U:S | I |1;4FAIL/62PASS/6SKIP;56.11s |
| green_controls_final | current worktree | R:U | I; `-k 'requires_actual_dummy or releases_rows or stream_failures'` |0;48PASS/3SKIP/21deselected;3.24s |

The first `red` command used the same scope/environment with `/usr/bin/time -v
-o "$scratch/resource.txt"` before Python and a read-only zsh variable `status`
afterwards. The scope starts; `/usr/bin/time` is absent and no tests run. Inner
exit was not captured; wrapper exit1 is observed. Preserve that raw log.

## Expectations versus observations

| Control | Passing condition | Observation |
| --- | --- | --- |
| Real MetricsStore retention | Old row object becomes unreachable after append | RED before repair; GREEN after repair for32 rows using weakref-able dictionaries at the real field |
| Actual validate whole-file read | EP-specific read_text guard is never reached | Three successful typed bindings RED before; all GREEN after |
| Successful output | Complete sorted JSONL bytes and row order match accepted writer | Accepted writer AST produces3,528B/2rows; candidate bytes match; SHA256 `409f50bf63b7a37c7324d1abcc7a16842c9ef3e1bfacb4716c3b01bc8b7afaf0` |
| Count and zero rows | Cumulative len32; no empty EP artifact | Persistent real-store test passes |
| Terminal output | Final writer closes without rewriting/truncating | Same32-row successful bytes after two final calls |
| Negative rows | Existing exception classes/messages remain effective | missing/empty/malformed/source-less/null/null-source/invalid-proof all pass |
| Streaming priority | Earlier invalid proof wins over later malformed syntax | AssertionError `Missing exact positive compute pair` wins; test passes |
| Row checks | Existing semantic loop body unchanged | AST byte-independent comparison passes |
| Producer | ep_wave_metrics.py remains byte-identical | Static Git comparison passes; list-backed existing probes pass |
| Existing performance digest | Hash only new persisted bytes; retain cumulative count/report payload | Exact main AST controls exercise zero, first rows and later file position before the broad supplement's auxiliary reporting fails |
| Source stream closure | Normal and earlier semantic-error paths close EP source | Exact actual validate controls execute both closure assertions before auxiliary reporting fails; no complete supplement exit0 is claimed |
| Native write/JSON failure | Original OSError object propagates; unsuccessful append not counted | Assertions execute before auxiliary reporting fails; raw exit1 remains |
| Existing save_state | Complete owned caller remains serializable | **FAIL**: actual save_state raises PicklingError for open mode `w`; not repaired in allocated source |

Supplemental commands use the same60s/systemd/single-thread environment and
`python -B - <<'PY'` in the current worktree. Their inline programs are recorded
in the execution trace; retained raw receipts are linked below. They inspect only
the allocated source, accepted Git text and typed/reporting fixture seams; no
production wave, required shape, CLI, census or platform operation occurs.

- `supplement`: execute the accepted final EP writer AST, current performance
  before/digest AST and actual validate/write-error controls. The control assertions
  complete; the auxiliary `memory.max` lookup then fails. Raw exit1.
- `caller_proposal`: execute actual existing save_state failure, and an unapplied
  four-line existing-save/load-seam proposal in memory. Restored cumulative count2,
  new row count1/digest and original source bytes pass before an incorrect16-gross
  assertion fails. Raw exit1; the wrapper's accidental local file `1` was inspected
  and removed as this session's own disposable output.
- `proposal_accounting`: `git show 0ca54025:<performance path>` plus difflib of the
  same in-memory proposal proves11insertions/3deletions=14gross. Static validator
  row-body comparison, producer comparison and retained successful prefix match
  pass. Raw exit0. The patch is retained as evidence only and is not applied.
- `archive`: exact Git numstat for the four owned paths against allocation and
  cumulative baseline; compare current blobs to code commit, check other source
  names unchanged, copy/hash raw evidence and compare JUnit IDs. Raw exit0.

## Raw status sets and failures

`evidence/implementation_2026-10-25/test_ids.json` records every focused test ID,
state and skip/failure message. `focused_status_comparison.json` records all93 old
IDs retained, all6 baseline SKIPs retained,9 newPASS IDs and four oldPASS->FAIL
constructor IDs. Those four IDs are
`test_position_command_prices_exact_plan_through_real_cluster` with
`offline/online × PREFILL/DECODE`. Their latest error is FileNotFoundError from
the existing backend checkout resolver. **The complete comparison remains FAIL.**

Earlier failures remain: wrong normalized output assertion; new test's IS_MOE
pollution, fixed using the existing monkeypatch seam; two missing unit-test imports
also reproduced on untouched baseline; missing collective_sim_core import; then
missing backend checkout despite read-only pinned Python imports. No donor/backend
or submodule source was edited to conceal the setup requirement. Pinned imported
submodule HEAD is `3dac53c544e546574cb6e8078cd14053d0916113`.

No full-unit/strict77 run is performed here. The task's already-recorded old raw
279FAIL/32ERROR/281SKIP and old-ID-presence FAIL are neither replaced nor admitted
as PASS by these focused results.

## Exact local accounting

| File | Step +/− | Step gross/cap | Cumulative +/− | Cumulative gross/cap |
| --- | ---: | ---: | ---: | ---: |
| frontier/metrics/metrics_store.py |26/7|33/38|26/7|33/38|
| tests/integration/run_step5_measured_main.py |12/3|15/20|52/4|56/67|
| tests/integration/test_step5_measured_main.py |61/1|62/65|320/0|320/325|
| tests/performance/step5_actual06_iteration.py |7/3|10/12|7/3|10/12|
| Total |106/14|120/135|405/14|419/four-path caps|

Commands: `git diff --no-renames --numstat <allocation> <code> -- <path>` and
the same command with cumulative baseline; `git diff --name-only <accepted>
<code>` excluding task records; `git show <code>:<path>` versus working bytes.
All local path caps pass. Increment112 projects CORE4373/4404,source9678/9942,
global9692/10000. `scope.json` explicitly marks this as **local four-path
accounting, not independent clean composed acceptance**.

Source `git diff --check` passes before the code commit. Whole staged evidence
`git diff --cached --check` exits2 because retained raw pytest/JUnit output and
blank diff context contain trailing whitespace. The raw evidence is not changed
to make this check pass; authored documents are checked separately.

## Minimum required primary action

The unapplied existing-state-seam proposal has performance14gross, requiring
**2 additional path-cap lines**. Transfer2 unused metrics allowance lines:
metrics38→36;performance12→14. The four-path135 total and owner/global totals
remain unchanged. Current production performance remains10/12; no silent cap
overrun or new serialization/public API/list-history proxy was added.

## Retained evidence and limits

All paths below are relative to `ledger_memory/`:

- `evidence/implementation_2026-10-25/scope.json` and `scoped_code.diff`.
- `evidence/implementation_2026-10-25/artifact_manifest.json`: exact39 copied file hashes.
- `evidence/implementation_2026-10-25/legacy_successful.jsonl`, `current_with_increment.jsonl`, `retention_32_rows.jsonl`.
- `evidence/implementation_2026-10-25/raw/`: raw exits/logs/JUnit and proposal receipts.
- `evidence/implementation_2026-10-25/raw/proposal_accounting/proposed.patch`: unapplied evidence only.

This demonstrates synthetic implementation controls, not successful whole-workflow
completion. Pending: allocated snapshot repair, checkout readiness and affected
focused rerun; independent audit/Standards/Spec review; final source freeze/full
regressions/integration; fullCLI/native/G3/E2E/runtime and full-workload RSS/storage
qualification. No budget/gate/resource expansion follows from this delivery.
