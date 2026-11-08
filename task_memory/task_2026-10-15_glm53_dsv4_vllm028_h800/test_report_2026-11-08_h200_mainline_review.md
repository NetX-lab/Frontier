# Test report 2026-11-08: H200 mainline review verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-08 | Created: lead verification of the codex-implement H200 mainline review, including the four `test_native_pd_worker.py` runs, the `-P` portability fix, and the independent re-checks of the review's metric and receipt findings. |

## Scope

This report records the checks the lead ran to verify the codex-implement review
`review_20261108_h200_mainline/codex-implement-4c000c1e-7ba4-4ef2-a77f-e341a5c8b551/implementation.md`
(model `gpt-6.1-sol-sys-paygo`, effort `ultra`, session `01a1ba95-8a5d-7c21-b128-2363134ee32a`,
executor self-report `implemented`, acceptance held by the lead).

The review itself ran CPU-only and read-only; its own environment had no `pytest`
(`handoff_audit.log` ENVIRONMENT: `"pytest": false`). The checks below therefore add direct
test execution, which the review could not perform.

Environment for every lead check:

- Host: local machine, Linux 5.10.25-nvidia-gpu.
- Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6, the declared project floor
  (`environment.yml` pins `python=3.10`; `pyproject.toml` declares `requires-python >=3.10`).
- Mainline worktree: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, HEAD
  `c30c338050fe8efb16e668fd574e496689eb3c90`.
- Harness worktree: `/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness`, HEAD
  `57caf2f79f79c8bc19e199ba64f75ca49e8baaf6`.
- No GPU, platform, or RJob operation was performed for this report.

Raw logs: `h200_mainline/p2_64_formal_retry08/heartbeat_tests_20261108/`.

## 1. Mainline native-worker unit tests, before the fix

Command (mainline worktree root):

```bash
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_native_pd_worker.py -q -p no:cacheprovider
```

- Expected: 41 passed, exit 0. The module is CPU-only and is the unit evidence for the native P/D worker
  that produced the P2 references.
- Actual: `4 failed, 37 passed in 1.19s`, exit 1. Each failure asserted on the child's stderr
  `Unknown option: -P` with child return code 2. Failing cases:
  `test_worker_launch_exports_staged_extension_path_to_real_child`, and the three
  `test_worker_child_ready_timeout_matches_configured_startup_budget` parameters
  `[3600-None]`, `[3600-600]`, `[37-7200]`.
- Verdict: **FAIL**.
- Log: `worker_unit_tests.log` (recorded 2026-11-08T10:34:34Z).

Diagnosis: `tests/unit/test_native_pd_worker.py:122` launched the real probe child with the `-P`
interpreter flag, which exists only from Python 3.11. Introduced on 2026-10-21 by `9c54f663`
("Export staged native extension path to serving child processes"), the commit that added these cases.
Recorded as **I-016**.

## 2. Mainline native-worker unit tests, after the fix

Fix applied: the `-P` flag was replaced by a portable prelude prepended to the probe source, which
removes `''` and the working directory from `sys.path` inside the child before the import. This keeps
the original assertion intact, namely that the probe resolves `native_full_structure` through the
exported staged extension path rather than through the working directory.

Command: identical to section 1.

- Expected: 41 passed, exit 0, and the staged-path assertion still exercised rather than skipped.
- Actual: `41 passed in 1.14s`, exit 0. The four previously failing cases pass, and the probe still
  asserts the staged path, because the prelude strips the working directory instead of removing the
  check.
- Verdict: **PASS**.
- Log: `mainline_worker_tests_after_fix.log` (recorded 2026-11-08T10:37:43Z).

## 3. Harness-worktree native-worker unit tests, re-verification

The review's finding F2 was that the records claim a passing heartbeat unit suite that it could not
locate (`handoff_audit.log` HEARTBEAT_SEARCH reports `files_read: 0, matches: []` for the harness
worktree, and no `heartbeat` match anywhere under the mainline package). The lead located that
evidence: it exists only in the harness worktree, whose `native_pd_worker.py` contains 28 occurrences
of `heartbeat` against 0 in the mainline copy.

Command (harness worktree root):

```bash
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_native_pd_worker.py -q -p no:cacheprovider
```

- Expected: the recorded `18 passed in 5.74s`, exit 0.
- Actual: `12 failed, 6 passed in 85.73s`, exit 1.
- Verdict: **FAIL**, i.e. the recorded result does not reproduce.
- Log: `harness_worker_unit_tests.log` (recorded 2026-11-08T10:37:59Z).

Failure breakdown:

| Failures | Cause |
| --- | --- |
| 4 | The same `-P` defect as section 1 (`test_worker_launch_exports_staged_extension_path_to_real_child` plus the three startup-budget parameters). |
| 8 | `TimeoutError: Native P/D rendezvous or workload deadline exceeded`: the test's `urlopen` -> `Health` stub was never entered, so each case waited out a real deadline. This also explains the 85.73 s wall time against the recorded 5.74 s. |

## 4. Order-dependence isolation probe

Purpose: establish whether the eight timeouts indicate a defect in the sealed worker, which produced
the accepted generation08 reference, or a test-isolation defect in the harness branch's test module.

Commands (harness worktree root, `PYTHONPATH=$PWD`, same interpreter):

```bash
# probe A
python -m pytest tests/unit/test_native_pd_worker.py -q -p no:cacheprovider \
  -k test_worker_forwards_canonical_clean_mode_and_idle_configuration
# probe B
python -m pytest tests/unit/test_native_pd_worker.py -q -p no:cacheprovider \
  -k test_worker_separates_auxiliary_report_faults_from_native_failure
```

- Expected, if the worker were at fault: the same timeout in isolation as in the full run.
- Actual probe A: `1 passed, 17 deselected in 0.19s`. The case that times out in the full run passes
  alone, in 0.19 s rather than a deadline wait.
- Actual probe B: `3 failed, 1 passed, 14 deselected in 30.36s`. Within the group, `[control]` passes
  and the next three parameters (`[report_write_failure]`, `[native_exit_at_replay_completion]`,
  `[native_shutdown_error]`) time out, reproducing the full-run pattern in isolation.
- Verdict: **PASS** as a diagnostic, i.e. it settles the question it was run to settle. Two distinct
  leaks are present: probe A's case depends on state left by earlier cases, and probe B's group leaks
  state from its own first parameter into the rest. Both point at a patch or module-state leak in the
  harness branch's test module, not at the worker. Not root-caused; recorded as **I-017**.
- Log: `isolation_probe.log` (recorded 2026-11-08T11:32:01Z).

Verification limit: the two probe commands were piped through `tail` when recording, so the `exitA`
and `exitB` values written into that log are `tail`'s status, not pytest's. The verdicts above come
from pytest's own summary lines, which are recorded verbatim.

## 5. Independent re-checks of the review's metric findings

These re-checks used the review's artifacts plus the task's own records; no new computation harness
was written for them.

| Check | Expectation | Actual | Verdict |
| --- | --- | --- | --- |
| Review's recomputed P2-32 worst per-repeat error | Matches the published 5.61% | `0.05607176023817675` at `repeat03 / ttft_ms` (`metrics_audit.log`), identical to the promotion receipt's `max` | PASS |
| Review's recomputed P2-64 generation08 worst per-repeat error | Matches the published 3.16% | `0.03162330523324778` at `repeat02 / ttft_ms` (`metrics_audit.log`), identical to `headline_recompute.json` | PASS |
| Review's recomputed generation08 pooled values | Match the report headline | vLLM `705.1974647474708 / 24.751650847855682 / 26026.136282103835 / 6001.121550818262`; Frontier `684.3138266042591 / 25.295336617652364 / 26561.443186462628 / 5975.251569855108` | PASS |
| Independence from the task's own script | The review reproduces the errors without reusing `headline_recompute.py` | `audit_metrics.py` recomputes from the row CSVs; its pooled throughput uses total tokens over the sum of per-repeat spans, matching the task convention | PASS |
| All-request and warm readings agree for generation08 | Both readings pass the 10% gate | Identical pooled values and identical worst error in both readings (`metrics_audit.log`) | PASS |
| Review's own assertion counts | All reported audits pass with zero failures | `metrics_audit` 157/157, `evidence_audit` 861/861, `qualification_audit` 906/906, `runtime_audit` 97/97, `runtime_supplement` 107/107, `handoff_audit` 28/28, all `FAIL 0` | PASS |
| Pre-launch promotion receipt binding | The P2-32 -> P2-64 receipt existed before the 64-GPU build and submission | `promotion/receipt_p2_32_to_p2_64.json`, sha256 `d505eece...`, issued 2026-11-02T07:38:49.523696+00:00, predicate limit 0.3, 24 checked errors, max 0.05607176023817675, `result: true`; the same digest is embedded in the generation08 package `config.json`, `case/inputs/native_config.json`, and run manifest, all built later | PASS |
| Old P2-64 reference is not retroactively promoted | The superseded reference keeps its 12.25% all-request error and its missing receipt | `metrics_audit.log` `p2_64_old` all-request worst `0.12254940665008353`; it remains recorded as not promoted | PASS |
| `a06a72a4...` as a repository anchor | Should resolve if it were a commit here | `git rev-parse --verify` fails with "Needed a single revision"; no `a06*` in `git log --all`. It is a native vLLM source SHA bound into `vllm_replay.runtime_version`, and the review independently resolved it in `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/p2-dp-startup-coordination` | PASS, as a cross-repository identity binding rather than a local anchor |

## 6. Record-correction checks

The review's finding F3 was that the generation08 formal report overstated its own verification
counts ("all sixteen per-repeat values ... all twenty errors"). The actual quantities are 4 Frontier
values, 12 per-repeat vLLM values, 4 pooled vLLM values, and 16 relative errors (12 per-repeat plus
4 pooled). Confirmed against `headline_recompute.json`, which contains exactly 12 `errors_per_repeat`
entries and 4 `errors_pooled` entries. The report prose was corrected. The lead's own assignment
brief had repeated the same wrong counts.

Finding F4's stale entries were corrected with case-specific scope, leaving dated historical
snapshots and the H800 cases untouched:

| File | Before | After |
| --- | --- | --- |
| `plan.md:207` | "New P2-64 reference acceptance remains open." | "P2-64 is formally accepted with the generation08 reference." |
| `plan.md:214` | "New P2-64 reference execution/admission and all four warm errors <=10% in each of three repeats remain pending." | Records the completed generation08 execution and admission and the worst 3.16% warm error, while keeping the old reference unpromoted. |
| `tickets/17-dsv4-canonical-case.md` Status | "No formal four-metric or 30% promotion receipt exists for V4." | Records the P2-H200-32 and P2-H200-64 formal passes and the dated receipt, and confines the "no receipt" statement to the H800 P0/P1 cases. |
| `frontier/model_architectures.py:690` | `"""Original V4 backbone identity with unsupported execution explicit."""` | `"""V4 backbone identity with its measured execution scope explicit."""` |
| `frontier/model_architectures.py:716` | "Native SP replicates shared weights; execution remains disabled." | "native SP replicates shared weights." (the stale disabled-execution clause removed) |

The two `model_architectures.py` changes are comment and docstring only; the `runtime_limitations`
tuple they contradicted is unchanged, and no executable line was touched.

Ticket 19's `PARTIAL` status and its GPU-hour accounting box were deliberately left open, per the
review's own constraint and the lead's independent reading of ticket 19 line 203.

## 7. Release sanity checks after the code edits

The only source edit in this pass is comment and docstring text in `frontier/model_architectures.py`, so this check
exists to confirm no regression and to exercise `AGENTS.md`'s declared starting checks.

Command (mainline worktree root):

```bash
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_model_architecture_registry.py tests/unit/test_native_pd_worker.py \
  tests/unit/test_pdd_public_surface_docs.py tests/unit/test_examples_pdd_scripts.py \
  tests/e2e/test_colocation_examples_layout.py -q -p no:cacheprovider
```

- Expected: all pass. The architecture-registry module is the one the edited file belongs to, and the other four are
  `AGENTS.md`'s starting checks.
- Actual: `3 failed, 126 passed in 10.38s`. The architecture registry, native worker, PDD example scripts, and
  co-location layout checks all pass. The three failures are all in
  `tests/unit/test_pdd_public_surface_docs.py` and all assert on `README.md`:
  `test_top_level_docs_advertise_supported_pdd_without_upcoming_claims`,
  `test_public_docs_require_sequential_pdd_release_mode`, and
  `test_public_docs_state_internal_parallel_pdd_boundary_and_current_evidence`.
- Verdict: **PASS for the edited module and for this task's scope; the three README failures are pre-existing and
  out of scope.**

Evidence that they are pre-existing and unrelated to this task:

- `git status --short` shows `README.md` unmodified, so the test read the file exactly as HEAD has it.
- `README.md` contains none of the asserted strings: 0 occurrences of `parallel PDD`, `pdd/run_all.sh`,
  `enable_parallel_clusters`, or `sequential`.
- `origin/main:README.md` is byte-identical in size (10,670 bytes) and also contains none of them, so the same three
  tests fail on upstream main.
- The last commit touching `README.md` is `bfa82de80` ("Update the release version to pre-release-v0.4"), which
  changed only the version badge: its parent's `README.md` already lacked the asserted strings.

Disposition: reported, not fixed. `README.md` is preserved unless explicitly asked for, and repairing an upstream
public-documentation contract is outside the DSV4 H200 calibration scope. This belongs to the release-documentation
owner.

## Summary

| Check | Verdict |
| --- | --- |
| 1. Mainline native-worker unit tests, before the fix | FAIL (4 failed / 37 passed) |
| 2. Mainline native-worker unit tests, after the fix | PASS (41 passed, exit 0) |
| 3. Harness-worktree unit tests, re-verification | FAIL (12 failed / 6 passed; recorded 18 passed does not reproduce) |
| 4. Order-dependence isolation probe | PASS as a diagnostic (test-isolation defect, not a worker defect) |
| 5. Independent re-checks of the review's metric findings | PASS (all 10 rows) |
| 6. Record-correction checks | PASS (5 entries corrected; ticket 19 left open) |
| 7. Release sanity checks after the code edits | PASS in scope (126 passed); 3 pre-existing upstream `README.md` doc-contract failures reported, not fixed |

Verification limits:

- No GPU or platform execution was performed. Sections 1 to 4 are CPU unit evidence; section 5 is a
  re-reading of existing artifacts.
- The eight harness-branch timeouts are localized to a test-isolation leak but not root-caused.
- The harness branch's worker change is still unmerged, so sections 2 and 3 describe two divergent
  copies of the same module rather than one verified module (**I-017**).
