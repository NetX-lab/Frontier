# Test report 2026-11-11 — I-017: root cause of the eight order-dependent timeouts, and the heartbeat merge into mainline

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-11 | Created. Root cause of the eight order-dependent timeouts, the pair-warmup reconciliation finding, the heartbeat port to mainline, and the verification of all of it. |

## Scope

YC approved, on 2026-11-11: locate and fix the test-state leak behind the eight order-dependent
timeouts, reconcile the two separately-implemented pair-warmup versions into one, merge the harness
branch's heartbeat work into mainline, and run a full verification pass.

## Environment

| Item | Value |
| --- | --- |
| Interpreter | `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6 |
| Mainline worktree | `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028` |
| Mainline HEAD before | `9a45a135` |
| Mainline HEAD after | `efbe09a9` |
| Harness worktree | `/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness`, branch `p2-64-startup-harness` |
| Harness HEAD before | `57caf2f79f79c8bc19e199ba64f75ca49e8baaf6` |
| Harness HEAD after | `0d7b6cb5` |
| Raw logs | `h200_mainline/i017_heartbeat_merge_20261111/` |

GPU, StepMind and cloud-volume operations were not used. Every check in this report is local CPU work.

## 1. Root cause of the eight order-dependent timeouts — CONFIRMED

### 1.1 The observed failure

Every one of the eight failures ends the same way, for example in
`test_worker_separates_auxiliary_report_faults_from_native_failure`:

```text
>       assert "native_health_200" in result["events"]
E       AssertionError: assert 'native_health_200' in ['server_launch', 'server_stop_called']
...
  File ".../tests/comparison/calibration/native_pd_worker.py", line 392, in run
    await_condition(healthy, config['startup_timeout_s'], native_required=True)
  File ".../tests/comparison/calibration/native_pd_worker.py", line 282, in await_condition
    raise TimeoutError('Native P/D rendezvous or workload deadline exceeded')
TimeoutError: Native P/D rendezvous or workload deadline exceeded
```

(line numbers as of harness commit `57caf2f7`.) The test's health stub was never entered, so the
worker's lane-0 health wait burned the whole 10 s startup budget.

### 1.2 The mechanism

The worker imports `urllib.request` **lazily**, inside the lane-0 branch of `run()`, and then calls
`urllib.request.urlopen(...)`. The test module did not import `urllib.request` at module scope.

`unittest.mock.patch.dict("sys.modules", {...})` restores by clear-and-update. A submodule imported
*during* the patched block is therefore **removed** from `sys.modules` on exit, while the parent
`urllib` package object **keeps** its `request` attribute. `mock._get_target("urllib.request.urlopen")`
resolves through `getattr(urllib, 'request')`, so from the second test onward it patched that
orphaned module object. The worker's lazy `import urllib.request` found no `sys.modules` entry,
re-imported a *different* module object carrying the real `urlopen`, issued a real HTTP request to a
port nothing was listening on, caught `OSError`, and kept polling until the deadline.

Direct probe output:

| Round | `patch` target id | `sys.modules` id | id the worker sees | worker gets the stub |
| --- | --- | --- | --- | --- |
| 1 | 140368560772480 | 140368560772480 | 140368560772480 | True |
| 2 | 140368560772480 | 94449546228704 | 140368553766288 | **False** |

After each block exit the probe reported `'urllib.request' in sys.modules = False` with
`urllib.request` attribute still present — exactly the stale-attribute state above.

### 1.3 Why mainline's 41 tests passed with the same latent pattern

Mainline's test module has the same `patch.dict("sys.modules", ...)` plus lazily-imported-`urllib.request`
pattern, and passed anyway. The reason is incidental, not a difference in the test:

- mainline's worker imports `vllm_replay` at module scope (`native_pd_worker.py:19`, after a
  `sys.path.insert`), and
- `tests/comparison/dp_placement_pp/vllm_replay.py:98` imports `urllib.request` at module scope.

So on mainline the key is already in `sys.modules` at collection time and every `patch.dict` snapshot
contains it, which keeps the module identity stable. Measured directly:

| Branch | `'urllib.request' in sys.modules` after importing the test module |
| --- | --- |
| mainline `feat/glm53-dsv4-vllm028` | `True` |
| harness `p2-64-startup-harness` | `False` |

Mainline therefore passed through a third-party import chain it does not control. Both test modules
now import `urllib.request` at module scope, which makes the guarantee explicit.

### 1.4 Hypotheses ruled out before this

| Hypothesis | Why it is not the cause |
| --- | --- |
| Leaked heartbeat daemon thread | the worker sets `stop_heartbeat` and joins the thread inside a `finally` block, so no thread survives a failing case |
| Module-level deadline | `deadline` is computed per `await_condition` call |
| Leaked `peer_heartbeats` state | the dict is created per `run()` call |

## 2. The second defect in the same module: the Python 3.11+ `-P` flag

The harness branch still launched its staged-child probe with `-P`, the same defect recorded as I-016
and already fixed on mainline. On the 3.10 floor the child exits 2 with `Unknown option: -P`. The
portable prelude mainline already uses was applied to the harness module as well.

## 3. Fix verification on the harness branch

| Stage | Command | Expectation | Actual | Verdict |
| --- | --- | --- | --- | --- |
| recorded 2026-11-08 | `pytest tests/unit/test_native_pd_worker.py -q -p no:cacheprovider` | — | `12 failed, 6 passed in 85.73s` | starting point |
| after the `urllib.request` import only | same | the 8 timeouts gone, the 4 `-P` failures remain | `4 failed, 14 passed in 12.37s` | PASS |
| after both fixes | same | all pass | `18 passed in 4.92s` | PASS |
| reverse collection order | same ids, reversed on the command line | all pass | `18 passed in 5.21s` | PASS |

`18 passed` matches the originally recorded `18 passed in 5.74s`, so that record was real; it simply
did not reproduce in the branch's later state. Committed as `0d7b6cb5`.

## 4. Pair-warmup reconciliation — already one implementation, nothing to merge

I-017 recorded the divergence as "two parallel evolutions". The reconciliation step found that the
pair warmup had already been forward-ported, and this is the evidence:

| File | Difference between the two branches |
| --- | --- |
| `tests/comparison/calibration/pd_proxy.py` | 0 diff lines |
| `tests/unit/test_native_pair_warmup.py` | 0 diff lines |
| `tests/comparison/calibration/native_pd_worker.py` | the two warmup lines are identical (`if config.get('pair_warmup'): replay_command.append('--pair-warmup')`) |
| `tests/comparison/dp_placement_pp/vllm_replay.py` | all 5 warmup-bearing lines identical; the divergence is mainline's extra `NATIVE_WORKER_EXTENSIONS`, `scheduler_cls` and `profiler_config` work |
| `tests/comparison/calibration/pd_replay.py` | every warmup-bearing line identical; the whole 148-line divergence is mainline's extra profiling-window feature |

Mainline commit `e2289283b` ("Integrate P2-64 pair warmup into the calibration harness", 2026-11-02
10:49 UTC) touches the same five code files with the same line counts as harness `4861e680`
(2026-11-02 08:41 UTC). The two versions are one implementation, so no reconciliation edit was needed.

## 5. Merge direction, established from the repository rather than assumed

| Fact | Evidence |
| --- | --- |
| The harness worker is exactly what produced the accepted generation08 result | `diff` against the sealed copy `h200_mainline/p2_64_formal_retry08/package_01/native_pd_worker.py` is **0 lines**; both 478 lines |
| The harness branch is *behind* mainline, not ahead | merge base `b62c155a` (2026-10-23); `git diff` between the branches covers 281 files with 35,716 deletions on the harness side |
| Only the heartbeat is genuinely missing from mainline | mainline carries 6 commits on these files that the harness branch lacks; of the harness branch's 4 unique commits, the pair warmup was forward-ported and the other two are harness bookkeeping |
| The accepted run really did use the heartbeat | generation08 `package_01/config.json` contains `heartbeat_interval_s: 10` and `peer_heartbeat_timeout_s: 180`; `h200_mainline/build_package.py` sets them |

A whole-branch merge would therefore have reverted mainline work. The port was done as a single
cherry-pick of `57caf2f7` onto mainline instead.

## 6. Conflict resolution

The worker auto-merged cleanly. The test module had two conflicts:

1. The `drained.json` suppression condition, a real semantic merge. Mainline suppresses it for the
   drain-quorum cases, the heartbeat side for the heartbeat cases; the merged condition suppresses it
   under either, for `index == 3`.
2. The test list, purely additive: mainline's cases plus the two heartbeat cases.

## 7. Integration check the harness branch could not have covered

Mainline added an ordered shutdown with a drain-quorum wait that did not exist when the heartbeat was
written, and `stop_heartbeat.set()` lands at the top of that shutdown block. Inspected rather than
assumed:

- every node writes `drained.json` at `native_pd_worker.py:451`, **before** the `finally` at line 464
  stops its heartbeat;
- the head's drain-quorum `await_condition` (line 455) runs while its own heartbeat is still alive,
  and `stale_peers()` (line 265) skips peers that have `drained.json`;
- the `failed.json` check (line 287) still precedes the staleness check (line 292), so a failing peer
  still raises the original error.

A peer that dies without writing either sentinel is flagged stale, which is the intended behavior.
No change was needed.

## 8. Verification on mainline

| Check | Command | Expectation | Actual | Verdict |
| --- | --- | --- | --- | --- |
| merged worker module | `pytest tests/unit/test_native_pd_worker.py -q -p no:cacheprovider` | mainline's 41 plus the 2 heartbeat cases | `43 passed in 6.61s`, exit 0 | PASS |
| order independence | same 43 ids, reversed | all pass | `43 passed in 5.71s`, exit 0 | PASS |
| release sanity | `pytest tests/unit/test_pdd_public_surface_docs.py tests/unit/test_examples_pdd_scripts.py tests/e2e/test_colocation_examples_layout.py -q` | the 3 known pre-existing README failures, nothing new | `3 failed, 17 passed in 0.64s`, same 3 ids | PASS (unchanged) |
| accuracy gate | `pytest tests/unit/test_e2e_gate.py -q` | all pass | `40 passed in 1.85s`, exit 0 | PASS |
| simulator fidelity | `git show --stat HEAD` | no `frontier/` file touched | 2 files, both under `tests/`, 111 insertions, 3 deletions | PASS by construction |

### 8.1 Regression check against an explicit baseline

The 15 collectible native/calibration unit modules were run twice: once with the two files reverted to
`HEAD~1`, once as committed.

| Run | Result |
| --- | --- |
| baseline (`HEAD~1` copies of the two files) | `32 failed, 442 passed, 62 skipped, 61 errors in 76.75s` |
| candidate (as committed) | `32 failed, 444 passed, 62 skipped, 61 errors in 83.13s` |

The 93 `FAILED`/`ERROR` node ids were compared as sets: **identical**, with no baseline-only and no
candidate-only entry. The difference is exactly the two new heartbeat tests. The port introduces no
regression in this group.

### 8.2 What the 93 pre-existing failures are

All are environment limits of the minimal release environment, not behavior:

| Count | Cause |
| --- | --- |
| 42 | `TypeError: TarFile.extractall() got an unexpected keyword argument 'filter'` (see issues.md I-018) |
| 33 | `ModuleNotFoundError: No module named 'torch'` |
| 16 | `ModuleNotFoundError: No module named 'aiohttp'` |
| 2 | `RuntimeError: vllm serve exited with 1 during startup` (needs a real engine) |

Nine further modules in the original 24-module selection could not be collected at all, for the same
missing `torch`/`aiohttp` and for `native_dp_wave_timeline` not being on `sys.path`.

## 9. The two previously unverified heartbeat claims

### 9.1 The `10.01 s / 1.01 s` stopped-peer timing comparison — now VERIFIED

Measured on mainline, same scenario, by neutralizing only the staleness limit
(`peer_heartbeat_timeout_s = 10**9`) so the wait can end only on its deadline, which is the behavior
before the heartbeat check existed. Evidence:
`h200_mainline/i017_heartbeat_merge_20261111/stopped_peer_detection_latency.json`.

| Case | Staleness check | Expected | Measured elapsed | Terminal exception | Exit | Thread left |
| --- | --- | --- | --- | --- | --- | --- |
| `peer_heartbeat_stopped` | disabled | waits the whole 10 s budget | **10.0215 s** | `TimeoutError: ... deadline exceeded` | 1 | none |
| `peer_heartbeat_stopped` | enabled | fails on the heartbeat first | **1.0172 s** | `RuntimeError: ... heartbeat unchanged for more than 1 s: ['host3']` | 1 | none |
| `peer_heartbeat_live` | enabled | completes normally | 3.0145 s | none | 0 | none |

This reproduces the recorded `10.01 s / 1.01 s` figures. At the real run's settings
(`startup_timeout_s=1800`, `peer_heartbeat_timeout_s=180`) a replica lost after the rendezvous is
detected in about 181 s rather than 1800 s.

### 9.2 The `101 passed in 19.50 s` related-harness suite — NOT REPRODUCIBLE HERE, and superseded

The recorded command includes `tests/unit/test_native_pd_canonical_http.py`, which cannot be collected
on **either** branch under this interpreter: it imports `aiohttp`, which the minimal release
environment intentionally omits. Both branches give `1 error during collection`. Every other
interpreter on this host was probed; the four that have `aiohttp` are other projects' 3.12
environments and all lack `plotly`, `sklearn`, `ddsketch` and `fasteners`, so none can run Frontier's
suite. The original figure was therefore produced in an environment that is no longer available here,
and the number itself is environment-specific.

The substance of that claim — that the heartbeat change does not break the related harness modules —
is covered more strongly by section 8.1, which compares an explicit baseline against the candidate
across 15 modules and finds identical failure sets.

## 10. Summary

| Item | Verdict |
| --- | --- |
| Root cause of the eight order-dependent timeouts | CONFIRMED and fixed, with the mainline/harness asymmetry explained |
| The `-P` defect on the harness branch | fixed, matching mainline's existing fix |
| Harness test module | `18 passed`, both collection orders |
| Pair-warmup reconciliation | already one implementation; evidence in section 4, no edit needed |
| Heartbeat ported to mainline | commit `efbe09a9`; `43 passed`, both collection orders |
| Regression risk | none observed: identical 93-id failure set against an explicit baseline |
| Release sanity and accuracy gate | unchanged / all pass |
| Recorded `10.01 s / 1.01 s` claim | verified by measurement |
| Recorded `101 passed` claim | not reproducible in any available environment; superseded by the baseline comparison |

## 11. Verification limits

- Every check is local CPU work on Python 3.10.6. No native engine, GPU or platform job ran, so this
  report says nothing about generation08's measured values, which are unchanged and separately
  reported.
- The heartbeat path is exercised only through the test module's process and platform stand-ins. Its
  behavior against a real lost replica remains evidenced by generation07/generation08 operationally,
  not by this report.
- Nine of the 24 native/calibration modules and 93 individual cases cannot run in the minimal
  environment. Their status is therefore unknown rather than passing, both before and after the port.
- Section 9.2's claim is closed as not reproducible, not as satisfied.
