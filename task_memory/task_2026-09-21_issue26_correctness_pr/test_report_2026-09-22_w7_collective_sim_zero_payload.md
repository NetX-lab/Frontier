# W7 Test Report — Zero-Payload Collective Through the collective-sim Backend

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | Fix review: `eb7bc4f` superseded by companion `ff11ee6` (cross-server empty all-to-all, W7-R1) and the Frontier test rewritten in `6d621c8` (W7-R2). Section "Superseded 2026-09-24" added; the measurements below are those taken at `eb7bc4f`. |
| 2026-09-22 | First issue: companion-repository fix, Frontier gitlink bump, both negative controls, clean-checkout validation, and the governance-scan repair the bump exposed. |

## 1. Scope

W7 is the zero-payload defect recorded in `plan.md` A7 and re-verified in
`review.md`. The user authorized the companion-repository option on
2026-09-22 ("1.授权"), so the work is: repair the backend in
`fwyc0573/frontier-htsim`, publish it, move Frontier's gitlink onto the
published commit, and add a Frontier-side regression test.

No Frontier source file changes. The Frontier diff is the gitlink, one new test
module, and one repair to three governance scans that the gitlink bump exposed.

## 2. What was broken, and that it is reachable

`moe_operator_times.py:512` computes `data_size_bytes = embedding_dim * 2 *
routed_tokens` and hands it to `predict_all_to_all`. An expert-parallel lane
that routes no token in a step makes that zero. `predict_reduce_scatter`
floor-divides by the device count and reaches zero the same way for any payload
smaller than the device count. `base_cc_backend._validate_data_size` rejects
only negative sizes, so zero reaches the collective-sim runner.

The published runner then rejected it. Three defects, each confirmed by
execution against the published `main` (`b8518af`) before any edit:

| # | Defect | Evidence |
| --- | --- | --- |
| 1 | An explicit zero is read as a missing field | `htsim_runner.py` tested `getattr(args, k) in (None, "", 0)` over a required-field list containing `tensor_bytes`. Running with `tensor_bytes = 0` and with the field deleted produced the identical `exit=2, Error: missing required fields: ['tensor_bytes']`. |
| 2 | A negative payload is accepted | `--tensor-bytes` is a bare `type=int` with no lower bound and no schema check, so `-1` passed validation and reached flow generation. |
| 3 | An explicit CLI zero loses to a positive spec value | `set_if_none_or_zero` reads `0` as missing on both sides, so `--tensor-bytes 0` against a spec of 32768 yielded 32768. |

A zero-byte collective is not a zero-cost collective, so Frontier cannot
short-circuit to `0.0` without changing the backend's synchronization
semantics. The repair belongs in the runner.

## 3. The companion-repository change

Repository `fwyc0573/frontier-htsim`, branch `fix/zero-payload-input-handling`,
commit `eb7bc4f`, companion draft PR
<https://github.com/fwyc0573/frontier-htsim/pull/1>.

| File | Change |
| --- | --- |
| `htsim_runner.py` | `tensor_bytes` merges through `set_if_none_or_empty`, which treats only `None` and `""` as unset, instead of `set_if_none_or_zero`. The argparse default for `--tensor-bytes` is already `None`, so scenario-file merging is unaffected. |
| `htsim_runner.py` | The required-field check became a table carrying, per field, whether zero is a legal value. Zero stays "missing" for `collective_type`, `domain_dims`, `topology`, `nodes`, `gpus_per_server` and `tp`. |
| `htsim_runner.py` | A negative payload returns `2` with `Error: tensor_bytes must be >= 0.` before flow generation. |
| `python/collective_sim_core/schema.py` | `Scenario.validate()` raises `collective.tensor_bytes must be >= 0`, so a serialized scenario cannot smuggle one past the schema. |
| `.gitignore` | `tests/` narrowed to `tests/*` plus an explicit entry for the published test, so git descends into the directory. Private working material under `tests/` stays ignored. |

No change to flow generation, topology modelling, or any latency arithmetic.

### Superseded 2026-09-24

- `eb7bc4f` let a zero payload reach flow generation, where the all-to-all
  generators were undefined once a pair crossed servers: multi-phase
  `pairwise_steps` raised `ZeroDivisionError`, `nccl_pairwise` emitted 0-byte
  flows that htsim reads as unbounded and never finished, and single-phase
  `pairwise_steps` and `full_mesh` emitted no flow and cost nothing. The tests
  in this report used one server with intra-server traffic excluded, so no
  flow reached the simulator.
- Companion `ff11ee6` (same branch, same draft PR 1) sends one byte per peer
  for an empty all-to-all, which is what the existing `ceil(tensor_bytes / n)`
  share already gives every payload of at most `n` bytes. Non-zero payloads are
  unchanged. The statement above that flow generation is untouched no longer
  holds for the zero payload.
- Frontier `6d621c8` moved the gitlink to `ff11ee6` and replaced the four
  Frontier tests with three `predict_all_to_all` cases (one single-server, two
  across servers). Companion 12 passed, Frontier 3 passed; against `eb7bc4f`
  the new cases fail (`/data/ycfeng/tmp/issue26-correctness-pr/review_20260924/w7_fix/`,
  `final_20260924/collective_sim_*.txt`).

## 4. Companion-side validation

Command, from the companion checkout:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/test_zero_payload_input.py -q -p no:cacheprovider --no-header
```

| Run | Sources | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| Repaired | `eb7bc4f` | all pass | **9 passed** | PASS |
| Negative control | `htsim_runner.py` and `schema.py` restored to `HEAD` | the zero-payload cases fail | **6 failed, 3 passed**, the failures reporting `Error: missing required fields: ['tensor_bytes']` | PASS |

The three that pass on pristine sources assert behavior that already worked: a
genuinely missing payload is an error, the scenario file still supplies an unset
payload, and the 32768-byte control arm. The two end-to-end cases skip when
`sim/datacenter/htsim_ndp` is absent.

The negative control was taken by copying the two edited files aside,
`git checkout HEAD --` on them, running, then restoring — deliberately not
`git stash`, whose stack is shared with other worktrees on this host.

## 5. Frontier-side validation

New module `tests/unit/test_collective_sim_zero_payload.py`, four tests on the
canonical `TP=4 x DP=2, EP=8` single-server pod from AGENTS.md, priced with
`intra_server_model=nvlink_analytic` and `nvlink_latency_us=0.5`.

Eight NVLink-connected ranks exchange over seven hops, so the payload-independent
synchronization term is `7 x 0.5 us = 0.0035 ms`.

| Test | Expectation | Measured |
| --- | --- | --- |
| `test_an_empty_all_to_all_keeps_its_synchronization_latency` | `0.0035 ms` | `0.0035` |
| `test_an_empty_reduce_scatter_keeps_its_synchronization_latency` | `0.0035 ms` for a 7-byte payload that floor-divides to zero | `0.0035` |
| `test_a_populated_all_to_all_costs_more_than_an_empty_one` | strictly greater | `0.0060486222 > 0.0035` |
| `test_a_negative_payload_is_still_rejected` | `ValueError` from Frontier's own guard | raised |

Command:

```bash
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_collective_sim_zero_payload.py -q -rA -p no:cacheprovider --no-header
```

| Run | Gitlink | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| Repaired | `eb7bc4f` | all pass | **4 passed** | PASS |
| Negative control | `b8518af`, the pre-fix commit both this branch and main pinned | the three zero-payload cases fail | **3 failed, 1 passed**, each failure reporting `Error: missing required fields: ['tensor_bytes']` | PASS |

The module skips at import when the optional submodule is not initialized and
built, which is the default state and the state AGENTS.md documents.

## 6. Clean-checkout validation

The point of this check is that the gitlink resolves from the published remote,
not from anything local to this host.

```bash
git clone --branch fix/issue26-correctness-pr <worktree> /data/ycfeng/tmp/w7_clean_checkout
cd /data/ycfeng/tmp/w7_clean_checkout
pytest tests/unit/test_collective_sim_zero_payload.py            # submodule absent
git submodule update --init frontier/cc_backend/backends/collective-sim
make -C frontier/cc_backend/backends/collective-sim/sim -j"$(nproc)"
pytest tests/unit/test_collective_sim_zero_payload.py            # submodule present
```

| Stage | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Submodule absent | the module skips | **1 skipped** | PASS |
| `git submodule update --init` | checks out `eb7bc4f` from `https://github.com/fwyc0573/frontier-htsim.git` | `Submodule path '...': checked out 'eb7bc4fda9744aa117063972eb0fbabbef91c453'` | PASS |
| `make -j` | builds | `build_exit=0` | PASS |
| Submodule present | all four pass | **4 passed** | PASS |

`.gitmodules` still records `branch = main`, which is correct for the state after
the companion PR merges. `git submodule update --init`, the command AGENTS.md
documents, uses the recorded gitlink and lands on `eb7bc4f`;
`git submodule update --remote` would follow `main` instead and drop the fix
until the companion PR merges.

## 7. Regression, and a governance scan the bump exposed

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit -q \
  -p no:cacheprovider --no-header --continue-on-collection-errors
```

First run after the bump: **85 failed, 3781 passed, 49 skipped, 11 errors**
against the standing baseline of 84 failed / 3778 passed / 49 skipped /
11 errors. Diffing the two `FAILED` lists named exactly one new failure:

```
FAILED tests/unit/test_model_architecture_registry.py::test_raw_model_profile_resolution_callsites_are_allowlisted
```

Cause, confirmed by running it alone: the scan walks
`(repo_root / "frontier").rglob("*.py")` and `ast.parse`s every hit. Once the
optional submodule is initialized, that walk reaches 38 vendored files, one of
which — `sim/EXAMPLES/in_and_out/process_data.py` — raises
`IndentationError: expected an indented block after 'with' statement on line 24`.

This is a latent defect in the governance test, not in the backend: any
developer who follows the AGENTS.md instruction to initialize the submodule
would hit it. Three test modules walk that tree:

| Module | Behavior before | Behavior after the bump |
| --- | --- | --- |
| `test_model_architecture_registry.py:575` | `ast.parse` with no guard | fails on the unparsable vendored file |
| `test_module_split_boundaries.py:106` | `ast.parse` inside `except SyntaxError: continue` | tolerated, but silently measures vendored files |
| `test_spec_decode_mtp_registry.py:169` | substring search, no parsing | tolerated, but silently measures vendored files |

Repair: `tests/frontier_sources.py` exposes `iter_frontier_sources(repo_root)`,
which yields the Python files under `frontier/` excluding the vendored subtree,
and all three scans call it. 413 of the 451 files under `frontier/` are
Frontier's own; the 38 excluded are the submodule's.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| The three governance modules | all pass | **88 passed** | PASS |
| Full `tests/unit` after the repair | back to the 84-failure baseline, plus the four new tests | **84 failed, 3782 passed, 49 skipped, 11 errors**; diffing the `FAILED` lists against the baseline gives an empty set in both directions | PASS |

## 8. Publication

| Artifact | State |
| --- | --- |
| `fwyc0573/frontier-htsim` branch `fix/zero-payload-input-handling` | pushed, `eb7bc4f` |
| Companion PR <https://github.com/fwyc0573/frontier-htsim/pull/1> | open, **draft** |
| Frontier commit `1b95187` | gitlink `b8518af` -> `eb7bc4f`, plus the new test module |
| PR 35 | remains **draft** |

## 9. Limits of this validation

- The synchronization figure `7 x 0.5 us` is the analytic NVLink intra-server
  model's own arithmetic, not a measurement against hardware. These tests hold
  the boundary and the monotonicity; they do not validate the model.
- With the default `intra_server_model=legacy_fabric` and
  `collective_exclude_intra_server=True`, a single-server pod prices both an
  empty and a 1 MiB all-to-all at `0.0`, because all traffic is intra-server and
  excluded. That is the configuration's own semantics and is unrelated to the
  payload; the tests therefore use the analytic model, where the payload term is
  observable.
- The companion PR is draft and unmerged, so the gitlink points at a branch
  commit. It must be re-pointed at `main` once that PR merges.
