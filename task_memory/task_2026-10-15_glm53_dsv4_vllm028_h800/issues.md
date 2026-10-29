# Issues: failures and resolutions

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | I-002 opened: two `test_balanced_routing.py` cases fail on main since B16b; the module is skipped without torch, so earlier suites did not run it. |
| 2026-10-28 | Created: unit-test fixtures wrote into this worktree's Git directory through an exported `GIT_DIR`. |
| 2026-10-28 | I-001 resolved: repair applied after YC's approval and verified. |

## I-001: unit-test fixtures wrote into the worktree's Git directory (2026-10-28)

Status: resolved 2026-10-28. YC approved the repair ("批准，继续"); the lead applied it with exit 0, and every check below passed. Owner: lead.

**Failure.** During the B8 review, the lead ran the whole unit suite twice in parallel: before (on a `git archive dd4c4317` snapshot) and after (on the working tree). The runner was `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b8_review/checks/run_unit_pair.sh`. The before half exported `GIT_DIR=/data/ycfeng/Frontier/.git/worktrees/glm53-dsv4-vllm028` to give the snapshot a Git identity. Fixture tests that build temporary repositories then sent `git init`, `git config` and `git commit` to that directory instead of their temporary directories:
- `tests/unit/test_moe_ep_baseline_replay.py:167`;
- `tests/unit/test_moe_ep_non_dummy_matrix.py:5615`;
- `tests/unit/test_native_pd_manifest.py:35`;
- `tests/unit/test_refactor_fidelity_gate.py:464`.

**Observed state** (writes between 10:43:04 and 10:45:31 UTC):
- `refs/heads/feat/glm53-dsv4-vllm028`: `dd4c4317` (B7) → `80aacd8e` ("baseline", Frontier Tests, parent `dd4c4317`, deletes 4994 files and adds `tracked.txt`).
- This worktree's HEAD is detached at `4a107a47` ("initial", author `t`), after 29 fixture commits. Its index holds the 9 fixture files.
- The shared `/data/ycfeng/Frontier/.git/config`:
  - `core.bare = true`, written by the reinitializing `git init`; the main checkout is a normal checkout, so the value must be `false`;
  - an appended `[user]` section, `name = t` and `email = t@example.com`, which would give every repository commit without its own identity the author `t`.
- Not affected:
  - the working-tree files (the fixtures' work tree was their temporary directory), so the B8 edits are on disk;
  - the main checkout's `HEAD` and `index`;
  - every other branch and every remote-tracking ref;
  - other worktrees' HEADs.

**Backups.** In `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b8_review/git_repair/`: `config.polluted`, `worktree_HEAD.polluted`, `worktree_index.polluted`, `branch_ref.polluted`.

**Repair.** The permission check first blocked it as a destructive Git action; it was applied after YC's approval. It restores the earlier state exactly, and the reflog keeps the old values:
1. `git config --file /data/ycfeng/Frontier/.git/config core.bare false`
2. `git config --file /data/ycfeng/Frontier/.git/config --remove-section user`
3. `git update-ref refs/heads/feat/glm53-dsv4-vllm028 dd4c4317606789e5cafcba148f6c1955f3b96f7f 80aacd8e282bd1ae2d877caeb794e5718dbf7686`
4. In `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`: `git symbolic-ref HEAD refs/heads/feat/glm53-dsv4-vllm028`, then `git read-tree dd4c4317606789e5cafcba148f6c1955f3b96f7f`. This touches the index only; the working tree is not changed.

**Checks after the repair:**
- `git -C /data/ycfeng/Frontier rev-parse --is-bare-repository` prints `false`;
- `git config user.name` resolves to the global identity;
- in this worktree, `git branch --show-current` prints `feat/glm53-dsv4-vllm028` and `git log -1` shows `dd4c4317`;
- `git status --short` lists only the B8 changes, the untracked task records and the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/`.

Observed after the repair:
- every check above as expected; the main checkout's `git status` works on branch `main`;
- `git status --short` in this worktree lists exactly the B8 changes and the known untracked paths;
- the whole unit suite rerun on the working tree without `GIT_DIR`: 133 failed, 6650 passed, 133 skipped, 109 errors. It differs from the first after-run only by the R4 test (failure → passed), so the 91 s overlap changed no result.

B8 was then committed.

**Prevention.** Never export `GIT_DIR` or `GIT_WORK_TREE` around a test suite or any command that runs `git init`. A before-snapshot that needs a Git identity gets its own worktree; otherwise its Git-reading tests are accepted as failing in the comparison.

## I-002: `test_balanced_routing.py` cache test fails on main since B16b (2026-10-29)

Status: open, deferred (outside B23). Owner: lead.

Observed during the B23 review. With CPU torch on the path (`PYTHONPATH=...:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages`, `VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0`), `tests/unit/test_balanced_routing.py` gives 126 passed and 2 failed, the same on main `2da2c186` and on the B23 tree:

```text
tests/unit/test_balanced_routing.py:316: KeyError: 'moe_gating_routing_topk'
    assert manager.get_models_for_cluster(role)["eager"][op] is cold[role]
```

Cause (from the code): B16b (`61c2f9d3`) makes `_get_measurement_types_for_cluster` return only the V4 measurement type, `KERNEL_ONLY`, for every DeepSeek-V4 role, so the `eager` view of `get_models_for_cluster` is empty for V4. The test still trains V4 gating models from `CUDA_EVENT` rows and reads them from the `eager` view. Without torch the module is skipped by `pytest.importorskip("torch")`, so suites run without torch do not report it.

Next action: update the test to the B16b contract (V4 roles read `KERNEL_ONLY`), in a later assignment; no simulator behavior is involved.
