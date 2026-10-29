# Worktree cleanup — 2026-10-25

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Recorded the D-WORKTREE-CLEANUP inventory, preservation, removal, restore and verification. |

**Authority.** [D-WORKTREE-CLEANUP](../requirements.md) (requirements item 11), approved by YC after the concurrent session was stopped.

**Result.** 107 of this task's 110 worktrees are gone. Every record they held is in the mainline worktree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Three worktrees were removed and then restored, because tracked tests read them by absolute path. No branch was deleted. Full per-file and per-worktree detail is in [manifest.json](manifest.json).

## Scope

The 218 registered worktrees were classified by name, by the task directories inside them, and by HEAD subject.

| Class | Count | Action |
| --- | ---: | --- |
| This task (DSV4/GLM H800 work, validation and fidelity checkouts) | 110 | Preserved, then removed |
| Other task (`step5*`, `verify-step5*`, `d29*`, `d30*`, `d5-ttft*`, step5 fidelity checkouts, `stepcode-codex-home`) | 103 | Not touched |
| Kept (`Frontier` main, `glm53-dsv4-vllm028`, `frontier-calibration-skill`, `remove-legacy-run-outputs`, `stage-admission-ordering`) | 5 | Not touched |

Before any removal, the newest write in an in-scope worktree was 11 hours old, and it was a launcher exit receipt. No process in this sandbox used an in-scope path.

## Preservation

| Step | Result |
| --- | --- |
| task_memory files missing from the mainline | 6,049 files (425.2 MB) copied to the same relative path with `shutil.copy2`; each SHA-256 checked after copying; 0 path conflicts between worktrees |
| task_memory files whose content differs from the mainline copy | 62 files from 12 worktrees archived under `divergent_records/<worktree>/<path>`; the mainline copy is unchanged |
| Uncommitted state (25 worktrees) and unreachable HEADs (2 worktrees) | 27 refs `refs/archive/worktrees/<name>`. Each dirty state was written through a temporary index (`read-tree HEAD`, `add -A`, `write-tree`, `commit-tree -p HEAD`). For each ref, reading its tree back into a temporary index showed no difference from the working tree and no untracked file. |
| Mainline symlinks into removed worktrees | 3 repointed to local relative copies (`candidate_h800_p0_02`, `analysis_candidate03/inputs`, `analysis_candidate03/manifest.yaml`); all resolve |
| Symlinks inside removed task_memory | 2 `glm-h5-integration` pytest links recreated as relative links; the 2 `v4-dp-sequence-*` links already existed in the mainline |

Ignored content other than task_memory was not kept: `cache/` (6.67 GB of predictor caches), `__pycache__`, `.pytest_cache`, `outputs/metrics` and ignored test outputs (about 24 MB). All of it can be regenerated.

## Removal

`git worktree remove` ran for each of the 110 worktrees. `--force` was used only for the 25 dirty worktrees, after their archive refs were verified. Then `git worktree prune` ran.

- 109 removed.
- `glm-full-dummy-h200`: unregistered, but its directory is left. 23 files under `task_memory/task_2026-10-21_glm_full_dummy_h200/graph_probe_*` are owned by uid 65534 (written by a GPU worker), and this user cannot delete them. All 23 are byte-identical in the mainline. Deleting the directory needs root.

## Restore

The first targeted test run after removal failed in `tests/unit/test_native_reader_binding.py` and `tests/unit/test_native_groundtruth_check.py`. These provenance tests check accepted native packages against the worktree each package was launched from: its absolute path, its commit, a clean `git status`, and the package files inside it. The failing tests named three worktrees:

| Worktree | Branch at the recorded HEAD | Files put back |
| --- | --- | ---: |
| `v4-p0-native` | `v4-p0-native` | 708 |
| `v4-balanced-native-model` | `feat/v4-balanced-native-model` | 51 |
| `v4-balanced-callers` | `feat/v4-balanced-callers` | 41 |

Each was recreated with `git worktree add <path> <branch>`. The branch tip was checked to equal the recorded HEAD. Then each worktree's task_memory files were copied back from the preserved copies, with their SHA-256 checked. `git status --porcelain --untracked-files=all` is clean in all three. They now count as kept worktrees.

`tests/unit/test_dsv4_graph_measured.py:783` read a package from the removed `v4-balanced-prefill-run` worktree. It now reads the byte-identical mainline copy (`diff -rq` showed no difference before removal).

## Verification

- Before removal: `tests/unit/test_dsv4_graph_measured.py` 194 passed.
- After the path change: the edited test passed.
- After the restore, the full unit suite was run with `/data/ycfeng/envs/frontier-py310/bin/python -m pytest -p no:cacheprovider -q -p no:logging --continue-on-collection-errors tests/unit -rfE` (Python 3.10.6, `PYTHONPATH=$PWD`, `TMPDIR=/data/ycfeng/tmp`). Result: 134 failed, 6112 passed, 208 skipped, 108 errors.
  - No failure message names a removed worktree. The only worktree path in the log is the mainline.
  - The remaining failures come from the environment, not from the cleanup:
    - missing `torch`, `aiohttp` and `matplotlib`;
    - no `caplog` fixture;
    - `TarFile.extractall(filter=)` and `python -P`, which need a newer Python than 3.10.6;
    - no `python` on `PATH`;
    - the absent `/data/ycfeng/stepfun-performance-optimization` Reference checkout;
    - `tests/debug/` scripts that are not in this checkout;
    - README content checks.
  - Log: `/data/ycfeng/tmp/wt_cleanup/unit_after.log`; JUnit XML: `/data/ycfeng/tmp/wt_cleanup/unit_after.xml`.
- No full-suite baseline was taken before removal. The claim that nothing else broke rests on that log scan, not on a before/after comparison.

## Notes for readers of older records

Older task records cite absolute paths under `/data/ycfeng/Frontier/.worktrees/<removed-worktree>/task_memory/...`. That content now lives at the same relative path under the mainline worktree. Where the mainline already had a different file at that path, the worktree's version is in `divergent_records/<worktree>/<path>`. Branches still hold all committed history, and `refs/archive/worktrees/<name>` holds the uncommitted states.

Scripts used: [scripts/](scripts/) (copies of `/data/ycfeng/tmp/wt_cleanup/{inventory,classify,preserve,remove,restore}.py`).
