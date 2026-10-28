## 1. Assignment and outcome

Implemented stage 1a assignment B5: match native DeepSeek-V4 attention kernel references by `(step class, CUDA Graph mode, padded rows, eager rows)`, report `NO_REFERENCE` when that shape is absent, and list shape coverage.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

Only these approved repository files were modified:

- `tests/comparison/calibration/dsv4_kernel_identity_check.py`
- `tests/analysis/v4_runtime_census.py`
- `tests/unit/test_dsv4_kernel_identity_check.py`

Read the root and ancestor `AGENTS.md` guidance; no applicable `CLAUDE.md` or additional guidance files existed in the affected directory ancestry. Read the named B5 assignment artifact, package 03 test report’s Kernel identity section, diagnostic `shape_matched.py`, and analysis-copy bridge configuration.

Preserved the settled decisions: FULL uses `None` for eager rows; PIECEWISE/NONE use scheduled rows; dummy eager execution uses all padded rows; both pilot and native row-count grids must be unique; pilot grid x must equal recorded dispatch tokens; candidate deduplication includes shape and kernel multiset; case and overall verdict priorities follow the assignment; only PASS exits 0; margin support is removed.

Outcome is **implemented**: all required checks finished successfully. Final review and acceptance remain with Claude.

The initial working tree contained only the named untracked `.cortexkit/`, `.pi/`, and `remaining_measurements/` directories, with no tracked modifications. Those directories were left untouched. No repository files were added. No Git state changes, task-record writes, GPU operations, network access, dependency changes, manual deletions, or moves were performed.

Observation: verification reported HEAD `42a86605`, while the brief names `56e06be2`. The cause was not established. The final tracked diff contains exactly the three approved files, and baseline copies were saved before editing. Claude should confirm the intended base during review; this discrepancy did not block the assigned checks.

## 2. Changes

- [dsv4_kernel_identity_check.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/comparison/calibration/dsv4_kernel_identity_check.py:94): `NativeCandidate` now carries padded/eager rows and exposes the complete `shape_key`. `native_candidates` derives native padded rows from the single named row-count kernel, consumes scheduled tokens from the census, and deduplicates by shape plus multiset. `_padded_rows` raises with trace and forward identification for missing or ambiguous native grids. `match_kernels` at line 202 selects only complete-key matches and retains closest-reference multiset containment and grid-mismatch diagnostics. `check_identity` at line 243 validates pilot grid x against dispatch tokens, computes eager rows, applies the requested status/verdict priorities, and adds step status counts and coverage. `main` at line 297 removes `--margin_cases`, prints referenced/unreferenced shapes, and returns 0 only for PASS. The module docstring states the coverage limit.
- [v4_runtime_census.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_runtime_census.py:197): `ForwardKernels.scheduled_tokens` defaults to `None`. `dispatch_kernel_multisets` fills it from the already parsed step range’s context plus generation tokens. No second trace parse is introduced. Existing `forward_census` output construction is unchanged.
- [test_dsv4_kernel_identity_check.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_kernel_identity_check.py:65): updated synthetic native/pilot fixtures with row-count kernels and scheduled-token metadata; removed margin tests; added the required shape, NO_REFERENCE, validation, dummy, deduplication, verdict, exit-code, and coverage checks. Also checks scheduled-token summation, absence of a step range, and preservation of census output fields. Existing role-expansion and closest-multiset tests remain covered.

Output choices: `shape_key` is a four-element JSON array. Coverage entries contain `pilot_case_ids`, `step_status_counts`, and `native_forward_count`. For NO_REFERENCE steps, `missing_kernel_count` and `missing_kernels` are null because no comparison exists. There were no deviations from the agreed matching behavior or scope.

## 3. Verification

All checks used `/data/ycfeng/envs/frontier-py310/bin/python`, verified as Python 3.10.6. Commands ran from the working root with:

```bash
ulimit -v 31457280
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export TMPDIR=/data/ycfeng/tmp/codex-implement/a1c6212e-bb73-48bd-b5e5-e0dc95cd4dda/b5
```

The initial inspection attempted bare `python`, which was not on PATH and exited 127. It was immediately rerun using the provided interpreter. No implementation or acceptance check failed.

**PASS — identity unit suite, exit 0.** Executed:

```bash
timeout 300s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_kernel_identity_check.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/codex-implement/a1c6212e-bb73-48bd-b5e5-e0dc95cd4dda/b5/pytest_identity
```

Output tail:

```text
..........................................                               [100%]
42 passed in 2.26s
```

**PASS — final combined identity and regression suites, exit 0.** Executed after the final source edits:

```bash
timeout 300s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_kernel_identity_check.py tests/unit/test_dsv4_composition_check.py tests/unit/test_dsv4_graph_bridge_check.py tests/unit/test_dsv4_composition.py tests/unit/test_dsv4_graph_replay_profiling.py tests/unit/test_dsv4_attention_profiling.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/codex-implement/a1c6212e-bb73-48bd-b5e5-e0dc95cd4dda/b5/pytest_regression --junitxml /data/ycfeng/tmp/codex-implement/a1c6212e-bb73-48bd-b5e5-e0dc95cd4dda/b5/regression.xml
```

Output tail:

```text
........................................................................ [ 97%]
........                                                                 [100%]
296 passed in 5.62s
```

Counts verified from the JUnit results:

| Suite | Passed |
| --- | ---: |
| `test_dsv4_kernel_identity_check.py` | 42 |
| `test_dsv4_composition_check.py` | 13 |
| `test_dsv4_graph_bridge_check.py` | 18 |
| `test_dsv4_composition.py` | 47 |
| `test_dsv4_graph_replay_profiling.py` | 98 |
| `test_dsv4_attention_profiling.py` | 78 |
| Total | 296 |

An executed `rg` search found no other unit suite importing `v4_runtime_census` besides the identity suite.

**PASS — cold imports, both exit 0.** Each command ran in a fresh interpreter and additionally asserted that no torch, vLLM, or CUDA modules were loaded:

```bash
timeout 20s /data/ycfeng/envs/frontier-py310/bin/python -c 'import sys; import dsv4_kernel_identity_check; assert not any(name == "torch" or name.startswith("torch.") or name == "vllm" or name.startswith("vllm.") or name == "cuda" or name.startswith("cuda.") for name in sys.modules); print("dsv4_kernel_identity_check: import PASS; no torch/vllm/cuda modules loaded")'
timeout 20s /data/ycfeng/envs/frontier-py310/bin/python -c 'import sys; import v4_runtime_census; assert not any(name == "torch" or name.startswith("torch.") or name == "vllm" or name.startswith("vllm.") or name == "cuda" or name.startswith("cuda.") for name in sys.modules); print("v4_runtime_census: import PASS; no torch/vllm/cuda modules loaded")'
```

Observed output:

```text
dsv4_kernel_identity_check: import PASS; no torch/vllm/cuda modules loaded
v4_runtime_census: import PASS; no torch/vllm/cuda modules loaded
```

**PASS — package 03 P2-32 and P2-64, both exit 0.** Executed sequentially, with native traces read one at a time:

```bash
timeout 2400s /data/ycfeng/envs/frontier-py310/bin/python tests/comparison/calibration/dsv4_kernel_identity_check.py --config /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge_config.json --native_trace_dirs /data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces --output /data/ycfeng/tmp/codex-implement/a1c6212e-bb73-48bd-b5e5-e0dc95cd4dda/b5/identity_p32.json
timeout 2400s /data/ycfeng/envs/frontier-py310/bin/python tests/comparison/calibration/dsv4_kernel_identity_check.py --config /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge_config.json --native_trace_dirs /data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces --output /data/ycfeng/tmp/codex-implement/a1c6212e-bb73-48bd-b5e5-e0dc95cd4dda/b5/identity_p64.json
```

Relevant log output:

```text
[dsv4_identity] PASS: 175 steps, {'PASS': 46, 'FAIL': 0, 'NO_REFERENCE': 129}
[dsv4_identity] PASS: 175 steps, {'PASS': 54, 'FAIL': 0, 'NO_REFERENCE': 121}
```

Each log also printed the complete referenced and unreferenced shape lists: 7 referenced and 16 unreferenced for each dataset. Both logs emitted the existing experimental-model architecture warning; neither reported an identity error.

| Dataset | Expected PASS / NO_REFERENCE / FAIL | Observed | Verdict | Case statuses |
| --- | --- | --- | --- | --- |
| P2-32 | 46 / 129 / 0 | 46 / 129 / 0 | PASS | 7 PASS, 18 NO_REFERENCE |
| P2-64 | 54 / 121 / 0 | 54 / 121 / 0 | PASS | 8 PASS, 17 NO_REFERENCE |

Both reports contain all 175 pilot steps: FULL 104, PIECEWISE 60, NONE 11. The final audit validated coverage counts and case IDs against the step records, native forward counts against the deduplicated candidates, and null missing-kernel diagnostics for unreferenced steps.

The previously failing P2-64 steps are now correctly NO_REFERENCE:

- `e0-decode-b8-p6144-piecewise9`, step indices 67–74, shape `(decode, PIECEWISE, 16, 8)`: all eight NO_REFERENCE.
- `e1-first_tail-b10-p6144`, step indices 9–11, shape `(first_tail_with_decodes, PIECEWISE, 16, 11)`: all three NO_REFERENCE.

There are no differing real-data results or FAIL steps to investigate.

Complete pilot shape coverage follows. P/F/N means PASS/FAIL/NO_REFERENCE; the number after the semicolon is the native forward count at that exact key. A native count of zero identifies an uncovered shape.

| Shape (class, mode, padded, eager) | Pilot case ids | P2-32 P/F/N; native forwards | P2-64 P/F/N; native forwards |
| --- | --- | --- | --- |
| `decode, FULL, 1, None` | `e0-decode-b1-p6144` | 8/0/0; 68 | 8/0/0; 162 |
| `decode, FULL, 2, None` | `e0-decode-b2-p6144` | 8/0/0; 48 | 0/0/8; 0 |
| `decode, FULL, 4, None` | `e0-decode-b4-p6144`, `e0-decode-b4-p7150` | 0/0/16; 0 | 0/0/16; 0 |
| `decode, FULL, 8, None` | `e0-decode-b8-p6144` | 0/0/8; 0 | 0/0/8; 0 |
| `decode, FULL, 16, None` | `e0-decode-b16-p6144`, `e0-decode-b8-p6144-full16` | 0/0/16; 0 | 16/0/0; 72 |
| `decode, FULL, 24, None` | `e0-decode-b24-p6144` | 0/0/8; 0 | 0/0/8; 0 |
| `decode, PIECEWISE, 4, 2` | `e0-decode-b2-p6144-piecewise4` | 0/0/8; 0 | 0/0/8; 0 |
| `decode, PIECEWISE, 16, 8` | `e0-decode-b8-p6144-piecewise9` | 0/0/8; 0 | 0/0/8; 0 |
| `dummy, FULL, 1, None` | `e0-dummy-full1` | 8/0/0; 257 | 8/0/0; 655 |
| `dummy, FULL, 2, None` | `e0-dummy-full2` | 0/0/8; 0 | 0/0/8; 0 |
| `dummy, FULL, 4, None` | `e0-dummy-full4` | 0/0/8; 0 | 0/0/8; 0 |
| `dummy, FULL, 8, None` | `e0-dummy-full8` | 0/0/8; 0 | 0/0/8; 0 |
| `dummy, FULL, 16, None` | `e0-dummy-full16` | 0/0/8; 0 | 0/0/8; 0 |
| `dummy, NONE, 1, 1` | `e0-dummy-none6144` | 8/0/0; 4 | 8/0/0; 16 |
| `dummy, PIECEWISE, 1, 1` | `e0-dummy-piecewise1` | 8/0/0; 1 | 8/0/0; 6 |
| `dummy, PIECEWISE, 2, 2` | `e0-dummy-piecewise2` | 0/0/8; 0 | 0/0/8; 0 |
| `dummy, PIECEWISE, 4, 4` | `e0-dummy-piecewise4` | 0/0/8; 0 | 0/0/8; 0 |
| `dummy, PIECEWISE, 8, 8` | `e0-dummy-piecewise8` | 0/0/8; 0 | 0/0/8; 0 |
| `first_tail_alone, PIECEWISE, 1, 1` | `e1-first_tail-b0-p6144` | 3/0/0; 2 | 3/0/0; 3 |
| `first_tail_with_decodes, PIECEWISE, 4, 4` | `e1-first_tail-b3-p6144` | 0/0/3; 0 | 0/0/3; 0 |
| `first_tail_with_decodes, PIECEWISE, 8, 8` | `e1-first_tail-b7-p6144` | 0/0/3; 0 | 0/0/3; 0 |
| `first_tail_with_decodes, PIECEWISE, 16, 11` | `e1-first_tail-b10-p6144` | 0/0/3; 0 | 0/0/3; 0 |
| `prefill, NONE, 6144, 6144` | `e0-prefill-p6144` | 3/0/0; 26 | 3/0/0; 35 |

**PASS — final consistency, syntax, size, and scope audit, exit 0.** Executed:

```bash
timeout 30s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/codex-implement/a1c6212e-bb73-48bd-b5e5-e0dc95cd4dda/b5/final_audit.py
```

This executed Python 3.10 AST parsing, module-size assertions, saved-report and JUnit assertions, and these read-only Git checks:

```bash
timeout 20s git diff --check
timeout 20s git diff --name-only
timeout 20s git rev-parse --short=8 HEAD
```

Observed: all assertions passed, `git diff --check` exited 0, and changed paths exactly matched the approved files. Final module sizes are identity 328 lines, census 458 lines, and identity tests 384 lines. Audit output ends with `B5 final audit PASS`.

Scratch root: `/data/ycfeng/tmp/codex-implement/a1c6212e-bb73-48bd-b5e5-e0dc95cd4dda/b5/`.

Saved artifacts include `identity_p32.json`, `identity_p64.json`, their `.log` files, `unit_identity.log`, `regression.log`, `regression.xml`, `imports.log`, `static_checks.log`, `final_audit.py`, `final_audit.log`, `coverage_p32.txt`, `coverage_p64.txt`, `coverage_table.md`, and `check_commands.txt`. The command record preserves the executed argument lists and output redirections. Baseline copies are `dsv4_kernel_identity_check.py.before`, `v4_runtime_census.py.before`, and `test_dsv4_kernel_identity_check.py.before`. Test temporary directories also remain under this scratch root.

## 4. Remaining work and decisions for Claude

No assigned implementation or verification work remains, and no acceptance blocker was found. Claude retains review, acceptance, commits, and task-record updates.

Confirm the intended checkout base because the observed HEAD differs from the brief. No inference is made about whether the brief is stale or the lead changed HEAD concurrently.

The demonstrated result is kernel containment for the shapes present in the retained native traces. The 16 unreferenced pilot shapes in each dataset remain unqualified by this check, as required. Their further qualification belongs to the planned end-to-end replay work. No GPU qualification was performed in this CPU-only assignment.

## 5. Review handoff

Inspect the complete-key candidate selection, row-count validation, dummy and FULL eager-row handling, null NO_REFERENCE diagnostics, case/verdict priorities, and coverage aggregation. The census diff is confined to the new field and its population; review that narrow boundary for preservation of other census outputs.

Rerun the combined 296-test command and both cold imports after integration. If matching/report code changes during review, rerun both package 03 identity commands and expect the exact counts above. The saved audit script can revalidate the reports and coverage without rereading native traces. Final acceptance has not been claimed.
