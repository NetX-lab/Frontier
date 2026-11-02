**1. Assignment and outcome**

Implemented stage 2 assignment B9 in `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, HEAD `0faf1aae70644aa4e535dbadddf94f9909fb2573`.

Writes were confined to the assigned collective modules, assigned unit tests, `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/`, and private scratch storage. Read the working and parent AGENTS.md files (identical), G5h design, C1 §4.5, task issues.md I-001, K5 launcher/configs, P2 package inputs, and the identified native source at commit 2cf0a69. No applicable CLAUDE.md was found.

Preserved the settled design: the seven-call V4 list, unpadded SP reduce-scatter keys, grouped EP dispatch, no V4 DP-sync call, ordered eager vectors, TP measurements restricted to uniform vectors, eager logits, hard G5h-16 gates, soft G5h-32 gates, and always-soft image vLLM comparisons. Reused K5 node_entry.py unchanged. No timing-method or simulator changes were made.

The outcome is **partial solely because the literal hygiene grep still matches pre-existing bytecode**. All implementation work and executable CPU acceptance checks are finished. Removing or moving that bytecode would violate the assignment's no-deletion/no-move restriction. Final acceptance remains with Claude.

**2. Changes**

Paths below are relative to the working root. `L` denotes `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch`.

| File | Status | Behavior and relevant references |
| --- | --- | --- |
| `frontier/profiling/collectives/vllm_forward_collectives.py` | Modified | Added the two SP ops, documented unpadded reduce-scatter rows, extended call_bytes, and renamed ceil_div. References: lines 52, 145, 185. GLM planning otherwise preserved. |
| `frontier/profiling/collectives/vllm_dsv4_forward_collectives.py` | New | DeepseekV4Shape, V4 call list, native-source references, scenario checks, and exact G5h grid selection. References: lines 46, 66, 116. |
| `frontier/profiling/collectives/vllm_collective_benchmark.py` | Modified | Lazy native SP function imports and tensor issuance; custom-collective metadata; architecture parser/planner table. References: communicators line 232, _issue line 263, plan_from_config line 490, main line 500. |
| `tests/unit/test_vllm_forward_collectives.py` | Modified | Four SP call_bytes cases, including non-divisible reduce-scatter inputs and lane-local indexing; line 137. |
| `tests/unit/test_collective_benchmark_helpers.py` | Modified | Main architecture selection/rejection, native SP routing with fake tensors, and custom communicator metadata tests; lines 222, 244, 255, 289. |
| `tests/unit/test_vllm_dsv4_forward_collectives.py` | New | Literal payload values, call order/counts/capture flags, both stage grids, filtering, config-derived shapes and rejection cases. |
| `L/stage_g5h16_tp8_dp2_ep16.json` | New | Requested H200/step_main two-node stage, full grid, hard gate, pinned image/P2 inputs, timing settings and output path. |
| `L/stage_g5h32_tp8_dp4_ep32.json` | New | Requested four-node stage, EP-only selection, empty logits grid and soft gate. |
| `L/runtime_check.py` | New | build_lock line 39; dependency-free worker_command line 66; check_runtime line 81. Hashes mounted sources/config, P2 NCCL and five native files. Records each comparison, publishes hard failures for K5 peers, and returns the required exit code. |
| `L/submit.py` | New | H200 resource/cap assertions, lock receipt before spawn_tasks, K5 resource/environment/receipt behavior, and unchanged K5 node entry. main starts at line 24. |
| `L/tests/test_runtime_check.py` | New | Isolated Git fixtures, committed-source locking/refusal, every hash match/mismatch, missing files, CLI exit codes, peer markers and CPU execution of the worker chain. |
| `L/dry_run/g5h16_tp8_dp2_ep16.txt` | New | Saved 82-case dry run, 83 lines. |
| `L/dry_run/g5h32_tp8_dp4_ep32.txt` | New | Saved 46-case dry run, 47 lines. |

The worker-command helper lives in runtime_check.py and is imported by submit.py. This permits inspection and CPU execution without steptron, as requested. The final mHC gather uses the documented bytes-equivalent contiguous 2-D tensor. DTYPE_BYTES is reused through CollectiveCall.row_bytes rather than duplicated or imported unused.

Pre-existing work remained intact: modified task `progress.md`, untracked `.cortexkit/`, `.pi/`, and `remaining_measurements/`, and the old collective bytecode. No worktree Git mutations, task-record updates, GPU operations, network access, submissions, installs, deletions or moves were performed. Git init/add/commit occurred only in the explicitly permitted temporary test repositories.

**3. Verification**

Tests used `/data/ycfeng/envs/frontier-py310/bin/python`, the supplied Python 3.10 environment without torch or vLLM. Commands ran from the working root unless stated otherwise, with `ulimit -v 31457280`, timeouts, bytecode disabled, and temporary outputs under scratch.

Command aliases used below:

```bash
ROOT=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
TASK=$ROOT/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800
L=$TASK/h3o_operator_costs/g5h/launch
K=$TASK/w5_glm_comm_20261020/launch
S=/data/ycfeng/tmp/codex-implement/c82022e6-3b51-4b19-99f6-92df67bc6810/b9
PY=/data/ycfeng/envs/frontier-py310/bin/python
export PYTHONPATH="$ROOT:$ROOT/tests/unit:$ROOT/tests/comparison/calibration:$ROOT/tests/analysis"
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export TMPDIR="$S" FRONTIER_TMP_ROOT="$S"
ulimit -v 31457280
```

Executed focused suite:

```bash
timeout 120s "$PY" -m pytest \
  tests/unit/test_vllm_forward_collectives.py \
  tests/unit/test_collective_benchmark_helpers.py \
  tests/unit/test_collective_kernel_evidence.py \
  tests/unit/test_vllm_dsv4_forward_collectives.py \
  "$L/tests/test_runtime_check.py" \
  -q -p no:cacheprovider --basetemp="$S/pytest_initial"
```

**PASS:** `135 passed in 2.96s`, exit 0. Collection-only verification confirmed these counts:

| Module | Passing tests |
| --- | ---: |
| test_vllm_forward_collectives.py | 15 |
| test_collective_benchmark_helpers.py | 18 |
| test_collective_kernel_evidence.py | 3 |
| test_vllm_dsv4_forward_collectives.py | 58 |
| launch/tests/test_runtime_check.py | 41 |

Executed both V4 and both GLM dry-run commands using this exact argument pattern, with stdout saved to the listed paths:

```bash
timeout 45s "$PY" -m frontier.profiling.collectives.vllm_collective_benchmark --config "$L/stage_g5h16_tp8_dp2_ep16.json" --list-cases
timeout 45s "$PY" -m frontier.profiling.collectives.vllm_collective_benchmark --config "$L/stage_g5h32_tp8_dp4_ep32.json" --list-cases
timeout 45s "$PY" -m frontier.profiling.collectives.vllm_collective_benchmark --config "$K/stage_k5_tp8_dp2_ep16.json" --list-cases
timeout 45s "$PY" -m frontier.profiling.collectives.vllm_collective_benchmark --config "$K/stage_k5_tp8_dp4_ep32.json" --list-cases
```

**PASS**, all exit 0. Output tails:

```text
G5h-16: cases 82 passes 3 measurements 246
G5h-32: cases 46 passes 3 measurements 138
K5 DP2: cases 324 passes 3 measurements 972
K5 DP4: cases 324 passes 3 measurements 972
```

Executed comparisons:

```bash
timeout 10s cmp "$S/k5_tp8_dp2_ep16.txt" "$K/dry_run/k5_tp8_dp2_ep16.txt"
timeout 10s cmp "$S/k5_tp8_dp4_ep32.txt" "$K/dry_run/k5_tp8_dp4_ep32.txt"
```

**PASS:** both exit 0, no output; both GLM dry runs are byte-identical.

The following rank-0 byte table was extracted from the saved dry runs and matches G5h design §4. Graph points follow T = 1, 2, 4, 8, 16, 24, 32; their shard rows are r = 1, 1, 1, 1, 2, 3, 4. Gather and EP graph columns show unique pairs in point order.

| Call | Graph input → output bytes | Uniform eager T=6144; T=1 |
| --- | --- | --- |
| embedding_all_reduce | 8192; 16384; 32768; 65536; 131072; 196608; 262144 → same values | 50331648→50331648; 8192→8192 |
| sp_attention_all_gather | 8192→65536; 16384→131072; 24576→196608; 32768→262144 | 6291456→50331648; 8192→65536 |
| sp_attention_reduce_scatter | 8192→8192; 16384→8192; 32768→8192; 65536→8192; 131072→16384; 196608→24576; 262144→32768 | 50331648→6291456; 8192→8192 |
| sp_final_mhc_all_gather | 32768→262144; 65536→524288; 98304→786432; 131072→1048576 | 25165824→201326592; 32768→262144 |
| EP16 dispatch | 8240→131840; 16480→263680; 24720→395520; 32960→527360 | 6328320→101253120; 8240→131840 |
| EP16 combine | 131072→8192; 262144→16384; 393216→24576; 524288→32768 | 100663296→6291456; 131072→8192 |
| EP32 dispatch | 8240→263680; 16480→527360; 24720→791040; 32960→1054720 | 6328320→202506240; 8240→263680 |
| EP32 combine | 262144→8192; 524288→16384; 786432→24576; 1048576→32768 | 201326592→6291456; 262144→8192 |
| logits_all_gather | No graph measurements | 32320×b→258560×b, b=1..24; endpoints 32320→258560 and 775680→6205440 |

SP reduce-scatter padded input is output×8; T=1 therefore has an 8192-byte measurement key and 65536 padded bytes.

For EP16 mixed vectors, rank 0 has 6328320→50692480 dispatch bytes on `eager_t6144_1`, and 8240→50692480 on `eager_t1_6144`. Combine inputs are 50397184 in both; rank-0 outputs are respectively 6291456 and 8192. Tests also verify rank 8's opposite lane.

All 16 ordered EP32 eager vectors are retained. Their shared gathered/combined sizes, grouped by the number q of 6144-token lanes, are:

| q | Dispatch output bytes | Combine input bytes |
| ---: | ---: | ---: |
| 0 | 263680 | 262144 |
| 1 | 50824320 | 50528256 |
| 2 | 101384960 | 100794368 |
| 3 | 151945600 | 151060480 |
| 4 | 202506240 | 201326592 |

Rank-0 dispatch input is 6328320 for its busy lane or 8240 for its one-token lane; combine output is respectively 6291456 or 8192.

**Cold imports — PASS:** each module was imported in a separate process using `timeout 20s "$PY" -c ...`, with an assertion that no torch, vllm or cuda package appeared in sys.modules:

- frontier.profiling.collectives.vllm_collective_benchmark
- frontier.profiling.collectives.vllm_dsv4_forward_collectives
- frontier.profiling.collectives.nixl_kv_read_benchmark
- frontier.profiling.collectives.ep_kv_read_overlap_benchmark

All four exited 0.

**Lock checks — PASS:** calling build_lock on the real G5h-16 config produced the required refusal:

```text
Uncommitted benchmark sources; commit before submission:
 M frontier/profiling/collectives/vllm_collective_benchmark.py
 M frontier/profiling/collectives/vllm_forward_collectives.py
?? frontier/profiling/collectives/vllm_dsv4_forward_collectives.py
```

The temporary-repository tests checked all six benchmark/config paths, all five native paths and overlay-derived NCCL hashes. An additional committed source snapshot in `$S/lock_snapshot` used the actual B9 source files and real P2 overlay/native inputs. Its lock was saved to `$S/lock_snapshot.json`, containing the temporary Git HEAD and all 12 entries. The NCCL entry is exactly:

```text
/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
618968ee277041751512e619190132a9bdf351c3ce7129c73484f6ef441ad692
```

The snapshot is CPU evidence, not a real submission lock.

**Launcher syntax — PASS:** executed from L:

```bash
timeout 20s python3 -c "import ast; ast.parse(open('submit.py').read()); print('submit.py AST PASS')"
```

Exit 0. All nine changed Python files also passed ast.parse with Python 3.10 feature_version.

The dependency-free helper generated this G5h-16 worker command. It was inspected; submit.py was not run:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTHONPATH=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028${PYTHONPATH:+:$PYTHONPATH}; timeout 2700 bash -c 'python3 /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/runtime_check.py --config /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/stage_g5h16_tp8_dp2_ep16.json --lock /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/receipts/dsv4-comm-g5h16-01_lock.json && python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/w5_glm_comm_20261020/launch/node_entry.py --config /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/stage_g5h16_tp8_dp2_ep16.json'
```

The launcher tests executed equivalent worker chains against temporary CPU scripts and confirmed that a hard failure prevents node entry.

Executed kernel-gap regression:

```bash
timeout 120s "$PY" -m pytest tests/unit/test_kernel_gap_pricing.py \
  -q -p no:cacheprovider --basetemp="$S/pytest_gap"
```

**EXPECTED EXISTING FAIL:** `1 failed, 20 passed in 1.82s`, exit 1. The sole failure was `test_every_collective_launches_one_kernel_by_default`, with `AttributeError: 'OperatorSpec' object has no attribute 'kernel_count'`. No new failures appeared.

**Hygiene:** `/data/ycfeng/tmp/unused_imports.py` reported `[]` for every changed Python file; git diff --check exited 0. Line counts:

| Python file | Lines |
| --- | ---: |
| vllm_forward_collectives.py | 293 |
| vllm_dsv4_forward_collectives.py | 145 |
| vllm_collective_benchmark.py | 555 |
| test_vllm_forward_collectives.py | 143 |
| test_vllm_dsv4_forward_collectives.py | 203 |
| test_collective_benchmark_helpers.py | 309 |
| L/runtime_check.py | 130 |
| L/submit.py | 66 |
| L/tests/test_runtime_check.py | 222 |

**FAIL, preserved external state:**

```bash
timeout 20s /usr/bin/grep -rn "_ceil_div" frontier tests
```

Exited 0 because it reported only:

```text
frontier/profiling/collectives/__pycache__/vllm_forward_collectives.cpython-310.pyc: binary file matches
```

Stat showed that file was created and last modified on October 21, 2026 at 03:28:09 +0800, before B9. It was not modified. The additional source-only command `timeout 20s /usr/bin/grep -rn --include=*.py _ceil_div frontier tests` exited 1 with no matches, as expected.

**NOT_RUN:** submit.py execution, steptron resource construction, image/runtime checks on H200, torchrun, GPU measurements and B10 comparisons.

**4. Remaining work and decisions for Claude**

- Resolve the literal grep hygiene exception. The evidence is a stale pre-existing .pyc file, with no Python-source references remaining. Recommended action: Claude handles bytecode cleanup under its authorization, then reruns the exact grep; alternatively explicitly accept the source-only hygiene evidence.
- Review and commit B9 before building real locks. The repository ignores new task_memory files via `.gitignore:171:task_memory/*`; include all seven new launcher/config/test/dry-run files explicitly in the commit.
- After committing, run build_lock on both real stage configs and verify the expected NCCL hash above. The real dirty-tree refusal is intentional and required.
- Run G5h-16, inspect node runtime checks and custom_collectives metadata in plan.json, then follow the agreed B10/review/G5h-32 sequence. CPU tests establish routing and shapes; they do not establish actual MNNVL kernel selection or GPU timings.

No native-source contradiction or material change to the agreed behavior was found. B10 and simulator entry 8 remain outside this assignment.

**5. Review handoff**

Inspect the narrow GLM module diff, the V4 call ordering and mixed-vector filter, lazy SP issue functions, and the runtime-check success/failure directory behavior. Check both stage configs against P2 and G5h §4-5, and verify that submit.py leaves ResourceConfig custom_resources at its default and uses the unchanged K5 entry.

Rerun the 135-test focused command, both GLM cmp checks, cold imports, and exact grep after resolving stale bytecode. The kernel-gap suite should continue to have only the declared existing failure.

Scratch retained at `/data/ycfeng/tmp/codex-implement/c82022e6-3b51-4b19-99f6-92df67bc6810/b9/`: GLM comparison outputs, pytest temporary repositories/artifacts, collection_counts.txt, lock_snapshot/ and lock_snapshot.json. The V4 dry runs are in the assigned launcher directory. No real receipt, submission lock or GPU output was created.
