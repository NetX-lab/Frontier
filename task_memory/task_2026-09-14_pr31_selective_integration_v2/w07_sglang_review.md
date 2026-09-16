## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Fixed routed sorting block ownership and routed replay correctness/reset/trace handling using existing orchestration; recorded CPU RED/GREEN evidence and native limits. |

# W07 SGLang Routed Profiling Repair

## Scope and ownership

- Reviewer/implementer: `/root/w00_review_inventory`.
- Specification: W07 in `Frontier_PR33_New_Execution_Plan_2026-09-16_EN.md`.
- Owned production files: `frontier/profiling/experimental/sglang/moe.py`, `routed_moe_replay.py`, and `graph_replay.py`.
- The parent explicitly approved extending ownership to `graph_replay.py` to reuse the existing normalization and replay orchestration. Ordinary `profile_graph` callers retain their return contract.
- Tests: existing `tests/unit/test_sglang_graph_replay_orchestration.py` and new `tests/unit/test_sglang_routed_sorting.py`.
- No timer, GDN, standard MoE wrapper, shared registry, or unrelated source file was edited. No commit or remote action was performed by this subtask.
- Applied skill: `diagnosing-bugs`, using deterministic CPU regressions through the real defective call paths. The already source-confirmed arithmetic and orchestration errors did not require a broad hypothesis search.

## Delivered behavior

### Multi-block expert sorting ownership

`make_moe_sorting_primitive` previously expected one sorted block owner per active expert. This disagreed with its own workload block count whenever an expert received more than 32 tokens. It also assigned rather than accumulated the number of observed routes for each expert.

The expected owner list now repeats each expert once per required block, validates all those blocks, and accumulates routes across blocks. No routing distribution, assignment reconstruction, AITER kernel call, quantized shape, or measured interval changed.

### Shared routed replay and real correctness/trace callbacks

`profile_routed_graph` previously owned a separate capture/replay loop. It never compared its returned tensors against the builder reference, did not reset mutable buffers, and emitted correctness and representative-trace flags without implementing their checks/probe.

The routed entrypoint now delegates to `profile_graph`. `make_primitive` uses the existing `MOE_ROUTED_PRIMITIVES` declaration to normalize the two routed builders through `_make_replay_call`. The established implementation now owns eager checks, warmup, reset before capture/replay, synchronized timing, replay checks, and the representative trace callback. Routed histogram and shape provenance remain attached to the row. `HIP_GRAPH_REPLAY` and experimental artifact boundaries remain unchanged.

The experimental routed API now returns `(row, trace_replay)`, matching the existing common replay API, instead of returning only a row. This permits the caller to execute the requested trace probe under its profiling context. Repository caller search found only invalid-input tests before this change, with no valid in-repository consumer requiring migration. Existing ordinary primitive callers keep their original tuple contract. External research callers of the routed API must unpack the tuple.

The shared comparison now uses exact equality for integral metadata while preserving the existing `atol=rtol=0.03` tolerance for floating/complex outputs. This is required for newly routed sorting metadata: the previous generic BF16-oriented tolerance incorrectly accepted a sorted-token count of 97 against an expected 96.

## Validation report

### Execution

- Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.
- Interpreter: `/usr/bin/python`, Python 3.12.3; no active conda environment.
- pytest: 9.1.1; torch CPU tensors are used, with graph/event/AITER boundaries explicitly simulated.
- All raw logs are under `/data/ycfeng/tmp/pr33-w05-preflight-regression/`; no raw outputs or caches were placed in versioned source.

Initial sorting reproduction:

```bash
PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp /usr/bin/python -m pytest tests/unit/test_sglang_routed_sorting.py -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w05-preflight-regression/w07-sorting-red.log 2>&1
```

Observed **2 failed, 3 passed in 2.54s**. The counts `(33, 31)` and `(64, 33, 0)` failed in the actual production sorting oracle; the latter error was:

```text
AssertionError: The values for attribute 'shape' do not match: torch.Size([4]) != torch.Size([2]).
frontier/profiling/experimental/sglang/moe.py:204
```

Initial routed replay reproduction:

```bash
PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp /usr/bin/python -m pytest tests/unit/test_sglang_graph_replay_orchestration.py -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w05-preflight-regression/w07-replay-red.log 2>&1
```

Observed **4 failed, 8 passed in 2.41s**. Routed sorting and experts lacked the common trace callback return; injected eager/replay corruption also reached that unchecked return instead of failing the numerical comparison. The observed exception was `ValueError: too many values to unpack (expected 2)`. The absence of comparisons itself was source-confirmed; this reproduction should not be described as a native numerical failure.

Integral metadata reproduction:

```bash
PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp /usr/bin/python -m pytest tests/unit/test_sglang_graph_replay_orchestration.py::test_replay_requires_exact_integer_metadata -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w05-preflight-regression/w07-integer-red.log 2>&1
```

Observed **1 failed in 2.38s**, with `Failed: DID NOT RAISE AssertionError` for actual integer metadata `[97, 64]` versus expected `[96, 64]`.

Final integrated targeted command:

```bash
PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp /usr/bin/python -m pytest tests/unit/test_sglang_routed_sorting.py tests/unit/test_sglang_graph_replay_orchestration.py tests/unit/test_sglang_experimental_increment13.py -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w05-preflight-regression/w07-final.log 2>&1
```

Observed **25 passed in 3.06s**. `git diff --check` passed for the owned changed tracked source/test files.

### Acceptance evidence

1. Sorting checks loads below, exactly at, and above the 32-token block boundary, including multiple active multi-block experts and an inactive expert. The fake AITER boundary independently builds packed block owners and padding from the actual top-k assignment tensor; the production validation logic is executed unchanged except for the repair.
2. The existing CPU graph harness now also exercises routed sorting and experts. Stateful tensors increment on every invocation, so the expected result is achieved only if capture/replay and repeated trace invocations reset their buffers correctly.
3. Separate eager and replay-only corruption cases must raise AssertionError before returning a success row. Both pass after migration to common correctness checks.
4. Representative trace callbacks execute twice and preserve reset/correctness checks. Rows retain `measurement_type=HIP_GRAPH_REPLAY`, explicit histogram provenance, two trace-probe invocations, and the common callback contract.
5. Existing ordinary primitive, invalid-count, shape/provenance and trace-import tests remain green.

### Native evidence limits

**Native AMD/MI355X/AITER/HIP graph execution: NOT RUN in this subtask.** The retained task hardware boundary remains unavailable AMD hardware; no GPU inspection, worker command, package installation, or fabricated native pass was attempted. CPU simulated graphs establish orchestration and validation correctness, not kernel numerical correctness, timing fidelity, or actual native trace coverage. Native T12/T14 lanes remain separate acceptance work.

## Other W07 findings and handoff

- Standard MoE DEVICE_EVENT admission and backend-label truth were reported to the parent and remain in its ownership.
- Existing accelerator visibility, collective dtype/rank evidence, and native compatibility checks remain covered by the W00 profiling ledger, not implicitly closed by these 25 tests.
- Serialized `graph_replay` plan validation/rank boundary gaps, routed packed-shape divisibility checks, and top-k native-order versus reference-order concerns from P08 remain separate bounded follow-up findings. They were not silently repaired or claimed complete here.
- No additional source changes are pending for the scoped sorting/replay fixes. Next: parent review and commit the verified sub-step, record this evidence in the primary W07 test report, and preserve separate native hardware dispositions.
