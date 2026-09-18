## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Started the authorized J1/J2 orchestration inspection; production edits are held pending main's P2 E2E source-freeze release. |
| 2026-09-17 | Main released P2 freeze. J1 completed: 5 before / 5 after tests; byte-identical metadata/output/timer snapshots; plan calls reduced from 5 to 3 over prefill, decode and empty begin. J2 remains pending main's J1 commit. |
| 2026-09-17 | Main committed J1 as 69eb7be8 and released J2. J2 completed: 39 before / 40 after focused tests; 12 complete replay snapshots containing 1,802 ordering events compare byte-identically. |

# P3 J1/J2 orchestration cleanup

## Scope and execution gates

Owner: scheduler/profiling sidecar. Authorized production paths:

- `frontier/profiling/attention/backends/vllm_rocm_attention_wrapper.py`
- `frontier/profiling/experimental/sglang/graph_replay.py`

Related dedicated tests are in scope; shared configuration/registry tests, shared progress records and commits are not. No standard profiling dependency on experimental code may be introduced. Required lifecycle `None` states remain supported.

Dependency: read-only inspection/baseline -> main releases P2 E2E source freeze -> J1 patch and focused/output/lifecycle checks -> main commits J1 -> J2 patch and focused/output/reset-order checks -> handoff. J1 was committed by main as `69eb7be8`. J2 is complete and ready for main's separate commit; this sidecar has not committed any changes.

Initial inspected HEAD: `4f3fc597a9dc0c426be0e5486bc2e921d66ad2e1`. Existing dirty `frontier/metrics/ep_wave_metrics.py`, `frontier/metrics/op_trace_utils.py`, and untracked `tests/unit/test_ep_wave_trace_context.py` belong to another worker and remain untouched.

## Inspected contracts and proposed minimal edits

### J1: reuse the admitted ROCm sequence plan

`begin_forward` builds the ordered prefill-plus-decode plan, then rejects a mixed batch. Therefore every admitted nonempty batch's active phase list is identical to the ordered list. Pass the already-built `plan` to `_materialize_metadata` instead of building the same plan again in either phase branch. Keep the partition, rejection order, token count, slot tensor, dual metadata fields and `end_forward` clearing unchanged.

Do not remove inactive timer scopes: `forward` enters both `ATTN_PREFILL` and `ATTN_DECODE` even when one metadata object is absent. Empty input currently materializes neither phase and retains an empty slot map; do not change that behavior incidentally. Before-begin/after-end slot-map `None` is a real invalid-forward lifecycle state.

Existing test: `tests/unit/test_vllm_rocm_attention_wrapper_increment9.py` verifies mixed sequence-plan slots, short-block-table rejection and lazy backend registration, but does not exercise wrapper begin/forward/end. Add a minimal CPU native-adapter/timer stand-in exercising actual wrapper initialization and repeated phase lifecycle. Compare materialized metadata, slot maps, outputs, timer entry/exit order and clearing before/after the refactor. CPU stand-ins establish orchestration only, not native ROCm math or timing.

### J2: explicit replay interface and existing builder metadata

All inspected `profile_graph` production and test callers supply the same seven positional arguments. Workload options are `logical_size`, `physical_context_lens`, `physical_expert_counts`, and `trace`; replace `args[:7]` / arbitrary forwarding with explicit parameters.

`_make_replay_call` owns independent mutable snapshots, reset and correctness checks. Those closures and their call ordering must remain unchanged. The current seven-position result carries GDN/MoE routing metadata under `attention_spec`, while the graph re-derives GDN metadata and reads routed metadata by position 6. A named internal result can clarify those existing outputs without changing native builder signatures or emitted row fields.

Builder evidence: `make_gdn_core_primitive` already returns its validated spec; attention and routed-MoE builders already return spec/workload; `make_moe_routing_primitive` returns a spec; `make_dense_primitive` returns only callable/reference/backend. Thus GDN metadata can be reused directly, but dense metadata cannot be recovered from the existing builder return contract. Preserve its existing derivation unless separately authorized; do not invent a standard-facing abstraction or expand primitive output schema.

Read-only caller inventory: `frontier/profiling/experimental/sglang/routed_moe_replay.py:profile_routed_graph`, `tests/unit/test_sglang_graph_replay_orchestration.py`, and `tests/integration/test_pr33_native_profiling_acceptance.py`. Routed replay already delegates to the common reset/check owner.

## Verification status

- Initial focused baseline: **35 PASS in 4.87 s**, with `/data/ycfeng/tmp/quality-review-env/bin/python` (Python 3.12.3), no conda activation, CPU-only stand-ins.
- Log: `/data/ycfeng/tmp/quality-p3-orchestration-before-20260917.log`.
- Expected acceptance: all existing checks remain green, output/schema/reset/timer/lifecycle comparisons remain equal, and no native-GPU claim is made from CPU tests.
- J1 implementation: complete, verified and committed by main as `69eb7be8`.
- J2 implementation: complete and verified; awaiting main commit.

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.

Exact baseline command:

```bash
set -o pipefail
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. \
  TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true \
  VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/ycfeng/tmp/quality-review-env/bin/python -m pytest \
  tests/unit/test_vllm_rocm_attention_wrapper_increment9.py \
  tests/unit/test_sglang_graph_replay_orchestration.py \
  tests/unit/test_sglang_replay_admission.py \
  tests/unit/test_sglang_experimental_increment13.py \
  -q -p no:cacheprovider --tb=short \
  --basetemp=/data/ycfeng/tmp/quality-p3-orchestration-before-20260917 \
  2>&1 | tee /data/ycfeng/tmp/quality-p3-orchestration-before-20260917.log
```

## J1 verified handoff

Main explicitly released the P2 freeze after reporting 159 integrated tests (including eight non-dummy cases), 90 stable comparisons, 106 symmetric files and 16 summary comparisons. Those are main-reported P2 results, not J1 verification.

J1 starting HEAD was `8e67fc25967cf5aaa01a87298a40b9b08ee13fc4`. At final verification HEAD was `3d29f6e46b4683dc8b40256a804bc494fd3af294`, reflecting a concurrent main commit; this sidecar's two paths remained uncommitted throughout the before/after checks. No shared progress or registry/configuration tests were edited.

| Path | Change | Invariant and observed verification |
| --- | --- | --- |
| `frontier/profiling/attention/backends/vllm_rocm_attention_wrapper.py` | Pass the existing `plan` into the two active-phase metadata branches. Production delta: two added / six deleted lines. | Every admitted nonempty batch has exactly one phase, so the active list equals the ordered plan input. Partition, mixed rejection order, slot mapping, token count, both timer scopes and begin/end sentinels are unchanged. |
| `tests/unit/test_vllm_rocm_attention_wrapper_increment9.py` | Add a real-initialization CPU native/timer stand-in, sequential prefill/decode lifecycle regression and mixed-rejection regression. | Full tensor output equality, metadata fields, KV-update mapping identity, active backend metadata identity, inactive timer entry/exit, after-end rejection and empty-batch metadata behavior pass before and after the production edit. |

The fixture uses `VllmRocmAttentionWrapper.init`, a real `ModelConfig` and `ParallelConfig`; it does not use `__new__` or manually populate required production attributes. It replaces only unavailable native vLLM/backend timing dependencies. The codebase-design skill's interface-first testing guidance informed this approach. The stand-in writes `query + 1`, so these assertions establish orchestration and tensor slicing/output propagation, not native attention numerical accuracy.

### Focused tests

- Before production change, including the two new regressions: **5 PASS in 2.88 s**, `/data/ycfeng/tmp/quality-p3-j1-before-20260917.log`.
- After production change, identical tests: **5 PASS in 2.80 s**, `/data/ycfeng/tmp/quality-p3-j1-after-20260917.log`.
- `git diff --check` on both owned code/test paths: **PASS**.

Exact command, run separately with `frontier_j1_phase=before` and `frontier_j1_phase=after` on the corresponding production sources:

```bash
frontier_j1_phase=after
set -o pipefail
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. \
  TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true \
  VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/ycfeng/tmp/quality-review-env/bin/python -m pytest \
  tests/unit/test_vllm_rocm_attention_wrapper_increment9.py \
  -q -p no:cacheprovider --tb=short \
  --basetemp=/data/ycfeng/tmp/quality-p3-j1-${frontier_j1_phase}-20260917 \
  2>&1 | tee /data/ycfeng/tmp/quality-p3-j1-${frontier_j1_phase}-20260917.log
```

### Output, ordering and unnecessary-work comparison

The same test helper was executed before/after, with a spy wrapping the real sequence-plan builder. Snapshots contain every materialized metadata field, the slot map, phase flags, all timer events, output shape and output sum. The helper also asserts exact full output tensor equality and lifecycle errors on every run.

| Observation | Before | After |
| --- | --- | --- |
| Prefill slots | `[64,65,66,67,96,97,98]` | identical |
| Prefill query starts / sequence lengths | `[0,4,7]` / `[20,3]` | identical |
| Prefill output shape / sum | `[7,4096]` / `28672.0` | identical |
| Decode slots | `[145,32]` | identical |
| Decode query starts / sequence lengths | `[0,1,2]` / `[18,1]` | identical |
| Decode output shape / sum | `[2,4096]` / `8192.0` | identical |
| Timer order, both phases | reshape -> KV save -> prefill -> decode -> output reshape; each enters and exits at layer 7 | identical, including inactive scope |
| End-forward state | both metadata fields and slot map become `None`; forward raises until next begin | identical |
| Empty begin | no active phase/metadata, empty slot map | identical |
| Plan builder calls across prefill + decode + empty begin | **5** | **3** |

Byte comparison: **PASS**, `cmp` produced no output. No numerical differences were observed. Builder calls demonstrate removal of duplicate host work only; no native GPU timing improvement is claimed.

Artifacts:

- `/data/ycfeng/tmp/quality-p3-j1-before-snapshot-20260917.json`
- `/data/ycfeng/tmp/quality-p3-j1-after-snapshot-20260917.json`
- `/data/ycfeng/tmp/quality-p3-j1-before-snapshot-20260917.log` (`sequence_plan_calls=5`)
- `/data/ycfeng/tmp/quality-p3-j1-after-snapshot-20260917.log` (`sequence_plan_calls=3`)

Exact snapshot command, run separately before and after the production edit:

```bash
frontier_j1_phase=after
set -o pipefail
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/ycfeng/tmp/quality-review-env/bin/python - <<'PY' \
  2>/data/ycfeng/tmp/quality-p3-j1-${frontier_j1_phase}-snapshot-20260917.log \
  | tee /data/ycfeng/tmp/quality-p3-j1-${frontier_j1_phase}-snapshot-20260917.json
from contextlib import redirect_stdout
import json
import sys
from unittest.mock import Mock
import pytest
with redirect_stdout(sys.stderr):
    from frontier.profiling.attention.backends import vllm_rocm_attention_wrapper as rocm
    from tests.unit.test_vllm_rocm_attention_wrapper_increment9 import _exercise_rocm_lifecycle
    with pytest.MonkeyPatch.context() as patch:
        builder = Mock(wraps=rocm.build_rocm_sequence_plan)
        patch.setattr(rocm, "build_rocm_sequence_plan", builder)
        snapshots = _exercise_rocm_lifecycle(patch)
        print(f"sequence_plan_calls={builder.call_count}")
print(json.dumps(snapshots, sort_keys=True, indent=2))
PY
cmp /data/ycfeng/tmp/quality-p3-j1-before-snapshot-20260917.json \
    /data/ycfeng/tmp/quality-p3-j1-after-snapshot-20260917.json
```

### Retained behavior, harness issues and remaining gate

- Keep both metadata fields and callable lifecycle `None` checks. This change does not redesign phase state or remove inactive timer scopes.
- Keep standard profiling independent of experimental graph replay; no dependency/import changes were made.
- Discovery misses: guessed `tests/unit/test_attention_chunked_prefill_profiling.py` and `data/model_configs` do not exist. The existing wrapper and `frontier/config/model_config.py` supplied the needed definitions.
- Initial snapshot output included timestamped pre-existing architecture warnings on stdout. That diagnostic-contaminated attempt is preserved as `/data/ycfeng/tmp/quality-p3-j1-before-values-20260917.json` and is not used for equality evidence. The corrected snapshot command redirects diagnostics during imports/construction to stderr; production logging is unchanged.
- No failed focused tests or new semantic discrepancy. J1 is ready for main's separate commit. J2 implementation remains gated on that commit signal; no commit was made by this sidecar.

## J2 verified handoff

Main reviewed/committed J1 as `69eb7be8883ff8f400035d1cd050f9e9728fd337` and explicitly released J2. Main subsequently reserved `routed_moe_replay.py`, `moe_ep_workload.py`, and `tests/unit/test_sglang_experimental_increment13.py` for S7. This sidecar read/ran related tests but did not edit those files, shared config/registry tests, or shared progress records.

J2 started at HEAD `69eb7be8883ff8f400035d1cd050f9e9728fd337`; observed HEAD after verification was `dcf66a4cc73074a30f4c0146aa6380ab10bf877c`, reflecting concurrent main/other-worker commits. Both J2 paths were clean before this sub-step and remained uncommitted during verification. This is shared-worktree focused evidence, not an isolated whole-tree or native-hardware campaign.

### Exact implementation scope and preserved contracts

| Path | Root cause / cleanup | Preserved behavior and verification |
| --- | --- | --- |
| `frontier/profiling/experimental/sglang/graph_replay.py` | `args[:7]` and arbitrary keyword forwarding concealed the supported interface. Define seven explicit parameters plus the existing keyword-only workload and trace options. | Every inspected production/native-test caller retains its existing call form. Invalid replay counts still fail before importing the GPU runtime. A new test verifies the explicit named-argument form. |
| Same file: `_ReplayCall` / `_make_replay_call` | Seven positional fields obscured callable/reset/check ownership and put GDN/MoE-routing specs in an attention slot. Replace the internal tuple with a frozen named result containing `fn`, `reset`, `check`, `backend`, and required `row_metadata`. | Native builder return tuples are unchanged. Independent reference tensors, cloned mutable-state baselines, numerical tolerances, finite checks and reset/check closures are preserved. Required mutable collections remain iterable; no `None` substitute is needed for the inspected builders. |
| Same file: primitive-to-row metadata | GDN metadata was already constructed by its builder, then derived again after replay; routed metadata was read by index 6. Pass the existing published specs through named row metadata. Do not retain unused attention/MoE-routing metadata in the internal replay object. | Published GDN and routed-MoE field names/values are unchanged. Dense retains its old spec derivation because its native builder returns only callable/reference/backend. No new standard-profiling dependency or row field is introduced. |
| `tests/unit/test_sglang_graph_replay_orchestration.py` | Existing tests checked aggregate completion but not exact reset/capture/replay order. Extend the existing CPU harness rather than add another execution harness. | Six builder paths now cover trace on/off; record reset/call/check/capture/replay/barrier/event ordering and final buffers. Keep eager/replay fault rejection and exact integer comparison. Add nonempty GDN metadata and explicit named-call coverage. |

The codebase-design skill informed the named internal result and testing through the live orchestration interface. The existing native builders remain the implementation seam; no replacement registry, compatibility path, runtime fallback or extra wrapper was added.

Deliberately retained:

- `probe = None` and its error on attempting trace replay without a trace graph are real lifecycle behavior; the returned callback is still present in both modes.
- `logical_size`, context lengths and expert counts are genuinely primitive-dependent workload options. Existing builder admission checks remain authoritative.
- Mutable snapshots and reset/check closures are necessary for recurrent/cache/output buffers; no snapshot, reset, barrier, warmup, correctness check or event timing scope is dropped.
- Primitive classification continues to use the existing `DENSE_PRIMITIVES`, `GDN_PRIMITIVES`, `ATTENTION_PRIMITIVES` and MoE primitive sets.
- No edits to native builder modules, `routed_moe_replay.py`, shared configuration/registry modules or standard profiling consumers.

### Focused verification

Interpreter/environment is the dedicated `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, no conda activation; all runs are CPU-only with existing simulated graph/native-callable stand-ins.

| Source / test state | Observed result | Log |
| --- | --- | --- |
| Original J2 source and original tests | **32 PASS in 4.86 s** | `/data/ycfeng/tmp/quality-p3-j2-original-20260917.log` |
| Original J2 source; added trace-off and metadata/order coverage | **39 PASS in 4.90 s** | `/data/ycfeng/tmp/quality-p3-j2-before-20260917.log` |
| Refactored J2 source; same coverage plus one explicit named-argument test | **40 PASS in 4.85 s** | `/data/ycfeng/tmp/quality-p3-j2-after-20260917.log` |

No focused test failures occurred. The direct private-helper test was migrated from `call[2]` to `call.check`, supplying explicit empty row metadata; its integer mismatch assertion is unchanged. The GDN fixture's old patch of the redundant spec resolver was removed after production stopped re-deriving that spec; builder-provided metadata assertions remain unchanged. `git diff --check` on the two owned paths passed.

Exact focused command, run separately with `frontier_j2_phase=original`, `before`, and `after` on the corresponding states:

```bash
frontier_j2_phase=after
set -o pipefail
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. \
  TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true \
  VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/ycfeng/tmp/quality-review-env/bin/python -m pytest \
  tests/unit/test_sglang_graph_replay_orchestration.py \
  tests/unit/test_sglang_replay_admission.py \
  tests/unit/test_sglang_experimental_increment13.py \
  -q -p no:cacheprovider --tb=short \
  --basetemp=/data/ycfeng/tmp/quality-p3-j2-${frontier_j2_phase}-20260917 \
  2>&1 | tee /data/ycfeng/tmp/quality-p3-j2-${frontier_j2_phase}-20260917.log
```

### Exact replay/output/reset-order comparison

Compared six existing builder paths with trace both disabled and enabled: dense activation, GDN core, attention RoPE, MoE routing, routed sorting and routed experts. Nonempty GDN shape metadata is included. **All 12 complete JSON snapshots are byte-identical**, including every row field, output buffer and all **1,802 ordered events**. `cmp` exited 0.

Each case keeps five retained samples of `1.0 ms` from the simulated event clock (three primitive calls per graph and `3.0 ms` simulated graph duration). Native-call counts are **38 without trace / 44 with trace** in both versions. Stateful calls have the same number of resets and begin every invocation at `[0.0]`; all final buffers are `[1.0]`. Without trace the callback still raises `RuntimeError`; with trace the first/last representative pair is replayed twice. These fixed clock values verify arithmetic/sequence preservation, not real device speed.

Artifacts:

- `/data/ycfeng/tmp/quality-p3-j2-before-snapshot-20260917.json`
- `/data/ycfeng/tmp/quality-p3-j2-after-snapshot-20260917.json`
- `/data/ycfeng/tmp/quality-p3-j2-before-snapshot-20260917.log`
- `/data/ycfeng/tmp/quality-p3-j2-after-snapshot-20260917.log`

Both snapshot logs report `snapshots=12 events=1802`. Exact command, run separately before and after the production edit:

```bash
frontier_j2_phase=after
set -o pipefail
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/ycfeng/tmp/quality-review-env/bin/python - <<'PY' \
  2>/data/ycfeng/tmp/quality-p3-j2-${frontier_j2_phase}-snapshot-20260917.log \
  | tee /data/ycfeng/tmp/quality-p3-j2-${frontier_j2_phase}-snapshot-20260917.json >/dev/null
from contextlib import redirect_stdout
import json
import sys
import pytest
with redirect_stdout(sys.stderr):
    from tests.unit.test_sglang_graph_replay_orchestration import _exercise_replay
    cases = [
        ('shared_expert_activation', 'make_dense_primitive', ()),
        ('gdn_core_decode', 'make_gdn_core_primitive', ({'conv_state_shape': (4, 3), 'recurrent_state_shape': (2, 2, 2)},)),
        ('attn_rope', 'make_attention_primitive', ({}, {})),
        ('moe_routing_topk', 'make_moe_routing_primitive', ({},)),
        ('moe_sorting', 'make_moe_sorting_primitive', ({}, {})),
        ('moe_experts_quant_gemm_combine', 'make_moe_experts_primitive', ({}, {})),
    ]
    snapshots = {}
    for name, builder, metadata in cases:
        for trace in (False, True):
            with pytest.MonkeyPatch.context() as patch:
                snapshots[f'{name}:trace={trace}'] = _exercise_replay(
                    patch, name, builder, metadata, trace=trace,
                )
    print(f'snapshots={len(snapshots)} events={sum(len(value["events"]) for value in snapshots.values())}')
print(json.dumps(snapshots, sort_keys=True, indent=2))
PY
cmp /data/ycfeng/tmp/quality-p3-j2-before-snapshot-20260917.json \
    /data/ycfeng/tmp/quality-p3-j2-after-snapshot-20260917.json
```

### Handoff limits

J2 implementation and requested bounded verification are complete; no unresolved semantic discrepancy was found. No commit was made. Main owns integration, S7 and any broader/native regression campaign. The public experimental replay row schema and native builder contracts were preserved; this report does not claim actual SGLang/AITER/ROCm execution or performance measurements.
