## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Started the authorized J1/J2 orchestration inspection; production edits are held pending main's P2 E2E source-freeze release. |
| 2026-09-17 | Main released P2 freeze. J1 completed: 5 before / 5 after tests; byte-identical metadata/output/timer snapshots; plan calls reduced from 5 to 3 over prefill, decode and empty begin. J2 remains pending main's J1 commit. |

# P3 J1/J2 orchestration cleanup

## Scope and execution gates

Owner: scheduler/profiling sidecar. Authorized production paths:

- `frontier/profiling/attention/backends/vllm_rocm_attention_wrapper.py`
- `frontier/profiling/experimental/sglang/graph_replay.py`

Related dedicated tests are in scope; shared configuration/registry tests, shared progress records and commits are not. No standard profiling dependency on experimental code may be introduced. Required lifecycle `None` states remain supported.

Dependency: read-only inspection/baseline -> main releases P2 E2E source freeze -> J1 patch and focused/output/lifecycle checks -> main commits J1 -> J2 patch and focused/output/reset-order checks -> handoff. J1 is complete and ready to commit; J2 production and tests remain unmodified by this sidecar.

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
- J1 implementation: complete and verified; awaiting main commit.
- J2 implementation: pending verified J1 handoff and main's commit signal.

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
