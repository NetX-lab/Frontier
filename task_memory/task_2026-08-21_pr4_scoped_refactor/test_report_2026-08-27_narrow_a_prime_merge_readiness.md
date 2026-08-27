## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-27 | Re-ran the complete focused and PR17-sensitive matrices after commit `0fb2781e`; recorded exact counts and the precise conflict-marker gate. |
| 2026-08-27 | Added SCOPE-042 dispatch/combine payload-order RED/GREEN evidence and typed fixture migration results. |
| 2026-08-27 | Added fresh 309-test focused evidence, 169-test PR17-sensitive evidence, interface static audit, and local PR20/PR21 merge-tree facts. |
| 2026-08-27 | Added fresh SCOPE-040b role-capability and SCOPE-041 topology mismatch probes, exact lookup counts, and focused GREEN results. |
| 2026-08-27 | Added SCOPE-039 dummy attention-only RED/GREEN evidence and the complete affected predictor matrix. |
| 2026-08-27 | Recorded the fresh narrow A' focused matrices and the source-effective versus lane-local collective payload direct case. |

## 1. Test Script Information

### Persistent regression scripts

- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_moe_ep_aggregate_admission.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_mixed_layer_decode_ffn_scheduling.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_moe_predictor_layer_id_semantics.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_mtp_terminal_overshoot_ep_replay.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_spec_decode_mtp_structural_moe_replay.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_typed_ep_predictor_contract.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_typed_ep_scheduler_consumers.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_moe_ep_workload_materializer.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_moe_routing_conservation.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_predictor_effective_tokens.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_mtp_token_ledger_repair.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_comm_operator_families.py`
- `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824/tests/unit/test_sklearn_disaggregation_execution_time_predictor.py`

### Exact focused commands

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824
PYTHONPATH=$PWD python -m pytest -q -p no:cacheprovider \
  tests/unit/test_moe_ep_aggregate_admission.py \
  tests/unit/test_mixed_layer_decode_ffn_scheduling.py \
  tests/unit/test_moe_predictor_layer_id_semantics.py
```

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824
PYTHONPATH=$PWD python -m pytest -q -p no:cacheprovider \
  tests/unit/test_mtp_terminal_overshoot_ep_replay.py \
  tests/unit/test_spec_decode_mtp_structural_moe_replay.py \
  tests/unit/test_typed_ep_predictor_contract.py \
  tests/unit/test_typed_ep_scheduler_consumers.py \
  tests/unit/test_moe_ep_workload_materializer.py \
  tests/unit/test_moe_routing_conservation.py \
  tests/unit/test_predictor_effective_tokens.py \
  tests/unit/test_mtp_token_ledger_repair.py \
  tests/unit/test_comm_operator_families.py \
  tests/unit/test_sklearn_disaggregation_execution_time_predictor.py
```

### Exact shared-collective direct case

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824
PYTHONPATH=$PWD python - <<'PY'
from types import SimpleNamespace

from frontier.moe_ep_workload import LayerEPWorkload
from frontier.operators.families import get_comm_operator
from frontier.operators.spec import CommPayloadContext
from frontier.types import ClusterType


class SourceBatch:
    total_num_tokens = 5
    num_tokens = [3, 2]

    def get_effective_total_tokens_rounded(self, _cluster_type):
        return 8


class IdentityQuantization:
    def adjust_tensor_size(self, _collective, data_size_bytes, _cluster_type):
        return data_size_bytes


source = SourceBatch()
model = SimpleNamespace(embedding_dim=8, num_experts_per_tok=2)
replica = SimpleNamespace(
    attn_tensor_parallel_size=1,
    moe_tensor_parallel_size=2,
    moe_expert_parallel_size=2,
    num_pipeline_stages=1,
    router_topk=2,
)
workload = LayerEPWorkload(
    target_replica_id=0,
    global_layer_id=4,
    routing_token_count=source.total_num_tokens,
    router_topk=2,
    total_routed_assignments=10,
    global_per_expert_tokens={0: 6, 1: 0, 2: 0, 3: 4},
    per_ep_per_expert_tokens={0: {0: 6, 1: 0}, 1: {2: 0, 3: 4}},
    per_ep_routed_tokens={0: 6, 1: 4},
    participant_ep_ids=(0, 1),
    expert_to_ep={0: 0, 1: 0, 2: 1, 3: 1},
)
lanes = tuple(workload.lane(ep_id) for ep_id in workload.participant_ep_ids)


def context(lane):
    return CommPayloadContext(
        batch=source,
        model_config=model,
        replica_config=replica,
        cluster_type=ClusterType.DECODE_FFN,
        quantization_manager=IdentityQuantization(),
        lane_workload=lane,
    )


shared = get_comm_operator("moe_tensor_parallel_allreduce")
ep = get_comm_operator("expert_parallel_alltoall")
print("source_raw_width", source.total_num_tokens)
print("source_compute_effective_width", source.get_effective_total_tokens_rounded(ClusterType.DECODE_FFN))
print("aggregate_routed_assignments", workload.total_routed_assignments)
print("participant_ep_ids", workload.participant_ep_ids)
print("lane_routed_counts", [lane.routed_token_count for lane in lanes])
print("shared_payload", shared.build_payload_bytes(context(lanes[0])))
print("ep_payloads", [ep.build_payload_bytes(context(lane)) for lane in lanes])

zero_lane = LayerEPWorkload(
    target_replica_id=0,
    global_layer_id=4,
    routing_token_count=5,
    router_topk=2,
    total_routed_assignments=10,
    global_per_expert_tokens={0: 10, 1: 0, 2: 0, 3: 0},
    per_ep_per_expert_tokens={0: {0: 10, 1: 0}, 1: {2: 0, 3: 0}},
    per_ep_routed_tokens={0: 10, 1: 0},
    participant_ep_ids=(0, 1),
    expert_to_ep={0: 0, 1: 0, 2: 1, 3: 1},
).lane(1)
print("zero_lane_id", zero_lane.ep_id)
print("zero_lane_routed_count", zero_lane.routed_token_count)
print("zero_lane_shared_payload", shared.build_payload_bytes(context(zero_lane)))
print("zero_lane_ep_payload", ep.build_payload_bytes(context(zero_lane)))
PY
```

### Environment

- Python executable: `/usr/bin/python`
- Python version: `3.12.3`
- Conda environment: `CONDA_DEFAULT_ENV` unset; system Python was used
- `PYTHONPATH`: integration worktree root
- Hardware/service requirements: CPU only; no GPU worker, Docker container,
  network service, or external model backend
- Captured logs:
  `/data/ycfeng/tmp/frontier_admission_mixed_20260827.log` and
  `/data/ycfeng/tmp/frontier_focused_20260827.log`

## 2. Validation Criteria

- EP>1 routed MONOLITHIC and PDD calls without `EPLaneWorkload` fail before
  dummy timing, measurement activation, model/backend/communication lookup;
  the instrumented downstream lookup count is exactly `0`.
- Dense mixed-model layers use dense MLP semantics in dummy mode:
  `_is_moe=False`, positive dense MLP components, and routed MoE/EP fields
  equal to `0`.
- Actual MoE layers retain typed physical lanes, EP=1 behavior, zero-routed
  physical participants, and the existing dummy numeric baselines.
- Aggregate routing uses raw width `5 * top-k 2 = 10` assignments.
- Shared pre-routing MoE-TP communication uses compute-effective width `8`.
  With embedding width `8` and fp16 width `2`, the expected payload is
  `8 * 2 * 8 = 128` bytes.
- EP all-to-all uses each lane's post-routing count. For lane counts `[6, 4]`,
  expected payloads are `[8 * 2 * 6, 8 * 2 * 4] = [96, 64]` bytes.
- A zero-routed lane remains in participant set `(0, 1)`, retains the shared
  `128`-byte collective, and has EP all-to-all payload `0`.

## 3. Test Results and Evidence

### Summary

| Check | Observed result | Expected result | Error | Verdict |
|-------|-----------------|-----------------|-------|---------|
| Admission/mixed-layer matrix | `199 passed in 4.05s` | `0` failures | `0` failures | PASS |
| Typed EP/MTP/communication matrix | `183 passed in 31.47s` | `0` failures | `0` failures | PASS |
| Aggregate assignments | `10` | `5 * 2 = 10` | absolute `0`, relative `0%` | PASS |
| Shared MoE-TP payload | `128` bytes | `8 * 2 * 8 = 128` bytes | absolute `0`, relative `0%` | PASS |
| Lane 0 EP payload | `96` bytes | `8 * 2 * 6 = 96` bytes | absolute `0`, relative `0%` | PASS |
| Lane 1 EP payload | `64` bytes | `8 * 2 * 4 = 64` bytes | absolute `0`, relative `0%` | PASS |
| Zero-lane shared payload | `128` bytes | `128` bytes | absolute `0`, relative `0%` | PASS |
| Zero-lane EP payload | `0` bytes | `0` bytes | absolute `0` | PASS |
| Missing-lane downstream lookup count | `0` | `0` | absolute `0` | PASS |

### SCOPE-039 dummy attention-only evidence

| Check | Observed result | Expected result | Error | Verdict |
|-------|-----------------|-----------------|-------|---------|
| RED shared-domain dummy selector regression | `2 failed, 32 deselected`; `post_attention=50.0 ms` for both `PREFILL` and unified `DECODE` | `post_attention=0.0 ms` | `+50.0 ms` per role | RED reproduced |
| GREEN shared-domain dummy selector regression | `2 passed, 33 deselected` | `2` passing role cases | `0` failures | PASS |
| Complete affected predictor matrix | `234 passed in 3.12s` | `0` failures | `0` failures | PASS |
| Dummy `DECODE_FFN` attention-only boundary | `1 passed, 34 deselected` after common-entry guard | explicit `ValueError` | `0` failures | PASS |

The GREEN probes retained positive attention and batch-level overhead while
returning `post_attention=0.0 ms`, `mlp_norm=0.0 ms`, all dense MLP projection
fields `0.0`, all MoE/EP fields `0.0`, and `is_moe=False`. The public
`DECODE_FFN + include_ffn=False` request now fails before the dummy branch,
matching the profiling-backed role contract.

### Token and participant evidence

```text
source_raw_width=5
source_compute_effective_width=8
router_topk=2
aggregate_routed_assignments=10
participant_ep_ids=(0, 1)
lane_routed_counts=[6, 4]
lane_local_token_counts=[(6, 0), (0, 4)]
shared_moe_tp_allreduce_payload_bytes=128
ep_alltoall_payload_bytes=[96, 64]
zero_lane_id=1
zero_lane_routed_count=0
zero_lane_shared_moe_tp_allreduce_payload_bytes=128
zero_lane_ep_alltoall_payload_bytes=0
```

The direct case confirms the architecture boundary: shared MoE work uses the
source pre-routing compute width, EP traffic uses one descriptor's routed
assignment width, and `LayerEPWorkload` owns aggregate conservation and the
physical participant set. No scaling factor, synthetic lane, raw-map fallback,
second source-batch resolver, or caller-specific payload flag is required.

Remote PR20/PR21 state was not modified by these checks.

## 4. Fresh Role and Topology Evidence (2026-08-27)

### Commands and environment

All commands ran from the integration worktree with `/usr/bin/python` 3.12.3,
no Conda environment (`CONDA_DEFAULT_ENV` unset),
`PYTHONPATH=$PWD`, and `PYTHONDONTWRITEBYTECODE=1`:

```text
python -m pytest tests/unit/test_sklearn_disaggregation_execution_time_predictor.py -q -p no:cacheprovider
40 passed in 2.71s

python -m pytest tests/unit/test_moe_ep_aggregate_admission.py -q -p no:cacheprovider
29 passed in 2.75s

python -m pytest tests/unit/test_typed_ep_predictor_contract.py tests/unit/test_typed_ep_scheduler_consumers.py -q -p no:cacheprovider
32 passed in 2.77s
```

The role-capability and topology mismatch probes were direct Python calls in
the same environment. They exercised the real constructor and public predictor
entry points while replacing only downstream timing/lookup owners with call
spies.

### Role-capability matrix

| Configuration shape | Routing calls | `_prefill_routing_details` | `_decode_ffn_routing_details` | `_decode_routing_details` |
|---------------------|---------------|-----------------------------|-------------------------------|---------------------------|
| PD aggregate | `PREFILL, DECODE` | populated | `None` | populated |
| PD-AF aggregate | `PREFILL, DECODE_FFN` | populated | populated | `None` |
| Explicit `PREFILL` | `PREFILL` | populated | `None` | `None` |
| Explicit `DECODE_FFN` | `DECODE_FFN` | `None` | populated | `None` |
| Explicit `DECODE` | `DECODE` | `None` | `None` | populated |
| Explicit `DECODE_ATTN` | none | `None` | `None` | `None` |

An explicitly requested unavailable `DECODE_FFN` role raised
`AttributeError: 'NoneType' object has no attribute 'moe_expert_parallel_size'`
with `routing_calls=0`; no routing/model lookup ran before the fail-fast
boundary.

### Strict topology admission matrix

| Predictor path | Mode | Descriptor mismatch | Observed error | Downstream calls |
|----------------|------|---------------------|----------------|------------------|
| MONOLITHIC | dummy | `EP=4` vs predictor `EP=2` | `ValueError`, `lane_workload EP size does not match predictor topology` | `0` |
| MONOLITHIC | dummy | `top-k=1` vs predictor `top-k=2` | `ValueError`, `lane_workload router_topk does not match predictor topology` | `0` |
| MONOLITHIC | non-dummy | `EP=4` vs predictor `EP=2` | same topology-specific `ValueError` | `0` |
| MONOLITHIC | non-dummy | `top-k=1` vs predictor `top-k=2` | same topology-specific `ValueError` | `0` |
| disaggregation | dummy | `EP=4` vs predictor `EP=2` | same topology-specific `ValueError` | `0` |
| disaggregation | dummy | `top-k=1` vs predictor `top-k=2` | same topology-specific `ValueError` | `0` |
| disaggregation | non-dummy | `EP=4` vs predictor `EP=2` | same topology-specific `ValueError` | `0` |
| disaggregation | non-dummy | `top-k=1` vs predictor `top-k=2` | same topology-specific `ValueError` | `0` |

The zero count covers the instrumented dummy timing, measurement activation,
model, backend, communication, and execution lookup hooks. This confirms that
the shared `_admit_routed_ep_aggregate()` check runs before mode-specific work.

### Verdict

The fresh role and topology gates PASS. The implementation keeps the existing
declaration-driven role capability source, immutable lane descriptor, scheduler
and materializer ownership, aggregate `LayerEPWorkload` conservation, and
registered communication payload domains intact. No synthetic descriptor,
representative-role fallback, duplicate caller guard, scaling factor, or
temporary compatibility branch was added.

## 5. Fresh final-scope verification (2026-08-27)

### Test Script Information

- **Focused command:**

  ```bash
  cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD
  python -m pytest \
    tests/unit/test_moe_ep_aggregate_admission.py \
    tests/unit/test_sklearn_disaggregation_execution_time_predictor.py \
    tests/unit/test_typed_ep_predictor_contract.py \
    tests/unit/test_typed_ep_scheduler_consumers.py \
    tests/unit/test_mixed_layer_decode_ffn_scheduling.py \
    tests/unit/test_spec_decode_mtp_structural_moe_replay.py \
    tests/unit/test_mtp_terminal_overshoot_ep_replay.py \
    tests/unit/test_comm_operator_families.py \
    tests/unit/test_predictor_effective_tokens.py \
    -q -p no:cacheprovider
  ```

- **PR17-sensitive command:**

  ```bash
  cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD
  python -m pytest \
    tests/unit/test_issue117_kv_stage_provenance.py \
    tests/unit/test_kv_transfer_completion_contract.py \
    tests/unit/test_stage_execution_context.py \
    tests/unit/test_prefix_cache_identity_ledger.py \
    tests/unit/test_prefix_cache_scheduler_frontier.py \
    tests/unit/test_pdaf_decode_handoff_semantics.py \
    tests/unit/test_pdaf_decode_attn_preemption.py \
    tests/unit/test_pdaf_decode_attn_online_cohort.py \
    tests/unit/test_pdaf_ep_stage_accounting.py \
    tests/unit/test_pdaf_m2n_metrics.py \
    tests/unit/test_pdaf_deferred_trace_contract.py \
    tests/unit/test_metrics_full_stage_scope.py \
    -q -p no:cacheprovider
  ```

- **Static owner-audit command:** An inline Python 3.12 AST check enumerated
  all `predict_moe_layer_time` definitions and call sites, verified active
  topology keywords at the three disaggregation routed call sites, checked
  admission ordering before dummy/measurement/lookup work, rejected positional
  raw-map lane inputs, and confirmed registry-derived MoE-gating classification.
- **Git topology command:**

  ```bash
  git merge-base origin/pr20-head origin/pr21-head
  git merge-base origin/pr20-head HEAD
  git merge-base origin/pr21-head HEAD
  git merge-tree --write-tree origin/main HEAD
  git merge-tree --write-tree HEAD origin/pr21-head
  ```

- **Environment:** `/usr/bin/python`, Python `3.12.3`; `CONDA_DEFAULT_ENV`
  unset; CPU-only; `PYTHONPATH` set to the integration worktree.
- **Logs:**
  `/data/ycfeng/tmp/pr20_pr21_final_focused_20260827_live.log` and
  `/data/ycfeng/tmp/pr17_sensitive_20260827_final.log`.

### Validation Criteria and Results

| Check | Observed result | Acceptance | Verdict |
|------|-----------------|------------|---------|
| Fresh focused matrix | `309 passed in 7.55s` | `0` failures | PASS |
| Fresh PR17-sensitive matrix | `169 passed, 115 warnings in 7.60s` | `0` failures; warnings limited to argparse deprecation | PASS |
| Interface AST audit | `3` definitions; `3` disaggregation routed calls; all carry `ep_size` and `router_topk` | one compatible signature and complete active-context forwarding | PASS |
| Admission ordering | first admission line `2653`; dummy branch line `2662`; operation lookup line `2696`; token lookup line `2727` | admission precedes mode/lookup work | PASS |
| Raw positional lane input | no `predict_moe_layer_time` call has a fourth positional argument | typed descriptor only | PASS |
| Name-based MoE classification | `0` classification heuristics; gating set derives from `MOE_FAMILY.profiling_ops()` | unified registry source | PASS |
| PR20/PR21 common base | `18d1a23e` | expected stacked base | PASS |
| PR21 ancestor of current HEAD | false (`merge-base=18d1a23e`) | PR21 requires local consolidation | PASS |
| PR20 vs post-PR17 main tree | no conflict in `git merge-tree --write-tree origin/main HEAD` | conflict already coordinated | PASS |

The first attempt at the PR17 command referenced historical filenames and
correctly failed with `file or directory not found`; the corrected command
above uses the current tree's actual test paths and is the authoritative result.
No production or remote PR state changed during either run.

## 6. SCOPE-042 payload admission ordering refresh (2026-08-27)

### Test Script Information

- **Command:**

  ```bash
  cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD \
    PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
    tests/unit/test_pdaf_step3_combine_payload.py \
    tests/unit/test_typed_ep_scheduler_consumers.py \
    tests/unit/test_pdaf_cluster_scheduler_invariants.py
  ```

- **Environment:** `/usr/bin/python`, Python `3.12.3`, CPU-only,
  `CONDA_DEFAULT_ENV` unset, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and
  `PYTHONDONTWRITEBYTECODE=1`.
- **Persistent regression files:**
  `tests/unit/test_pdaf_step3_combine_payload.py`,
  `tests/unit/test_typed_ep_scheduler_consumers.py`, and
  `tests/unit/test_pdaf_cluster_scheduler_invariants.py`.

### Validation Criteria

- Both dispatch and combine reject a missing `EPLaneWorkload` or an exact
  entity-width mismatch before architecture collective resolution,
  communication predictor/backend lookup, trace publication, or final-lane
  commit.
- `predict_alltoall_time` call count remains `0` on each malformed-lane case;
  the earlier valid lane remains the only waiting-room participant.
- Valid typed lanes preserve the existing max-lane payload bytes and event
  timing. Predictor-error and non-finite-time tests reach those downstream
  assertions after descriptor admission.

### Test Results and Evidence

| Matrix | Result | Evidence |
|--------|--------|----------|
| Dispatch/combine payload + typed consumers + PD-AF invariants | **PASS** | `392 passed, 19 skipped in 4.48s` |
| Missing descriptor, dispatch | **PASS** | `ValueError` at typed payload owner; `predict_alltoall_time` calls `0`; waiting-room lanes `{0}` |
| Missing descriptor, combine | **PASS** | Same typed `ValueError` before `resolve_ep_collective_kind()` can reach a predictor; calls `0`; lanes `{0}` |
| Entity width mismatch, dispatch/combine | **PASS** | `ValueError` matching `total_num_tokens.*routed_token_count`; calls `0`; lanes `{0}` |
| Legacy fixture migration | **PASS** | Successful and predictor-error lanes carry valid immutable `EPLaneWorkload`; identity-only invalid fixtures remain raw by design |

The initial RED matrix had `14` fixture failures because old successful and
predictor-error cases supplied only raw `total_num_tokens`. After migration,
the intended predictor-failure and non-finite-time assertions were restored.
The selected option-1 boundary leaves an earlier valid lane in the waiting room
when a final malformed lane fails; no collective event or final-lane commit is
published. Remote PR20/PR21 state remains unchanged.

## 7. Post-implementation commit verification (2026-08-27)

### Test Script Information

- **Focused command:**

  ```bash
  cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824
  PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -m pytest -q -p no:cacheprovider \
    tests/unit/test_moe_ep_aggregate_admission.py \
    tests/unit/test_mixed_layer_decode_ffn_scheduling.py \
    tests/unit/test_moe_predictor_layer_id_semantics.py \
    tests/unit/test_mtp_terminal_overshoot_ep_replay.py \
    tests/unit/test_spec_decode_mtp_structural_moe_replay.py \
    tests/unit/test_typed_ep_predictor_contract.py \
    tests/unit/test_typed_ep_scheduler_consumers.py \
    tests/unit/test_moe_ep_workload_materializer.py \
    tests/unit/test_moe_routing_conservation.py \
    tests/unit/test_predictor_effective_tokens.py \
    tests/unit/test_mtp_token_ledger_repair.py \
    tests/unit/test_comm_operator_families.py \
    tests/unit/test_sklearn_disaggregation_execution_time_predictor.py
  ```

- **PR17-sensitive command:**

  ```bash
  cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr20-post-pr17-merge-20260824
  PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -m pytest -q -p no:cacheprovider \
    tests/unit/test_pdaf_cluster_scheduler_invariants.py \
    tests/unit/test_pdaf_ep_stage_accounting.py \
    tests/unit/test_pdaf_step3_combine_payload.py \
    tests/unit/test_mixed_layer_decode_ffn_scheduling.py \
    tests/unit/test_moe_ep_aggregate_admission.py \
    tests/unit/test_moe_predictor_layer_id_semantics.py \
    tests/unit/test_spec_decode_mtp_structural_moe_replay.py \
    tests/unit/test_typed_ep_scheduler_consumers.py \
    tests/unit/test_comm_operator_families.py
  ```

- **Static commands:**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -m compileall -q frontier tests
  git diff --check
  rg -n '^(<<<<<<<|>>>>>>>)( |$)|^=======$' --glob '!outputs/**' .
  ```

- **Environment:** `/usr/bin/python` 3.12.3, CPU-only, no Conda environment,
  `PYTHONPATH` set to the integration worktree, and
  `PYTHONDONTWRITEBYTECODE=1`.
- **Logs:**
  `/data/ycfeng/tmp/pr20_pr21_final_focused_20260827_rerun.log` and
  `/data/ycfeng/tmp/pr17_sensitive_20260827_rerun.log`.

### Validation Criteria and Results

| Check | Observed result | Acceptance | Verdict |
|------|-----------------|------------|---------|
| Narrow A' focused matrix | `397 passed in 5.12s` | `0` failures | PASS |
| PR17-sensitive matrix | `643 passed, 19 skipped in 5.37s` | `0` failures | PASS |
| Python compilation | exit code `0` | all `frontier/` and `tests/` compile | PASS |
| `git diff --check` | exit code `0` | no whitespace errors | PASS |
| Exact conflict scan | no matches | no standard Git conflict marker | PASS |
| Implementation commit | `0fb2781e` | production/tests only; metrics output excluded | PASS |

The first broad marker expression also matched decorative lines containing
long runs of `=` in existing scripts. The exact expression above requires a
standalone `=======` separator and standard start/end markers, and returned no
matches. The two test logs contain the complete fresh output; no remote PR
state changed during this verification.
