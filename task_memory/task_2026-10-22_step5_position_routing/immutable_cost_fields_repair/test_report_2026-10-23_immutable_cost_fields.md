# D29 immutable cost fields verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Parent verifies committed source, closes both independent review axes, builds missing HTSIM locally and completes affected checks. |
| 2026-10-23 | Record exact current-source differential, negatives, unchanged pytest failure and Git-derived accounting. |

TaskID: `D29-IMMUTABLE-COST-FIELDS`.
Authority/baseline: `48d34106c026b824097f9910a0bbf89b253a8669`.
Code commit: `7540414341d5b81a86b18b9562d1506c0ba74285`.
Worktree: `/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields`.
Scratch: `/data/ycfeng/tmp/d29_immutable_cost_fields_repair/`.
Environment: CPython3.10.6 at `/data/ycfeng/envs/frontier-py310/bin/python`,
brainpp,60s timeout/30GiB systemd scope, existing dependencies only.

## Outcome and exact scope

Replace only the two cost `asdict` expressions and their local imports with
`fields` reads into fresh dictionaries. This removes recursion that cannot add
detachment for the supported immutable cost values. All surrounding numerical
work, source snapshots, metadata proof objects, plan identity/evidence/count
checks, trace flushing and exception handling are unchanged. No helper/API/cache,
new source/test path, persistent new test or allowance transfer.

| Source path | Step + / - | Step gross / cap | Retained before -> after / cap |
| --- | --- | --- | --- |
| `frontier/metrics/op_trace_utils.py` | 2 / 2 | 4 / 4 | 45 ->45 /45 |
| `frontier/metrics/ep_wave_metrics.py` | 2 / 2 | 4 / 4 | 22 ->22 /22 |
| Total | 4 / 4 | 8 /8 | 67 ->67; delta0 |

Retained comparisons use `git diff1aee0af`; both paths remain entirely D29-added
lines. The step is8gross but consumes0additional retained allowance. Given the
root's supplied baseline and zero delta in the only two changed source paths,
PRICING3920/3956, CORE4105/4404, source9375/global9389 and112allocated paths stay
unchanged. This is an exact local delta audit, not a new full-union gate.

## Actual immutable domain and source boundaries

`CollectivePlanCost` (`frontier/cc_backend/collective_plan.py:103-136`) declares
three scalar timing floats, three string identity/receipt fields, a tuple of
local timing floats and a tuple of integer kernel counts.
Its constructor normalizes `local_rank_ms` and `kernel_counts` to tuples,
validates finite nonnegative local service, exact integer counts and the full
receipt/plan/model identities. This justification does not rely only on frozen.

The existing producer at `collective_sim_cc_backend.py:825` builds this class
from validated JSON receipt values. The native-model producer at877-880 uses
numeric JSON result fields, tuple-normalized local ranks/counts and JSON/string
identities. In the actual208-event replay, all64 cost objects are the same
concrete class; recursive inspection observes only float/string/scalar-number
tuples, with no mutable dictionary/list or nested dataclass value. Fresh cost
dicts remain independently owned; tuple leaves can be shared safely because
the supported numeric elements cannot mutate in place. Future mutable cost
fields would require re-evaluating this optimization.

Existing deep snapshots in `ExecutionTime` and separate trace/ledger mutable
plan/cost mappings remain unchanged. Neither component borrowing nor a joined
write nor duplicate plan-key encoder was introduced.

## Source and probe binding

| File | Baseline SHA256 | Candidate SHA256 |
| --- | --- | --- |
| `op_trace_utils.py` | `c5c888935cf98790903f7498b1ef67b91c85de7b0862512fcfcfa0ad310f5003` | `fb89d1c3406df2fe55372e2f3055b9abcbd484d7566fa11a3410d76af4e5aec5` |
| `ep_wave_metrics.py` | `ef0d74058650d1508a0acbd84003912a1d6431c1b2d2a739d6672525de3e895d` | `d438286b0611e0c23c8fd9037c29d4fac6286edcdcb820f2ec2f843bb8396e1e` |
| `trace_store.py` | `07d7cbbb081829f558b6a4c101201baf112e5387e2045ee244436853fb463a0d` | Unchanged |
| `moe_routing_workload.py` (prepared source) | `d9423f3f318cd355f2773ea6a65a9bf9d297cdc3ec7c8c46230b38456ccacbfc` | Unchanged |

New `current_setup_baseline.py` derives from the accepted residual setup, changing
only the two worktree-path constants and expected9ae60ebc HEAD to48d34106. All
business setup/assertions stay. Full actual pricing uses the current prepared
source once before emission timing. The old scripts are unchanged.

`baseline_probe.py`/`candidate_probe.py` explicitly guard current module hashes
and compile the same new setup/body. Candidate differences are only its output
label/location and the two actual source hashes. No source is synthetically
patched inside the probe, no diagnostic reuse switches, and no frozen79 code is
relabeled current. Measurement occurred with dirty candidate source under48d34106;
the subsequent code commit's blobs match those measured hashes exactly.

`source-binding-baseline.json` and `source-binding-candidate.json` retain all
original/derived script/source SHA256 and transformations. `probe_body.py` retains
the accepted real-wave, snapshot, independent-mapping and invalid-key checks,
uses only actual imported emission functions, omits unallocated controls and
tests wrong-plan-key errors for all64 actual collective contexts.

## Exact commands

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_measured_attention_root_metrics.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_cache_zero_execution.py tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_metrics_stage_execution_time.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_immutable_cost_fields_repair/pytest-baseline --junitxml=/data/ycfeng/tmp/d29_immutable_cost_fields_repair/pytest-baseline.xml > /data/ycfeng/tmp/d29_immutable_cost_fields_repair/pytest-baseline.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_immutable_cost_fields_repair/baseline_probe.py > /data/ycfeng/tmp/d29_immutable_cost_fields_repair/baseline-probe.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_measured_attention_root_metrics.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_cache_zero_execution.py tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_metrics_stage_execution_time.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_immutable_cost_fields_repair/pytest-candidate --junitxml=/data/ycfeng/tmp/d29_immutable_cost_fields_repair/pytest-candidate.xml > /data/ycfeng/tmp/d29_immutable_cost_fields_repair/pytest-candidate.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_immutable_cost_fields_repair/candidate_probe.py > /data/ycfeng/tmp/d29_immutable_cost_fields_repair/candidate-probe.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_immutable_cost_fields_repair/final_verify.py > /data/ycfeng/tmp/d29_immutable_cost_fields_repair/final-verification.log 2>&1
git diff --numstat
git diff 1aee0af 48d34106 --numstat -- frontier/metrics/op_trace_utils.py frontier/metrics/ep_wave_metrics.py
git diff 1aee0af --numstat -- frontier/metrics/op_trace_utils.py frontier/metrics/ep_wave_metrics.py
git diff --check
```

Pytest commands exit1 because the preserved baseline failure remains. Both probes
and final verification exit0. Timed probes ran after the corresponding pytest
finished, avoiding intentional contention. Only this worktree's submodule checkout
was locally initialized at the existing gitlink3dac53c without network or peer edits.

## Observed comparisons

| Criterion | Baseline | Candidate | Result |
| --- | --- | --- | --- |
| Full actual layer60 events/proofs/order | 208 | Identical208 | PASS |
| Full JSONL incl fixed header | SHA1850a9b6... | Exact bytes equal | PASS |
| Event bytes | 2,889,655 | Exact bytes equal | PASS |
| Full ledger bytes | 2,309,545 | Exact bytes equal | PASS |
| Public getter/source/pending snapshots | Detached | Detached | PASS |
| Pending metadata mutation affects source | No | No | PASS |
| Trace/ledger mutable plan/cost mapping independence | Distinct | Distinct | PASS |
| Wrong plan_key negatives | 64 exact ValueErrors | Same64 | PASS |
| Late JSON TypeError restore / first-row disk prefix | All events /3902B | Same | PASS |
| Exact literal and AST change | Four approved replacements | No other delta | PASS |
| Step/cumulative Git/whitespace | Bound8/45/22 |8/45/22, clean | PASS |
| Existing pytest suite |98PASS/1FAIL | Same99 ID/statuses, identical message | Comparison PASS; raw suite FAIL |

Full event body SHA256:
`18d320438ee1fb5f077b12aaa83da8055b47d8cf256d88aedcca159a5d7b819c`.
Full JSONL SHA256 with fixed header:
`1850a9b6a90cc7bb087b881e72d8cc00f5a669fa0be0f54bb25f3361a38d7e11`.
Full ledger SHA256:
`006dc7e8f76e2c8d2f59a42d3163aae75d474a4149eb08a3e85d688e948482bf`.

Known unchanged failure ID:
`tests/unit/test_mla_core_native_op_tracing.py::test_frontier_stage_batch_component_ledger_dense_attention_uses_shared_mapper`.
Exact baseline/candidate message:
`TypeError: test_frontier_stage_batch_component_ledger_dense_attention_uses_shared_mapper.<locals>._fake_get_attention_trace_op_times() got an unexpected keyword argument 'allow_structural'`.
No fixture/guard repair, suppression or weaker test was attempted.

## Emission-only performance and limits

15 real current-source emissions per side in sequential processes, no profiling
inside timing. Baseline median58.067739ms, candidate51.181030ms:6.886709ms or
11.859785% lower. Raw timings retain cold/GC spikes. Pricing, verification,
ledger JSON encoding and trace flush are outside the measured interval; this
is not pricing/flush/wholeCLI acceleration.

Separate actual-emission profile: asdict336->208 (-128), deepcopy26,336->21,472
(-4864), plan.to_dict192->192, plan.cache_key64->64, compute_op_trace_meta208->208.
All full proof containers are retained; no numeric/proof caching or omission.

Selected receipts, exact diff and script identities are archived in this task
directory; complete traces/ledgers, logs/XML and scripts remain in the new scratch.
No full CSV/trace scan, fullCLI, GPU/platform/network/dependency change or further
agent. Parent independent review, final source freeze/union accounting/regressions
and wholeCLI1024/2400s acceptance remain open. No whole-workflow PASS claim.

## Parent acceptance on committed source75404143

Environment:frontier-py310 Python3.10.6, GCC11.3.0, local CPU only. Parent archive
check validates8JSON/12Markdown records,7raw evidence hashes, exact8gross lines,
source AST and primary/source byte equality. Raw diffs remain unchanged.

Both independent reviews returned **No findings**: D29-COST-FIELDS-STANDARDS
(`ses_e96be0e6affexfYUjqdynhvKdK`) and D29-COST-FIELDS-SPEC
(`ses_e96bdc8c2ffehsyfMJwVI39x6W`). They inspected the exact two-file diff,
requirements, admitted producers, ownership boundaries and retained evidence;
neither independently reran tests or Git measurements.

Parent re-executed the unchanged probe body after changing only its setup HEAD
assertion from the pre-commit revision to75404143. New output is isolated at
`/data/ycfeng/tmp/d29_cost_fields_parent01`; `binding.json` records the original
and derived hashes. Actual208events and the entire2,309,545-byte ledger match
the owner baseline byte-for-byte. All64 wrong-plan-key checks, detached source
snapshots, independent mutable mappings and3902-byte late-error prefix pass.
Parent emission median is50.490645ms; this remains a phase-only observation.

Initial direct tests:131PASS/14SKIP across test_op_trace_utils,
test_step5_moe_pricing and test_step5_comm_cost_export. Inspection showed13
skips were missing HTSIM in the new worktree, not code failures. Parent built
the existing3dac53c sources, then reran the affected exporter file:54PASS,
zero skips/failures/errors. All54 IDs belong to the initial suite; replacing
their earlier statuses gives144PASS/1existingTorchSKIP across145 unique IDs.
This is an ID-based reconciliation, not a sum of overlapping suite totals.
The owner98PASS/1baselineFAIL comparison remains separate.

Exact commands (cwd is the cost-fields worktree, except the two make commands
which use its submodule sim and sim/datacenter respectively):

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_cost_fields_parent01/run.py > /data/ycfeng/tmp/d29_cost_fields_parent01/run.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_op_trace_utils.py tests/unit/test_step5_moe_pricing.py tests/unit/test_step5_comm_cost_export.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_cost_fields_parent01/pytest --junitxml=/data/ycfeng/tmp/d29_cost_fields_parent01/pytest.xml > /data/ycfeng/tmp/d29_cost_fields_parent01/pytest.log 2>&1
timeout 180s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env TMPDIR=/data/ycfeng/tmp make -j4 libhtsim.a > /data/ycfeng/tmp/d29_cost_fields_parent01/build_lib.log 2>&1
timeout 180s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env TMPDIR=/data/ycfeng/tmp make -j4 htsim_ndp > /data/ycfeng/tmp/d29_cost_fields_parent01/build_ndp.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_comm_cost_export.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_cost_fields_parent01/pytest_htsim --junitxml=/data/ycfeng/tmp/d29_cost_fields_parent01/pytest_htsim.xml > /data/ycfeng/tmp/d29_cost_fields_parent01/pytest_htsim.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_cost_fields_parent01/archive.py
```

`parent_acceptance.json` retains all145 latest statuses, both review identities,
source/probe/XML/log hashes and the locally built binary hash. Tracked submodule
sources stayed clean; primary source/test blobs match75404143. Code is merged
on feat/step5-preview. The API quota failure interrupted the owner's records
delivery after its code commit; the parent recovered and archived the actual
files. An attempted duplicate worktree creation found the already-existing
owner tree; its extra unused fix/d29-cost-fields branch remains untouched.
Final union/freeze/regressions and complete CLI/native qualification stay open.
