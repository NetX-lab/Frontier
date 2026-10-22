# D26 Prefill Target Test Report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record red/green lifetimes, affected checks, generated fixture identities and limits. |
| 2026-10-22 | Record read-only retained-state P dummy workflow investigation. |

## Environment and commands

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-prefill-target`.
Base `ebd5baf`; branch `feat/step5-prefill-target`.
Python `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6, GCC11.3.0,
Linux5.10.25-nvidia-gpu, glibc2.35. NumPy2.2.6, pandas2.3.3,
scikit-learn1.7.2, pytest9.1.1. CPU-only local execution; no conda environment
switch or package installation.

All commands below use this common environment and interpreter prefix:

```bash
PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data/ycfeng/envs/frontier-py310/bin/python -B
```

The original red/green and affected04 commands used `PYTHONPATH="$PWD"`;
the additional tests/unit path only resolves existing unit-test imports.
For each pytest suffix, append `-q -p no:cacheprovider > LOG 2>&1`.

| Check | Exact command suffix | Log under `/data/ycfeng/tmp/` | Expected / actual | Result |
| --- | --- | --- | --- | --- |
| Red before source correction | `-m pytest tests/unit/test_step5_prefill_target.py` | `d26_prefill_red.log` | Six failures at actual P6143/6144/6145 versus6112/6128/6128 | PASS red; pytest exit1 |
| First green | same suffix, after source correction | `d26_prefill_green01.log` | Six boundary/offline-online lifetimes pass |6 PASS,59.79s |
| Adjacent first iteration | `-m pytest tests/unit/test_nixl_read_cost.py tests/unit/test_step5_measured_pd.py tests/unit/test_step5_structural_pd.py tests/unit/test_glm53_structural_pd.py` | `d26_adjacent01.log` | Legacy full-P4105 assertion is obsolete under its measured Step contract; actual4080 |94 PASS /1 FAIL,114.97s; assertion corrected |
| Final affected lifecycle/fixtures | `-m pytest tests/unit/test_step5_prefill_target.py tests/integration/test_step5_measured_main.py tests/unit/test_step5_measured_pd.py tests/unit/test_nixl_read_cost.py tests/unit/test_step5_structural_pd.py tests/unit/test_glm53_structural_pd.py` | `d26_affected04.log` | All affected/nonqualifying/legacy/source-reservation checks pass |118 PASS,177.85s |
| Actual forward-key/root replay and legacy request checks | `-m pytest tests/integration/test_step5_measured_main.py tests/unit/test_step5_mrv2_forward.py tests/unit/test_request_preempted_step.py tests/unit/test_request_spec_decode_zero_commit.py` | `d26_exact_forward06.log` | Agreed real/dummy features match exact fitted root/count; legacy request checks pass |103 PASS,22.92s |
| Existing round/root/cache regression | `-m pytest tests/unit/test_thinking_round_boundary.py tests/unit/test_step5_measured_replay.py tests/unit/test_step5_measured_training.py` | `d26_round_roots07.log` | Round requeue and inclusive-once measured root/count unchanged |42 PASS,11.25s |
| Semantic max-model bound | `-m pytest tests/unit/test_step5_prefill_target.py::test_prefill_target_does_not_relax_the_client_model_length_bound` | `d26_model_bound08.log` |6145/1024 rejects before P target selection at7168 |1 PASS,5.51s |

The affected04 run preceded the added two exact-fit cases and semantic-bound
test; exact_forward06 and model_bound08 subsequently verified those additions.
Counts overlap; they are not a unique test/scenario census or strict77 result.

## Actual lifetime expectations and observations

| Semantic prompt / output | Old useful P width | New useful P width / D tail | New P source logical pages | New initial P physical pool blocks | Client outputs |
| --- | ---: | --- | ---: | ---: | ---: |
|6143 /1024 |6143 |6112 /31 |382 |1528 |1024 |
|6144 /1024 |6144 |6128 /16 |383 |1532 |1024 |
|6145 /1023 |6145 |6128 /17 |383 |1532 |1023 |

Canonical old P6144 plus D16 computed6160 prompt positions. Corrected P6128
plus D16 computes6144. The canonical useful forward census is P6128 once,
D16 once and Dq1 exactly1023 times. P does not credit an output; D's prompt tail
credits the first output. Semantic/client6144 and1024 remain unchanged.

The real scheduler/event test runs both offline and online at attentionTP8,
DP2, EP16/PP1. P completes and retains source allocation until READ ends;
D schedules nothing before readiness. READ readiness and source release share
the transfer-end timestamp. Duplicate transfer-start/end callbacks reject;
one metric start/end is recorded. Dummy lanes own no requests. D execution
contexts are `[prefix, prompt, ..., prompt + outputs -2]`; all allocations and
load-balancer request ownership are empty after the last output.

`source_prompt_common_blocks` is logical prefix pages per group, not the four
group pool total. The old main-run validator incorrectly expected1536 for this
field; the existing source API returned384. The corrected field is383.
Existing D remote-prefix reservation initially holds1532 pool blocks, then grows
for the local tail. SWA physical lifetime/payload correction is D27, not this task.

At fixed max_model_len7168,6145/1024 is invalid. The6145 lifecycle check uses1023
outputs, while the separate rejection test preserves the full semantic bound.
Partial P computation/preemption retains target6128 and grants no output early.
Dense/DSA families have no registered Step target; dummy timing, no connector,
co-location and prefix-cache paths retain the old semantic prompt target.
Mismatched exact READ coverage fails before placement/reservation/metric mutation.

## Fresh fixture generation and validation

Exact generation suffix:

```bash
tests/integration/step5_measured_compute_fixture.py \
  --output-dir /data/ycfeng/tmp/d26_main_fixture01 --device h800 --gpus-per-role 16
```

Log:
`/data/ycfeng/tmp/d26_fixture_generation01.log`; exit0 and manifest created.
All retained CPU timing rows say `cpu_fixture`, `native_admitted=false`.
No rows were copied or renamed from fixture02 or called native profiles.

Manifest: `/data/ycfeng/tmp/d26_main_fixture01/fixture_manifest.json`, schema
`step5_measured_compute_cpu_fixture.v2`, SHA256
`a5e116390664d03a74e4da5a34ef06d1761eb7ea913aa06f5a90933e72a8e455`.
Topology H800 P16/D16, attentionTP8/DP2, MoETP1/EP16, PP1, model7168,
batch budget8192, capacity128, FULL_AND_PIECEWISE and balanced seed42.
All inputs, hashes, bundle IDs and package versions are in `fixture_receipt.json`.

Bundle identities (all34 models each):

| Role | Family | Bundle ID |
| --- | --- | --- |
| PREFILL | eager | `dd1433171077731e87b1c3999e08721511ac263793f19cc694b6eb3a1ee9293f` |
| PREFILL | kernel_only | `68fb28039f25596c2466462c1c027801ec015197db9bbde70a0b2146c8ff47d2` |
| DECODE | kernel_only | `4cbae587aff75b829a8c480f09430edec37a767b05efcad969e61fe8bf2613c3` |
| DECODE | eager | `ac4e8873d1c5d1ae4051858d88520e6e7a58b0d48c7b4e6fb77f4926f8bc79e0` |

Exact read-only generated bundle validation command:

```bash
ulimit -v 31457280
PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 60s \
/data/ycfeng/envs/frontier-py310/bin/python -B \
  /data/ycfeng/tmp/d26_validate_fixture01.py \
  > /data/ycfeng/tmp/d26_generated_fixture_validation11.log 2>&1
```

PASS: existing `load_model_bundle` validates hashes/identities and reloads all
four bundles;2944 actual agreed real/dummy root/count queries across92 layers
match exact keys and `cpu_fixture` proof. Local widths1/16/24/32/6112/6128 and
aggregate2/32/48/64/6113/6129 use existing balanced Hamilton materialization;
unlisted shapes still fail. D tails31/17 pad to32/24 without changing requests.
Sparse/root physical ownership and inclusive-once cost assembly remain intact.

Validation iteration09 could not start systemd scope:
`Failed to start transient scope unit: Interactive authentication required.`
The timeout + virtual-memory limit above provided the bounded inspection.
Iteration10 reader assumed unlabelled model keys and failed `KeyError:
'step5_layer_e2e'`; inspected bundle entries use registered `cache_model_name`
labels. Only the one-shot reader was corrected; no registry/perf/source behavior
was changed. A fixture-test draft also incorrectly used dummy P width1 as the
real NONE width and strict binary-float equality; actual per-lane descriptors and
float comparison corrected the test, with exact key membership still mandatory.

Historical fixture02 remains untouched by task writes. Its read-only manifest
hash is `417492f6223f83bc702b815cd8a4280323afb60600fd68fe9b9636cc0077fffd`.
Actual runner read rejects v1; refresh also rejects v1 before any rewrite.
No pre-task full-tree digest was recorded, so unrelated concurrent byte changes
are not claimed to have been excluded by cryptographic comparison.

## Acceptance limits

No full2400 CLI, GPU/native configuration, overhead analysis, timing calibration,
parent balanced/native assignment reconciliation, D27 SWA correction, independent
review or strict77 gate was run here. Synthetic3.5ms READ and compute/gap costs
are fixture evidence only, not measured native fidelity. After parent integration,
retain workflow -> GPU/opgap -> CPU overhead last ordering. Parent must run the
fresh-fixture full CLI only once its perf stream is ready, verify actual request,
transfer and stage ledgers, perform native comparison, and complete strict77.

## D29 read-only workflow follow-up

Parent requested examination of existing complete-q1 evidence only. No simulator
events were executed, no trace or fixture generated, and no production source or
fixed snapshot was changed. Observations below are from retained historical
fixture02 evidence, not D26 or native performance qualification.

### Retained time/request evidence

Both `/data/ycfeng/tmp/d23_parent_actual06_wall01` and
`/data/ycfeng/tmp/d23_parent_identity_verify01` show:

| Evidence | Simulated time / state |
| --- | --- |
| Last useful P batch0, request0, end |85.51257516799994s |
| READ end, D-ready, source lease release |85.51607516799994s |
| First retained D q1, request0, width1 |85.95974032799978..86.00261039840062s |
| Second retained D q1, request0, width1 |86.00261039840062..86.04548046880146s |
| P scheduling at iteration12 |85.98403594240912s; running0, waiting0, request IDs[], useful tokens0 |
| P scheduling at iteration13 |86.02689601280996s; running0, waiting0, request IDs[], useful tokens0 |

Both `events.json` files have582 P and586 D events. The receipt attributes291 P
and293 D events to each complete D q1 interval; this is existing evidence, not a
new overhead measurement. `event_trace_prefix.json` and
`stage_ledger_prefix.json` are byte-identical across these two runs, SHA256:
`51bbd6b18a1d0e9bea42aa8c33a37f2cb3d917273b7af7e8bd3ef6cfb6b9c51d` and
`f5e5f4e99b486cc9e0594ac77f5a9f408c1302e469c299696a89fb3c3ab5c858`.

The retained `/data/ycfeng/tmp/d23_actual06_exact_q1_state/state.pkl` is
117392622 bytes, SHA256
`2482363cf6d2fe65464e499d038e3dd8c438d531d9f8b0bf84fb5f72272ac68c`.
It was inspected with old compatible source7e8bea3 without running an event or
calling the load-and-resume script. At85.95974032799978s, both P lanes have:

- `_dp_wave_running=True`, `_dp_wave_steps=11`, `_iteration_open=True`.
- `_dummy_forwards_in_flight=1`, `_num_running_batches=0`, and no useful
  `_in_flight_batch_ids` or `_unpopped_outputs`.
- Empty request/preempted/running/waiting queues, pending-transfer set,
  pending-transfer-start list, and allocation map.
- `_waits_at_dp_sync=False`, `_unfinished_at_dp_sync=False`.

Request0 is P-complete, semantic prompt6144, processed6145, not globally complete,
and owned by D running queue only. The retained case has PP1. There is no useful
P pipeline overlap or source lease remaining at the q1 boundary.

### Exact read-only state command

Working directory:
`/data/ycfeng/Frontier/.worktrees/fidelity-candidate-7e8bea3`, HEAD
`7e8bea3cd2362f8e46c4b7cd0e712d2f2fe392fb`; same Python3.10.6 as above.
The persistent model/backend references are opaque placeholders because only
stored scheduler/request state is inspected; no predictor is called or changed.

```bash
PYTHONPATH="$PWD:/data/ycfeng/Frontier/.worktrees/step5-mrv2-main/frontier/cc_backend/backends/collective-sim/python" \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
import resource
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
import pickle, types
from pathlib import Path
path = Path('/data/ycfeng/tmp/d23_actual06_exact_q1_state/state.pkl')
class Reference:
    pass
refs = {}
with path.open('rb') as stream:
    blocks = pickle.load(stream)
    class Unpickler(pickle.Unpickler):
        def persistent_load(self, identity):
            if identity[0] == 'block':
                return blocks[identity[1]][identity[2]]
            if identity[0] == 'predictor_class':
                return Reference
            if identity not in refs:
                refs[identity] = types.SimpleNamespace(identity=identity)
            return refs[identity]
    state = Unpickler(stream).load()
sim = state['simulator']
print('STATE KEYS', list(state), 'TIME', sim['_time'])
for role, scheduler in sim['_global_scheduler']._cluster_schedulers.items():
    print('ROLE', role)
    for key, lane in scheduler._replica_schedulers.items():
        print('LANE', key)
        names = ('_dp_wave_running', '_dp_wave_steps', '_waits_at_dp_sync',
                 '_unfinished_at_dp_sync', '_dummy_forwards_in_flight',
                 '_dummy_pass_due', '_iteration_open', '_num_running_batches',
                 '_max_concurrent_batches', '_in_flight_batch_ids',
                 '_pending_kv_transfer_requests', '_pending_kv_transfer_start_events',
                 '_request_queue', '_running_requests', '_waiting_requests',
                 '_preempted_requests', '_unpopped_outputs', '_allocation_map')
        for name in names:
            value = getattr(lane, name, None)
            if name in ('_request_queue', '_running_requests', '_waiting_requests',
                        '_preempted_requests'):
                value = [x.id for x in value]
            elif name in ('_unpopped_outputs', '_pending_kv_transfer_start_events') and value is not None:
                value = len(value)
            print(name, value)
print('REQUESTS', [(r.id, {k: getattr(r, k, None) for k in
      ('_num_prefill_tokens', '_num_processed_tokens', '_is_prefill_complete',
       '_num_recomputed_tokens', '_completed')}) for r in sim['_all_requests']])
print('QUEUE', [(time, type(ev).__name__, str(getattr(ev, '_cluster_type', '')),
                getattr(getattr(ev, '_batch', None), 'id', None),
                getattr(getattr(ev, '_batch_stage', None), 'batch_id', None))
                for time, ev in sorted(sim['_event_queue'])[:12]])
PY
```

Result: PASS inspection, exit0. Initial inspection lacked the existing
collective-sim Python path and failed `ModuleNotFoundError: collective_sim_core`;
adding the existing path resolved it. A reader assumed the newer snapshot's
`iteration_boundary` key and failed `KeyError`; inspecting the old stored keys
resolved that reader-only assumption. Neither failure modified source or snapshot.

### Source mechanism and conclusion

At D26 HEADaee2562 (the same mechanism is present in retained source7e8bea3):

- `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:527-601`
  runs a dummy if no useful batch is formed while the DP wave is running; wave
  termination is checked only at the32-step DP group finish-sync.
- `:608-612` limits that group to lanes of this cluster. D requests cannot keep
  the P wave running through cross-role membership.
- `:634-637` checks running/waiting requests only. Shared Request's global
  completion and retained KV source ownership do not enter that predicate.
- `:794-808` removes a completed P request from the running queue and separately
  retains its source lease; `:485-500` frees that lease on READ completion.
- `frontier/entities/batch.py:1340-1357` defines dummy forwards with no requests
  and no serving KV/output credits. Role accounting in
  `tests/performance/step5_actual06_iteration.py:501-504` reads event cluster
  type, not EP subrole; P sync/MoE events here really belong to P.

D26 changes `Request.on_batch_end`'s P completion target to6128, which reaches
the existing running-queue removal and transfer trigger earlier; it changes the
transfer width in `frontier/events/cluster_batch_end_event.py:582-617`. It does
not alter `_dp_wave_running`, dummy issuance or finish-sync cadence. Changed P
costs may change overlap timing, but D26 alone does not establish disappearance
of this early P activity.

**Conclusion:** These retained P events are request-free DP wave-drain forwards,
not legitimate useful/in-flight P serving work and not stale shared-request or
source-lease ownership. Do not delete their work/costs based on this evidence.

One separate native-alignment question remains for parent: locked native
`/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm/v1/engine/core.py:2405-2415`
also syncs step1 and uses configured `dp_sync_interval`; its
`config/parallel.py:162` defaults to16, whereas Frontier hardcodes32 and checks
only multiples of32. The deployed native effective interval was not established
here. Parent must reconcile that workflow contract before deciding a separate
alignment change; this finding authorizes neither a config change nor an
optimization. No native/GPU/full CLI or CPU-overhead acceptance is claimed.
