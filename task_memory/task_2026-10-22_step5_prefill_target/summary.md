# D26-PREFILL-TARGET Completion Archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archive approved source correction, fixture proof and parent handoff. |

## Outcome

Implemented the approved CPU lifecycle correction on `feat/step5-prefill-target`,
based on `ebd5baf`. Step's registered measured grouped NIXL/PP1 contract selects
a P computed prompt target before admission. Request semantic and user-facing
lengths remain unchanged. Admission width, P completion, source prefix ownership
and transfer batch width consume the target. Exact READ coverage must agree;
on READ completion D retains semantic prompt progress with the existing local
recompute cursor and produces the first client output.

Canonical census changes from P6144 + D16 + 1023 q1 forwards to
P6128 + D16 + 1023 q1 forwards, for client prompt6144/output1024.
Boundary6143 uses P6112/D31, and6145 uses P6128/D17. The6145 lifetime uses1023
outputs to preserve native max_model_len7168;6145/1024 still rejects.

## Commits and changed files

- `e8d9c6c`: CPU source correction and real lifecycle regression.
- `d56f2dd`: new immutable fixture schema, canonical/boundary keys and balanced argv.
- `68ab45c`: semantic model-length rejection and documented Request size exception.
- Final record commit: contains this archive, requirements/plan/progress/checkpoint,
  design, test report, receipt and `changes.patch`.

Changed CPU source:

- `frontier/attention/{families.py,ops.py}`: one registered target resolver,
  separate from memory retention.
- `frontier/entities/request.py`: P target, P completion and D recompute anchoring.
- `frontier/scheduler/replica_scheduler/{vllm_v1_engine_replica_scheduler.py,vllm_v1_kv_allocation.py}`:
  selection/admission/scheduling/source logical pages.
- `frontier/events/cluster_batch_end_event.py`: computed transfer width.
- `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py`:
  exact transfer-prefix agreement.

Changed verification/integration:

- `tests/unit/{test_step5_prefill_target.py,test_step5_measured_pd.py}`.
- `tests/integration/{step5_measured_compute_fixture.py,run_step5_measured_main.py,test_step5_measured_main.py}`.
- `docs/general/module-size-exceptions.md`.

No attention memory, native profiling producer, measured registry/performance
identity, GPU/settings, unrelated source or original fixture02 writes occurred.

## Deliverables and observed verification

Task records: `task_memory/task_2026-10-22_step5_prefill_target/`.
New fixture: `/data/ycfeng/tmp/d26_main_fixture01/fixture_manifest.json`,
schema `step5_measured_compute_cpu_fixture.v2`, `cpu_fixture`, not native-admitted.
Manifest SHA256 `a5e116390664d03a74e4da5a34ef06d1761eb7ea913aa06f5a90933e72a8e455`.
All four bundle IDs and input hashes are in `fixture_receipt.json`.

Observed: six red lifetimes, then six green; affected tests118 PASS;
actual forward-key/request tests103 PASS; existing round/root/cache tests42 PASS;
semantic bound1 PASS. Existing bundle loader reloaded34 models for each of four
bundles;2944 actual real/dummy root/count queries across92 layers passed exact
lookups and inclusive-once proof. Tests overlap; this is not a unique scenario
census. Full commands, errors, environment and limits are in
`test_report_2026-10-22_prefill_target.md`.

## Parent handoff and remaining acceptance

1. Review and integrate source/fixture/bound commits and task records on the final
   parent branch. No further delegation was used in this task.
2. Integrate D27 SWA live pages. Existing standalone D26 source ownership is383
   logical pages; initial P/D remote-prefix pool allocation is1532. These physical
   pool/payload expectations and synthetic READ bytes are pre-D27, not final
   native values. Reconcile fixture READ rows and main validator with D27 actual
   ownership after integration; do not relabel synthetic rows as native.
3. Once parent perf is ready, run fresh-fixture full offline/online CLI and inspect
   real stage/request/READ ledgers and root costs. No full2400 CLI was launched.
4. Parent/native stream reconciles balanced assignments and collects native
   profiles. Continue workflow -> GPU/opgap -> CPU overhead last.
5. Parent owns independent review and final strict77 artifact/numeric fidelity gate.

Synthetic costs and CPU fixture proof make no native latency/fidelity claim.
Historical fixture02 is rejected for new-target runs and remains untouched;
no pre-task full-tree digest exists to exclude unrelated concurrent changes.
