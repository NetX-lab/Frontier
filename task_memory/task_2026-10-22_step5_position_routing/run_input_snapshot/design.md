# D29 run-fixed collective input design

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Select shared run ownership and allocate the fixed-input replacement after YC's decision and lifecycle inspection. |

## Requirement and observed lifetime

The controlling request is the 2026-10-24 Q&A in `../requirements.md`.
One run uses one version of each configured `collective_local.json`; a new run
loads updates. Actual-point validation and full layer-specific output remain.

`simulator.py:110-182` constructs one manager before giving it to both role
predictors. Each role has a separate, lazily constructed backend
(`entities/cluster.py:71-86`, `cc_backend_factory.py:121-127`). A backend-only
snapshot would permit mixed versions if the file changed between role bindings.
The shared manager is recreated for a new Simulator, including in one process.

## Selected module and interface

`frontier/cc_backend/collective_profile.py` contains `CollectiveProfileInputs`,
a small stdlib-only module for configured-path lookup and immutable file bytes.
Its `read(profiling_data_dir, device_type)` hides the existing device-directory
and filename rule. Cache only a successful strict UTF8 and JSON read. Do not
cache exceptions or publish bytes before parsing succeeds. A failed attempt
may be retried; the first successful value is fixed for that owner lifetime.

Use a stable absolute configured-path key. File replacement or a symlink target
change must not select a second version of the same configured path mid-run.
Different configured device/path inputs are separate entries. The existing
same-path P/D case must read once; no filesystem identity polling is needed.

The existing shared prediction manager owns one `CollectiveProfileInputs`.
Expose this object through a small read-only property. Both role predictors bind
their backend to it in the existing `policy.requires_source` construction path.
Use the constructor's `model_manager`/`cc_backend` arguments before `super()`:
the corresponding `self` fields do not exist at lines595-607. Existing optional
manager/backend call shapes remain; standalone backends own an instance of the
same input module. Both contexts use exactly the same loading implementation.

Base backend interface:
- `bind_collective_profile_inputs(inputs)`: attach the run owner and establish
  this backend's input through the regular getter before role use.
- `collective_profile_bytes()`: return the bound immutable bytes.

Other backends reject unsupported collective-profile capability through the
existing NotImplementedError convention. Non-position paths perform no reads.
There is no live/fixed switch, refresh interface, global state, or generic I/O
adapter. Binding is initialization-only; no speculative post-use rebind guard
is needed for a state the actual construction path cannot reach.

CORE obtains initial bytes from the backend. It prepares structural receipts
once for its predictor; remove the superseded per-query file read and artifact
hot-update check. Keep transactional candidate-map copies and fresh per-point
selection. `_step5_position_queries(lane)` stays unchanged. Backend pricing
parses the fixed bytes once locally. Separate local parses of immutable bytes
are acceptable; exposing a shared mutable parsed dictionary is unnecessary.

The input module belongs to CORE input-lifecycle work; it does not change the
communication model. Its location permits both the manager and backend to use
it without a backend-to-predictor import or file I/O inside collective-plan
value types. The existing `profiling/collectives/collectives_input.py` describes
GPU benchmark arguments and is a different responsibility. No registry or
model-bundle identity absorbs the new state.

## Verification through the real interface

1. Baseline RED: a retained real pricing call rereads/mixes changed input instead
   of honoring the new lifetime. Missing-method failures alone are insufficient.
2. Same run: bind P, change the file before D construction, then exercise both
   role bindings and pricing. Both use the first version. Change the file
   between CORE query and backend pricing; they still agree.
3. New run in the same process: construct a new real manager/backend lifetime;
   it observes the new valid or invalid file. Standalone backend calls retain
   their existing signature and use an independent fixed lifetime.
4. Initial missing/permission/UTF8/JSON failures propagate without successful
   snapshot publication; repeated attempts fail and a later successful bind
   can establish a value. Keep native/CPU admission, plan/count/context/buffer
   negatives. Existing file-mutation negatives move to a fresh lifetime.
5. Existing estimator/row-tuple/measurement-identity replacement tests remain.
   Non-position constructors must not access the collective file.
6. The real layer60/16-EP replay retains every one of208 costs/proofs/full keys,
   ordering, JSONL and ledger bytes, snapshots and error prefix. Count file
   reads over the run binding plus cold/warm waves: one successful read for the
   one configured input and no warm file reads. Any manually reconstructed
   cached manager in a replay is disclosed; constructor wiring is tested
   separately through actual constructors, not merely field assignments.

Use existing test files. Test constructor wiring and results, not just a new
object's map. Stub only unrelated expensive training at an explicit seam.
Preserve the original fixture/cache, source pins and all numerical checks.
The existing paired78PASS/24 evidence hashes were checked by root; this is
evidence for the byte intermediate, not snapshot acceptance.

## Allocation and sequence

`../repairs/run_input_snapshot_scope_amendment.json` is authoritative: ten paths,
340 gross step maximum (120 source,220 tests),114 effective paths, unchanged
owner/source/global caps. Both new paths have CORE ownership. Donor files are
untouched. `../repairs/evidence/run_input_snapshot/budget_before.json` derives
the delivered-byte planning baseline: CORE4105,PRICING3922,native1350,
9377 source/9391 global. Non-superproject numbers reuse the accepted79 audit;
this planning receipt is not the final union audit.

Allocation -> RED and implementation -> focused checks/real wave -> independent
Standards and Spec review plus parent verification -> integration/union audit
-> new source freeze/regressions/full workflow -> T2/native/G3/E2E gates.
Output encoding analysis is independent. No full CLI is justified by this
design alone, and no340-line allowance is a measured consumption claim.
