# Capture scope and graph lifetime with keep_graph=False

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Selected the corrected bounded standalone proof; no implementation or submission yet. |

## Purpose and boundary

The accepted tiny callback probe proved retained-source graph correspondence.
The native graph constructor uses Torch's default `keep_graph=False`, whose
source graph cannot be queried after capture. The next experiment tests whether
a graph-node-created callback can directly read the active real Python scope
before that source graph disappears. It does not run the native V4 model or
change serving ownership, calibration predicates or operator-cost admission.

Authority is `D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY`; concrete allocation is
`D-V4-SCOPE-LIFETIME-PROBE` in ../../requirements.md. The implementation has
not begun. Corrected feasibility estimates are not measurement results.

## Smallest selected implementation

- Fresh module `probe_scope_lifetime.py`, at most250gross new lines.
- New CPU controls `tests/unit/test_cupti_scope_lifetime.py`, at most160lines.
- Import unchanged accepted `graph_probe/probe_diagnostic.py` primitives:
  resource envelopes, ABI, ID queries, owned subscription/close lifecycle,
  fsynced stage/fault sink and strict target replay extraction.
- Exercise the unchanged `ServingProfile.range` and its `threading.local`
  scope, with only the minimal inventory/breakable-module scaffold needed to
  construct the standalone diagnostic. Do not replace the real range method.
- Five named pointwise operations, each launching one kernel. Warmup occurs
  before subscription/capture; no extra CUDA events or retained-source DAG.
- Extend the existing callback class with resource callback13. Copy payload
  values and query valid node/graph/exec IDs only inside their callbacks.
- Store op-slot/name/start/end monotonic windows in the same retained stage
  record stream. Callback rows retain their own time/scope/native_id/get_ident.

## Predicates

| Predicate | Required evidence |
| --- | --- |
| P1 delivery | Selected source rows cover all five clone originals exactly, typekernel, one source graph joined to the exec callback; unmatched created rows explicitly counted, not presumed instantiate-time. |
| P2 thread observation | Recordnative_id/get_ident equality with the active capture frame separately; neither equality supplies P3. |
| P3 freshness | Each selected source row lies inside exactly one open op window and reads that exact op's scope; the five scopes each own one selected source row. |
| P4 lifetime | Default graph constructor; one documented raw_cuda_graph expectedRuntimeError boolean aftercapture_end; no returned value retained or source dereference; secondary rawexec handle equals copied exec callback handle. |
| P5 closure | Each clone's originalhandle/originalnodeID identifies exactly one selected created row; five unique cloned handles/full IDs; no creation-order or cross-run mapping. |
| P6 replay | One target annotation, one contained cudaGraphLaunch, five directly correlated kernels; each full graph-node ID joins exactly one clone and one measured source scope; no unmatched or duplicate target kernel. |

Source namespace and executable namespace differ: `clone.original_node_id`
must equal the selected `created.node_id`; `clone.node_id` is the full ID used
by the replay kernel. Never equate source and clone IDs or subtract an offset.
Graph-ID checks use only directly recorded callback identities, not numbering.

Post-capture API calls on copied source handles are prohibited, including
cudaGraphGetNodes/NodeGetType/EventRecordNodeGetEvent and CUPTI node/graph-ID
queries. The one negative `raw_cuda_graph()` observation records only whether
the documented RuntimeError occurred; other errors are retained as errors.

## Runtime and control

After CPU/source/review/package/image/SDK/seal gates, at most one personal-local
StepMind RJobBackend job:1H800/codesign/non-preemptible,16CPU/64mem_gb,600seconds,
attempt1of1. Use existing local-NFS plus personal-cloud mounts and a fresh
personal destination. Primary owns all platform actions; no launch yet.

Observe and control are phases of that one job, not separate submissions.
Unsubscribe before either Kineto trace. Control has no CUPTI subscription and
checks identical five-kernel topology and numerical output, not timing or
non-perturbation. No cost or throughput quantity is claimed.

## Verdicts and next action

CPU positive/negative fixtures must settle the actual callback, freshness,
lifetime restrictions and full-ID join rejection conditions. Package generation
is separately allocated after the frozen source is verified and reviewed.

A clean delivered-but-stale/deferred scope is a negative finding only for the
tested live-TLS callback route/image/driver/shape. Setup, subscription, query or
trace failures are UNVERIFIED and preserve raw failure evidence. Neither result
settles resource-based ownership generally. Untested alternatives remain
runtime-API/activity correlation, clone/exec-only mapping, explicit per-op
push/pop independent of TLS and post-capture retained-activity attribution.

Future native integration would separately extend ServingProfile's own capture
wrapper/hooks in a fresh generation. It is not part of this experiment; sealed
profile04 and the original U1–U4 bytes remain immutable.
