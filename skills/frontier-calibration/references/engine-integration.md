# Reference Engine Integration

Use this procedure before the first case for an engine, fork, or version.
The reference may be vLLM, a user's fork, SGLang, TensorRT-LLM, or another
serving engine. The calibration method is shared; collection commands, probe
formats, and supported Frontier behavior must be checked for the chosen build.
An engine name alone establishes none of those capabilities.

## 1. Bind the Target

Use the engine and revision the user selected. Record its identity under
`contract.md`, Ground-Truth Engine and Build, together with the actual serving
configuration, model/tokenizer, hardware layout, workload, and metric endpoints.
For a fork, distinguish intentional serving changes from instrumentation.
Those intentional changes are part of the target; an upstream release is a
comparison source only when the case names it as one.

Keep machine paths and launch commands in `site.local.md` or the case inputs.
A local default is a convenience for that host, not a restriction on the skill
or authority to replace an engine the user selected.

## 2. Check the Capabilities Needed by This Case

Record an `engine_binding` in the existing case manifest: for each capability,
name the command or implementation, its version, input/output paths, supporting
evidence, and `available`, `missing`, `unsupported`, or `not_applicable`.
This is a case record, not a new runtime plugin API or a CLI option.

| Capability | What must be verified |
| --- | --- |
| Clean collection | The selected engine can replay the workload and retain per-request identities, token counts, timing boundaries, completion status, and repeated clean runs. |
| Metric conversion | A reader maps native records to the comparison's units, clocks, endpoints, and formal-request population without hiding missing or duplicate rows. |
| Frontier semantics | The chosen Frontier scheduler, topology, KV behavior, execution path, and profiling inputs represent the target configuration. Similar setting names are insufficient. |
| Workflow observations | Scheduler, batch, placement, KV, and preemption evidence covers the behavior being investigated. |
| Full-model layer timing | Probes and a timing-input adapter support the optional layer experiment without flattening event dependencies or double-counting time. |
| Fine timing | Operator/kernel, CPU, routing, and communication observations map to the scopes needed by the selected analysis. Check each required family separately. |

Clean collection, metric conversion, and relevant Frontier semantics are needed
for an E2E accuracy claim. Missing optional probes do not block an otherwise
supported E2E comparison; leave their diagnostic claims unverified. A missing
collector or converter can be implemented as case-local work within the
task's authorized scope, then validated before use. Missing simulator behavior
needs the evidence and core-change decision in `repair-approval.md`. Record
unsupported scope and continue independent supported work.
Never select a vLLM scheduler merely to make another engine's case runnable.

## 3. Validate the Comparison Path

Inspect the actual collector, converter, and metric consumer before a large
run. On a small case, reconcile raw records with normalized rows: formal and
warmup IDs, token counts, arrival/first-token/completion endpoints, clock units,
TPOT denominator, and throughput window. Check hand-computed examples against
the output and retain unmatched records. A client first-token timestamp cannot
be renamed to a prefill-completion timestamp without evidence of equivalence.
Aggregate benchmark summaries alone do not provide this request-level join.

The existing `run_gate.py normalize` reads vLLM-specific raw files. A new
engine needs a verified reader; passing its directory as `--vllm-run-dir` does
not adapt its format. The numerical gate can be reused through a converter
that satisfies `gate.py` and the normalized schema, once the cell is registered
and its metric semantics and gate kind are applicable. Validate that converter
against raw samples before using its output for a verdict.

The current gate uses the legacy side name `vllm/<run_tag>` even when a verified
converter supplies another reference engine. Preserve the actual engine/build
and raw-source mapping in the case and conversion receipt; the directory name
is only a storage convention. Do not fabricate mode labels, endpoint offsets,
or routing evidence to fit the reader. If the existing gate cannot express the
chosen metric boundary, record the limitation and scope the required change;
a manually adjusted number cannot stand in for the native gate.

## 4. Run the Applicable Branches

Return to `parity-run` after semantic alignment and the needed capability checks.
Use engine-native observations through the recorded readers. Each analysis
keeps its existing evidence and completion criteria; probe names may differ.
For example, clean SGLang rows can support E2E comparison after conversion is
validated even when a layer-timing probe is absent. A TensorRT-LLM build may be
pinned by package/image and compiled artifacts without a source checkout, but
still needs an accurate Frontier semantic mapping and metric reader.

For the repository's existing vLLM collectors and probes, read
`tool-boundaries.md`, Existing vLLM Collection Profile. Those recipes are useful
when their formats and behavior match the selected build. They are not default
requirements for other forks, versions, or engines.
