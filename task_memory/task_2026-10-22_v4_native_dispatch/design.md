# Host dispatch observation design

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded W128-V4-NATIVE-DISPATCH ownership before implementation. |
| 2026-10-22 | R1 added source-required sum <= count*max classifier input constraint. |

## Ownership and boundary

This agent owns `tests/comparison/calibration/native_full_structure.py`, the existing
`tests/unit/test_native_v2_serving_inventory.py`, `tests/unit/test_native_reader_binding.py`,
and the versioned successor in this task directory. Root owns review, source freeze, generation06 builder changes,
packaging, image checks and all GPU/platform operations. No native source is edited.

The collector is 861 lines and R2 is 1,382 lines, below the repository 2,000-line
limit. Tests already exercise CPU native-body extraction and soft observer errors.
Estimated physical additions: collector 100–130, successor 170–210, tests 120–150;
maximum 500 changed code/test lines. A full R2 copy alone exceeds that bound.

## Association

Use the existing V2 execute wrapper's thread-local scope to hold one actual
classifier call, one imported dispatch call and one DP all-reduce. Observe the
classifier arguments/return directly, not recomputed request state or observed
mode. Capture dispatch scalar arguments and the native descriptor return. A
module-local `dp_utils.dist` proxy forwards all other attributes and the native
all-reduce unchanged, and snapshots only its CPU 4-by-DP tensor after return and
before native padding mutates row0. No torch-global distributed monkeypatch.

Successful forward/replay counters retain the complete immutable dispatch facts
in their existing metadata key. One successful forward per execution is required;
piecewise can have many successful replay returns. Failed forwards are never
credited, including replay-only rows after a failed outer call. Scope cleanup
prevents cross-execution reuse. Missing/multiple operands, observation errors,
counter loss and capacity overflow remain UNVERIFIED. A replay followed by a failed
outer forward is explicitly logged as unassociated, including when its aggregate
key already contains an earlier successful forward. Existing MAX_ROWS bounds
the combined representation; cardinality is unverified until the fresh run.

## Versioned checker

Load the frozen R2 only after its exact SHA256 is verified. Replace three unique
source sections: V2 source/hook pins and the invalid local-only oracle branch.
Retain every other byte, including all seven other predicates and reader origin,
shadow, startup, count and terminal guards. Compile the resulting namespace with
the successor's file identity. Report the effective-source delta at handoff
as a diff artifact; no full duplicate Python file is written. The successor executes unedited
pinned classifier and DP native functions on CPU carriers and compares the native
descriptor against both dispatch return and concrete forward metadata. Observed
mode is never an input to the expected descriptor.

Required additional source pins: gpu/dp_utils.py, worker/utils.py, gpu/states.py,
NIXL pull_scheduler.py (utils may already be pinned; actual count checked by root).
Frozen R2/native05 and the global checker are immutable. No builder or package is
created until final code acceptance.

## Observed CPU results

72 observer/native-body controls and 43 reader controls pass. The collector, V2
test file and successor are 954, 611 and 136 lines respectively. R1's existing
negative case observed RED then18 targeted checks PASS; no legitimate None/zero
encoding changed. Physical maintained
code/test changes occupy four files and remain below 500 changed lines. R2 is a
read-only external dependency at its authority path, verified before execution;
root must preserve and bind it during freeze (not silently relocate or copy code
without counting it). Native cardinality/overhead remain unverified.
