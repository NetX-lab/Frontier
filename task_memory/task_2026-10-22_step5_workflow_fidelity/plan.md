# Plan: merged Step legacy fidelity

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Define the isolated clean77 verification and evidence handoff. |

## Dependency path

`read contract -> fresh exact-SHA checkout -> baseline provenance/input audit ->
candidate clean77 run -> strict artifact/cache comparison -> evidence audit ->
scoped records -> records-only commit -> parent handoff`

The parent independently owns source integration, unit tests and full Step CLI.
No nested agents are authorized. Capture runs before records are written into
the checkout so that source and harness provenance remain clean at the requested
SHA. During execution, draft records remain under the new temporary output root.

## Steps and completion conditions

1. Inspect the request, root instructions, main requirements and plan. Create
   `verify/fidelity-candidate-fa5bae2` at the exact requested SHA without replacing
   an existing worktree.
2. Verify one prior clean77 label: full case definitions, uniform clean
   provenance, intact recorded artifacts, identical runner/cases/wrappers/input
   files and matching Python/dependency versions. Retain a preflight receipt.
3. Execute the unmodified candidate matrix once with the requested environment
   and flags. Check results and manifest rather than relying on exit status;
   continue-on-failure deliberately permits a zero exit for failed cases.
4. Run `run_matrix.py compare` using the absolute existing baseline label and
   the new candidate label. Do not use partial-comparison options. Inspect all
   failure/missing/definition/provenance/cache and artifact-difference fields.
5. Rehash candidate artifacts and verify exact-SHA provenance and unchanged
   tracked source. Preserve selected manifests, comparison and receipts in the
   new task records. Record verification limits and a mainline self-check.
6. Re-read requirements, commit only the new task records, verify the commit's
   changed-file list and return the tested SHA, records commit and report paths.

## Failure handling

Preserve failed records and raw logs. Identify the first actual mismatch or
missing prerequisite. Do not rerun or repair source, tests or fixtures to force
agreement. If a completed strict comparison fails, return its evidence to the
parent without claiming acceptance.
