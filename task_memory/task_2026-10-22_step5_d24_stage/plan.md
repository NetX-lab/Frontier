# D24 offline stage plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Defined the independent offline staging dependency graph. |

Dependency: authority/source inspection -> fresh builder package -> {offline verifier, scoped capability tests, payload/syntax/origin checks} -> seal manifest/receipts -> parent handoff.

1. Inspect task authority, local instructions, calibration contracts and StepMind mount/launcher rules. Confirm clean source and fresh output paths.
2. Use `r2_canonical.definition()` and `build_stage.build()` from the owned checkout. Bind imported file origins and Git blobs; copy only model metadata and pinned probes from capture01. Preserve exact native pins, engine/resources/waves and failure policy.
3. Complete stage-local package/submit artifacts, refresh canonical tool hash pins, and run `verify_r2_stage.py`. Compare imported staged modules and exact configured preflight with their frozen sources.
4. Independently run existing bounded reference, HTTP, startup census and native staging tests. Compile staged Python source and check bash syntax. Validate offline payload and (if inert) `ResourceConfig` defaults without constructing any backend.
5. Record evidence, check mainline drift, seal the stage/metadata manifest, reread requirements and commit task artifacts. Stop at offline-ready; all live/native validation remains parent-owned.

Acceptance is offline package readiness only. Native startup, loaded source/GPU identities, actual raw bytes, pressure/abort/overlap witnesses, state agreement, qualification and formal timing remain UNVERIFIED until parent execution and judgment.
