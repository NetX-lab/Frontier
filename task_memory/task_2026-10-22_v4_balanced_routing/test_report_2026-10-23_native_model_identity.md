# Native06 declared model identity

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded the actual builder failure and restored the existing native engine model-name input for this fixed V4 case. |

## RED and diagnosis

The sole e9e79d4 native06 build failed at `build_candidate.py:123 -> verify_balanced_frontier():38` with `KeyError: 'model_name'`. Full command/error and unchanged-file evidence: `/data/ycfeng/Frontier/.worktrees/v4-balanced-native-run/task_memory/task_2026-10-22_v4_balanced_routing/test_report_2026-10-23_native06_package_cpu.md`. Raw log: `/data/ycfeng/tmp/glm53_dsv4/native06_cpu_e9e79d4/build.log`.

The actual retained engine template has no model_name, and native serving receives the explicit local model directory. The new binding checker requires a declared model label to compare with the independent Frontier input. Existing tests supplied this label, so their17PASS did not establish builder integration. No checker relaxation or inferred Frontier-to-native identity is appropriate.

## Bounded correction

Set the existing engine.model_name input to `deepseek-ai/DeepSeek-V4-Flash` in this fixed V4 case builder, alongside its already-explicit TP8/DP2/EP16 and H800 settings. This is a case input, not a shared runtime default or model classification branch. The label comes from the retained model metadata manifest; all four declared native model-file hashes must match that manifest entry and the actual local files. The Frontier input is not its source. Native server command/model-directory behavior and R3/frozen R2 bytes remain unchanged.

The correction adds three cumulative routing lines (+2/-1), replacing a previously unchanged baseline line: A432/B319/C202=953. Root used delegated authority to raise the ceiling by one from952. Preserve the incomplete e9e79d4 native06 directory as failed-build evidence. Build generation06 in a new clean worktree after verification; no earlier package is overwritten or removed, and no GPU resource budget changes.

Next validation uses the actual builder/configuration flow, exact metadata-label/hash agreement, unchanged server argv, complete R3 configuration verification, then the complete generated package and locked-image checks. No additional broad simulator rerun is needed for this task-local case-input-only change after e9e79d4 strict77 passed77/77 and all frontier/test sources remain identical.

## GREEN and receipt iteration

Python3.10.6 imported the actual builder and called main('06'), overriding only HERE to a new temporary directory and REPO to clean e9e79d4. Full staging, real R3 reconstruction,74 source/18 within-stage tool/12 module pins and3600/1800/1800 deadlines passed. The retained metadata manifest uniquely matches all four native model-file hashes. Existing server_command produces identical argv before/after the new label.

The outer receipt script then failed its expected30/7 diff-count assertion; functional checks had passed. Actual31/8 was recovered from retained files without rebuilding. `evidence/native_model_identity.json` records hashes, counts and temporary package `/data/ycfeng/tmp/glm53_dsv4/native06_model_identity_8njcbdze/candidate_h800_p0_06`. This is a CPU regression fixture, not the final package. No persistent harness or runtime fallback was added.
