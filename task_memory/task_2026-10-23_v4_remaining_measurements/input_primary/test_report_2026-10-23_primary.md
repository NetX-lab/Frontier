# Root validation of V4 isolated attention input

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded fixed-source verification and actual isolated representation admission. |

## Environment and source

Code `4a1b53e6033134c59b4561306eb2d586bd3d8300`; worktree `/data/ycfeng/Frontier/.worktrees/v4-attention-input`, delivery HEAD `04a72b08`. Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6). Commands use `PYTHONPATH=.`, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode`. No package installation or GPU operation.

## Results

| Check | Expectation | Observed | Result |
| --- | --- | --- | --- |
| Affected unit module | All controls pass on final source | 194 tests, zero failures/errors/skips; source digests unchanged | PASS |
| Strict matrix | All 77 scenarios and cache identities identical | 77/77, no differences/missing evidence/provenance findings; 451 artifacts rehashed | PASS |
| Original accepted inputs | Original file digests and package pins unchanged | 529 sealed package files and copied input/evidence digests match | PASS |
| Independent representation | Twelve table/layer layouts from non-timing facts | Seven engine/step records reproduce both six-layer maps after removing 282 call records | PASS |
| Explicit input | Original 94 medians/layouts | 50 CUDA_EVENT and 44 KERNEL_ONLY scopes, isolated-only | PASS |
| Legacy input | Original 102 positives / 72 typed negatives | Same positive records and controls; only ten temporary-path error strings differ | PASS |
| Missing admission | Both candidate methods refuse automatic use | Both raise DeepSeekV4CoverageError referencing the absent admission | PASS |
| Real externally admitted input | Automatic path equals explicit path | All 94 layouts and medians exactly equal, serving_admission=false | PASS |

These counts identify separate checks; they are not added together.

## Commands and retained evidence

Unit invocation from the source worktree:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/glm53_dsv4/root_attention_input_4a1b53e6_units.xml
```

Strict comparison (no new capture):

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/glm53_dsv4/root_attention_input_4a1b53e6_compare --baseline-label /data/ycfeng/tmp/glm53_dsv4/prefill_contract_576d278_20261023/candidate --candidate-label /data/ycfeng/tmp/glm53_dsv4/v4_attention_input_4a1b53e6_20261023/candidate
```

Actual CLI preparation and explicit/legacy lookup used the commands in owner `input_bundle/test_report_2026-10-23_input_bundle.md:34–37`, with root output `/data/ycfeng/tmp/glm53_dsv4/root_attention_input_4a1b53e6_checks/`; the automatic call uses the actual candidate02, not the fixture. Its complete argv, timestamps and evidence digests are in `input_admission.json`.

Independent representation command:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/root_attention_input_representation_check.py
```

`representation_verification.json` pins that script, source, unit receipt, matrix result, lookup outputs and source review. `input_admission.json` binds the separate external review and actual candidate admission. The original 11 candidate files were unchanged; only the new admission file was added.

## Root-check corrections and limits

Three root assertion errors were diagnosed before correction: (1) requiring the fresh sidecar's serialization to match despite unordered metadata-column keys; parsed JSON and all original files were exactly equal; (2) comparing complete legacy error strings despite fresh temporary directory names; every other field was equal; (3) comparing `identical_cases` (a list of 77 IDs) to integer77 instead of taking its length. These errors were in the ad hoc root checks, not the source or CLI execution. No measurement, source, old output or test control was altered to pass.

This admits a private isolated-input component only. Complete kernel ownership, GPU dependency accounting, native graph/TP8 profiling, serving totals and four-metric calibration remain unfinished. The collector is a separate workstream.
