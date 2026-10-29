# D29 CPU performance component evidence

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive existing parent JUnit/log evidence,command/environment provenance,owner performance observations and pending combined gates. |

Task ID: D29-PERFORMANCE-COMPONENT-ARCHIVE.
This archive ran **no tests,pricing probes,source scans or dependency installs**.
It read owner records and existing parent logs,parsed/rehashed five existing
JUnit files and checked only its owned documentation before committing.

## Accepted revisions and environments

| Component | Code commit | Records commit(s) | Execution workdir |
| --- | --- | --- | --- |
| Group | 3763e5b87a845eac186cfd481f0b56cffb2dd045 | 88fe6c65db675609ee48043ca63b7673c5f739c8;37c7832ee5ee2e9588ba6fed411a9e9894949ab8 | `/data/ycfeng/Frontier/.worktrees/step5-d29-group-reuse` |
| Loader | ee7e66ee439054978897c91fa32598fc13804883 | 35d2f74ed60855e48e5af4ea97c49721b9aba7ea | `/data/ycfeng/Frontier/.worktrees/step5-d29-loader-family` |
| Parser | 51e1f7205d07b269db2291c5f8076431a86d29e5 | 0fa22e49417b80ee315518f9ea6843b77663fe09 | `/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-parse` |

Primary reports all three trees clean and code byte-identical to code-only
commits. This is parent-supplied identity evidence,not a fresh archive scan.
Group test/pricing execution remains bound to88fe6c65,not relabelled as a rerun
on the later records-only37c7832e.

Four parent selections (group shared,group trainers,parser and loader MoE/routing)
use `/data/ycfeng/envs/frontier-py310/bin/python`,Python3.10.6:

```text
PYTHONPATH=$PWD:$PWD/tests/unit
PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
TMPDIR=/data/ycfeng/tmp
FRONTIER_TMP_ROOT=/data/ycfeng/tmp
python -B -m pytest -q -p no:cacheprovider
FRONTIER_TEST_CPU_WRITER_FILE and FRONTIER_TEST_CPU_WRITER_SHA256 unset
for selections using the actual current-repo writer.
```

Group shared timeout300s;other selections120s. Owner records identify existing
NumPy2.2.6,pandas2.3.3 and scikit-learn1.7.2;these dependency versions were not
freshly queried by the archive. No conda environment was changed.

Parent loader governance uses
`/data/ycfeng/envs/asyncopd-py312/bin/python` (existing Python3.12.13):
`PYTHONPATH=$PWD:$PWD/tests/unit:/data/ycfeng/tmp/d29_py312_cpu_extras01/site`
and `VLLM_TARGET_DEVICE=cpu`,with no new installation. This is distinct from
the owner's earlier same-ABI Torch supplement and failed initial Python3.12
attempt;their exact commands/failures remain in the unmodified owner report.

## Parent JUnit receipts: expected versus observed

Expected:zero failures/errors in each selected regression;only the existing
Torch-dependent group tensor-interface skip. Each XML was reparsed and SHA256
rehashed during archival;all counts,times and supplied hashes match (PASS).
JUnit hostname is `kun-workspace-vgen2` for each receipt.

| Selection | Observed PASS / SKIP / FAIL / ERROR | JUnit seconds | XML path | Verdict |
| --- | --- | ---: | --- | --- |
| Group six-file shared selection | 81 / 1 / 0 / 0 | 164.431 | `/data/ycfeng/tmp/d29_group_reuse_parent01.xml` | PASS with existing skip |
| Separate group actual trainer/preflight and independent fit/cache | 2 / 0 / 0 / 0 | 10.327 | `/data/ycfeng/tmp/d29_group_reuse_trainers_parent01.xml` | PASS |
| Parser four whole files | 128 / 0 / 0 / 0 | 6.171 | `/data/ycfeng/tmp/d29_artifact_parse_parent01.xml` | PASS |
| Loader MoE/routing files | 29 / 0 / 0 / 0 | 32.316 | `/data/ycfeng/tmp/d29_loader_family_parent01.xml` | PASS |
| Loader typed governance,Python3.12 supplement | 52 / 0 / 0 / 0 | 19.366 | `/data/ycfeng/tmp/d29_loader_governance_parent01.xml` | PASS |

```text
d29_group_reuse_parent01.xml
a37987c7b72f64a87be43623b3136d93e12c8b103b00856cf525b8918177cb4f
d29_group_reuse_trainers_parent01.xml
6b411b2457883e46b96c136ffa5b43f420fcac465c8481bbe2ec798f3186e5d9
d29_artifact_parse_parent01.xml
33ffaf6016ec49a718c099c9f23f10700a5b6007e4edb235e07cca6749cd4fe9
d29_loader_family_parent01.xml
d7db803efc54dabf9b22e56e15d59413c5d644947901142c314aae40690fdacf
d29_loader_governance_parent01.xml
9a2c72314570ef9be14644c10a809811378471db81e43e5ebcb66e833488fdd1
```

Logs have the same directory/stem with `.log`. Parser log says
`128 passed in 6.17s`;loader MoE log `29 passed in 32.31s`;governance log
`52 passed, 2 warnings in 19.37s`,with Swig type deprecation warnings and a
shutdown `swigvarlink` deprecation warning. These are not code failures.
The group skip remains
`tests.unit.test_moe_position_routing.test_one_source_tensor_interface_preserves_masks_and_request_intervals`
because the Python3.10 environment lacks Torch.

The archive's exact receipt read/parse/rehash command,cwd
`/data/ycfeng/Frontier/.worktrees/step5-preview`,was:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
names = [
    'd29_group_reuse_parent01',
    'd29_group_reuse_trainers_parent01',
    'd29_artifact_parse_parent01',
    'd29_loader_family_parent01',
    'd29_loader_governance_parent01',
]
for name in names:
    path = Path('/data/ycfeng/tmp') / (name + '.xml')
    payload = path.read_bytes()
    root = ET.fromstring(payload)
    suites = list(root.iter('testsuite'))
    cases = list(root.iter('testcase'))
    skips = [case.attrib for case in cases if case.find('skipped') is not None]
    print(json.dumps({'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(), 'suites': [suite.attrib for suite in suites], 'testcase_count': len(cases), 'skipped_cases': skips}, sort_keys=True))
PY
```

Observed:exit0;five JSON output records exactly match the hashes and JUnit fields
listed above. This reads existing evidence only and does not execute pytest.

**Never sum overlapping selections.** Group81 is shared-only;the separate2
establish actual trainer/preflight and independent fit/cache evidence. Parser
focused/backend cases are subsets of128. Loader29 includes tests also used in
other selected commands;there is no combined unique-case total in this archive.

## Command provenance

Exact previously recorded raw group shared,actual trainers and pricing shell
commands are preserved in
`../resumed_parent/test_report_2026-10-23_group_reuse.md`. The two-test raw
command is also preserved in the owner's
`cli_group_reuse/test_report_2026-10-23_model_group_reuse.md`. No launcher,
test,script or command artifact was edited or executed by this archive.

For parser and loader parent runs,the primary supplied the following command
specifications rather than a raw serialized shell transcript. Preserve the
provided executable,argv,environment,timeout and artifact names;do not claim
that JUnit/log parsing independently recovered unrecorded launcher fields.

```text
Parser parent command specification
cwd: /data/ycfeng/Frontier/.worktrees/step5-d29-artifact-parse
timeout: 120s
executable: /data/ycfeng/envs/frontier-py310/bin/python
argv: -B -m pytest -q -p no:cacheprovider
  tests/unit/test_collective_sim_cc_backend.py
  tests/unit/test_op_trace_utils.py
  tests/unit/test_ep_wave_trace_context.py
  tests/unit/test_comm_operator_families.py
  --basetemp=/data/ycfeng/tmp/d29_artifact_parse_parent01
  --junitxml=/data/ycfeng/tmp/d29_artifact_parse_parent01.xml
stdout/stderr: /data/ycfeng/tmp/d29_artifact_parse_parent01.log
environment: Python3.10 contract above;actual-writer overrides unset.

Loader MoE/routing parent command specification
cwd: /data/ycfeng/Frontier/.worktrees/step5-d29-loader-family
timeout: 120s
executable: /data/ycfeng/envs/frontier-py310/bin/python
argv: -B -m pytest -q -p no:cacheprovider
  tests/unit/test_moe_grouped_gemm_padded_features.py
  tests/unit/test_moe_routing_runtime.py
  --basetemp=/data/ycfeng/tmp/d29_loader_family_parent01
  --junitxml=/data/ycfeng/tmp/d29_loader_family_parent01.xml
stdout/stderr: /data/ycfeng/tmp/d29_loader_family_parent01.log
environment: Python3.10 contract above;actual-writer overrides unset where applicable.

Loader typed-governance parent command specification
cwd: /data/ycfeng/Frontier/.worktrees/step5-d29-loader-family
timeout: 120s
executable: /data/ycfeng/envs/asyncopd-py312/bin/python
argv: -B -m pytest -q -p no:cacheprovider
  tests/unit/test_profiling_governance_minimal_red.py
  --basetemp=/data/ycfeng/tmp/d29_loader_governance_parent01
  --junitxml=/data/ycfeng/tmp/d29_loader_governance_parent01.xml
stdout/stderr: /data/ycfeng/tmp/d29_loader_governance_parent01.log
confirmed environment:
  PYTHONPATH=$PWD:$PWD/tests/unit:/data/ycfeng/tmp/d29_py312_cpu_extras01/site
  VLLM_TARGET_DEVICE=cpu
no new installation
```

Owner exact RED/GREEN,affected-suite,baseline comparison and bounded probe
commands remain unchanged at the absolute owner report paths below. Do not
replace their60s/RLIMIT_AS30GiB commands with parent120s regression commands,
or erase their failed systemd guards/environment attempts.

- `/data/ycfeng/Frontier/.worktrees/step5-d29-group-reuse/task_memory/task_2026-10-22_step5_position_routing/cli_group_reuse/test_report_2026-10-23_model_group_reuse.md`
- `/data/ycfeng/Frontier/.worktrees/step5-d29-loader-family/task_memory/task_2026-10-22_step5_position_routing/init_family_repair/test_report_2026-10-23_init_family_repair.md`
- `/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-parse/task_memory/task_2026-10-22_step5_position_routing/artifact_parse_repair/test_report_2026-10-23_artifact_parse.md`

## Performance observations and provenance

Group parent verification:original P6128+idle1 point,without diagnostic
`--reuse-groups`,cold0.924303847s/warm0.354695124s,88/0 descriptions,exit0.
Expected unchanged cost/proof/full-plan-key payload;observed SHA256
`d2c141f088b52063edb539094fe3e0967f35e53303d70ebe2e8fbdbe4c86c952`,equal to
the pre-fix parent payload (PASS for that component point). Exact raw command,
script/launcher hashes and result JSON hash remain in the parent group report.

Loader prior parent3000-row typed-map control:2.837165373s baseline to
0.274640171s hoisted,one derivation,equal payload SHA256
`f460359008814440c80a40fd2e97b557e7bcbb802c72a4076c6ebe3c8a2cdd3f`.
Raw command remains in
`../resumed_parent/test_report_2026-10-23_input_validation.md`. This is prior
bounded diagnostic evidence,not a fresh full-loader/startup timing.

Parser **owner evidence only**,not an integrated parent reproduction:steady
baseline0.380598s to actual-source0.144797s;20 reads remain20,large decodes
20 to1. Complete cost/proof/plan payload remains byte-identical,full-file SHA256
`c2aa250d2103db50140e3f19c3579e9221dcfaf6251df1b44ed5f580982c5582`.
Resident pair18,617,675 bytes;peak RSS baseline1,195,412KiB/source1,212,976KiB.
Exact source timing was observed with the owned diff on24f970cb before its
code-only commit;it is not relabelled as a clean committed-source parent run.
The owner's raw timing/control/source commands and residency method remain in
its report. Parent128PASS checks correctness,not this timing claim.

None of these short-loop timings predicts completed full6144/1024 offline/
online CLI time. No error-threshold or native/GPU qualification is inferred.

## Archive documentation verification

The owned-document check parsed `primary_acceptance.json` and confirmed three
accepted components,152 step gross,separate0-finding review axes,pending actual
merged audit,false final accounting acceptance and all workflow gates pending.
It checked only the six owned files for final newline,trailing whitespace,
Markdown Modification History and balanced fenced blocks:all PASS.
Exact final syntax/whitespace commands,cwd the archive worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m json.tool task_memory/task_2026-10-22_step5_position_routing/resumed_performance/primary_acceptance.json > /dev/null
git diff --check -- task_memory/task_2026-10-22_step5_position_routing/resumed_performance/primary_acceptance.json task_memory/task_2026-10-22_step5_position_routing/resumed_performance/review.md task_memory/task_2026-10-22_step5_position_routing/resumed_performance/test_report_2026-10-23_components.md task_memory/task_2026-10-22_step5_position_routing/resumed_performance/summary.md task_memory/task_2026-10-22_step5_position_routing/resumed_parent/review_2026-10-23_group_reuse.md task_memory/task_2026-10-22_step5_position_routing/resumed_parent/test_report_2026-10-23_group_reuse.md
git diff --cached --check
```

Expected and observed:exit0,no JSON or whitespace errors. Staging is limited to
the six owned files. Existing unrelated primary record edits remain unstaged.
These are documentation checks,not new component regression/source scans.

## Scope and remaining gates

| Component | Source gross | Test gross | Step total / ceiling | Retained paths / ceilings |
| --- | ---: | ---: | --- | --- |
| Group | 17 | 54 | 71 / 105 | 143/175;111/128;254/260 |
| Loader | 4 | 34 | 38 / 45 | 13/17;231/237 |
| Parser | 8 | 35 | 43 / 50 | 84/93;233/233 |

Three-step churn152/200;source29,test123. Expected combined retained global
gross9254 = existing9116 + retained increments59+38+41. This is arithmetic
from component receipts only:actual merged per-path/global audit is **pending**,
not accepted. Do not add step churn152 to the prior retained total.

Standards and Spec are independently closed No findings,0 findings/worst none;
group records P2 is closed after37c7832e. Primary accepts components only.
Integration,final freeze,integrated parent performance probe,full CLI,fresh
strict77/full-unit,T2/native/GPU/G3/E2E remain pending for the combined source.
Historical e41 receipts keep their original revision. No extra early-init fix
is approved;partial diagnosis cannot authorize changed cache hashes/filenames.
