# D29 native-key verification report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Accept independent clean scope9580 and integrated c668a938 after124-binding/33-artifact primary verification; retain all frozen/raw verdicts. |
| 2026-10-24 | Compare8620unitIDs/states with no change,retainrawexit1;sourcecommit376983d2andclean frozen execution follow the precommit checks. |
| 2026-10-24 | Record corrected canonicalwavePASS,9580scope/124bindings and the expectedraw precommitprovenanceFAIL;final frozen gates stay pending. |
| 2026-10-24 | Add observed RED/GREEN,focused status comparison and exact-expression timing; retain first wave-preparation failures and bound the corrective replay. |
| 2026-10-24 | Record the passing allocation check; implementation/runtime checks are pending. |

## Allocation check

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-preview`.
Source/records HEAD: `18cc3e9e1008d796a08f3b5637c9b10311979735`, with only the
scoped task records modified. Invocation timeout20s; small JSON/TSV reads only.

Exact command:

```sh
python3 -B - <<'PY'
import csv, json
from collections import Counter
from pathlib import Path
root = Path('task_memory/task_2026-10-22_step5_position_routing')
proposal = json.loads((root/'repairs/native_collective_key_scope_amendment.json').read_text())
rows = list(csv.DictReader((root/'repairs/evidence/named_query_scope/named_query_paths.tsv').open(), delimiter='\t'))
by_path = {row['path']: row for row in rows}
assert len(by_path) == proposal['effective_unique_paths'] == len(rows)
before = Counter()
after = Counter()
actual = Counter()
for row in rows:
    before[row['owner']] += int(row['cap'])
    after[row['owner']] += int(row['cap'])
    actual[row['owner']] += int(row['actual'])
assert dict(before) == proposal['owner_gross_ceilings'], dict(before)
assert sum(change['delta'] for change in proposal['path_ceiling_adjustments']) == 0
for change in proposal['path_ceiling_adjustments']:
    row = by_path[change['path']]
    assert int(row['cap']) == change['before']
    assert change['before'] + change['delta'] == change['after']
    assert int(row['actual']) <= change['after']
    after[row['owner']] += change['delta']
assert before == after
assert sum(proposal['path_step_ceilings'].values()) == proposal['repair_delta_ceiling'] == 3
assert proposal['source_delta_ceiling'] + proposal['test_delta_ceiling'] == 3
print(json.dumps({'verdict':'PASS_ALLOCATION_ONLY', 'allocated_paths':len(rows), 'caps_before':dict(before), 'caps_after':dict(after), 'accepted_actual':dict(actual), 'accepted_source':sum(actual.values()), 'accepted_global':sum(actual.values())+14, 'step_gross_ceiling':3, 'projected_growth':1, 'donor_actual':int(by_path['frontier/cc_backend/collective_plan.py']['actual'])}, indent=2))
PY
```

Expected:114 unique rows, unchanged owner ceilings, exact before/after values,
zero-sum allowance, donor code usage within139 and at most3gross/1net change.
Actual: **PASS_ALLOCATION_ONLY**, exit0; owner caps4404/3956/1582 and accepted
actual4261/3954/1350, source9565/global9579. The fixed doc12/gitlink2 contribution
is retained from the accepted ledger. No successor source or performance result
has been measured by this command.

## Implemented delta and RED/GREEN

Candidate directory: `/data/ycfeng/Frontier/.worktrees/step5-d29-native-cache-key`.
Allocation HEAD98705977,baseline18cc3e9e; source is not committed. Three gross
changed lines/net1: backend1add/1delete,test1add. Exact AST check: only
`("plan",plan.cache_key(),json.dumps(artifact,sort_keys=True))` becomes
`("plan",plan.cache_key(),self.collective_profile_bytes())`; every other module
AST node is identical. Fixed-base1aee0af1 numstat is backend93/0,test237/0,
plan139/0,all within amended caps. Whole114-path successor audit is still pending.

BackendSHA256 `e4a77ab21176e4e29fe65e6cdae1919a3ab2a9a659ea3d9bc49408a06a037cb5`;
testSHA256 `f20ecc4ba95b2ff5d63c6ae199774bff27642d509f26f5a56e7fb2190c7fd041`.

Scratch root `R=/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb`.
Actual RED:1FAIL/rawexit1; GREEN:1PASS/rawexit0,through the existing test
`tests/unit/test_collective_sim_cc_backend.py::test_importer_groups_frozen_policy_but_keeps_complete_count_and_cache_identity`.
Expected RED is the old private native-key token beingstr,not the exact heldbytes.
Existing synthetic native-schema fixture labels are not actual GPU measurements.

Common exact execution prefix for these checks (ROOT isR above; SOURCE is each
listed source directory) is:

```sh
timeout SECONDS sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=SOURCE TMPDIR=ROOT XDG_CACHE_HOME=ROOT/cache MPLCONFIGDIR=ROOT/mpl NUMBA_CACHE_DIR=ROOT/numba /data/ycfeng/envs/frontier-py310/bin/python -B
```

RED/GREEN useSECONDS60 followed by `-m pytest NODE -q -p no:cacheprovider
--basetemp=ROOT/{red,green}_tmp --junitxml=ROOT/{red,green}.xml`;stdout/stderr
are preserved inROOT/{red,green}.log. The red run contains the one test assertion
but the old backend expression;the green run uses both candidate files.

## Focused status comparison

Using the same prefix withSECONDS180,each source runs:

```sh
-m pytest tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_moe_pricing.py tests/unit/test_step5_contract_reuse.py -q -p no:cacheprovider --basetemp=ROOT/focused_CASE_tmp --junitxml=ROOT/focused_CASE.xml
```

BaselineSOURCE=`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-08`,
candidateSOURCE=the isolated directory above. Both actual124PASS/1SKIP/rawexit0.
The unmodified skip is the existing absentTorch case. XMLclassname/name and
PASS/FAILURE/ERROR/SKIPPED dictionaries compare exactly:0missing/0new/0changed.
BaselineXMLSHA256 `ff76b1bb1ad29751790917d5435fccf3cc507f4d3f79c442eaf893588cdefb47`;
candidateXMLSHA256 `b8c9447de1c690fce5864f444713ce58a43a63c2bc60a860aa050ea04ab3822e`.

## Key-expression-only paired measurement

Exact script: `R/key_seam.py`,SHA256
`b98e265372dc92066a6b80a636f2c84a081f609e0b4231bfb1be9c4186fa09b8`.
Command uses the above60s/30GiB/runuser/BLAS1 prefix withoutPYTHONPATH/cache
overrides,then `R/key_seam.py`,redirected toR/key_seam.log.
Python3.10.6/CPython,executablefrontier-py310;GCenabled throughout. The script
compiles the exact old/new native-key AST and existing getter/LRU-get method
bodies; retained plan-key strings and placeholder cached values isolate only
this work. No qualification,scenario,count or numerical simulation executes.

Input: unchanged
`/data/ycfeng/tmp/d29-parent-position-fixture03/fixture_collectives/h800/collective_local.json`,
8,033,853B,SHA256 `447e697340d77c205e7457eb50c7b63c47ab103ba77c7f52e7c5fe844498ca90`,
explicitcpu_fixture/native_admittedfalse. Eight distinct keys,64warmLRUlookups
per sample,12alternating paired samples per variant;all24rawsamples are retained
inR/key_seam.json. Source and original file bytes,size andmtime are unchanged.

| Observed metric | Baseline | Candidate |
| --- | ---: | ---: |
| Median wall seconds/64lookups | 2.806010747 | 0.000094784 |
| Mean wall seconds/64lookups | 2.833245370 | 0.000093234 |
| Median CPU seconds/64lookups | 2.805994594 | 0.000095001 |
| GC collections/time | 0/0 | 0/0 |
| Unique artifact-token objects | 8 | 1 |
| Additional retained token bytes | 64,271,208 | 0 |

The candidate's one8,033,886B bytes object is already held by the input owner.
These are key-seam observations,not native prediction latency,full-wave speed,
total process peak-memory or successfulGPU/native/CLI qualification.

## Actual-wave preparation failure and correction

First baseline and candidate attempts both exit1 during initial binding with
FileNotFoundError for `R/wave-CASE/baseline/collective-fixture/h800/collective_local.json`.
The new preparation omitted the copied input. PreserveR/wave_baseline.log and
R/wave_candidate.log; neither pricing nor emission ran. This is a harness
preparation failure,not a production-regression finding or an accepted waveform.

Correction: retainedrun.py,fixture_wiring.py,install_control.py andcontrol.json
remain byte-identical to the existing registry baseline driver. `R/prepare_wave.py`
creates freshwave-CASE-02roots,adds an exact SHA-verified original-input copy,
and updates only source/root/hash bindings. Each source still uses thebaseline
label,so diagnosticinstall returnsappliedfalse. `wave_preparation02.json` records
these explicit adaptations. Original source,body,fixtures and first failures stay.

Bounded next command: one corrected invocation per source,60s/MemoryMax30G,
frontier-py310-B/BLAS1,using `R/wave-CASE-02/run.py baseline`;fresh logs
R/wave_CASE02.log. Expected208events/fullcosts/proofs/keys/order,288numerical/
144freshproof/64collective/16pointcalls,one binding read and0cold/warmreads;
full emitted/ownership/contexts files compare bytes,latefailurebody compares
all3902B,profilebodyempty,raw auxiliary headers retained unnormalized.

## Remaining checks

Corrected canonicalwave and primarycomposed scope check have passed below;
independent review and required complete/frozen regression receipts are pending.
Unit andmatrix precommit commands were launched
through `R/precommit_checks.py`;their launch/execution envelopes retain exact
commands,Python/dependency versions,source hashes and raw exits. Precommit dirty
source is not a clean frozen-target admission. Sourcecommit/integration and
native hardware,wholeCLI andcleanE2Equalification are not complete.

## Corrected canonical-wave observation

Second baseline/candidate invocations eachrawexit0,without source/body/wiring
changes or any installed JSONcontrol. Exact driver files remain those copied
in`R/wave_preparation02.json`;input copies are8,033,853B with the fixed SHA.
`R/compare_wave.py` (SHA256
`f23103b61ae669d51386f51467acdb34fb8b18103a2d29f03e40e939b34abf02`)
runs through `timeout60s sudo-n systemd-run--scope--quiet-pMemoryMax=30G
runuser-ubrainpp--python3-B R/compare_wave.py`. Actualrawexit0,verdict
PASS_CANONICAL_CPU_WAVE_AND_PROTECTED_BOUNDARY.

- 208events,64collectivecost objects,144freshproofs;metadata/proofs/order match.
- 19whole-file comparisons:17pinned emitted/ownership/context traces plus
  ledger andpricingpayload. PayloadSHA
  `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`.
- All64wrong-key negatives andgetter/pending mutable snapshot isolation pass.
- Latefailure restores all3902bodybytes,SHA
  `f5e7f5a67660096a71d14abfe690dbc432d09a480c85c9999dea0cd16979f1ec`;
  profilebodyempty. Auxiliaryheaders preserve real timestamps and are not
  claimed whole-file equal;header schema/non-timestamp metadata match.
- Each source retains288numerical/144freshproof/64collective/16pointcalls,
  192livekeycalls,loads1136/dumps704;onebindingread,0cold/warmreads,
  cold8/warm0/profile0/emission0keyencodes.
- Each source keeps23filebindings and originalfixturemetadata/input hashes.

Selected machine-readable result:R/wave_comparison.json;raw source-bound
wave reports underR/wave-{baseline,candidate}-02/baseline/wave_result.json.
This is the canonical cachedFP8DECODE/layer60/16EP CPU fixture,notP6128/D16/
BF16/fullSimulator/native/CLI qualification. The changed native-key branch is
not entered here,so no whole-wave speedup is claimed.

## Primary composed successor accounting

`R/scope_check.py`,SHA
`80614bb67e6d0acc618518482bca5d7af3fead57de9bfc20351f9052d699040f`,
runs through the same60s/systemd30GiB/runuser/Python3-B envelope. It reuses the
unchanged accepted TSV/receipt/manifest, checks the exact two-path name set,
recomputes all105superproject and5submodule path usages,retains the4native
patch row billings after fresh base/derived hashes,and applies the zero-sum
native-key cap overlay. No dataset/trace scan or live image inspection occurs.

ActualPASS_COMPOSED_SCOPE_WITH_DIRTY_THREE_LINE_PATCH/rawexit0/0violations:
114allocated/113changed,CORE4261/4404,PRICING3955/3956,native1350/1582,
source9566/9942/global9580/10000. Submodule333/nativepatch99/doc12/gitlink2
unchanged. All124bindings rechecked;122non-key bindings have equal baseline/
successor digest
`eaf97ec1746ad70a1e2bb97fcd9d9e4a8f176dcb166c09c65e6ac26d74ba01b6`.
Backend93/93,test237/237,donorplan139/139;donorcode unchanged. Artifact
R/scope_check.json contains every path/binding and the explicit limits.
This primary accounting check is not independentreview,final frozen acceptance,
live native derivation or GPU qualification.

## Precommit matrix: observations and raw provenanceFAIL

`R/precommit_checks.py matrix` reuses the existing read-only namespace and
unchanged matrixrun/compare commands,with only source/output/environment
bindings changed. Exact argv and versions are inR/precommit-matrix/{launch,
execution}.json;source before/after is identical and SHA-binds the two own
working-tree files. Actualrunexit0/compareexit1. The retained comparator prints
77/77identical,0mismatch/missing/evidence/definition/cachefilename findings,
then correctly rejects dirty98705977 as non-commit-specific provenance.

PreserveR/precommit-matrix/compare.log andcomparison.json;this is **not** an
accepted strict77PASS. Pending:complete rawunit/status comparison,verifiedcode
commit,cleanfrozen matrix andoriginalauditor rerun,independent review and
integration. No source-provenance predicate or comparator is relaxed.

## Precommit complete-unit comparison and source commit

Actualpytest counts8028PASS/279FAIL/32ERROR/281SKIP,rawexit1. The preceding
dbe19f53 baseline counts and all8620JUnitidentities/states match exactly:
0missing/0new/0changed. OriginalbaselineXMLSHA
`64371d6394026aedb9fa8d9c409a9263bfc2f23d72a063886581b61827b8b5b3`;
candidateXML2,120,475B,SHA
`8dd3c42c6f12847d967413e266722f56bc14dd1931fb47bab8c8fee36996f102`.
Source_before/after hashes match; canonicalnativebinary remains absent.
Rawexistingfailures/errors/skips and the older79approved-but-unwaived ID
replacement are not normalized,aliased,waived or reported as alltestsPASS.

`R/compare_unit.py` first fails before any comparison withSyntaxError because
`pass` was used as a dictionary keyword. Repair only thatliteral,preserve
R/unit_comparison.log,thenrun the60s/systemd30GiB/runuser/Python3-B envelope
withfreshR/unit_comparison02.log. Actualexit0,
PASS_ALL_RETAINED_UNIT_IDS_AND_STATES;scriptSHA
`306d71b396cea3b17eef61e68d8d3aa747a8ccb1c4879e3025872de2a140e98b`.
R/unit_comparison.json retains complete identity/status and inputfile evidence.

After these checks,commit only thetwo ownedsource/test paths as376983d2,
“Reuse run-fixed bytes in native collective cache keys” (2add/1delete).
Gitdiff--checkpasses; no taskrecord,donor or unrelatedcode enters this commit.
This sourcecommit is isolated,not primaryintegrated or native/CLIaccepted.

## Clean frozen follow-up envelope

Run `python3-B R/frozen_checks.py {unit,matrix}` withfreshR/frozen-CASE roots.
The wrapper adapts the unchangedretainedrunner only at outputpaths and strict
clean376983d2/sourceSHA guards plus the provenance text. It does not alter
namespace,math/measurement configuration,unit/matrix argv,comparator,presence
gate,canonicalmissingbinary or output. Each maincommand1800s/MemoryMax30G,
matrixcompare180s,BLAS1/frontier-py310-B. Exactcommands/sourceversions/launch
bindings are emitted before execution. Source must stayclean and hash-identical
through both runs. Both runs and their comparisons are complete below;
independent final accounting/review and integration remain pending.

## Observed clean frozen results

Source: `376983d2ecc441d1e0a8cb8b943e65a2dcb217b5`, tree
`c8f8657f650e4645ec0fc0c86b84283766c09b3c`, submodule
`3dac53c544e546574cb6e8078cd14053d0916113`. The exact two changed source/test
hashes and clean before/after state are in `frozen-{unit,matrix}/execution.json`.

Commands, with `R=/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb`:

```bash
python3 -B "$R/frozen_checks.py" unit > "$R/frozen_unit_launcher.log" 2>&1
python3 -B "$R/frozen_checks.py" matrix > "$R/frozen_matrix_launcher.log" 2>&1
timeout 60s sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- python3 -B "$R/compare_frozen_unit.py" > "$R/frozen_unit_comparison.log" 2>&1
python3 -B "$R/run_frozen_audit.py" > "$R/frozen_matrix_audit03_launcher.log" 2>&1
```

The frozen runner retains the recorded1800s/30GiB envelopes and canonical
missing native binary. Python3.10.6/pytest9.1.1, BLAS/OMP/MKL/NumExpr threads1,
`-B`, writable scratch and read-only source are unchanged. Torch/aiohttp remain
absent. Full argv/environment/version records are in the execution receipts.

| Check | Passing condition | Observed result | Verdict |
| --- | --- | --- | --- |
| Full-unit retained identities | No missing/new IDs or changed statuses against dbe19f53 | All8620 equal;8028PASS/279FAIL/32ERROR/281SKIP; raw pytest exit1 | PASS identity preservation; not a clean unit suite |
| Strict77 comparison | Complete77-case comparison with zero differences and clean committed provenance |77/77 identical; zero numeric/provenance/evidence/cache/definition differences | PASS |
| Strict77 retained evidence | Unchanged artifacts and cache inventory in each actual run namespace |451 artifact hashes and519 cache filenames per side | PASS |
| Environment | Baseline/candidate/actual versions equal | All recorded package versions equal | PASS |
| Source guard | Exact candidate commit/tree/submodule and two blobs unchanged before/after | Clean376983d2 and unchanged source hashes | PASS |

Candidate unit XML SHA256:
`bb1703a9cb5ca0d28f4fbdb732b391c8f9d38717330e375d48e6e098fe8df344`.
The frozen unit comparator changes only the retained input/output paths and
clean-source description, with explicit before/after state assertions. It keeps
the original comparator and all raw test failures. The previously approved79
metadata-ID absence remains unwaived; no new alias or exemption is introduced.

### Retained auditor failures and correction

The first auditor command used the same180s/30GiB/Python3.10.6 envelope, but ran
outside the generated repository namespace. The unchanged auditor rehashed451
artifacts per side and found no77-case differences, while both manifest-absolute
cache directories were absent in that namespace. Its raw verdict was correctly
FAIL (`cache_inventory_changed=true` on both sides). Retain
`frozen_matrix_audit.log` and `frozen-matrix/postrun_receipt.json` unchanged.

The next command, `python3 -B "$R/run_frozen_audit.py"` with output directory
`frozen-audit-02/`, tried mounting only the baseline cache. It failed before
auditing: `bwrap: Can't mkdir .../verify-step5-d29-integrated-08/cache: Read-only
file system`. The existing root mount contains no cache directory to mount over.
Retain that launch receipt and log, and do not label it a test failure.

The corrected command uses a new `frozen-audit-03/` directory and mounts the
baseline's entire generated repository read-only at its original absolute
location, alongside the retained candidate namespace. Only output-directory
and mount bindings change; `audit_evidence.py` and its PASS predicates remain
unchanged. `frozen-audit-03/launch.json` records exact argv and namespace mapping.
The unchanged comparison SHA256 remains
`02a558b25c3184b87b0bd32ed9a2aa626811c13b58c932b2edc5044b0e950b19`.
The corrected audit and package-version check both PASS. No matrix case or
training/simulation run is repeated during these auditor corrections.

These checks settle frozen behavior preservation only. Synthetic native-schema
tests are not hardware admission, and the canonical CPU wave does not enter the
changed native-key branch. Independent final accounting/review and integration
are completed below; native/GPU/G3/E2E and the2400s whole-CLI gate remain separate.

## Independent clean audit and primary integration

The independent audit on clean376983d2 passes with zero violations. Its9 bound
artifacts plus the pinned manifest are byte-preserved under evidence/frozen_scope/;
owner1dd96179/65a2766d are integrated as04d4319a/9584941c. Source376983d2 is
integrated as c668a938. Both independent review axes have No findings.

Primary command, from `/data/ycfeng/Frontier/.worktrees/step5-preview`:

```bash
timeout 60s sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/native_cache_key/verify_integration.py > /data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/primary_integration.log 2>&1
```

Python3.10.6, no project imports or dataset/model/trace scan; no suite/probe rerun.
Actual exit0, PASS_PRIMARY_SOURCE_AND_FROZEN_BINDINGS:

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Integrated source | All non-task source equals clean376983d2; exact two-path source commit | c668a938, zero non-task differences | PASS |
| Frozen evidence |9 scope artifacts and24 frozen artifacts/scripts retain hashes | All33 checked | PASS |
| Source/native bindings |124 exact tree/file/pin bindings;122 non-key digest unchanged |124, digest eaf97ec1…74ba01b6 | PASS |
| Cumulative two-path/donor bill | Backend93, test237, donor139; donor code unchanged |93/237/139 | PASS |
| Independent full allocation |114 allocated/113 changed; all caps/17 zero-sum overlays valid | CORE4261/4404, PRICING3955/3956, native1350/1582; source9566/9942, global9580/10000 | PASS |

The primary receipt is evidence/primary_integration.json; it records exact
source/tree, interpreter and script SHA256. Tracked source/index are clean;
unrelated untracked.cortexkit/.pi files are preserved. Frozen8028PASS/279FAIL/
32ERROR/281SKIP(rawexit1) and strict77/77 are carried by exact source equivalence,
not described as newly rerun or as an entirely passing unit suite. Native
hardware and the full6144/1024 CLI within2400s remain unaccepted.
