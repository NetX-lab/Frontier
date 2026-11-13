# D29 exact-text artifact reuse verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record independent exact-text control,actual source loop,RED/GREEN and exact accounting. |

TaskID D29-COLLECTIVE-ARTIFACT-PARSE. Workdir
`/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-parse`;immutable source
`24f970cbd6cae059467f68e0e1cf771b07aeecfe`. Python3.10.6 from
`/data/ycfeng/envs/frontier-py310/bin/python`;existing env/dependencies,thread
caps OMP/OPENBLAS/MKL=1,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/data/ycfeng/tmp.
Normal constructor bootstraps independent local submodule
`3dac53c544e546574cb6e8078cd14053d0916113`;no predictor/bootstrap mock.

## Scope and frozen inputs

Retained fixture collective artifact:
`/data/ycfeng/tmp/d29-parent-position-fixture03/fixture_collectives/h800/collective_local.json`,
8,033,853 bytes,SHA256
`447e697340d77c205e7457eb50c7b63c47ab103ba77c7f52e7c5fe844498ca90`.
CPU/unadmitted candidate168 counts,no models. Paired bundle:
`/data/ycfeng/tmp/d29-parent-position-fixture03/cache/bundle_3e46232621a39c1c5f598921435bd41da1a4c1915b424bc53f3eba09a5bb3e37.json`.
Existing main07 CLI config:
`/data/ycfeng/tmp/d29-parent-position-main07/offline/command.json`.
Original physical source q6128+idle1,gathered6129,layer3,EP0;four actual plans per
point,five repeated points. Real config/Cluster/backend/current prediction methods
and verified paired model/count loading;test predictor construction seam is inherited
from retained loop,not Simulator initialization/full workflow admission.

Original scratch script preserved byte-identically:
`/data/ycfeng/tmp/d29_collective_json_scope/loop.py`,13,384 bytes,SHA256
`7617211b60e0c3b9dfb48854f8155890b546509bd99abcd3b5ba0523698bfae4`.
New scratch `exact_text_loop.py`,14,097 bytes,SHA256
`0dfebf96ac7234f496c1ffb181528a0ba4397b7f8c1cbc424b61c206be7d8d61`.
The only baseline/control arm difference is artifact_exact_text. Source bound to
24f970cb in both;group reuse ON,no profile. Module-local json shim forwards all
other JSON functions untouched;module-local Path still performs every read.
Single latest text/parsed pair replaces the previous scout's stat cache. No global
json/pathlib mutation. Complete payload serialization occurs after measured loop
iteration timing;same instrumentation in both arms.

Bounded diagnostic:each loop script sets RLIMIT_AS=30GiB,outer timeout60s.
systemd-run --user --scope -p MemoryMax=30G timeout60s true failed (scope job);
systemd-run --scope failed interactive authentication. Therefore actual guard was
script RLIMIT_AS plus timeout,not a claimed systemd scope. Observed peak<1.2GiB.

## Ranked predictions and control result

Before control:1.redundant large decode dominates and equality reuse removes19
decodes without removing reads;2.CORE query/count scan remains after this local
fix;3.any changed checks/payload would disqualify the control. Predictions1/2
supported;all complete payload fields unchanged,not merely selected timings.
No new semantic tradeoff beyond explicitly approved additional bounded residency.

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_collective_json_scope/exact_text_loop.py --execution --reuse-groups --no-profile --iterations 5 --output /data/ycfeng/tmp/d29_collective_json_scope/exact_baseline01 > /data/ycfeng/tmp/d29_collective_json_scope/exact_baseline01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_collective_json_scope/exact_text_loop.py --execution --reuse-groups --no-profile --artifact-exact-text --iterations 5 --output /data/ycfeng/tmp/d29_collective_json_scope/exact_reuse01 > /data/ycfeng/tmp/d29_collective_json_scope/exact_reuse01.log 2>&1
```

| Actual observation | Baseline | Exact-text scratch | Actual source repair |
| --- | ---: | ---: | ---: |
| Iteration times,s | .439790,.382661,.379535,.376020,.386237 | .192310,.246225,.143172,.140703,.138683 | .191362,.253107,.145528,.146009,.142854 |
| Mean,s | .392849 | .172219 | .175772 |
| Steady mean iterations3–5,s | .380598 | .140853 | .144797 |
| Artifact reads |20|20|20|
| Reads,total s |.212310|.170699|.180754|
| Artifact decodes |20|1|1|
| Artifact decode,total s |1.162125|.034984|.034672|
| All backend JSON loads |80|61|61|
| Explicit shim reuse hits |0|19|0 (source bypasses decoder on19 reads)|
| Peak RSS,KiB |1,195,412|1,213,336|1,212,976|
| RSS before loop,KiB |1,177,424|1,177,608|1,176,512|
| Retained text,bytes |0|8,033,950|8,033,950|
| Deep unique resident pair,bytes |0|18,617,675|18,617,675|

Control steady2.702× faster/62.99% lower;source steady2.628×/61.96% lower. These
are fresh exact-text results,NOT scout .354→.097 stat timings. Reads remain live;
minor read-time variation is not attributed to optimization. Pair measured by
recursive sys.getsizeof with unique-object IDs includes text/parsed containers/
strings/numbers,not allocator overhead. Peak RSS delta17,924/17,564KiB versus
baseline is a process observation,not an isolated allocator theorem. Source
loop ran alongside required test checks,so exact timing is limited to this sample.

Payload PASS:complete_payload.json byte-equal across all three runs. Serialized
full-file SHA256 `c2aa250d2103db50140e3f19c3579e9221dcfaf6251df1b44ed5f580982c5582`;
extended canonical payload SHA256
`3888b075b5af3fd63fac1c7c0ef9ef27a4f65732a874742875c48f02cd3cd1a7`;
original times/proofs/keys payload SHA256
`d2c141f088b52063edb539094fe3e0967f35e53303d70ebe2e8fbdbe4c86c952`.
Includes every vars(CollectivePlanCost),exact plan/count receipts,numerical
composition,kernel counts,compute proofs and operator times. Equality asserted
within every run and across all arms,not inferred from exit status.

Actual source loop with shim reuse OFF:

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_collective_json_scope/exact_text_loop.py --execution --reuse-groups --no-profile --iterations 5 --output /data/ycfeng/tmp/d29_collective_json_scope/exact_production01 > /data/ycfeng/tmp/d29_collective_json_scope/exact_production01.log 2>&1
```

Before source SHA256 `27b575f6e34ef525f01f97435fa67e2184e59661659928e4d4e65e9c5a5013d3`;
repaired source SHA256 `49d98a212fa0ca8555da339d29acb0ad9f66b2b0432ff69f26044e1f8fa93e73`.
Source HEAD still24f970cb with explicit owned diff at this measurement;not
misreported as a clean source commit. Result JSON records HEAD and source SHA.

## Actual-seam RED/GREEN and negatives

New test uses real current-repo cpu_writer_payload and constructor,then observes
only backend-module bindings. RED:two equal plan calls read twice/decode twice;
expected two reads/one decode,1 FAIL/0 ERROR/SKIP,1.342s. Test was initially37
lines during RED preparation,reduced to35 before production implementation;
final retained regression stays within allocation. No code step was committed
outside its cap. GREEN1 PASS/0 ERROR/SKIP,1.285s. Regression proves:

- Equal current text skips parse but still reads;different count/key still fails.
- Same-size .03→.04 composed cost update with original nanosecond mtime is seen.
- Malformed current JSON throws twice and never replaces last successful pair.
- Restoring that exact last successful text returns .04 without another parse.
- FileNotFoundError and PermissionError objects are propagated unchanged on hits.
- Existing original same-instance native count rewrite test remains unchanged.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py::test_collective_artifact_reuses_only_identical_current_text -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_artifact_parse_red --junitxml=/data/ycfeng/tmp/d29_artifact_parse_red.xml > /data/ycfeng/tmp/d29_artifact_parse_red.log 2>&1
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py::test_collective_artifact_reuses_only_identical_current_text -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_artifact_parse_green --junitxml=/data/ycfeng/tmp/d29_artifact_parse_green.xml > /data/ycfeng/tmp/d29_artifact_parse_green.log 2>&1
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_comm_operator_families.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_artifact_parse_affected --junitxml=/data/ycfeng/tmp/d29_artifact_parse_affected.xml > /data/ycfeng/tmp/d29_artifact_parse_affected.log 2>&1
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_artifact_parse_backend --junitxml=/data/ycfeng/tmp/d29_artifact_parse_backend.xml > /data/ycfeng/tmp/d29_artifact_parse_backend.log 2>&1
```

Actual parsed JUnit:whole backend46 PASS (all45 old+new1),3.820s;whole affected
128 PASS/0 FAIL/ERROR/SKIP,6.113s=backend46+trace19+EP-context9+families54.
Focused1 and separate46 are subsets,not added into128. Existing native fit
cost-injection,CPU25 identity negatives,current-repo sealed writer,frozen runtime
grouping/native context/count checks remain untouched. Native local TP formula
executes real submodule;no GPU calibration inferred from it.

## Accounting,patch and unchanged semantics

Source+7/-1=8<=15;test+35/-0=35<=35;total+42/-1=43<=50. Retained source78→84/93,
test198→233/233;owner3879+6+35=3920/3956,remaining36. Global111 paths and10000
ceiling unchanged;root owns overlay,donor caps and final integrated accounting.
No source except allocated two files,new production flag,global/stat/hash cache,
numerical cost cache/key,CORE/shared API,read exception fallback or new validator.
Pair replaced only after json.loads succeeds;validation remains per call.

Exact two-path patch:artifact_parse_repair/source.patch (same as scratch
artifact_parse_source.patch),SHA256
`bcbc52c52e767eab4b69854c78a565ffb3a7e72116a79a2b13746b8f59ebc674`.
git diff --check and both Python AST parse PASS. Submodule clean3dac53c,gitlink
unchanged;no external fetch/build/commit. Original script/source data hashes
rechecked unchanged. No full CLI/GPU/platform/network operation or extra agent.

Limit:actual retained CPU/unadmitted partial point only,not offline/online full
run or native/G3/E2E qualification. CORE per-query artifact parsing/count scans
and native whole-artifact cache-key serialization remain unchanged/deferred.
No whole-workflow speedup or final freeze claim. Parent performs independent
review,integration,residual check,source freeze and required full runs.

Verified immutable source commit51e1f7205d07b269db2291c5f8076431a86d29e5
contains only the two allocated paths.
All method ASTs outside __init__/predict_collective_plan equal24f970cb;changed
method hunks are only pair initialization and current-text read/parse reuse.
Exact source.patch retained for independent review,records committed separately.
Raw source.patch is intentionally byte-identical to git diff,including the normal
single-space prefix on empty context lines. Staged git diff --check therefore
flags that artifact's line11 as trailing whitespace;source-only check and records
check excluding the raw patch PASS. Do not rewrite the review artifact to hide
this expected diff-format line.
