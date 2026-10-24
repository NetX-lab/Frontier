# D29 post-named-query regression test report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Complete8028/279/32/281rawunit,two-baseline joins and allafterguards without gatewaiver. |
| 2026-10-24 | Complete strict77source/scenario/artifact/cache/environment audit with unchanged original comparator. |
| 2026-10-24 | Completegovernance52PASS with existingCPUextras and exact46fe/parent52 IDs. |
| 2026-10-24 | Complete actual197/53PASS memberships against46fe/79 without raw-ID aliases. |
| 2026-10-24 | Close101 with actual XML101PASS/rawexit0 and exact preceding46fe identity/status. |
| 2026-10-24 | Bind required independent suite recipes and precise actual-result reporting. |

Frozen source dbe19f531ff68d9edafc480418e8927c40eb1c60/tree
6c43090097e92c6fc0c33aa7358e69c2cf092705/submodule3dac53c.
Canonical old binary absence preserved;no test/fixture or skip change.
Execution envelopes and exact underlying commands/env/packageversions are emitted
to each fresh suite launch.json/execution.json. Runner source digests and existing
unchanged recipes are bound in preflight;all raw exits/warnings remain intact.

Operational invocation: python3-B /data/ycfeng/tmp/d29_post_named_query_regressions_20261024/run_suite.py SUITE.
Inner command: sudo-n systemd-run--scope--quiet-pMemoryMax=30G runuser-ubrainpp
--env explicitPYTHONPATH/BLAS1/NUMEXPR1/PYTHONDONTWRITEBYTECODE1 bwrap read-only
source and writable unique scratch;timeout1800s for full-unit,retained caps for
others. Full-unitpytest tests/unit-q-pno:cacheprovider--continue-on-collection-errors,
unique basetemp/JUnit/log. Actual results are listed below;failedrawgates are not relabelled.

Separate outputs: /data/ycfeng/tmp/d29_post_named_query_regressions_20261024/;
strict /data/ycfeng/tmp/d29_workflow_fidelity_20261024_final07/;
unit /data/ycfeng/tmp/d29_unit_parent07.xml,.log and basetemp without suffix.

PASS conditions: strict77equality;no lost/changed46fe statuses,newIDsallPASS;
explicit79rawloss unchanged;no unexplained new failure after same-source101/52;
source/input/script/scope bindings unchanged. Oldrawexit1/skips/history remain.
No totals summed across overlapping suites orqualification beyond observedregressions.

## Supplement101 observed result

Actual XML101PASS/rawexit0,noFAIL/ERROR/SKIP. CandidateSHA256
1463ac9747cc4af431d881c917c60748c5032688c023a5e7273e396becced384.
All101IDs and statuses exactly equal46fe. No missing/newIDs orPASS-to-SKIP.
Rawreceipt: scratch/supplement101/{execution,comparison}.json with full exact
command,Python3.12/package versions and sourcebefore/after. This is the selected
same-source environment supplement,not a GPU/native/E2E qualification.

## Metadata/catalogue observed results

Metadata file selection actually197PASS/rawexit0,current XMLSHA256
69ce0103e28663ba975ba420c53e554983529802906e2c034ad724ffd42df80c.
All197IDs/statuses equal46fe;against79,exactlyold
tests.unit.test_collective_sim_cc_backend::test_collective_artifact_reuses_only_identical_current_text
is absent and new
tests.unit.test_collective_sim_cc_backend::test_collective_artifact_is_fixed_and_failed_initial_reads_are_not_published
passes. No commonstatus change orPASS-to-SKIP,raw79presence not waived.

Catalogue53PASS/rawexit0,XMLSHA256
56d26f9844c7e34ea482653f5426cefbc69364e5609c820b3e50efd1a73059ec.
All53IDs/statuses equal46fe and79. Both suites preserve source/root/3dacclean.
Exact commands/env/versions and comparisons retained in per-suite receipts;
these independent counts are not added to101 orfull-unit counts.

## Governance52 observed result

Actual52PASS/rawexit0,currentXMLSHA256
c146cb4b6fbe92dc974d16d63b6a98978d20bc13882ceacb6696532aa69592e4.
Exactlysame52IDs/statuses as preceding46fe andaccepted52parent;no missing/newIDs
orPASS-to-SKIP. ExistingPython3.12CPUextras/Torch,no install,canonicalbinaryabsent.
Exact command/env/version and XMLbaseline hashes are in independently archived
evidence/governance52/{execution,comparison}.json. This joins actual fullunit
dependency failures only after fullunit completion;no oldFAIL/SKIP is relabelled.

## Strict77 final07 observed result

Run0,compare0,audit0. All77cases complete/equal;every difference,missing-case,
definition,provenance andcachefinding count0. This exceeds50scenario settings.
Bothbaseline713e6a8 andcandidatedbe rehashed451artifacts/519cachefilenames,full
cleancaptures,unchangedcacheinventories. Candidate/source/gitlinkcleanafterrun.
Python3.10.6/numpy2.2.6/pandas2.3.3/scipy1.15.3/sklearn1.7.2/plotly7.1.0/
fasteners0.20/ddsketch3.0.1equalbaseline/candidate/currentinterpreter.
ComparisonSHA256680c535ee73b14fffb1ed7e0d622b547ad6e8ac341502f9e70fce2510e51a447.
Scenario run/compare/audit commands are in evidence/strict77/execution.json and
rawfinal07audit.log;originalauditorSHA8d985f0f...unchanged,noalteredgates.
RawauditorhistoricalTaskID preserved;currentfinal07receipt bindsnewTaskID/source
andtreeexplicitly. Selectedreceipts durable,large77outputs/caches referenced.

## Canonical full-unit and exact status gates

Actual XML8028PASS/279FAIL/32ERROR/281SKIP,rawexit1. Pytest1231.99s,
launcher1234.355750s<1800s. Python3.10.6/pytest9.1.1;Torch/aiohttp absent.
Exact retained command (inside the archived namespace/resource envelope):

```bash
timeout 1800s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit -q -p no:cacheprovider --continue-on-collection-errors --basetemp=/data/ycfeng/tmp/d29_unit_parent07 --junitxml=/data/ycfeng/tmp/d29_unit_parent07.xml
```

`PYTHONPATH=/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-08:/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-08/tests/unit`;
`PYTHONDONTWRITEBYTECODE=1`,BLAS/OMP/NUMEXPR1,TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp.
Outer sudo-n/systemd-run/MemoryMax30G/runuserbrainpp and every exactargument,
version andenvironment are archived in evidence/unit/execution.json.

- Against46fe:all8612retainedIDs/statuses present andunchanged;8newIDsallPASS.
  Missing/statuschange/newFAIL-ERROR-SKIP/PASS-to-SKIP0;exact46fegatePASS.
- Against79:allcommonstatusesunchanged;11newPASS andoneknownapprovedhotupdate
  test-IDloss. Raw79presence(andparserparent128presence)FAIL remainsunmodified.
  No alias/filter/waiver orclaim thatrawunitisallPASS.
- Original267FAIL/31ERRORallretained. Twelve addedmissing-aiohttpFAILandone
  missing-TorchcollectionERRORexactlysameaspreceding46fe/79;all13coveredby
  same-source101PASS. FivegovernanceTorchFAILcoveredby52PASS. No newsourcefailure.
- Requiredmetadata197/catalogue53IDsallPASSalsoinfullunit. IndependentPython3.12.13/
  pytest9.0.2/Torch2.9.1/aiohttp3.13.3supplements do notrelabelrawoutcomes. NumPy
  2.2.6/Pandas3.0.1/SciPy1.17.1andexistinggovernanceCPUextrasretained,noinstallation.

RawXML `/data/ycfeng/tmp/d29_unit_parent07.xml`,2111062B,SHA256
64371d6394026aedb9fa8d9c409a9263bfc2f23d72a063886581b61827b8b5b3;
rawlog `/data/ycfeng/tmp/d29_unit_parent07.log`,4614412B,SHA256
9d055a740ae01a6428c5c13daca60cf3ee15bad934e6bcf61b73d017549a873b.
Allrawfailures/errors/skips/warningsretained. Precise8/11newIDs,oldloss,known
FAIL-ERROR-SKIPsets andsource/environmentjoins are in evidence/unit/ JSONreceipts.

## Final source/input/script/scope guards

PASS:2755streamedfilehashes/14inputs/12scriptsunchanged,source/tree/3dacclean,
historicalbinaryabsencepreserved. Reused124bindingguardrerunaftertests,byte-identical
receiptSHA8efc7bce63ff956ef4c03962acd13afc35c76229885a0e078fbe051c92e5d989;
120non-querydigest19c5718dfc1aa4cb11eef0306b8b877dbd9f234f1192053cb77ce4de7c9337fb.
Scope9565source/9579global,CORE4261/PRICING3954/native1350,zero violations.
Originalguardpredicatesunmodified,onlyoutputdestinationrebound. Selecteddurable
receipts/sourceinventory/finalverificationplusmanifests under evidence/;large
raw/cachesreferenced. No missing/interruptedrun orunfinishedverification.
These are observedregressions,not fullCLI/native/GPU/platform/G3/E2Equalification.
