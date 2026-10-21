# D23-ACTUAL-MAIN Progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Primary reverified8 fixture tests, merged8e1d0a1 and reproduced the root ledger failure; core repair delegated. |
| 2026-10-22 | Begin exact fixture generation. |
| 2026-10-22 | Correct exact compute coverage; scoped8 PASS; actual04 pending. |
| 2026-10-22 | Actual04 paired core ledger failure; archive exact blocker without bypass. |

- completed: created new worktree from d8e3e941; immutable forward branch unchanged.
- completed: inspected actual native capture plan7168/8192/128,FULL_AND_PIECEWISE,
  max capture256. Native vllm/config/vllm.py:2331 supplies capture sizes1,2,4,8..256.
- in-progress: exact single-request rows P6144/NONE, D16/PIECEWISE,cached6128,
  D1/FULL contexts6144..7166 inclusive, metadata-bearing MRv2 dummy1/16/1.
- pending: offline/online complete actual-main acceptance and final fixture regression.

Commands:
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/step5_measured_compute_fixture.py --output-dir /data/ycfeng/tmp/d23_main_fixture01 --device h800 --gpus-per-role 16 > /data/ycfeng/tmp/d23_main_fixture01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/integration/test_step5_measured_main.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_main_fixture_unit.xml > /data/ycfeng/tmp/d23_main_fixture_unit.log 2>&1
```
Fixture generation in progress. Scoped tests PASS4 in1.64s: padded real17->24
preserves one request/context, plus NONE/PIECEWISE/FULL MRv2 identity and raw null.

### Check-progress checkpoint
- PASS: worktree/isolation and fixture-shape regressions. TODO: fit,offline,online,archive.
- Mainline: exact fixture completion -> actual subprocess -> artifacts/metrics.
- D1–D7 CLEAR: fixture/trace work directly verifies acceptance; no core edits, old runs,
  delegation or unrelated fixes. S1–S6 CLEAR; no route/spec correction needed.
- Next: run both modes from the same exact fixture, bounded900s each, then inspect
  READ/source release and1024 D useful forward census. Initialization does not pass.

Generation01 stopped with SIGTERM before any output after9min: inspected10-row
sample costs0.01392s/row because model contracts were rebuilt for every context.
Fix: reuse one uniform FULL typed row per layer/shape, retaining all1023 actual
context fields. `attention_scopes` depends on shape/capture/capacity for these
uniform FULL rows, not live context; independent boundary tests compare complete
row dictionaries. No production changes or dropped exact coverage.

- completed: final fixture tests6 PASS in1.83s, including full row equality for
  reused contexts6145/6500/7166 on both attention variants; no fake request padding.
- in-progress: generation02 through shared cache/manager.100 actual CUDA_EVENT
  rows take0.2488s in production loader; retain all exact keys despite load cost.
- pending: actual main subprocesses once manager fixture manifest exists.

Generation02 command equals01 with output/log `d23_main_fixture02`. Final tests
equal the earlier pytest command with XML/log `d23_main_fixture_final_unit`.

Loader-cost diagnosis only (not completion):
`/data/ycfeng/tmp/d23_main_loader_probe100.log`:100 rows loaded in0.2487736s.
Probe CSV:`/data/ycfeng/tmp/d23_main_loader_probe100.csv`.

Actual fresh CSV census:
- CUDA_EVENT94392 rows/92 layers,94208 request-owned+184 dummy.
  PIECEWISE184,FULL94208; decode contexts1023 distinct6144..7166 inclusive.
 92 FULL dummy raw max_query_len=null/uniform1 and92 PIECEWISE num_reqs=null.
- KERNEL_ONLY184 rows/92 layers,92 request-owned+92 MRv2 dummy, allNONE.
  Native request scope and dummy scope remain distinct; no isolated rows.
Receipts:`/data/ycfeng/tmp/d23_main_fixture02_{rows,kernel_rows}.json`.

- completed: `git merge --no-edit c9e9388f` fast-forward, no integration conflict.
  Core source equals the immutable supplied correction; main fixtures unchanged
  by the merge. Final CLI request capacity128 comes from the existing scheduler.
- in-progress: original generation02 PID828161 retained, no duplicate fit/output.
  Manager is validating/fitting exact mixed attention union. Full CLI pending.

- completed: corrected-source fixture regressions6 PASS; XML/log
  `/data/ycfeng/tmp/d23_main_correction_unit.{xml,log}`.
- environment:Python3.10.6,numpy2.2.6,pandas2.3.3,sklearn1.7.2;
  torch unavailable and unnecessary for CPU DES. Initial version receipt tried
  importing torch and failed; rerun records that verified limit rather than
  installing dependencies. `/data/ycfeng/tmp/d23_main_environment.json`.

## Resumed actual main

- completed: generation02 notification is successful. Existing manifest and all
  shared caches/bundles retained; no duplicate fit. Log ends with initialized
  manager and roles2/5. Manager fit elapsed02:33:18..02:51:06 (1068s).
- completed: inspected resumed source; HEAD is
  `c9e9388f117c625c6d546d1d083f8aaf3f824f51`. Fixed batch_size_cap follow-up is not
  included; parent owns that delta. No independent core edits.
- failed: actual01 offline/online stop in config before any request event:
  `decode_cuda_graph_mode cannot be combined with use_cuda_graph=True`.
  Cause: runner retained PD-AF global flag alongside the native PD graph policy.
  Fix only runner to explicit `--no-use_cuda_graph`; FULL_AND_PIECEWISE/capture
  sizes unchanged. Added command assertion for the mutually exclusive flags.
- bounded runtime revised to2400s per subprocess: recorded original manager fit
  takes1068s even before DES execution, so900s was not sufficient. This changes
  harness wall timeout only, not simulator workload/capacity/cost.
- completed: resumed scoped regressions6 PASS; `/data/ycfeng/tmp/d23_main_resumed_unit.*`.

Actual01 command:
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d23_main_fixture02/fixture_manifest.json --output-root /data/ycfeng/tmp/d23_main_actual01 --mode all > /data/ycfeng/tmp/d23_main_actual01.log 2>&1
```

## Continuation after provider failure

- completed: generation02 fit and manifest exist. Existing profile/cache output
  unchanged; no repeat fixture generation. Scoped CLI flags tests6 PASS.
- failed: actual02 offline/online each exits1 in1.366s before simulation:
  `ModuleNotFoundError: No module named 'collective_sim_core'` from
  `parallel_semantics.py:283`. Exact logs/execution/source hashes are under
  `/data/ycfeng/tmp/d23_main_actual02_{offline,online}`. No request events/READ or
  final metrics exist; completion is unverified, not zero-valued measurements.
- diagnosis: this new worktree's gitlink was uninitialized; parent preview has
  local checkout at exact gitlink `d28fe917370c2d55230c3ea226e5660785b44aca`.
  Initial `submodule update --init --no-fetch` still attempted default HTTPS clone
  (failed immediately at localhost proxy); no successful external access.
  Correct local-only init explicitly supplies temporary submodule URL override:
  `git -c protocol.file.allow=always -c submodule.frontier/cc_backend/backends/collective-sim.url=/data/ycfeng/Frontier/.worktrees/step5-preview/frontier/cc_backend/backends/collective-sim submodule update --init --no-fetch frontier/cc_backend/backends/collective-sim`.
  No tracked core/gitlink changes. Actual H800 scenario import now passes.
- in-progress: build existing pinned htsim dependency with2 CPU threads; no
  dependency source changes. Command:
  `make -j2 libhtsim.a && make -C datacenter -j2 htsim_ndp > /data/ycfeng/tmp/d23_main_htsim_build.log 2>&1`
  in `frontier/cc_backend/backends/collective-sim/sim`.
- pending: run actual03 both modes with existing fixture02 immediately after
  local build; inspect full acceptance. Parent owns fixed batch_size_cap follow-up;
  tested core remainsc9e9388f unchanged.

- completed: local htsim build succeeds; binary SHA256
  `812fb3d86489d76a4d50e2f0ea936bdb6d25dfe740c06050a0d0c1a1f9f4631e`.
- in-progress: actual03 offline and online launched concurrently using the same
  retained fixture02 and cache, each child bound2400s. No source edits during CLI.
  Child config receipt independently verifies7168/8192/128,0.92,TP8/DP2/EP16/PP1,
  FULL_AND_PIECEWISE and native capture sizes. Config is evidence of inputs only,
  not request completion. `actual03_config_receipt.json` is the selected receipt.
- completed: final resumed scoped regression6 PASS in1.86s:
  `/data/ycfeng/tmp/d23_main_actual03_scoped.{log,xml}`. No runner/fixture/core
  source changes or commits while actual03 children are running (source hashes
  are pinned by the runner).
- observed: existing compute fixture has token domain1/8/16/32/6144. Under
  measured native-bound NONE, `ep_wave_inputs.py:70-90` preserves local useful6144
  and dummy1 -> aggregate6145, while both cached gating estimators bound max6144.
  Direct existing on-demand lookup at6145 rejects exactly:
  `ValueError: On-demand prediction feature 'num_tokens' for probe is above declared maximum 6144.0: 6145.0`.
  Receipt:`aggregate_coverage_probe.json`. This is a fixture-source gap, not yet
  an observed actual03 event or final request verdict. Do not change source
  during running children; if event confirms, supplement only compute tables
  through existing typed table/manager path, retaining original attention output.
- observed: direct existing `materialize_layer_ep_workload` +
  `MoELoadImbalanceInput` for balanced352 experts/EP16/topk8 at aggregate2/32/6145
  differs from the old unit fixture's global-topk load distributed across only22
  experts.116 nonempty probe lanes;104 violate current cache bounds, none is an
  exact key. `routed_coverage_probe.json` retains the fields and violations. This
  is the second part of the same scoped fixture-source mismatch, not a reason
  to widen domains or disable exact checks. If confirmed by actual event, use
  the existing materializer for the exact nonempty lane counts and preserve
  the zero-lane native skip; no alternate router or cost fallback.

Actual03 exact commands (same worktree/environment; distinct output roots):
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d23_main_fixture02/fixture_manifest.json --output-root /data/ycfeng/tmp/d23_main_actual03_offline --mode offline > /data/ycfeng/tmp/d23_main_actual03_offline.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d23_main_fixture02/fixture_manifest.json --output-root /data/ycfeng/tmp/d23_main_actual03_online --mode online > /data/ycfeng/tmp/d23_main_actual03_online.log 2>&1
```

### Check-progress self-check

- Plan1 PASS; actual2/3 FAILED due to recorded local dependency; archive4 TODO.
- Mainline: local pinned build -> offline/online in parallel -> artifact validation.
- D1–D7 CLEAR: each fix removes an observed actual CLI blocker; no unrelated work,
  core edit, additional fixture fit, old run access, or extra matrix.
- S5 HIT: earlier summary/progress still said generation02 running despite
  successful notification. Updated active status here and summary below; plan2
  explicitly owns local dependency. Other S1–S6 CLEAR. BranchB, no scope changes.

## Actual03 result and exact compute correction

- failed: actual03 offline1141.94s/online1160.34s, both exit1 without timeout.
  Setup cache identity differs from original fixture scheduler shape policy and
  kv_cache_dtype; both actual CLIs rebuilt their exact attention root/counts.
  DES reaches firstP routed layer3 then rejects:
  `ValueError: On-demand prediction feature 'num_tokens' for share_expert_up_proj:fbb4ccb3 is above declared maximum 6144.0: 6145.0`.
  No terminal request/READ metrics: neither is acceptance.
- completed: correction only integration fixture: local widths1/16/6144;
  aggregate widths2/32/6145; linear table covers their union. Routed MoE features
  use existing balanced Hamilton materializer over352 experts and16 EP lanes,
  not the unit template's global routing assignments distributed over22 experts.
  Nonempty lane rows only; native zero-load skip remains intact. Scope/type/count
  contracts stay unchanged. FULL/PIECEWISE/NONE attention rows are untouched.
- completed: existing fixture02 compute refreshed in place, backed up four
  source CSVs with `.before_mrv2_compute` suffix. Existing attention/cache retained.
  Original bundles were marked stale in manifest; actual CLI will rebuild affected
  compute bundles via the unchanged manager/cache identity pipeline.
- completed: scoped7 PASS in2.34s. New regression firstRED (missing compute_tables),
  thenGREEN checks actual materializer lane counts/features, source6145/2/32,
  labelled CPU provenance and per-replay count7. XML/log:
  `/data/ycfeng/tmp/d23_main_compute_{red,green}.{xml,log}`.
- in-progress: actual04 offline/online from corrected compute inputs; source still
  c9e9388f. No source changes until both complete. No duplicated fixture fit.

Exact refresh command:
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/step5_measured_compute_fixture.py --output-dir /data/ycfeng/tmp/d23_main_fixture02 --refresh-compute > /data/ycfeng/tmp/d23_main_compute_refresh.log 2>&1
```
Actual04 commands are actual03 commands above with output/log04; same fixture02,
unchanged2400s child bound and native7168/8192/128 workload.

Checkpoint after verified compute correction:
- Plan1 PASS for fixture source and original fit; actual2/3 TODO awaiting04,
  archive4 TODO. Distinct scoped error corrected, no core workaround.
- D1–D7 CLEAR; only frontier-serving work, both modes run in parallel. S5 old
  actual03-running text corrected in summary/test report/checkpoint; plan1 now
  explicitly owns observed compute fixture correction. BranchB applied.
- No further fixture or runner edits during04. Inspect real terminal artifacts;
  bounds remain2400s and no more generic exploration/extra matrix is needed.
- Input lock receipt:`actual04_input_receipt.json`; compute old/new hashes and
  unchanged attention hashes:`compute_refresh_receipt.json`. Actual03 selected
  failure receipt:`actual03_failure_receipt.json` (7 scheduler decisions/mode,
  no terminal metrics; exception atP layer3).
- completed: scoped8 PASS in5.84s. Existing typed shared fit/cache and Step FFN
  lookup replays both routed precision groups, each nonempty physical EP lane at
  2/32/6145; exact duration and count7, zero-load native skip, shared-expert6145
  exact key. Missing7168 source width still rejects. No simulator class replacement
  in actual CLI. XML/log:`/data/ycfeng/tmp/d23_main_exact_compute_replay.*`.
- completed: re-read requirements/plan and ran check-progress after this verified
  milestone. Plan1 PASS, actual2/3 TODO awaiting terminal04, archive4 TODO.
  D1–D7 CLEAR; S5 stale checkpoint next action corrected in place. No broadening
  or extra run is needed: inspect04 outputs at terminal notification and archive
  their actual verdict. `actual04_config_receipt.json` verifies native capacity
  and both exact topologies; this is an input receipt, not completion evidence.

## Actual04 terminal result: core blocker

- failed/blocking: offline1032.6621223730035s and online1032.4319722638465s,
  both actual child returncode1, timed_out=false. No extra wall timeout or new
  actual run. Computation now reaches P-stage completion, then production
  `metrics_store.py:4504` rejects the supported scalar-free measured attention
  root with `KeyError: 'step5_layer_e2e'`. Same stack in both logs:
  DenseLayerCompleteEvent -> prefill_collective:250 -> metrics component ledger.
- observed: nine P scheduler records/mode, one useful `[6144]` batch, no D records.
  Last scheduler event times85.20541516799993/85.45541516799992s. No terminal
  request/system/event/transfer/stage artifacts were written, so1024 output,
  READ/readiness/source release and final metrics remain UNVERIFIED. No zero
  metrics are substituted for missing completion artifacts.
- completed: exact failure/config/source/hash receipt:
  `actual04_failure_receipt.json`; retained root source exact duration.6244ms
  and count23:`actual04_root_source_receipt.json`. This is retained cache source
  evidence, not a captured event cost. Actual CLI source remainedc9e9388f.
- completed: isolated CPU reproduction confirms the layer and stage ledger both
  throw the same KeyError. Direct existing structural attention projection also
  rejects this root after that lookup with ValueError. Therefore merely using
  `.get()` or disabling the ledger cannot satisfy accurate component accounting.
  `core_blocker.md` contains executable command, complete source request and
  observation/inference boundaries; `/data/ycfeng/tmp/d23_main_ledger_repro.log`
  contains observed traces. No core changes or bypasses made.
- observed limit: generic mixed FP8/BF16 source metadata warnings mention.500
  approximation. Typed Step scoped replay remains exact with count7; live final
  reporting is blocked. Do not claim that those warnings prove scaling is safe
  or applied to live typed costs; parent must inspect this during core repair.
- completed: reread current requirements. Check-progress: plan1 PASS, offline2
  BLOCKED, online3 BLOCKED; archive4 completes with scoped commits and selected
  failure evidence. D1–D7 CLEAR. S5 active docs updated from running to blocked;
  BranchB applied. Shortest next route is parent-owned metrics root projection
  repair -> unchanged fresh paired CLI from retained fixture02/cache -> full
  terminal artifact acceptance. No further independent run or unrelated matrix.
- completed: implementation commit`e54f9457aed7e31336cb6cb553aec016a737ae2a`.
  Only three authorized integration files changed from suppliedc9e9388f.
  Committed runner/fixture SHA256 still match actual04; exact proof in
  `committed_source_receipt.json`. Scoped8 PASS and paired demonstrated blocker
   preserved. Archive selected receipts now; full actual acceptance staysBLOCKED.

## Primary integration and core repair

- completed: primary reviewed the three integration files and independently ran
  the8 scoped tests:8PASS/6.45s. Log/XML:
  `/data/ycfeng/tmp/d23_main_parent_fixture01.{log,xml}`.
- completed: merged the integration/source records at `8e1d0a1`; all three
  integration files are byte-identical to `e54f9457`. The only production-source
  difference from actual04 is the verified fixed graph-capacity binding.
- confirmed blocker: primary ran the exact minimal root/Stage repro in
  `core_blocker.md` on the integration tree, obtaining the same
  `KeyError: 'step5_layer_e2e'` at metrics_store.py4504. Log:
  `/data/ycfeng/tmp/d23_ledger_parent_red01.log`.
- in-progress: `D23-ROOT-METRICS`, session
  `ses_e9f9fd0bdffeZH9Q2Ux5NbCYko`, owns the existing metrics/trace projection
  repair and once-only root/count/evidence regression. Mixed-precision warning
  investigation is independent; no timing/precision substitution is authorized.
- pending: after parent verification/merge, fast-forward the clean existing
  `step5-mrv2-main` worktree to accepted source and keep it unchanged during
  both actual subprocesses. Reuse fixture02/cache and the built pinned
  collective-sim dependency. Keep2400s bounds and the full acceptance criteria;
  preserve all actual04 evidence in place.
- completed warning diagnosis: source audit plus primary factor-injection
  control confirm generic0.5 factors do not alter this typed Step path.
  Layer3/88 remain122.9/184.35ms, count7, FP8/BF16, exactly equal to the saved
  source; legacy positive control10ms->5ms works. This is a CPU lookup result,
  not a completed actual-run measurement. See `quantization_warning.md` and
  `precision_factor_receipt.json`; no warning suppression or source edit.
- completed local prerequisite check: the existing main worktree imports
  `collective_sim_core.predictor.predict_collective_time`; collective-sim is
  pinned atd28fe917 and executable `sim/datacenter/htsim_ndp` retains SHA256
  `812fb3d86489d76a4d50e2f0ea936bdb6d25dfe740c06050a0d0c1a1f9f4631e`.
  No dependency rebuild, new fit or new actual CLI was started during diagnosis.
- partial root repair: primary161PASS/1 explicitly deselected; original
  root/ledger/stage=.6244ms and23 kernels now pass on mergedf237362. The inherited
  cache-order failure has identical testcase/error before and after this change.
- blocked source acceptance: independent review identified a reachable P1 in
  stage-only owner projection. The owner's irrelevant attention has no proof
  after public first/last boundary replacement and must not be validated.
  Same owner is implementing the public-call-site RED and stage-owned-only fix;
  see `root_metrics_review.md`. No actual05 or candidate matrix has launched.
  Only the clean pre-fix77 baseline is running independently.
