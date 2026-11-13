# D23-ACTUAL06-WALL public iteration diagnosis

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded exact public iteration RED, whole-path CPU profile and source-equivalence evidence. |

## Outcome and scope

Diagnostics-only on accepted `7e8bea3cd2362f8e46c4b7cd0e712d2f2fe392fb`. No production edits, full CLI retry, native/GPU work, original cache/fixture mutation, timeout change or disabled metrics. Task/session: `D23-ACTUAL06-WALL` / `ses_e9f519775ffe3Cv0bhEzJSKNhH`.

Complete actual public q1 iterations remain RED. Original actual06 setup337.9405/339.5054s and2400s bound imply an **optimistic** per-output budget `(2400-339.5054249763489)/1024=2.012201733s`, before P/tail/final export. This is an iteration budget, not the older maximum-query heuristic or final6144/1024 acceptance.

## Environment and reusable RED command

- Worktree: `/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf`, branch `feat/step5-mrv2-perf`.
- Python3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python`; existing `cloudpickle`, NumPy/sklearn; no dependency installation.
- Existing collective-sim/htsim build reused read-only from `step5-mrv2-main`; runner outputs/cache writes redirected into each diagnostic output directory.
- Reads/copies and commands limited to60s. Memory scope unavailable; no full94k CSV materialization or unbounded inspection.68 artifact files146374462 bytes and4 source bundles copied/hash-validated; original exact94392/184-key coverage preserved.

Run from the perf worktree, with a **new** output directory:

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_actual06_iteration.py --output /data/ycfeng/tmp/d23_actual06_verified_red01 --max-runtime 30 --resume-state /data/ycfeng/tmp/d23_actual06_exact_q1_state/state.pkl --start-emitted 1 --outputs 3
```

Observed exit1 **RED**, not an execution error: setup13.260835s, loop19.222818s, two complete intervals9.540253/9.676296s. Log `/data/ycfeng/tmp/d23_actual06_verified_red01.log`; receipt/boundaries/events/source setup under that output directory. Process is seconds-scale (~35s including imports/copies/state) and remains below60s. Timing includes observer overhead; identical source/event/ledger truth is checked separately. The snapshot is trusted project-owned diagnostic state, not a native artifact or a production resume path.

### Explicit setup and state fidelity

The script parses original `d23_main_actual06_offline/offline/command.json` using `SimulationConfig.create_from_cli_args()`. It restores SHA256/identity-validated original bundles into the production family/precision/contract registries instead of unrelated broad training/init. All runtime predictor, event, scheduler and metrics methods remain production methods. The loop matches the original sequential handler/add-events/JSON-trace/checkpoint path; actual cluster logging/performance profiler/Chrome trace/checkpoint observer are false. Final export and terminal assertions are deliberately not invoked on an incomplete prefix.

The saved state is captured exactly after the first D `GlobalBatchEndEvent`, simulation85.95974032799978, emitted1. It retains:

- Original92 physical layers, P/D PP1/Attn_TP8/Attn_DP2/MoE_TP1/EP16 topology,6144/1024 request and real active/dummy MRv2 descriptors.
- Full103675-block KV pool per lane, exact block refcounts/hash/epochs and free-list links. Generic pickle originally failed on the103k-node doubly linked list; explicit scalar/link-ID representation preserves its contents, not a reduced pool.
- Exact entity/event class counters, original queued events/requests/routing/predictor state; predictor/estimator references resolve to verified bundles and correct role classes. Configuration equality is asserted before restoration.
- Pending stage and transfer ledger object-ID maps rebound to the same restored objects; output paths alone move to scratch. Initial heuristic snapshots are not used by the final command, and their recovery code was removed.

Regenerate this snapshot without native work in three bounded segments, preserving complete events:

```bash
timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_actual06_iteration.py --output /data/ycfeng/tmp/d23_actual06_exact_prefix01 --max-runtime 30 --backend-cache /data/ycfeng/tmp/d23_actual06_boundaries01/backend_cache.json --outputs 3 --save-state
timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_actual06_iteration.py --output /data/ycfeng/tmp/d23_actual06_exact_tail01 --max-runtime 30 --resume-state /data/ycfeng/tmp/d23_actual06_exact_prefix01/state.pkl --outputs 1 --save-state
timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_actual06_iteration.py --output /data/ycfeng/tmp/d23_actual06_exact_q1_state --max-runtime 30 --resume-state /data/ycfeng/tmp/d23_actual06_exact_tail01/state.pkl --outputs 1 --save-state
```

These exact commands were run before adding the read-only assertions/report fields; use new directory names and matching references if regenerating. Existing backend snapshot contains only real collective-sim results. Cold collection began with original public prefix `/data/ycfeng/tmp/d23_actual06_prefix02`, no backend snapshot,35.099s bounded loop; subsequent prefixes reused exact cache entries. No original cache API writes occurred.

## Whole-path observed work

| Complete interval | All roles wall | DECODE wall/events | PREFILL wall/events | Verdict |
| --- | ---: | ---: | ---: | --- |
| emitted1→2 |9.540253s|4.418710s /293|5.121543s /291|RED|
| emitted2→3 |9.676296s|4.876795s /293|4.799501s /291|RED|

1168 events/two intervals,352 EP waves,**5632 lanes** and all92 layers. Counts:368 D sync,368 P sync,176 D collective,176 P collective,32 dense completion,8 batch-stage-end,8 replica-schedule,8 batch arrival,20 stage-schedule,2 cluster-end,2 global-end. No lane/operator result is reused or dropped.

Original early actual06 q1 starts are14.514s then9.895/9.136/9.633/9.410s. Sampled original iteration8/12/16/20 gaps9.522/9.152/9.476/9.478s; iteration24/32/35 gaps5.053/5.097/5.082s. This focused first36-row inspection explains why the early exact replay is9.5s while parent whole-run median is5.1s. **Inference:** the measured full D component4.4–4.9s explains most later5.1s cadence after P workload falls. We did not replay an exact later-context state and do not claim it.

### Source/artifact/output truth: PASS

Add `--audit`, new output name `d23_actual06_verified_audit01`; exact observed command otherwise unchanged. Audit loop24.450029s, intentionally excluded from performance baseline. Every query is compared against the currently selected original estimator exact key, paired count, model hashes/evidence and cost:

-41288 FFN+960 attention queries; all92 layer IDs; precision BF1635316/FP325632/FP8340;34 unique model hashes.
-Current actual q1 query familyCUDA_EVENT42248 queries; KERNEL_ONLY bundles are loaded/identity-validated but not forced into a phase that selects CUDA_EVENT.
-Repeated full source-truth dictionaries identical across `complete_audit02`, `final_audit01`, `verified_audit01`; digest `c2dfc11132106d5ac0cdb3eef8b29f08dd54b490f311b47d009751b86aed648a`.
-Original grouped-cache event/role/iteration/simulation time/tokens equal at all observed original matching boundaries, including P completion85.51257516799994, tail85.95974032799978 and q1 completions86.00261039840062/86.04548046880146. Original terminal ledger is absent due to timeout, so direct original full-ledger equality cannot be claimed.
-Baseline/audit JSON event trace **bytes equal**, two completed stage-ledger rows **full JSON equal**,5632 EP lane ledger rows **hash equal**: `bdd74d1e1ec310fa94f8ad60af888d61bbc0e4279ba92349cdaed3fd33cb2526`.
-All original actual metrics flags retained: `write_metrics=true`,request/batch/utilization metrics,stage ledger,metrics-ground-truth andJSON event trace. Original `store_operation_metrics=false`,op-level trace=false; not silently described as enabled. No metrics are disabled by the diagnostic.

## Ranked falsifiable hypotheses and results

Shown after first RED and before profiling:

1. **Public predictor attention/FFN or full-stage rebuilding dominates.** Prediction: public predictor/layer trees dominate whole-path CPU. Supported:6410 public stage calls24.054/34.100s profiled, per-role wallD6.86/P6.98s of19.15s loop. This includes per-lane queries and completed-stage assembly; not only inner FFN.
2. **Scheduler P dummy/EP population dominates outside prediction.** Prediction: high wave/entity population and role-specific wall explain cadence. Partly supported:5632 lane builds3.754s,123904 fullRequest constructors2.635s; D4.4–4.9s plus earlyP4.8–5.1s. Coordinating every real lane is required; no proposal to suppress P or lanes.
3. **Ledger/metrics/trace serialization dominates.** Prediction: production metrics boundaries exceed predictor. Falsified for original settings: EP metrics0.653s and stage metrics1.007s in34.1s profile; wall~0.52s combined. Additional enabled op trace is expensive but was not actual06 original setting.
4. **Repeated cold backend subprocess explains warm cadence.** Prediction: misses/subprocess time in warm loop. Falsified for warm loop:12224 queries all hits/zero misses; profile backend2.050s incl topology/rank preparation, no subprocess. Cold real collectives remain expensive separately.
5. **Host load or restored state mismatch explains slow replay.** Prediction: repeated verdict/identity/event counts differ or original early cadence differs. Falsified for current early state: repeatRED, exact config/artifact/proof/event/ledger equality and original early9.1–9.9s cadence. Later-context equivalence remains untested, not assumed.

## cProfile and one-variable wall probe

Profile command: same RED command with `--profile --max-runtime 40`, output `d23_actual06_complete_profile01`. It completed two outputs in34.678726s plus14.037020s setup, within60s. Artifact `iteration.pstats`, summary `profile.txt`, receipt/events/boundaries.42885839 calls/40997599 primitive in34.100s.

| Production function | Calls | Cumulative seconds |
| --- | ---: | ---: |
| `_prepare_moe_ep_wave_plan` |352|27.820|
| public `predict_stage_execution_time` |6410|24.054|
| `_predict_step5_ffn_operator` |41288|9.896|
| `_get_on_demand_prediction` duration/count |84496|5.933|
| `build_ep_lane_batch` |5632|3.754|
| fixed identity `json.loads` |42248|3.313|
| full `Request.__init__` |123904|2.635|
| collective backend `_predict` |12224|2.050|
| full Step config validation via attention workload |998|1.735|
| `_parse_prediction_bounds` |179600|1.193|
| `_as_prediction_feature_names` |168992|0.914|
| stage ledger schedule |2|1.007|
| EP wave metrics |352|0.653|
| current family/model selection |41328|0.551|

These are **nested cumulative** times; do not sum them. Observer wrapper self time~1.006s profiled. Previous fixed metadata/precision construction is not recurring and is not revisited.

To distinguish parser cost without cProfile, the diagnostic wraps only canonical-marker `json.loads` and returns its unchanged result. `d23_actual06_final_red01/boundaries.json`:42248 parses3.135263s in19.145855s total loop. CPU callers:41288 FFN proof reads3.213s profiled,960 attention identity reads0.100s. The repeated whole source audit remains byte/dictionary equivalent. This is measured fixed parsing, **not the sole5.1s cause**.

### Cold and trace separation

- `d23_actual06_collective_check/receipt.json`: ATTN_TP8/8192B real allreduce cold1.883520s,hit0.111104ms; EP16/16384B real all-to-all cold2.486331s,hit0.162369ms; repeated real costs0.122465/0.0199104ms exactly equal. Same original topology; no fake cost.
-Full P/tail with warmed backend cache: P until transfer7.897355s; fullP+firsttail60.788119s cumulative across three bounded process segments, excluding repeated setup. Source boundaries match. Cold public P/tail total remains uncovered; original parent receipt retains cold firsttail62/65s and setup337/339s.
-Additional `--op-trace` output `d23_actual06_final_trace01`: loop31.610762s vs19.145855s baseline; two intervals16.157514/15.446672s.51988 persisted events76067468 bytes; all92 layers,25696P/26292D,finite nonnegative durations, same1168 public events/5632 lanes andzero backend misses. This is added trace overhead, not original timeout attribution.

## Smallest source proposal, not implemented

Reuse parsed **fixed canonical estimator measurement identity/evidence-kind** at the existing registry attach/load boundary, consumed by Step proof construction. It has a clear gain:~3.135s fixed parsing removed per two earlyP+D rounds, roughly0.8s perD-only round by query population. It does not avoid a dynamic lookup or suppress proof output.

Supported lifetime evidence:

-`PredictionModelRegistry._set_measurement_identity` (`prediction_model_registry.py:657`) attaches canonical JSON and rejects conflicting reattachment.
-`load_model_bundle` (`:999`) checks bundle/hash/registration/measurement and duration/count links before publishing; `_bundle_model_metadata` (`:734`) already parses canonical marker.
-Predictors retain fixed model/TP/EP config; however family registry may be replaced in memory. Keep every `_get_family_model` call and use the selected estimator's view, not a predictor-wide model cache.
-If attached marker changes, reparse or reject by exact marker comparison; malformed identity must still fail. Preserve duration/count marker equality, target/statistic/runtime architecture/quant/device/TP/block/kind checks and all dynamic feature/bounds/coverage/count/proofs. No numeric feature/output cache, family guessing or global invalidation layer.

Do **not** expect this alone to make the iterationGREEN: dynamic schema/bounds validation,123904 syntheticRequest constructions, layer/stage object assembly, attention's998 config scans and backend topology/rank work remain. The next optimization requires parent decision and the complete iteration loop, not a small lookup gate. A source proposal for these broader costs is deferred rather than bundled into this narrow suggestion.

## Verification limits and retained failures

-Syntax AST PASS; `git diff --check` PASS. No persistent production regression suite rerun because this is diagnosis-only and accepted224/77 parent gates are unchanged.
-Initial snapshot generic pickle recursion and initial missing-ID snapshot were diagnostic failures; corrected with exact scalar/link/counter/object references and regenerated state. A subsequent completion detector used a nonexistent GlobalBatchEnd trace `cluster_type`; corrected after inspecting actual event format. Final verified loops repeat deterministically.
-Not covered:337/339s production initialization, exact later iteration24 context/P shutdown, online whole-path replay, cold fullP/tail total, long-run metric growth/final export,1024 completion, native accuracy or strict77. Parent owns acceptance/review/merge/terminal records. Original source fidelity fixtures remain CPU fixtures, not native evidence.
