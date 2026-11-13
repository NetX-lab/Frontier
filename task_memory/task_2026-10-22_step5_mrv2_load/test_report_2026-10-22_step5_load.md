# D23-MRV2-LOAD-DIAG Test Report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archive bounded actual-API measurements and full-wall limits. |

## Outcome
Diagnosis PASS. The unchanged manager loads and validates the same attention
source before warm-cache lookup, and repeats this work before returning an
already-trained role signature. The command intentionally exits1 with
`RED_REPEATED_LOAD`: the specific inefficient pattern was reproduced. No
production fix or full-run acceptance is claimed.

## Exact fast command
Working directory: `/data/ycfeng/Frontier/.worktrees/step5-mrv2-load`.
The output directory must be new; use another suffix to rerun.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/profile_step5_load.py --config /data/ycfeng/tmp/d23_main_actual05_offline/offline/metrics/stepfun_ai_step_5_preview/offline_batch/step5_measured_cpu_main_offline/config.json --output /data/ycfeng/tmp/d23_mrv2_load_diag04 --rows 100 --event-cache-hash 23b7026a --assert-no-repeated-loads > /data/ycfeng/tmp/d23_mrv2_load_diag04.log 2>&1
```

The two-size control used the same command with output `diag03`, default rows100
and200, and no event-cache-hash. Both completed in seconds under60-second tool
limits with the expected RED. Environment: Python3.10.6, numpy2.2.6,
pandas2.3.3, sklearn1.7.2, CPU only, OMP/OpenBLAS threads1. No dependency install,
GPU/platform operation, full manager initialization, simulator or estimator fit.

## Source truth and checks
- Loader/trainer/registry/producer source is byte-identical between actual05
  26db654 and diagnostic baseec5de15 (`git diff` empty for those five files).
- Diagnostic model's complete JSON equals the retained actual config model.
  Role replicas retain TP8/DP2/EP16/PP1. Actual scheduler/graph/predictor options
  retain7168/8192/128, block16, autoKV, full_and_piecewise and capture sizes.
- Source CUDA_EVENT:5,606,492,481 bytes,94,392 rows,125 columns. Prefix100/200
  preserves all columns and every source string field. Prefix100 contains one
  layer0 PIECEWISE request row and99 FULL request rows; prefix200 contains one
  PIECEWISE and199 FULL. These prefixes do not cover other layers/variants/dummies.
- Full small KERNEL_ONLY184 source is used unchanged:92 layers, both variants,
  request and MRv2 dummy rows. No production validation is patched or weakened.
- Warm cache APIs use independent scratch copies; all native lock writes stay
  there. Missing-cache fitting is forbidden. Original and scratch artifact
  SHA256 values agree after calls. Original config/CSV size and modification
  times are unchanged across final command. No original fixture/cache writes.
- Existing source timing values are CPU synthetic fixture values, unchanged;
  not native admission or accuracy evidence. No substitute timing is introduced.

## Measured profiles
Times are cumulative under cProfile except plain measurements; nested times
overlap and must not be summed. Comparable values below are fromdiag03.

| API / boundary | Seconds | Calls / result |
| --- | ---: | --- |
| Event100, plain |0.255748|100 rows |
| Event200, plain |0.453526|200 rows |
| Event100, cProfile |0.563309|CSV1, layer description1 |
| Event200, cProfile |1.033775|CSV1, layer description1 |
| Event200 feature generation |0.233025|400, twice per row |
| Event200 typed metadata validation |0.198476|1800, nine per row |
| Event200 targets validation |0.187091|one complete pass |
| Event200 typed contract parse |0.115503|200; nested deepcopy0.099624s |
| Event200 CSV parse |0.074160|one read |
| Event200 expected contract construction |0.003332|200, already inexpensive |
| Full KERNEL_ONLY184 loader |2.222436|92 layer descriptions |
| Event100 role union |1.181785|same source loaded twice |
| Warm PREFILL KERNEL_ONLY |4.942950|two cache hits/two models, zero fit |
| Warm PREFILL loaders |4.449957|two full184 loads |
| Warm PREFILL hash |0.139270|three DataFrame serializations |
| Warm PREFILL cache reads |0.003785|two calls |
| Already-trained DECODE KERNEL_ONLY |4.681278|returns no models |
| Already-trained DECODE loaders |4.486186|still two full184 loads |
| Already-trained DECODE hash |0.051305|one hash before signature return |

Finaldiag04: plain1000.255889s; already-trained DECODE4.611782s/two loader calls,
zero fits. Current actual-policy event rootcache `step5_layer_e2e_23b7026a.pkl`
reads in0.529611s,71,149,163 bytes and94,392 exact coverage keys. Stored identity
agrees with both actual role input paths/SHA256 and7168/8192/128 shape policy.
Earlier isolated cache control: current root0.524986s; old fixture-policy
root0.573500s. No full-table cache hash/fit occurs in those controls.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Config identity |complete model JSON equal |equal |PASS |
| Subset truth |all125 columns/values equal |100/200 equal |PASS |
| Warm cache |two hits, no fit |two hits, no fit |PASS |
| Red-capable signal |second role repeats load |two calls, diag02/03/04 |PASS, expected RED |
| Artifact preservation |original SHA256/stats equal |equal |PASS |
| `git diff --check` |empty |empty |PASS |

Firstdiag01 stopped before profiling at Python tuple versus JSON list equality.
The harness now compares actual JSON serialization; no model change. A preliminary
streaming full CSV census exceeded10 seconds and was stopped without output/writes.
Successful profiles never read the full5.6GB event CSV.

## Full-init attribution: source counts, not measured full phases
Both retained actual05 logs contain zero `MODEL CACHE MISS ERROR` and `CACHE
MISS: Training` warnings. The unchanged fit branch emits these unconditionally
at warning level on every miss. This supports warm initialization; the17-minute
interval cannot be assumed to be new fitting.

Call chain: `_train_all_required_models` -> each role/measurement ->
`_train_attn_models_for_cluster` -> `load_step5_attention_input_union` -> each
declared role -> loader -> DataFrame hash -> signature check -> first signature's
targets validation/cache reads. Relevant sources: manager435–519,
family trainer702–724, Step loader184–302 and union305–395, registry127–165.

The unchanged exact config requires four event-table loads and four small
kernel-table loads per process; the last two of each occur before the
already-trained return. Source arithmetic for94,392 event rows implies:
- 377,568 row validations and expected operator contract constructions;
- 3,398,112 typed metadata validations (nine per row);
- 755,136 feature generations (twice per row);
- 1,132,704 workload constructions (three per row);
- 368 layer descriptions (92 per load, not per row);
- five target-validation passes (four loaders plus first training call);
- four full DataFrame hash/JSON serializations and four source SHA256 scans.

Four event parses logically scan22.426GB; four SHA256 scans scan22.426GB again.
These are call-derived byte counts, not actual physical disk traffic. First-row
quant_signature has53,767 characters of59,209 total value characters (~91%).
The source representation is preserved rather than replaced by a compact hash.

Plain100/200 incremental slope0.001977783s/row arithmetically gives approximately
747s for four full event loads. **This is prefix-only extrapolation, not a wall
profile or an accepted allocation of998.503711/1004.875261s.** Prefix covers only
SWA layer0; full allocations, hashes/dedup, other variants/dummies,92-layer
descriptions, parallel offline/online contention, generic compute loaders, bundle
publication and predictor/replica setup were not profiled. cProfile roughly
doubles prefix wall and cannot directly estimate full wall. Remainder unallocated.

## Smallest proposal: parent decision, no production implementation
Hypothesis1 is only partly confirmed: per-row typed parse/validation and feature
work repeat, but layer descriptions already run once per layer per load.
Hypotheses2/3 are reproduced directly. Hypothesis4 is not supported as the
dominant sampled boundary: cache reads are milliseconds to half a second, zero
fits occur, and CSV parse is7% of the200-row cProfile wall. This does not establish
the full-table CSV parse fraction or rule out all other initialization costs.

1. Reuse the identical fully validated attention input union and its existing
   content hash within one manager initialization for the same declared sources,
   topology, shape policy and measurement. Preserve each role's source record,
   path/SHA256, identity checks, malformed-row rejection, exact coverage and
   current estimator cache hashes. Distinct source/policy remains separately
   validated. Share the identical role load before concatenation and avoid
   reloading the union before the already-trained signature return. No cross-run
   state, weakened validation or cache invalidation.
2. Retain the first feature dictionary instead of calling `step5_row_features`
   twice; reuse layer geometry/expected contracts per layer. Every source row
   still receives typed parsing/metadata checks. Expected contract construction
   costs only3ms/200rows; whole validated input reuse is the higher-value seam.
3. Do not change quant_signature formatting, target values, typed validation,
   cache identity or the runtime per-EP seam from this evidence. More extensive
   schema/parser changes require separate scope/evidence. Stop profiling here.

## Artifacts
Durable evidence in this task directory: `profile_receipt.json` (diag03 counts
and environment), `final_command_receipt.json` (diag04),
`event_cache_receipt.json` (two actual artifact controls), this report.
Raw JSON/pstats/subsets: `/data/ycfeng/tmp/d23_mrv2_load_diag03/`,
`/data/ycfeng/tmp/d23_mrv2_load_diag04/`, and event cache control
`/data/ycfeng/tmp/d23_mrv2_load_event_cache01/`. Parent's actual05 timeout receipt
and original run directories remain unchanged. Parent owns source integration
and original2400-second full offline/online acceptance.
