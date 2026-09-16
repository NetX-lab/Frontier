## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded explicit scoped D02 acceptance after final18 unprofiled runs, causal ablations and concrete design alternatives. |

# Performance RCA

## Result and source boundary

Final production candidate: `c9f8f904e3550c11aad3cc5d851d75d648cef6e1`. Baseline: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`. **All18 final executions succeeded with unchanged request/event counts. Dense run-phase residuals remain; D02 is explicitly ACCEPTED for this measured scope.** The code-level correction is not a performance waiver. Request/system semantics changed only in the separately documented D01 dummy cases; a performance ratio is not a fidelity result.

Final small/long dense median paired ratios are **1.775045x / 1.679023x**; representative MoE is **0.978816x**. The median absolute run durations are14.149→25.225ms,29.309→49.268ms and9.730→9.524s. The tiny run phase makes dense relative costs large, but the absolute cost is still reported and is not dismissed using process initialization. Three pairs establish the sign and scale of the dense residual in this fixture; the MoE difference does not establish a general speedup.

## Reproduction and controls

```bash
env TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 python tests/performance/measure_pr33_paired.py --baseline ../pr33-r12-baseline-20260915 --candidate . --output /data/ycfeng/tmp/pr33-w11-final-paired-2 --repetitions 3 > /data/ycfeng/tmp/pr33-w11-final-paired-2.log 2>&1
```

Script: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/performance/measure_pr33_paired.py`, reusing the existing `tests/performance/sim_walltime_scaling/run_case.py` without different simulator behavior. Python `/usr/bin/python`3.12.3, no conda, same CPU host, no allocated GPU, OMP/OpenBLAS/MKL/NUMEXPR threads1, WARNING logging, seed42, online sequential PDD, analytical communication, dummy1ms. One fresh subprocess per sample; cold process state and identical dummy cache behavior. No profiler or other agent test/benchmark ran during the window. Pair order is baseline→candidate, candidate→baseline, baseline→candidate. No sample exclusions or reruns were selected for a favorable result.

The complete per-process commands, output flags, runner identity, init/run/process/outer-runner time, RSS and child CPU are preserved in `w11_final_paired_samples.json`. Source stayed frozen; dirty files in the manifest are task documentation only. Raw evidence remains `/data/ycfeng/tmp/pr33-w11-final-paired-2/`.

| Case | Model / simulated topology | Prefill/decode tokens per request | Completed requests | Events | Requested total tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| small_dense | llama3.3-70b;16GPU, attentionTP4, PP2 |16/8|2|176|48|
| longer_dense | llama3.3-70b;16GPU, attentionTP4, PP2 |64/16|4|352|320|
| representative_moe | Qwen3-235B-A22B;32GPU, attentionTP8, EP8, PP2 |32/8|2|3560|80|

All paired processes retain these counts and the exact W00 reporting-disabled flags. Identities, numerical predictions, stages and logical events remain present; no required work moved into initialization. For D01-corrected dummy semantics, numerical simulated timestamps need not equal historical main; identical event/token work and the W10 causal classification are the relevant controls.

## Every final unprofiled sample

All times below are seconds. Ratios are candidate/base for each paired attempt; displayed medians elsewhere are median of paired ratios, not an unlabeled ratio of separate medians.

| Case | Pair | Base run | Candidate run | Ratio | Base init | Candidate init | Base process | Candidate process |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|small_dense|1|0.014148701|0.025114580|1.775045|2.041796|2.084949|2.058843|2.113186|
|small_dense|2|0.014249908|0.025275817|1.773753|2.067671|2.075486|2.085355|2.104114|
|small_dense|3|0.014063886|0.025224790|1.793586|2.088619|2.077760|2.105923|2.106169|
|longer_dense|1|0.029226815|0.049072485|1.679023|2.045758|2.077766|2.077981|2.129802|
|longer_dense|2|0.029309148|0.049268273|1.680986|2.100971|2.052439|2.133570|2.104921|
|longer_dense|3|0.031383208|0.049540262|1.578560|2.093328|2.080647|2.127996|2.132983|
|representative_moe|1|9.886554207|9.372422262|0.947997|2.220421|2.129115|12.110467|11.504892|
|representative_moe|2|9.730011290|9.523890981|0.978816|2.100831|2.125496|11.834005|11.652617|
|representative_moe|3|9.707917634|9.562267278|0.984997|2.091710|2.183912|11.802928|11.749240|

## Resource and scale observations

| Case | Run ratio range | Median paired run delta | Init paired median ratio | Process paired median ratio | Median RSS base→candidate MiB | Child CPU median base→candidate s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
|small_dense|1.773753–1.793586|11.026ms|1.003780|1.008996|207.754→208.273|2.427864→2.425324|
|longer_dense|1.578560–1.680986|19.846ms|0.993942|1.002343|207.770→208.496|2.444246→2.469296|
|representative_moe|0.947997–0.984997|-206.120ms|1.011741|0.984672|213.574→214.574|12.157453→11.967437|

RSS increases are about0.5–1.0MiB in these process maxima and do not establish per-layer retained memory or an unbounded leak. No allocation/leak claim is inferred from RSS alone. The longer dense fixture has2x events and6.67x requested tokens; its residual is about20ms versus11ms in the microcase. This suggests event/stage work matters more than token count here, but is not a fitted universal scaling law. Its third baseline sample is slower than the first two and remains included.

## Before/after campaigns and causal findings

| Campaign | Source | Small dense run ratio | Longer dense run ratio | Representative MoE run ratio | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| W00 initial |7b59d1fd|1.462255|1.436879|1.007479|Initial recorded median-duration ratios;18 successful processes before repairs |
| Intermediate final |d514f417;runtime f9099f85|1.895170|1.829327|0.954938|Median paired ratios after uniform snapshots, before homogeneous block reuse; all18 samples retained |
| Final |c9f8f904|1.775045|1.679023|0.978816|Median paired ratios after profile-supported homogeneous block reuse; all18 samples retained |

`w00_paired_samples.json`, `w11_intermediate_paired_samples.json` and `w11_final_paired_samples.json` preserve separate campaigns. Relative to intermediate candidate medians, final candidate run times drop27.035→25.225ms and53.825→49.268ms. These are separate-window observations, not a randomized causal percentage estimate. The call-count change and independent arithmetic tests prove removal of repeated work. Uniform finalized publication now costs more than the initial W00 implementation in the dense microcases; this remains explicit rather than hiding it behind the historical9.42x number.

The controlled W04 experiment in `w04_performance_review.md` is a different comparison:21 successful subprocesses, real three-request trained synthetic hybrid,336 attention queries/90 hits. Bypassing attention reuse increased median paired Simulator.run time22.34%, with exact normalized outputs. Both real EP-workload-cache controls havezero calls and prove no EP speedup. Component microbenchmarks separate key formation0.333us, attention payload copy5.218us, immutable EP hit2.142us, miss plus explicit clear79.277us, one mutable snapshot17.413us, eight finalized-layer stage5.186us and eight mutable-layer stage137.846us. These component costs must not be added up as measured E2E overhead.

## Separate profiling and allocation ownership

The profile is diagnostic and never participates in acceptance timing. Exact final command:

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m cProfile -o /data/ycfeng/tmp/pr33-w11-small-dense-final-2.profile tests/performance/sim_walltime_scaling/run_case.py --case-json /data/ycfeng/tmp/pr33-w00-20260916/paired/small_dense/candidate/attempt-0/case.json --result-json /data/ycfeng/tmp/pr33-w11-small-dense-final-2-profile-result.json > /data/ycfeng/tmp/pr33-w11-small-dense-final-2-profile.log 2>&1
```

| Small-dense diagnostic | Before homogeneous snapshot reuse | After snapshot reuse | Final block reuse |
| --- | ---: | ---: | ---: |
| Stage assemblies |36|36|36|
| Mutable component snapshots |1440|36|36|
| Physical identity attachment calls |1440|1440|1476 including36 bound source owners|
| Single-layer block calculations |1440|1440|36|
| Stage model-time reads |72|72|72; cached scalar reads|

Final profiling shows the72 `model_time_ms` reads consume a cached scalar without rescanning layer versions. Model spec lookup/range validation and distinct finalized physical identities remain O(N) construction work. The extra36 owner bindings come from reusing the existing homogeneous constructor; they do not repeat mutable snapshots. Stage owners continue to charge PP/draft/terminal once. Reporting is disabled in the acceptance workload, so no heavy EP diagnostic materialization explains this residual.

The profile locates remaining construction work, chiefly shallow identity attachment and validation; its instrumented timings vary and cannot allocate the entire11/20ms wall-time delta to those functions. Source and spies prove that repeated snapshots and identical block arithmetic are removed. They do not prove that every remaining microsecond is logically unavoidable or that the current object representation is globally optimal. `w11_profile_counts.json` preserves all diagnostic counts/timings; raw profiles remain under `/data/ycfeng/tmp`.

## D02 concrete alternatives and recommendation

1. **Retain the current reviewed representation and accept the measured residual for this scoped handoff (recommended).** Preserve required explicit per-layer identity, finalized isolation and the public ordered stage contract. Cost on these fixtures: roughly+11ms/+20ms run time, paired ratios1.775x/1.679x, about+0.5/+0.7MiB median peakRSS; MoE has no observed regression. This is acceptance of these measured cases, not a universal percentage budget, performance guarantee, merge approval or native verification.
2. **Keep D02 open and undertake a separate representation optimization before acceptance.** A compact immutable numerical payload with lightweight layer views could reduce copying the full ExecutionTime object per identity. This must preserve `StageExecutionTime.layer_execution_times`, actual global IDs/family/variant, component isolation, per-layer MoE routing and all reporting consumers. It changes shared entity construction/representation and needs the same arithmetic/isolation/58-case/non-dummy/final performance verification. Its gain is unmeasured; no predicted speedup is promised. Skipping identities or restoring mixed single-layer/stage semantics is not a valid alternative under D01.

Recommendation: option1, because the known repeated work is removed, the scoped residual is measured and small in absolute terms, and the next plausible optimization changes a shared representation rather than fixing a demonstrated correctness defect. The relative dense cost is nevertheless substantial and was presented for the user's explicit decision under plan§6. **User decision received: accept the measured residual, retain the current contract and complete this delivery. D02 is resolved; no broader budget, representation change, native verification or merge authorization is inferred.**
