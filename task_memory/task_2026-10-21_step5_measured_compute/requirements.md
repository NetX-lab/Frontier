# D20 Step measured compute requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record explicit CPU implementation approval,exclusive ownership and unchanged external gates. |

## [Original Request]

YC answered **批准 CPU 实现（推荐）** and released the D3 CPU implementation wait. Stop preparation/diagnostic loops. Finish only the already-collected D19 single-shape judgment,then implement executable measured compute/bundle in `/data/ycfeng/Frontier/.worktrees/step5-measured-compute`,branch `feat/step5-measured-compute`,base `6db2250211a3cf312d17138abdc0cbdcb92449ce`. Preserve old profiling branch. Existing judgment is complete:protocol PASS,quantitative fixed-cost H1 FAIL,original strict FAIL unchanged;11-file64541-byte promotion manifest under the old D18 task.

Own `frontier/attention/families.py`,existing training/typed dataframe loaders,shared prediction manager/registry,and execution-time-predictor compute/typed-cost files plus unique associated tests. C01/C04 trainer/cache/bundle includes timing selection. Do not edit scheduler,events,Request,Batch,comparator or native-client files. Measured-PD owner `ses_ea37359c5ffeoQkLY6yYkhqkx8` owns lifecycle/scheduler only;coordinate necessary interfaces through primary.

Deliver actual **fit -> existing cache publication -> load/lookup -> executable cost replay** from Step producer-shaped measurements,first a complete CPU minimum,then required coverage. Reuse the single current trainer/cache chain and physical ownership. No publisher/framework,flag-only enablement,fake profiling,CUDA_EVENT renamed KERNEL_ONLY,duplicated generic attention projections,or invented kernel counts. Explicitly select native FULL event runtime latency versus eager KERNEL_ONLY GPU-active physical costs,and retain duration/kernel-count/host-gap proof without double counting. Actual unavailable timing/kernel coverage is an explicit error. Fixtures test code only and never admit native evidence.

Integrate BF16 dense/shared,routed FP8/exclusions and all92 SWA/DSA layers through existing registry. Preserve bundle identity/cold-warm and A->B->A cache contracts. The old H200 data may inform shapes but is unadmitted. Native/profile admission and four10% accuracy gates remain unchanged and separate from CPU implementation authority. Do not wait for GLM raw/full profiles before writing CPU code.

CPU env `/data/ycfeng/envs/frontier-py310/bin/python`;profiling-only env oneshot-opd-py312. Primary initialized pinned collective-sim `d28fe917` may be reused through PYTHONPATH if needed;no other-model dataset cloning. No GPU/auth/push. Commit completed verified code steps. Parent owns integration and77-case fidelity;avoid redundant full matrices per intermediate commit.

## Read authority and acceptance

- Primary read-only gap inventory:`/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/d19_pipeline_preparation/commands_and_inputs.md`. Its earlier CPU-wait dependency is superseded by the explicit approval above;external native admission is not superseded.
- First milestone:producer-shaped labelled CPU fixtures pass real trainer/cache/lookup and existing cost interface,with actual count targets,explicit modality identity,positive finite costs,missing-coverage errors and no native-admission claim.
- Completion:source-integrated Step attention/typed FFN cost coverage,existing-chain bundle isolation/reuse,physical-owner ledger regressions and scoped baseline consistency;all production-input gaps reported honestly. Parent may continue CPU pipeline work independently of unresolved native quality.
