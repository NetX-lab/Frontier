# Root MoE package and runtime verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Bound independent image/SDK/resource/runtime/data checks and retained the inspector-label failure. |

`root_seal.json` binds receipts containing the exact image/SDK commands, source hashes and results. Locked image2286e853 used Python3.12.3/Torch2.13.0+cu130. Five image witnesses passed:440source/20binary/1003inputs, real worker entry, two four-row dry runs and actual native API imports without CUDA initialization. Receipt `/data/ycfeng/tmp/glm53_dsv4/moe01_image_8266q40i/receipt.json`. Actual GPU selection remained pending at that CPU boundary and was later observed in the run logs/kernels.

Fresh launcher under StepMind Python3.10.6 reached the real ResourceConfig before spawn:1H800/16CPU/mem_gb128/replica1/codesign/non-preemptible, expected local source/cloud mount, timeout1800. No submission during this test. Receipt `/data/ycfeng/tmp/glm53_dsv4/moe01_sdk_owvwod60/receipt.json`. Git-byte equality covered512snapshot files;1003package and527original hashes passed. Measured maintenance is204/300gross lines. Existing77-case comparison is explicitly rebound, not recaptured.

Actual submission command is in `root_submission.json`. Personal job `exp-1023-235250-540507` finished Succeeded15:54:35Z; container15:53:00–15:54:32Z, Completed/exit0/restart0. Submission-to-terminal105seconds is0.02917GPUh, an elapsed upper bound rather than verified billing. Correct memory quantity is125Gi=128000Mi. The initial inspector failed because it expected a qualified quota label and `false`; actual platform values are `codesign` and `no`, consistent with the runbook. The first snapshot is retained in `platform_launch.json`; corrected checks are in `root_run_check.json`. No job setting was changed.

All19 output files/3,261,347bytes were hash-identical to cloud originals. The actual H800/runtime/MARLIN and local rank0 metadata are in `results/data_audit.json`; the twelve original traces remain at the cloud root recorded by `root_seal.json`. Root's full CPU audit invocation is in `results/root_verification.json`: exit0, all non-clock/non-RSS fields equal,112statistic fields with zero difference. Independent review No findings. `results/primary_acceptance.json` records isolated kernel-data acceptance and preserves the CUDA_EVENT and native-tuple/serving limitations.
