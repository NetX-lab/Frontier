# Balanced q1 decode preparation archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archived the bounded source/package/CPU deliverable for root review and seal. |
| 2026-10-23 | Added root integration, real prelaunch verification and initial running-worker evidence. |
| 2026-10-23 | Recorded successful GPU execution, actual constructor values and hash-identical outputs; data admission remains pending. |

## Root execution update

Source64133239/records9fb4bd7e are integrated at56e9405d. Two independent source reviews found no actionable issue. Root checked530 package inputs,512 Git snapshot files and preserved originals; actual locked-image60native/520tool/20binary checks and real SDK resource entry passed. `root_seal.json` records the independent seal; the original package location and bytes are unchanged.

The1H800/codesign/non-preemptible run `exp-1023-213905-916183` finished Succeeded at13:48:57Z. Root verified containerCompleted/exit0/restart0,569s container and592s submission-to-terminal (conservative0.16445GPUh upper bound, not billing), exact locked image/H800, unchanged530inputs/originals, and actual nativeV2/eager/physicalTP1-DP1/balanced_topk. Native constructor and resolved scheduler budget6144/max_num_seqs1 now have runtime evidence. Actual memory125Gi equals SDK128000Mi; its initial string-comparison failure remains recorded.

Eight outputs/1,601,865,978bytes were downloaded hash-identically from the exact personal cloud root. Independent per-row/routing/raw-to-CSV/trace validation is active; `root_run_check.json` is `PASS_RUN_EXECUTION_DATA_ACCEPTANCE_PENDING`. No operator-data/physicalTP8/graph/serving admission. At14:18:37Z the32-GPU profile02 job remains Pending/no replicas with launcherPID3236764 alive. See `test_report_2026-10-23_root.md` for commands and observed evidence.

## Historical owner-delivery archive

The following sections preserve the owner-delivery state before root sealing/submission. Their pending launch/runtime checks are superseded by the root update above; they do not describe the current run status.

### Outcome

Task W128-V4-BALANCED-DECODE-PACKAGE completed CPU preparation. Branch `feat/v4-balanced-decode-run`; builder/config commit `64133239c4f4c7fdc50b1b0b85d57d2d72fe3097`. Exactly two maintained non-core files /214 gross added lines against `97803f07`. No core/native/C26/collector modification. Generated immutable package and records are excluded from that budget. No external publish, cloud, StepMind, Docker, GPU or seal operation occurred.

The runtime source remains pinned to `97803f07f72454030a4d3fe538d7cdf473906b8a`; native Git `2cf0a6915ce544dc493a0990f2ea38d81601128a`, locked vLLM0.28 image `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`, accepted C26 SHA256 `d3556a47056bd52680b295f1585906149310397ec53ab90aa0008ff7d57e86ee`, model config SHA256 `b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818`, balanced strategy SHA256 `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`.

## Deliverables

Task directory `/data/ycfeng/Frontier/.worktrees/v4-balanced-decode-run/task_memory/task_2026-10-23_v4_remaining_measurements/balanced_decode/`:

- Maintained builder/config: `prepare_package.py`, `package_plan.json`.
- Executable fresh package: `h800_decode_balanced_02/`; 530 manifest inputs plus manifest. Manifest SHA256 `9714cdb749e8504df32955bb62e080467434e33e20da1d23ae653da2ce8e09cf`. It is local and intentionally untracked/unsealed; root must use this exact path, not a relocated reconstructed path.
- Full input hash map and root handoff: `package_binding.json` and package `manifest.json`.
- CPU verification: `cpu_checks.json`, `test_report_2026-10-23_preparation.md`, `matrix_binding.json`.
- Requirements/plan/progress/checkpoint: `requirements.md`, `plan.md`, `progress.md`, `checkpoint_2026-10-23.md`.
- Historical candidate `h800_decode_balanced_01/` retained unchanged, unsealed and not runnable as submitted: its extra directory level makes the copied relative launcher WORKSPACE locator wrong. Candidate02 corrects only that generated assignment. Candidate01 manifest SHA256 `9ac0e6c09df9755d44980ff448f2e71244dde4b4c6772fc0525213a7f4899910`.

Original q1 527 manifest inputs and balanced-prefill529 inputs remain byte-identical. The corresponding manifest digests are `7929afaef9deced67543262ad9a3eec398048b68914fd0322b00c97416cbe360` and `c6a009f87220c96d19f9b89097762b2f92e3ccdf05199398eaab331510331c8c`.

## Executable plan and expected outputs

Job JSON: `h800_decode_balanced_02/jobs/Q1_WARM_6143_7166_BALANCED.json`.
Job tag: `v4-h800-q1-balanced-6143-7166-20261023-02`.
Resources:1 H800 /codesign, non-preemptible requested,16 CPU /128 GiB SDK setting, timeout1800s /0.5GPUh ceiling. Historical664s /0.184444GPUh is a reference only.

Existing case `e0-decode-q1-c6143-7166`: local seed6141 tokens, two unmeasured decode warmups,1024 measured contexts6143–7166 per CUDA_EVENT/KERNEL_ONLY, six leading layers, physicalTP1/DP1/EP1/PP1/eager with targetTP8-local attention dimensions. The plan retains max sequences128 while the existing isolated engine config is1. CLI chunk budget6144 and the source-observed LLM constructor argument6144 are separately recorded; actual native max_num_batched_tokens is UNKNOWN. This does not establish remote-prefill first-tail or physicalTP8 equivalence.

Current plan derivation and CPU producer aggregation agree:

| Artifact | CUDA_EVENT | KERNEL_ONLY |
| --- | ---: | ---: |
| Measured steps | 1024 | 1024 |
| Calls | 51200 | 45056 |
| CSV rows | 6144 | 6144 |

Raw JSONL expected records:1 engine +2048 measured steps +96256 calls =98305 records, assuming healthy producer completion. Keep the existing kernel trace `profiler_traces/e0-decode-q1-c6143-7166.json`; the producer creates one trace for this KERNEL_ONLY case, not a second CUDA_EVENT trace. CSV routing columns are `routing_runtime_path=balanced_topk`, `routing_strategy_sha256` and full `moe_routing`; origin/native dependency evidence is collected by the unchanged existing runner.

Personal cloud output root is `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h800_v4_decode/97803f0_balanced_warm2_02/Q1_WARM_6143_7166_BALANCED`. The unchanged worker appends phase name `Q1_WARM_6143_7166_BALANCED`, so tables/raw/traces would reside under that repeated phase directory. Expected paths beneath it:

- `raw/dsv4_raw_e0.jsonl`, `raw/dsv4_aggregate_summary.json`.
- `compute/h800/deepseek-ai/DeepSeek-V4-Flash/attention.csv`, `attention_kernel_only.csv`.
- `profiler_traces/e0-decode-q1-c6143-7166.json` and two command logs.
- `status.json` lives directly under the job output root.

Root-only submission command, after independent image/prelaunch/seal acceptance:

```bash
bash /data/ycfeng/Frontier/.worktrees/v4-balanced-decode-run/task_memory/task_2026-10-23_v4_remaining_measurements/balanced_decode/h800_decode_balanced_02/launch/launch_glm_producers.sh /data/ycfeng/Frontier/.worktrees/v4-balanced-decode-run/task_memory/task_2026-10-23_v4_remaining_measurements/balanced_decode/h800_decode_balanced_02/jobs/Q1_WARM_6143_7166_BALANCED.json
```

## Observed validation

- 131 existing profiler/routing CPU tests PASS, zero skips;256 related workload/measured-input tests PASS, zero skips. Initial minimal run's65 PASS /one skipped routing module is explicitly superseded, not claimed complete.
- All530 package inputs unchanged after CPU checks and builder commit;512 source snapshot files match the pinned Git source. 520 tool hashes,60 local native Git pins and20 recorded binary pins checked/bound at the appropriate level.
- Fresh actual generated launcher imports reached submit boundary in both Frontier/StepMind interpreters with no dependency preloads/platform calls. Actual worker stdlib CLI import PASS. Fresh actual bootstrap callback/flat wrapper reached native-run boundary with correct paths/routing using CPU installed-package/binary stand-ins; source/selector/hash negatives and unchanged eight bootstrap controls PASS.
- Synthetic full-window aggregation used existing producer record factories, yielded6144rows/method with correct routing/source/case/contexts, and rejected corrupted routing identity. No consumer layout/admission was introduced; no synthetic timing is hardware evidence.
- Existing fixed77-case matrix comparison explicitly binds unchanged simulator/data/examples/matrix domains to accepted4a1b53e6 evidence; no new capture.

## Pending root acceptance and limits

Root must independently inspect the actual locked image/native source/binaries/bootstrap, confirm current quota and fresh cloud destination, verify actual SDK non-preemptible/platform resources, perform an external seal and then launch under existing D-V4-H800-RJOB-AUTHORITY. The existing launcher does not explicitly pass `preemptible` into ResourceConfig; its default is intentionally unchanged and not established by CPU launcher import. No collector/source change is authorized here.

Actual H800 allocation, native constructor-effective budget, business outputs, retained trace integrity, terminal status, measured routing origin and serving/graph/TP8 qualification remain unverified. Collection does not depend on a decoder sidecar. The delivering owner stops writing after this archive/records commit so root can review and seal stable bytes.
