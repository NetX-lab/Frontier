# D18-NATIVE immutable stage handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Parent merged db5c99f, repeated585 tests and stage02 hashes; clarified Engine NNN logging context and pending resource dependency. |
| 2026-10-21 | Recorded both offline-verified stages, exact parent commands and unverified runtime dependencies. |
| 2026-10-21 | Superseded stage01 after MRv2 review; built and isolated-verified both stage02 packages and recorded graph evidence limitations. |

## Outcome and identity

**Do not submit qualification01 or canonical01.** They remain immutable historical artifacts from`4a708ff`; their additional source proof incorrectly cited MRv1. Both replacement qualification02/canonical02 packages were built from clean source **2a1070be38ed6fb75dc223c46b0ea407ee908ce6** and independently verified. No native qualification is claimed. Frozen inputs, original Step metadata, historical Step/SharedGPT packages, environment tooling, simulator core and R2 verdict/observations remain unchanged.

| Item | Qualification | Canonical |
| --- | --- | --- |
| Workspace | `/data/ycfeng/tmp/step5_d18_native_20261021/qualification02` | `/data/ycfeng/tmp/step5_d18_native_20261021/canonical02` |
| Stage | workspace + `/stage` | workspace + `/stage` |
| Cloud output, not accessed here | `/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d18_qualification_h200_02` | `/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d18_canonical_h200_02` |
| Job tag | `d18qualh20002` | `d18canonicalh20002` |
| Client population |16 same-shape warmups at QPS1, then8 concurrent same-shape diagnostic requests |Three repeats ×(16 warmups +128 formal requests at QPS1) |
| Fixed shape |6144 prompt /1024 output tokens |6144 prompt /1024 output tokens |
| Config SHA-256 |`b8c6b1e500b5bc44343a59ae56bfad2c03388e39fff10c6b62d5c2c3dbf0b2af` |`8b5cabad51548bff837f9b641e2e6b3cdadd9701c9cf74f334fc7cf87d6bb746` |
| Offline result |27 staged file hashes;47 native-source identity pins, including26 additional proof bindings;3 unchanged model metadata pins PASS |Same counts PASS |
| Worker / replay deadlines |3600s /1200s |5400s /3000s |
| Estimate, not measurement |20–35 active minutes plus queue;21.3–37.3 GPU-hours |35–60 active minutes plus queue;37.3–64.0 GPU-hours |
| Timeout GPU budget |64 GPU-hours |96 GPU-hours |

Each run requests **8 nodes ×8 H200 =64 GPUs**, `step_main`, non-preemptible,32 CPUs and128GiB per node. Each role is P32/D32 TP8/DP4/EP32/PP1. Run qualification and canonical sequentially; they cannot coexist under the70-GPU non-preemptible task cap. Other active jobs must leave the64 GPUs available. Parent checks live quota and creator/source mounts through the approved backend; no quota or authentication call occurred here. Estimates are planning allowances, not extrapolated R1 DP2/chunked throughput claims.

## Offline verification

Stage02 build receipts:`/data/ycfeng/tmp/d18_qualification_build02.json` and`/data/ycfeng/tmp/d18_canonical_build02.json`; isolated verification receipts:`/data/ycfeng/tmp/d18_qualification_verify02.json` and`/data/ycfeng/tmp/d18_canonical_verify02.json`. Each build/verify pair matches exactly. Verification ran from each stage's own workspace without repository`PYTHONPATH`. The durable receipt is this task's`stage_receipts02.json`, including source/config/bindings hashes, resources and unchanged-stage01 evidence. Neither new stage has a`launcher/` directory.

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python \
  /data/ycfeng/tmp/step5_d18_native_20261021/qualification02/stage/tests/integration/step5_pd/canonical_stage.py \
  verify --stage-dir /data/ycfeng/tmp/step5_d18_native_20261021/qualification02/stage
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python \
  /data/ycfeng/tmp/step5_d18_native_20261021/canonical02/stage/tests/integration/step5_pd/canonical_stage.py \
  verify --stage-dir /data/ycfeng/tmp/step5_d18_native_20261021/canonical02/stage
```

Both isolated CLI`--help` checks and both`bash -n submit.sh` checks passed. All eight regenerated node plans match, including native graph statistics,7168/8192/128 limits, disabled chunk/prefix flags and omitted MTP flag. Both stages contain the same26-source MRv2 proof manifest, SHA-256`c8184dcefd0ed30d213f4c6f34edcb8ac771f65cf680a36a567723d68203f914`. Frozen common manifest/SHA256SUMS pins and original metadata match. Stage01 workspace file-name/hash maps are byte-identical before/after new builds; original snapshot SHA-256`9c6ae1c1b4fcc9a955188f54e0bd6bc105c9634d5b4bdc0db549f512918dbc01`. Metadata SHA-256:

- `config.json`: `e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63`
- `tokenizer_config.json`: `78202af487f4d4360e8d15cb0506d60718f8599770599c6f28f7f1fa045a591f`
- `tokenizer.json`: `b564c620eb77fa11d0926011c2202347d6cfc358d79724ee04ae7007e13636f0`

## Parent-only submission commands — NOT executed

The immutable submit wrapper verifies offline hashes **before** loading the existing personal credential environment, sets`STEPMIND_BACKEND=rjob`, and executes the original submitter/launcher callback. Do not background the Python launcher inside the wrapper or kill its process while the native run is healthy. Retain each outer launcher until terminal status. Read the current StepMind handbook and quota first.

```bash
setsid nohup bash /data/ycfeng/tmp/step5_d18_native_20261021/qualification02/stage/submit.sh \
  > /data/ycfeng/tmp/step5_d18_native_20261021/qualification02/launcher.log 2>&1 < /dev/null &
```

After the parent has judged the separate R2 evidence and the H200 qualification, run the canonical package only if its prerequisites are satisfied:

```bash
setsid nohup bash /data/ycfeng/tmp/step5_d18_native_20261021/canonical02/stage/submit.sh \
  > /data/ycfeng/tmp/step5_d18_native_20261021/canonical02/launcher.log 2>&1 < /dev/null &
```

The wrappers source `/data/ycfeng/tmp/step5_native_pd_r1/stepmind_env.sh`; they contain no credentials. Both packages reuse the existing launcher callback copied from the primary task's `native_role_tp16/launch_capacity_20261018_03/launcher_scope.py`. No old frozen stage is modified.

## Required returned evidence and admission boundary

1. Parent R2 content/lifetime disposition remains separate; this branch changes no R2 verdict or observation.
2. Obtain native `nodes/*/result.json` and`nodes/*/server.log` plus platform terminal/creator/resource receipts. Verify all64 distinct H200 devices, both TP8/DP4/EP32 roles, installed hashes, native FP8 exclusions/cache capacity, effective MRv2 selection (Triton available, no MRv1 override) and resolved context/budget/concurrency/graph configuration. MRv2 FULL runtime rows require subsequent healthy serving; config/capture/PIECEWISE alone is insufficient. Enable effective log stats/INFO and retain complete raw process-labelled logs: native `metrics/loggers.py:329–338` prints **Engine NNN** before that engine's graph table. This context supplies lane attribution without additional hooks or table-order guesses. Output-rank statistics do not replace complete rank/forward accounting, but64 separate replay counters are not required. See`source_contract.md`.
3. Obtain `replay/canonical_summary.json`, `replay/client_requests.jsonl` and`replay/proxy_requests.jsonl`. Qualification must retain all24 diagnostic requests. Canonical must retain all432 requests, all384 formal outputs, exact native prompt/output counts and repeated prompt identities. Record lane participation, D's actual recompute tail, transfer/failure deltas, fresh idle boundaries and repeat variation.
4. Any client/observation error leaves reference admission **UNVERIFIED**. The consumer continues healthy requests and later repetitions; auxiliary checks never stop native workers. A non-clean boundary does not become clean merely because the next phase runs. Output metric candidates are not admitted ground truth until parent qualification and the existing ticket14/23 admission checks succeed.

## Reproducible build record

Do not rebuild or overwrite existing stages. Both executed stage02 builds used the command below with`KIND=qualification`/`canonical`,`DIR=qualification02`/`canonical02`, and`TAG=d18qualh20002`/`d18canonicalh20002` respectively. Source`2a1070b` was clean and committed before both builds. Substitute a **new** unused directory/tag/output for any later generation.

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python \
  tests/integration/step5_pd/canonical_stage.py build \
  --run-definition tests/integration/step5_pd/canonical_run_h200.json --kind "$KIND" \
  --workspace "/data/ycfeng/tmp/step5_d18_native_20261021/$DIR" \
  --stage-dir "/data/ycfeng/tmp/step5_d18_native_20261021/$DIR/stage" \
  --output-dir "/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d18_${KIND}_h200_02" \
  --model-dir /data/ycfeng/tmp/step5_native_pd_h800_r1w/reference_inputs \
  --image-source-hashes /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/native_pd_20261020/image_source/sha256_in_image.txt \
  --probe-dir /data/ycfeng/tmp/step5_native_pd_h800_r1w/stage02 \
  --runtime-source-root /data/ycfeng/tmp/step5_profiling_20261020/image_src \
  --launcher-scope /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/native_role_tp16/launch_capacity_20261018_03/launcher_scope.py \
  --job-tag "$TAG"
```
