# Test report: P2-64 formal native reference, generation08 (2026-11-07)

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-08 | Recorded the removal of the redundant duplicate platform-route copy after re-verifying both copies. |
| 2026-11-08 | Added the independent recomputation of the four metrics from the row CSVs (`headline_recompute.py`), which reproduces the table exactly. |
| 2026-11-08 | Recorded that the planned CPU copy job later succeeded and that both copy routes hold identical job output (`PAYLOAD_IDENTICAL`). |
| 2026-11-07 | Created: generation08 native run, terminal receipt, hash-identical copy, pair coverage, R5 and the four-metric readings; P2-64 formal PASS. |

## Scope

D-P2-64-SIXTH, first job: the generation08 package (`h200_mainline/p2_64_formal_retry08/package_01`, manifest
`1a2bbc13339b7c741cb8151611864c7e7815d8bb180588ffff2700ae888d3508`, harness `57caf2f7` with the peer heartbeat check)
submitted at a night hour. Layout P32+D32, TP8/DP4/EP32/PP1 per role, 8 x 8 H200, all preemptible, step_main, 3600 s.
Workload 3 x 128 requests at 6144/1024 tokens, QPS 1, 16 pair warmups per repeat. Native code, workload, promotion
receipt, R5 and the numerical criteria are unchanged. The Frontier side is the completed formal replay
`/data/ycfeng/tmp/glm53_dsv4/formal_p2_20261102/runs/p2_64` (unchanged; its file hashes are checked before and after).

## Run

| Item | Value | Source |
| --- | --- | --- |
| Job | `exp-1108-002857-059466`, UID `58a67493-ecdc-47e6-abb2-ab042a718957`, creator `i-fengyicheng`, step_main, preemptible label `yes` | `platform_terminal_receipt.json` |
| Submitted | 2026-11-07T16:28:57Z, no queue wait; all 8 replicas Running at 16:34:19Z | `p2_64_formal_retry08/lead_submission_acceptance.json`, monitor |
| Rendezvous | 8 `ready.json` between 16:32:23Z and 16:32:50Z | cloud tree read (label `p64-formal8-inspect-1107s`) |
| Heads ready | decode head gpu-h200-0472 16:42:45Z, prefill head gpu-h200-0031 16:42:53Z | `server_ready.json` |
| Replay | `REPLAY_DONE {"status": 0, "requests": 432}` (3 x (128 + 16)); `replay_finished.json` returncode 0 at 16:53:47Z | `nodes/gpu-h200-0031.../replay.log` |
| Drain | all 8 `drained.json` 16:53:47Z-16:53:49Z, no `failed.json` | cloud tree |
| Terminal | Succeeded, finishTime 16:54:31Z; 8/8 replicas Succeeded, exit code 0, restart count 0 | terminal receipt (label `p64-formal8-post-1107r`) |
| Heartbeats | all 8 nodes refreshed every ~10 s during startup and replay (counts 83-86 at 16:46:47Z) | cloud tree |

## GPU hours

Exact container clocks for all 8 replicas (`container_clocks_complete: true`): 1286-1332 s each, 10,488 s in total.
8 GPUs x 10,488 s / 3600 = **23.31 container GPUh**. Under D-P2-64-SIXTH: 23.31 of 128 counted, within the 64 GPUh
job ceiling. The ended D-P2-64-FIFTH chain counted 68.90 GPUh (upper bounds). Queue wall time is excluded.

## Copy

- The planned CPU copy job `exp-1108-010820-461640` (created 17:08:20Z) was late to start: the platform placed it on
  gpu-h200-0577, which did not have the image; the pull failed at 17:16:25Z with `failed to read expected number of
  bytes: unexpected EOF`, followed by ImagePullBackOff (37 back-offs by 17:29:56Z, phase Starting). Read-only evidence:
  `/data/ycfeng/tmp/ds4_runs/p64-formal8-copyread-1107t/scratch/` (`copy_job_1.json`, `events_1/`). The job holds no GPU.
  A later pull succeeded: the job copied and ended Succeeded at 17:32:43Z, and its own launcher stopped it and exited
  (`p64-formal8-post-1107r/scratch/copy_launcher.log`, `FINAL_STATUS succeeded`). No stop action was needed.
- The copy instead used the read-only SSH route of `h200-ssh-copy-1025c` (label `p64-formal8-sshcopy-1107u`, script
  `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal8-sshcopy-1107u.py`, the 1025c script with this one run and a new target
  directory): remote `sudo -n` hash inventory and `tar` stream of the one run root on shai-ycfeng, local rehash.
- Result: `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261107_08_ssh/dsv4-h200-p2-64-formal-20261107-08`,
  `HASH_IDENTICAL_DOWNLOAD`, 130 files, 75,585,300 bytes, inventory sha256
  `14dc2ae8b5f5f4a0fcdfc28a3a5591471b579d6b06d8cbfde8e998baa66c1ddb`.
- Lead check: an independent local rehash of all 130 files against the inventory found 0 mismatches and no extra files.
- `platform_terminal_receipt.json` (sha256 `08933759...47ca2c`) was copied into the controller run directory next to the
  launcher's `launcher_receipt.json`; `post_run_command.json` now points at the SSH copy and records the reason.
- Because the planned route did finish, the two copies of the same cloud run were cross-checked on 2026-11-08
  (`p2_64_formal_retry08/copy_route_crosscheck.py`, report `copy_route_crosscheck.json`, status `PAYLOAD_IDENTICAL`):
  both hold the same 130 job-output files, 75,585,300 bytes, joint sha256 `5ef547e0a1206076...`, with no file present in
  one copy only and none differing. Only the two files each copy tool writes about its own route differ by design
  (`download_receipt.json` route and timings; `cloud_inventory.json`, whose root records the same cloud directory under
  the `/mnt/ws-jfs/ycfeng` and `/mnt/codesign-exp/ycfeng` mount views). The bytes the comparison consumed are therefore
  confirmed by the originally planned platform route as well, and the copy deviation carries no numerical consequence.
- The duplicate platform-route copy was then removed on 2026-11-08 as redundant scratch, after re-running the
  cross-check immediately beforehand (same `PAYLOAD_IDENTICAL` verdict and the same joint sha256; only its
  `checked_utc` differs). Receipt: `p2_64_formal_retry08/platform_copy_removal.json`, 135 files, 76,001,280 allocated
  bytes, `REMOVED_SURVIVOR_INTACT`, with the SSH copy's joint hash unchanged across the removal. The cross-check script
  therefore no longer has two sides to compare; `copy_route_crosscheck.json` is the surviving evidence, and the cloud
  source under `/mnt/codesign-exp/ycfeng` is untouched and remains the origin if that copy is ever needed again.

## Comparison

Command (`h200_mainline/p2_64_formal_retry08/post_run_command.json`), from the repository root:

```bash
PYTHONPATH=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/formal_p2_64_native_20261102/compare.py \
  --case task_memory/.../h200_mainline/p2_64_formal_retry08/package_01/case \
  --download /data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261107_08_ssh/dsv4-h200-p2-64-formal-20261107-08 \
  --controller-run task_memory/.../package_01/case/runs/groundtruth_clean/dsv4-h200-p2-64-formal-20261107-08 \
  --reference-checker r5 --r4-python /data/ycfeng/envs/openmopd-py312/bin/python \
  --output task_memory/.../h3o_operator_costs/formal_p2_64_native_20261102/generation08_run2
```

The first attempt (output `generation08`) ran without `PYTHONPATH`. It passed the download, policy, coverage and R5
stages, then `run_gate.py` stopped with `ModuleNotFoundError: No module named 'tests'`, so its status is UNVERIFIED. The
retained command record (`formal_p2_64_native_20261102/report.md:108`) sets `PYTHONPATH="$REPO"`; the rerun used that
environment. The first output is invalid and goes to the next cleanup.

Result `generation08_run2/result.json` (sha256 `04a22daf...6f58a8`): status `TOOLS_FINISHED`, no errors, all seven
stages PASS (hash-identical download, run manifest, case and policy binding, coverage, R5, persisted inputs,
comparison). The formal replay bytes are unchanged (command record sha256 `702e9d0a...2d3ac1`).

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Pair coverage soft gate | each repeat warms all 16 P/D lane pairs before formal requests | repeat01/02/03 `COVERED`, 16/16 pairs, no missing pair; `evidence_verified: true` | PASS |
| R5 reader | `PASS` | `{"status": "PASS", "failed": []}` | PASS |
| Routing and tokens | match, no gaps | `routing_status: MATCH`, no request ID gaps, no token count mismatches | PASS |
| Formal openers | none needed after pair warmup | empty for all three repeats; warm reading keeps 384/384 requests | PASS |

Four metrics, relative error `abs(frontier - vllm) / abs(vllm)`, limit 0.10. Frontier values are the same in every
repeat: TTFT 684.31 ms, TPOT 25.30 ms, E2E 26,561.44 ms, THPT 5,975.25 tokens/s.

| Repeat | vLLM TTFT ms | err | vLLM TPOT ms | err | vLLM E2E ms | err | vLLM THPT tok/s | err | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| repeat01 | 703.64 | 2.75% | 24.75 | 2.18% | 26,027.55 | 2.05% | 6,006.10 | 0.51% | PASS |
| repeat02 | 706.66 | 3.16% | 24.74 | 2.26% | 26,010.74 | 2.12% | 6,005.15 | 0.50% | PASS |
| repeat03 | 705.29 | 2.97% | 24.77 | 2.14% | 26,040.12 | 2.00% | 5,992.14 | 0.28% | PASS |
| pooled | 705.20 | 2.96% | 24.75 | 2.20% | 26,026.14 | 2.06% | 6,001.12 | 0.43% | PASS |

The all-request reading (`gate/reading.json`, verdict PASS) and the warm reading (`warm_reading.json`, verdict PASS,
D-WARM-TARGET) are identical here because no formal request needed exclusion. Largest error: TTFT repeat02, 3.16%.

Every number in that table was recomputed from the row CSVs alone, independently of `compare.py`'s own summary, by
`h200_mainline/p2_64_formal_retry08/headline_recompute.py` (result in `headline_recompute.json`). It reads
`generation08_run2/gate/rows/frontier/a10.csv` and the three `gate/rows/vllm/*.csv` files, applies the TPOT filter
`num_decode_tokens > 1`, and pools latency as the mean of the per-repeat means and throughput as total tokens over
total span. All sixteen per-repeat values, all four pooled values and all twenty errors reproduce the table to the
digits printed above; verdict `PASS`, worst `repeat02/ttft` 3.16%.

**Verdict: P2-64 formal PASS** (all three repeats, all four metrics, all-request and warm readings, within 10%).

## Environment

- compare.py: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10), R5 reader with
  `/data/ycfeng/envs/openmopd-py312/bin/python` (Python 3.12); R5 checker sha256 `7a7d6c31...97e30`.
- Platform reads, terminal receipt and copy: Pi on HOST (`custom/deepseek-v4.1-flash-sys`, thinking high), personal
  i-fengyicheng StepMind client, `/data/ycfeng/tmp/stepmind-env/bin/python`; SSH copy with `/usr/bin/python3`.

## Limits

- The SSH inventory root records the shai-ycfeng view (`/mnt/ws-jfs/ycfeng/...`) of the same cloud directory the job
  wrote as `/mnt/codesign-exp/ycfeng/...`. The cross-check above compares two local copies of that directory; it does
  not re-read the cloud source, so it shows the two routes agree rather than independently re-verifying the cloud bytes.
- Two replica audit reads in `p64-formal8-post-1107r/scratch/events/` timed out (ReadTimeout); the job and replica
  phases, container clocks and exit codes come from the terminal receipt, which read without error.
- This verdict covers the P2-64 case with this workload; other ticket 19 cases and items stay as recorded there.
