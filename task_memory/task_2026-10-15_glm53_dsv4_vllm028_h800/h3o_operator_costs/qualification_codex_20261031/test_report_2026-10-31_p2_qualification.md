# P2 qualification evidence verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Completed the continued evidence verification and scoped Git delivery checks; retained the original task date and source receipts. |
| 2026-10-31 | Recorded CPU-only retained-artifact capture, arithmetic verification and initial authority drift. |
| 2026-10-31 | Recorded final capture/verification PASS, exact native source-pin checks and scoped Git verification. |

This report verifies the local investigation deliverable. PASS here means evidence consistency and report validation, not native device-timing, predictor qualification, promotion or final accuracy acceptance.

## Environment and exact commands

Working directory: `/data/ycfeng/Frontier/.worktrees/b30b-ex-situ-link-probes`.

Python: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6, GCC 11.3.0. Use `-B`, `PYTHONDONTWRITEBYTECODE=1` and `TMPDIR=/data/ycfeng/tmp`. The capture uses only Python standard-library local file reads and `git`/`rg`; it imports no torch or simulator package and performs no cloud download or platform API request.

Capture command:

```bash
env TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/qualification_codex_20261031/evidence/audit.py
```

Verify command:

```bash
env TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/qualification_codex_20261031/evidence/audit.py --verify
```

Source identity snapshot uses `git rev-parse HEAD`, `git status --porcelain=v1`, `git diff --binary` and `git diff --cached --binary` in this worktree. `/proc/<pid>/cwd` was read only to detect another writer in this worktree; no lead job/session was inspected. The exact empty diff hashes and observed process are in `evidence/resume_worktree_snapshot.json`.

## Expected and actual results

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Initial capture frozen-authority assertion | All current main authority hashes equal initialization | Main plan/progress changed while the lead updated the route; initial index retained | FAIL, assumption corrected |
| Existing capture collected after resume | Complete named retained sources, no new run | exit0, 160 sources, 96 rank inventories, 12 node results, 12 startup logs, 24 metric rows | PASS |
| Initial retained-byte verification | Exact source bytes, selected line/JSON/rank pointers | exit0 at 2026-10-31T15:49:34Z; 160 hashes, 73 selectors, 384 rank pointers | PASS |
| Final capture and report verification | Correct current Ticket09 excerpt; exact hashes, packed rows, graph logs, 14 report criteria and source references | exit0 at 2026-10-31T15:58:36Z; 160 hashes, 73 selectors, 384 rank pointers, 96 packed rows, 96 graph-memory rows, 37 report source keys, 14 criteria, 24 metric rows | PASS |
| Native packed extent arithmetic | 46 blocks * 1,002,240 B = 46,103,040 B/lane | 46,103,040 B/lane | PASS |
| Component accounting arithmetic | Sum of [24,2,2,2,16] * component pages = 39,521,664 B/lane | 39,521,664 B/lane; shortfall 6,581,376 B, 14.27536231884058% of packed extent | PASS; wire observation missing |
| Native startup memory identity | requested - available_KV = consumed + transient headroom at every rank | 96/96 exact identities | PASS, startup scope |
| F4 planner formula | Reproduce 116,034/120,623 with existing terms; compare native 109,934/112,914 | +6,100/+7,709 blocks; +5.548783815743992%/+6.827319907186% | PASS arithmetic; diagnostic MISMATCH |
| Existing F4/B28 comparison JSON | Reuse two FINISHED comparisons, five equal metrics per scale | Both statuses FINISHED; 10/10 recorded metric equality entries true | PASS record verification only |
| Existing native R4 | Preserve both original verdicts and launcher/manifest/terminal binding | 2/2 PASS; 96 ranks; Succeeded, exit0, restart0 | PASS, original admission only |
| Existing B29 error arithmetic | 24 finite abs(Frontier-native)/abs(native) values match reading | 24/24 match; P2-32 maximum 5.607176023817675%; P2-64 repeat01 TTFT 12.254940665008354% | PASS arithmetic; preserved P2-64 numeric FAIL |
| Limited promotion/admission filename check | Report only the named roots and exact result | rg return1, no matches; original manifests have no promotion/admission fields | PASS limit, no global absence claim |
| Sidecar worktree before edits | Preserve existing edits; no independent writer observed | Empty tracked/index diff, clean porcelain; only this capture process with sidecar cwd | PASS, observation scope |
| One-off memory display | Show workspace inventory values | First display TypeError on list set-membership; corrected JSON display shows one-entry lists 1.5/3 GiB | FAIL display, then PASS; collector unaffected |
| Reproducible source-bound verification | Run the final audit.py --verify command, including the three native git-object checks | exit0 at 2026-10-31T16:02:56.350285+00:00; 160 source hashes, 384 rank pointers, 96 packed rows, 96 graph-memory rows, 37 source keys, 14 criteria, 24 metric rows and 3 native git objects PASS | PASS |
| Native source-pin binding | Captured NIXL and memory/counter source bytes match original native source commit | Native HEAD 2cf0a6915ce544dc493a0990f2ea38d81601128a; 3/3 git objects match local captured bytes | PASS |
| Scoped Git verification | Stage this qualification directory only; no source/README/main docs | Pending staging and commit | Pending |

The initial authority mismatch is a document concurrency observation. Initial main plan SHA-256 was `34a45e6f4a7116a3707cade2a02fcfc5854afab04ea6e9b1206c0baf5cfc6d0f`; captured current plan is `939f953ba278cf62413e7ae00f208460374d3ffeba2dda38c38ca31f55887e5e`. Progress changed from `93796a9476792a7dac36dac62babc532d79798136159c90293d6aae143c3d35e` to `cfe15f6d87f593047142f474d02d9425550932cc6eacadb981b06e74d9ff5c75`. The initial capture guard was replaced with current source bytes and explicit changed-since-initial flags after the current route was read. No old measurement or receipt was changed.

## Predicted and actual values

| Quantity | Predicted/accounted | Native/actual reference | Absolute difference | Relative difference |
| --- | ---: | ---: | ---: | ---: |
| T6144 component vs source-derived packed extent, B/lane | 39,521,664 | 46,103,040 | 6,581,376 | 14.27536231884058% |
| P2-32 F4 pool, blocks | 116,034 | 109,934 | 6,100 | 5.548783815743992% |
| P2-64 F4 pool, blocks | 120,623 | 112,914 | 7,709 | 6.827319907186% |
| P2-32 B29 repeat03 TTFT, ms | 344.8789913386803 | 365.3656886308454 | 20.486697292165104 | 5.607176023817675% |
| P2-64 B29 repeat01 TTFT, ms | 684.313826604259 | 779.8887274002482 | 95.57490079598915 | 12.254940665008354% |

The packed extent is source-derived; registration geometry is observed, but the request's wire count is not. F4 and B29 values are existing artifacts, not reruns. All denominators above are nonzero. A zero native denominator would be undefined and could not qualify.

## Verification limits

- This audit does not establish actual per-request wire bytes, rank receive/deduplication scope, transport timing or device completion.
- Native counters overlap and independent per-phase peaks are explicitly unverified. Graph GiB values are rounded log observations. No exact quantized capacity framework or disjoint category sum is introduced.
- Existing F4/B28 equality reports are reused; actual metric files are not compared again and the workload does not exercise a capacity limit.
- Existing B29 errors are arithmetically checked without rerunning gate.py, A6, A10 or any replay. All old verdicts and all open tickets stay unchanged.
- Original R4/basic authorization is historical evidence. This investigation is not a pre-launch promotion receipt and does not change future resource admission.
- No GPU, platform, cloud, UUID selector, F2 analysis or simulator source operation was performed. Additional GPU-hours incurred by this audit: 0.

## Final native source identity check

The final source check uses `git -C /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 rev-parse HEAD` and compares local bytes to these exact read-only objects:

```bash
git -C /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py
git -C /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/worker/gpu_worker.py
git -C /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/utils/mem_utils.py
```

The comparison is byte-for-byte in the local Python command, and `evidence/verification.json` records all three object paths and PASS. This binds the descriptor and memory explanations to the original clean source rather than relying on the directory name alone. No native source was edited or executed.
