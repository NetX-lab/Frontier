# Independent Spec review: interval metrics correction

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Reviewed the complete candidate, tests, owner evidence and frozen primary driver using read-only source, JSON, AST and hash inspection. |
| 2026-10-25 | Added actual decode PASS / prefill SIGSEGV control evidence, the pre-metrics boundary and failed acceptance limits; no rerun. |

Component: V4 profile04 P0 offline trace-audit interval representation, `metrics_correction03`.
Reviewer: Codex, independent Spec reviewer; no implementation or primary execution role, no spawned agents.
Checkout: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`; review-start HEAD `8ecc3f93d7540fd54375ba7b3ab1ca08af031a46`.
Concurrent primary work advanced closeout HEAD to `781e902c9df867ae4170209421d29b6a007eba3c`; all three reviewed candidate/test/driver hashes remained unchanged. This reviewer made no commit.
Verdict: **NO FINDINGS within the stated source review scope**. No currently evidenced source defect was identified. The required two-control gate is **FAILED / NOT ACCEPTED**; this review grants no source/control acceptance.
Authority: main task `requirements.md:347–350`, D-V4-P0-AUDIT-INTERVAL-REPRESENTATION-CORRECTION; current `plan.md:56–65`; primary allocation below. The follow-up adds only complete read-only primary-driver review.

Paths below resolve against the checkout; `T` means `task_memory/task_2026-10-23_v4_remaining_measurements`, `C` means `T/profile01/profile04/results/p0_cpu_loop/metrics_correction03`. Absolute scratch paths remain literal.

| Inspected path | Reading range | SHA-256 |
| --- | --- | --- |
| `AGENTS.md` | 1–857 | `82354337fec8131194f31ad8f09d298d2ef6e3ad48282b0b4bd46e87b5a57249` |
| `T/requirements.md` | final decision 347–350; preceding correction/diagnosis context | `b0188a0e32b6197d4cfbc0937aa65a04a7d3e3ab0cef2f37b7ebe857e5ca39f9` |
| `T/plan.md` | current continuation 56–65; dependency/history context | `bbe91969bd83abbf0e04de03d225914f9ff139a82140a6f7f48470a37acb607f` |
| `C/primary/allocation.json` | complete, 1–172 | `d2963e1ee9e4bda21840d2e1234b066ddefc3df07ee232c0f2cb33ee859ab822` |
| `tests/analysis/v4_trace_audit_interval_metrics.py` | **complete, 1–81** | `33fee81fbaad80f89f9c971acf81870c6c6ac78f8e1bbe0d81a6de361f1a17f6` |
| `tests/unit/test_v4_trace_audit_interval_metrics.py` | **complete, 1–243** | `642ab9a4c41c37a94dc3472401dd8fb312e81ada60a35c6a321747ffb601f00b` |
| `C/primary/verify_interval.py` | **complete, 1–150** | `3fca9965b085d898ea2c190dbce634de08ef06f3083d07c2038a5e362bb794d0` |
| `tests/analysis/v4_trace_audit_group_metrics.py` | complete, 1–80 | `4dec265e20426a20e437588a4e44828e257247c82897cc8188621f93369a5251` |
| `tests/unit/test_v4_trace_audit_group_metrics.py` | complete, 1–194, source only | `d9747f5b8bc4371a0ed60c309632a367bb7931ef8c6677d370ef88c10f713827` |
| `tests/analysis/v4_trace_audit_metrics.py` | complete, 1–120 | `6dff04f36dbcd8a66a298e4f6ae88d382bb29813c8a98f8fea91372b271b788a` |
| `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/audit.py` | complete, 1–197; original pure metrics 28–35 | `7ed0ea45a61d5150e2f45bbc1cdf245f2331e40a351c235228b209b064ede0c8` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024/stage_audit.py` | complete, 1–128 | `f7f1253ce9ebae5dcb3b60e2aeb97d7f17f58f34a7d68af56611918f809b9fe4` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024/observe_gc.py` | complete, 1–43 | `3ae64dce5e1498418c3489d48bb74393f08c4dbbcb95231bab4e11a414a5a83c` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024/load_without_gc.py` | complete, 1–34 | `6b91e25fc592d4f1340ef22e6d6e73c4dd8ed6e7ce6239f7a113d946a74edd02` |
| `tests/comparison/calibration/forward_device_timeline.py` | complete, 1–210; JSON loader 79–81 | `fe4c3f82500df79ce097ce762808bcaf2b78040e41cbab942ccc746a723b3a52` |
| `C/owner/source_notes.md` | complete, 1–25 | `e4e4282f725ef129359df4ff8d0e227dfd3a8f6b1515194927ff1120e85a81b6` |
| `C/owner/unit_execution_allocation.json` | complete JSON, 1–183 | `3d145640dfe664366957bcaa83b5c88d4cb8f56725d8fbb72f60c7da97ab4375` |
| `C/owner/unit_receipt.json` | complete JSON, 1–109 | `3994411cc5648e78fcc0766729042c760727f97dba9f101065507dccd981769e` |
| `C/owner/unit_assertions.json` | complete JSON, 1–8222; all 90 actual/expected pairs | `cffa30667c60455e5f4569d095f86c2887c9c37808315a21e81c3c358545c39b` |
| `C/owner/unit_final_verification.json` | complete JSON, 1–86 | `d5d05e592229fc56e8629643dfd83592403f4c7e98b1eeec90f88413ceb8418e` |
| `C/primary/unit_allocation.json` | complete JSON, 1–118 | `18f0de920bd0190cdf33abd561dbdac1cdc30dce4def371d6d3058ba157226b0` |
| `C/primary/unit_execution.json` | complete JSON, 1–24 | `82152bb4395974824178197758ae457d44bb32fa43bbd01182cf86917b225d16` |
| `C/primary/unit_result.json` | complete JSON, 1–3603; all 36 checks | `82e28106a57d167de2ab862eebefa7bd9980769ba017c84b85c018cda147a1a7` |
| `C/primary/control_decode_execution.json` | complete JSON, 1–24 | `27d6f72e425a3114e556b3d20c957ef1e804d6471f681bc93998a8bec733204a` |
| `C/primary/control_decode_result.json` | complete parsed JSON, 1–94503; all 37 checks | `2f1bda47290745e4d9545fa8aa96a7479fffae5466ef97427948b4e2fb48004b` |
| `C/primary/control_prefill_execution.json` | complete JSON, 1–24 | `4df3fced9966a7b08016baa8897ef75459100bae3c7154ddc83048106c4847a1` |
| `C/primary/control_prefill_result.json` | complete JSON, 1–214; all 29 checks | `21d727bfcfec694daeff3b18af8f4b9f39c77de6839305aebe8a3e366fd16059` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_metrics_interval_controls_20261025_01/control_prefill_dp1_rank0.hb.log` | complete, 1–48 | `cab659d03703d3b50a65cbcf8537578ff86b7f12f1309d6e54713e34cc3166d4` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_metrics_interval_controls_20261025_01/control_prefill_dp1_rank0.stderr` | complete, 1–2 | `f5d3a3c9827fa6433fef5106e79296f749455563bb861076dfa089a247031fb8` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_metrics_interval_controls_20261025_01/control_prefill_dp1_rank0.obs.jsonl` | complete, 1–29 | `37008d8b3c121b0cc3f8fd579383cceb56ae066d5dcab430a7385f5be794a12d` |

## Findings and requirement coverage

No actionable Spec findings. The following observations explain the reviewed acceptance boundary and source line locations.

- **Count/sum and input preservation:** candidate 31–34 keeps the accepted `len(members)` and `sum(e['dur'] for e in members) / 1000` AST and trace order, including repeated members; it neither sorts nor mutates the input lists. Original pure metrics 34 uses the same accumulation expression. Candidate 42–44 preserves metric fields and group insertion order.
- **Exact recurrence:** candidate 35 and 40–42 preserve initialization, both original `union += max(0.0, b - max(a, end))` / `end = max(end, b)` statements and final division. Both statements are exact AST matches to auditor 30–35, with no tolerance or arithmetic rearrangement.
- **Positions, ties and duplicate identity:** candidate 26–29 retains every event occurrence in the stable global `(ts, ts+dur)` order and caches unchanged numeric endpoints. The identity map selects the last occurrence of a repeated object, whose endpoints remain identical. Candidate 37 emits a position for every subgroup member, including repeats. Distinct events tied on both endpoints can change identity order relative to subgroup-local sorting without changing the interval recurrence values. No member is deduplicated.
- **Full-list reuse:** candidate 36 selects `range(len(ordered))` only when `members is events`, preserving every global occurrence. The accepted batch construction at `v4_trace_audit_metrics.py:27–34` supplies this identity for `('total', None)`; an equal copied list takes the subgroup path. Empty lists remain safe. Auditor 56–76 and 97–118 build groups from the original JSON-loaded kernel objects; hierarchy may append the same kernel multiple times, which remains supported.
- **Composition and outside behavior:** candidate 14–21 pins/reuses the accepted adapter; 47–60 deep-copies the composed AST, checks exactly one matching accepted definition, and replaces only `batched_metrics`. Lines 57–58 retain both sparse observations. Lines 63–81 preserve composition and entry. All five substitutions and group construction in accepted metrics 16–35/59–110 remain unchanged, as do auditor attribution, raw IDs, overlap/reversal logic and index writes. The frozen stage and loading/GC chain remains unchanged.
- **Unit coverage:** new tests 63–179 check original pure metrics floathex, count/key order, input values/list identities, empty/mixed keys, ties, reversed sum order, global duplicate identities, repeated memberships, recurrence and composed AST. Lines 181–204 inspect frozen sensors and rejected AST shapes. Lines 207–243 retain exclusive owner evidence writes and the allocated one-shot entry. Owner assertions are deep-copied, not mutable result aliases.
- **Primary full JSON check:** driver 25–32 recursively converts every JSON float value to `.hex()` and keeps dict/list order. Control 99–111 compares the entire payload after removing only elapsed/RSS and generated index path/hash; fresh path/hash, elapsed/RSS, useful steps and child memory are separately checked at 101–107. Count/sum/union, hierarchy/leaf/per-stream and every other retained non-clock field participate; there is no float tolerance or selected-metric-only comparison.
- **Primary gzip completion:** driver 113–121 reads both indexes in binary chunks, asserts byte equality **before** the empty-chunk break, and therefore requires both streams to reach EOF together. Standard-library gzip reads surface truncation/CRC errors. Lines 119–124 retain decompressed digest/size/newline count, check rows against metadata and return `clean_gzip_eof=True` only after successful completion; 142–144 propagates errors as FAIL. This describes the checking mechanism; actual control limits are recorded below.
- **Primary execution scope:** driver 125–150 records one-shot start/result artifacts, checks frozen pins, parent 30 GiB, original Python/environment, then dispatches the allocated mode. Unit 48–73 uses four fixtures against original pure metrics; scale 74–90 uses the pinned accepted pure function once before the candidate and reports only that synthetic pair.

## Evidence, method and limits

Observed retained owner evidence: 9 tests, 90/90 actual/expected assertions equal and marked PASS, exit 0, 1/1 spent, Python 3.10.6, child 32212254720 bytes; all six receipt artifact hashes/sizes and all 18 reviewed source/runtime pins matched. Primary retained unit evidence: four fixtures, 36/36 checks equal/PASS, exit 0, 1/1 spent, parent 32212254720 bytes; all 18 primary pins matched. These are read-only receipt observations, not executions by this reviewer.
Read-only calculation command: `/usr/bin/python3 -I -B -` with inline standard-library `ast`, `copy`, `hashlib`, `json`, `pathlib` code; no local module imports, `compile`, `exec`, metrics calls, trace decompression or harness runs. Independent static reconstruction from frozen AST/literals passed 18/18 comparisons, including both recurrence statements, all five substitutions, outside-metrics AST, two observations, aliases, entry and five cgroup expressions. Source/test/driver hashes were rechecked before writing.
Review scope self-check used `check-progress`: source/evidence inspection advances the allocated independent review; primary execution remains separate. Only this report is written; no source/pin/main-record edits, commits, spawn, unit/scale/control/diagnostic or GPU invocations.
Actual controls, retained evidence only: decode parent exit 0 / enclosing 6.947991s; result PASS, 37/37 checks, exact non-clock JSON and floathex comparison, 142567 equal index rows / 10462613 decompressed bytes, clean gzip EOF recorded. Prefill parent exit 1 / enclosing 20.366353s; driver result FAIL after 20.295309s, with 28 passed preflight checks and `control-returncode` actual -11 versus expected 0. Parent 30 GiB is recorded; the subsequent child-memory, final JSON and gzip comparisons at driver 99–124 were not reached. Heartbeat 45–48 records Fatal Python error: Segmentation fault / `<no Python frame>`, distinguishing SIGSEGV from timeout.
Pre-metrics boundary: observations 18–29 enter kernel pass at 15.331s and end at 19.943s / `kernel_pass_progress idx=1175000`; this is an original traceEvents index, not a count of completed kernels. There is no `kernel_pass_done`, metrics start/end or final JSON. Heartbeat 35–44 last periodically samples JSON encoding at original auditor 119, which serializes kernel-index rows. Frozen stage 53–54 orders flushed `kernel_pass_done` before the batch inserted at accepted metrics 26–35; candidate metrics starts with the copied observation. Retained execution is clearly pre-metrics and provides no metrics equality evidence for prefill. The fatal native instruction/writer is unmeasured; the preceding encoder sample is not a fault-site attribution and does not rule out earlier startup/composition effects.
Acceptance limits: both control invocations are spent, 2/2, with no rerun. Decode PASS cannot substitute for failed prefill; required source/control acceptance remains NOT ACCEPTED. No failed/partial prefill index EOF, complete row count, payload equality or candidate performance is claimed. Scale-result acceptance is outside this review. No harness was executed by this reviewer; decode EOF is a retained receipt observation, not independently replayed trace evidence. Source review covers the frozen JSON trace pipeline and recorded numeric fixtures; no arbitrary mutable-caller or new parsing contract is inferred. A single synthetic pair does not establish real P0 performance or general speedup.
R5 remains 24/32 under the allocation. New affected/GPU allowance is zero. Original allocator corruption writer remains UNKNOWN; unit/source/scale/control success cannot establish allocator cure, native ownership/dependencies, serving-cost or reference admission.
