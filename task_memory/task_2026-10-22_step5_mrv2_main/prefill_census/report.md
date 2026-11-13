# Native P computation versus the current CPU fixture

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Confirm native6128/current CPU6144 with parent evidence; record a separate SWA payload discrepancy and pending decisions. |
| 2026-10-22 | Parent verifies three exported block lists:33 SWA slots include one null;32 pages carry KV. |

## Confirmed computation mismatch

The canonical prompt has6144 unique tokens. Native computes6128 on P and16 on
D. Current actual06 schedules6144 on P and16 on D:6160 scheduled prompt-token
computations, with16 positions computed twice.

`parent_receipt.json` verifies four native source hashes against the frozen
capture01 configuration and preserves its actual business-forward row104 from
`prefill_pid_2199.jsonl`: query6128, context0, padding6128, graphNONE. The same
receipt preserves both actual06 scheduler row5 records: P6144 on source7e8bea3.
These are observed partial-run records, despite both full requests timing out.

Locked `nixl/base_scheduler.py:393-456` computes the remote prefix at a replay
block boundary and truncates the P scheduler request before scheduling. Frontend
request-length metrics may retain6144; they are not compute-width evidence.
`payload_parent_receipt.json` also checks the native counter deltas:
P local6128; D external6128 and local16.

| Prompt length | Native P computation | Full-history prefix pages | D local tail | D physical width |
| ---: | ---: | ---: | ---: | ---: |
| 6143 | 6112 | 382 | 31 | 32 |
| 6144 | 6128 | 383 | 16 | 16 |
| 6145 | 6128 | 383 | 17 | 24 |

The6145 native boundary control uses1023 outputs to stay within7168 tokens;
the canonical workload remains6144/1024. Logical prefix pages are not the same
as aggregate physical pool ownership across full-history and sliding groups.

## Pending P correction

P and D share one Frontier `Request`. Simply reducing its semantic prompt length
would change D's completion arithmetic. A correction must distinguish the P
compute target from the6144-token client prompt, complete P at that target,
derive source ownership/transfer from the computed prefix, and preserve D's
received-prefix/local-tail split and1024 output credits.

The existing P admission and completion/transfer path is the proposed seam:
`VLLMv1EngineReplicaScheduler.add_request`, `vllm_v1_kv_allocation.py`, `Request`,
and `cluster_batch_end_event.py`. Reuse the existing family/prefix contract;
sliding-window presence alone is not evidence of Step's replay rule. Do not
introduce another independent boundary formula or mutate and restore semantic
prompt length across clusters without an explicit lifecycle design.

Fresh CPU coverage must include P6128 and the actually reached real/dummy
aggregate widths, then go through the existing fit/lookup path. Preserve the
original fixture02. New lifecycle behavior and changed fixture acceptance are
pending a user decision; the current fixture is not native-equivalent.

## Separate payload discrepancy

Parent execution of the current TP8 payload resolver for a6128-token transferred
prefix produces140,734,976 bytes/rank. Native canonical counters record
1,119,096,832 bytes and158,608 descriptors over8 reads: means139,887,104 bytes
and19,826 descriptors/read. See `payload_parent_receipt.json`.

The numerical difference closes exactly to one SWA page in each of69 layers:
`69 * 12,288 = 847,872 bytes` (0.6061%). Current components use33 SWA pages;
32 SWA pages plus383 sparse-raw and383 summary pages close both native totals.
This is independent of changing P6144 to6128: the current SWA cap is33 for both.

The native clip and the shared Frontier helper both contain a conservative
`ceil(window/block)+1` bound. Parent receipt `swa_parent_rows02.json` now binds
proxy rows17/18/19 and their complete source-file hash. Each exported SWA group
has33 slots with its first slot0 and exactly32 non-null blocks. The full-history
groups have383/382/383 pages for canonical6144/boundary6143/boundary6145.

Locked native `nixl/base_scheduler.py:153-161,264-305` declares and applies the
conservative clip. `pull_scheduler.py:266-278` documents leading null blocks;
`base_worker.py:3481-3487` end-trims the source group to the D group length before
READ descriptors are built. D's runtime group list itself was not captured, so
that final list is source-derived and corroborated by the independent descriptor
and byte totals. The separate native CPU allocator also reserves32 SWA blocks
per group for the same6128 external prefix.

The confirmed discrepancy is cap-as-payload: the current Frontier component
resolver bills all33 slots as pages carrying KV. A Step-scoped correction must
use the actual retained page range, keeping conservative capacity separate.
Do not remove every shared `+1`, assume arbitrary unaligned windows equal the
cap, or derive Step backoff eligibility from sliding retention alone. The
DeepSeek-V4 family also has sliding groups with a different transfer contract.

## Status and scope

Computation mismatch and payload arithmetic: **CONFIRMED**. P lifecycle fix:
**PENDING DECISION**. Effective native SWA-page selection: source/raw-list
mechanism established within the stated runtime-list limit; correction remains
**PENDING DECISION**. Full CLI remains limited by CPU runtime and its2400s cap; native
qualification and accuracy remain unverified. No old row is relabeled and no
native setting, criterion or sealed output is changed by this investigation.

Verification note: initial `swa_parent_rows.json` is an empty failed output from
using unavailable `hashlib.file_digest` in system Python. The corrected bounded
streaming SHA256 command writes only `swa_parent_rows02.json`; no failed artifact
is promoted or overwritten.
