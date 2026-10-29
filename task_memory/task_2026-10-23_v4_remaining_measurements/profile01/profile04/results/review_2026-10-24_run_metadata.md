# Profile04 run and corrected metadata acceptance

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded independent review, corrected primary replay and limited R3/R4 metadata acceptance; R5 and all cost/reference gates remain open. |

## Outcome
The primary accepts the retained-run-data lane and the corrected metadata lane only.
Authoritative decision: [run_metadata_acceptance.json](run_metadata_acceptance.json).
This is not complete profile04, raw-trace, serving, reference or cost acceptance.

## Independent review
The reviewer inspected the new run/metadata adapters, retained raw evidence,
requirements and derived observations under `W128-PROFILE04-RUN-METADATA-REVIEW`.
R3's limited claims were supported. R4 had one P2: absent `unpublished_tmp_*`
partitions were populated from the formal snapshot. No temporary files existed.

The owner deleted only the 48 inapplicable categories in the new adapter.
The old auditor and sealed/native sources remain unchanged. A real-pairing
negative control and a before/after semantic comparison protect all other observations.
`W128-PROFILE04-METADATA-P2-REVIEW` closed the P2 with no new in-scope findings.
Both reviews were static; their borrowed test/hash counts are not independent execution.

## Primary execution
- [root_cpu_replay_01.json](root_cpu_replay_01.json): real run audit 193 checks,
  99 input bindings, exit0; token-count negative control exit1 as expected.
- Its metadata part is pre-fix history, not current semantic acceptance.
- [root_cpu_replay_02_metadata_p2.json](root_cpu_replay_02_metadata_p2.json):
  corrected 32-rank coordinator exit0 and semantic comparison exit0, output byte-identical,
  SHA256 `8ba69802bdbef62f1dd8cd6e5c979cbf4a60f1a1eb317c84f060b5399712a587`.
- Commands, source hashes, Python3.10.6 and timing are in those receipts.
  Each rank is limited to60seconds/30GiB; at mosttwo ranks run concurrently.
  No aggregate timeout was substituted for the per-rank cap.
- No temporary categories/evidence remain. The 80 direct decode vector joins
  and 1320 counter-matched sampling annotations are unchanged.
- Recording command: `python3 -B /data/ycfeng/tmp/glm53_dsv4/profile04_primary_metadata_record_20261024.py`.
  It checks completed receipts/source/output hashes; it does not repeat traffic or trace capture.

## Limits retained
432 requests completed, but13 children have SIGKILL escalation logs.
32 formal metadata windows closed with192 collector stream entries and zero drops/errors;
stream inventory is not actual trace-ID correlation. Raw DP partitions remain asymmetric,
so80 direct joins do not establish whole-window alignment. R5 remains24/32 ranks complete;
P0's eight original60second timeouts are preserved. Captured ownership, dependencies,
actual transport, cross-host ordering, binary rebinding and cost/reference admission remain open.
