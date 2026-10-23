# D29 trace encoding seam completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive verdict, deliverables, validation and open work. |
| 2026-10-24 | Correct 15-gross artifact accounting, repeated-wave buffer and 8 supported + 1 RED cases; replace cleanup recommendation with exact read-only manifest. |

TaskID: `D29-TRACE-ENCODING-SEAM`. Read-only analysis on frozen source `75404143`.
Records branch `records/d29-trace-encoding`
(`/data/ycfeng/Frontier/.worktrees/d29-trace-encoding`).

## Verdict

Conditional **SAFE_SEAM**: per-flush exact-value reuse of the stdlib JSON string
escape through `json.encoder.c_make_encoder`, with byte-identical output, errors,
3902 B prefix and buffer restoration. Flush savings:

- 5.3-6.1 ms per fresh-parse wave;
- 8.9-9.4 ms per real emitted wave;
- 160 ms per real-sized 5000-event buffer of freshly parsed repetitions of one
  wave (-30 %); cross-layer equality was not proven.

The scratch proposal is 15 gross lines (14+, 1-, net +13) in
`frontier/metrics/trace_store.py`, from `git apply --numstat evidence/proposal.diff`.
Preconditions: a private CPython API (verified on 3.10.6 and 3.12.13), and string
leaves that keep `str` equality (true for every observed leaf and for every
`frontier` str subclass).

It does **not** meet the 12-17 ms per-wave whole-output budget. Emission alone is
about 50 ms per wave. The projected zero-pricing total stays at about 7.0-7.6 ks
per mode.

## Deliverables

- `report.md`: hypotheses, controls, negatives, proposal shape, budget statement.
- `test_report_2026-10-24_trace_encoding_seam.md`: commands, hashes, PASS/FAIL.
- `requirements.md`, `plan.md`, `progress.md`.
- `evidence/`: result JSON files, `proposal.diff`, derived P1 flush text, script
  and buffer hashes.
- Scratch: `/data/ycfeng/tmp/d29_trace_encoding_seam/` (scripts, raw outputs, one
  reference 5000-event buffer body plus five root-owned duplicates).

## Observed validation

Every timed run wrote byte-identical output. The RED-capable negatives worked:
eight supported cases are equal for P1; the ninth lying-str case is deliberately RED.
a lying str subclass and a wrong meta-identity cache both went RED. The unchanged
accepted probe body passes with the proposal (whole JSONL `1850a9b6...` and
ledger `006dc7e8...` equal to the parent). Escapes per wave fall from 57,248 to
597. The extra peak allocation is +238 KB per 5000-event flush.

## Open / deferred

No allocation was granted, and no production edit was made. The filesystem
inventory lists 284 git-ignored root-owned `.pyc` files and 33 directories,
consistent with the runs but lacking a pre-run manifest for exclusive attribution.
Tracked source is unchanged. `evidence/pyc_manifest_2026-10-24.json` gives exact
paths/owners/sizes/timestamps/SHA; cleanup advice is withdrawn, with no deletion
or chown. Still open: the
parent's decision on the private-API trade-off, emission work for the budget,
full CLI, strict77/full-unit and native acceptance.
