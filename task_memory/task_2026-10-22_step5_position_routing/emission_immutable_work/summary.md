# D29 emission immutable work completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive bounded diagnostic findings, rejected public conversion and parent-owned decisions. |

TaskID `D29-EMISSION-IMMUTABLE-WORK`; session
`ses_e96a5c1c7ffeoPBK7V08CMas7v`; records branch `records/d29-trace-encoding`.
The trace correction is independently committed as `01a1c58d`.

## Findings

- Actual getters: 16 MoE and 16 communication snapshots, not two physical-frame
  getter calls. Recursive baseline deepcopy count 21,472.
- Checked immutable `__deepcopy__` protocol: 49.60 → 46.84 ms emission; mutable
  bytearray/nested-child negatives remain detached. No broad protocol proposal.
- Exact derived-string key memo: →39.96 ms (-9.65 ms); 64 identity reads/checks
  remain, with zero warm emission key encodes. Narrow scratch render only:
  `frontier/cc_backend/collective_plan.py`, 5 gross/net +3.
- Fresh fields conversion: →40.49 ms; diagnostic H2+H3 →31.08 ms. Canonical
  208-event bytes, tuple/order and mutable dictionaries match, but the same H3
  function aliases constructor-accepted raw bytearray receipts. **Do not allocate
  unguarded H3/combined**; preserving general public semantics needs parent design.
- All five unchanged body runs pass complete JSONL/ledger equality, current
  snapshot/ownership tests, 64 wrong-key checks and the 3902 B late-failure prefix.

## Deliverables

This directory: `requirements.md`, `plan.md`, `progress.md`, `report.md`,
`test_report_2026-10-24_emission_immutable_work.md`, checkpoint and `evidence/`.
Decisive evidence: `evidence/evidence.json`, `observations.json`, `h2.diff`,
`filesystem_receipt.json`, plus original per-label result/inventory files.
Scripts/raw outputs only in `/data/ycfeng/tmp/d29_emission_immutable_work/`;
selected scripts retained as `.py.txt` in evidence.

## Limits and open work

No production/test/peer edit, pricing investigation, encoder change, new
dependency, GPU/network or full CLI. Source remains frozen `75404143` and tracked
clean; all 284 previously inventoried root-bytecode files and 33 directories
remain unchanged. No delete/chown; no universal filesystem-cleanliness claim.

Parent owns final interface/design/allocation. H2's safe domain is the actually
deeply immutable reached plans, not arbitrary mutable dataclass extensions.
H3/combined are diagnostic only. The 12–17 ms whole-output budget is still unmet,
and the earlier 15-gross trace escape candidate remains unallocated.
