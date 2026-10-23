# Test report: D29 trace encoding seam (2026-10-24)

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Commands, environments, expectations and observed results for all scratch controls. |

## Environment

- Source: `/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields`, HEAD
  `7540414341d5b81a86b18b9562d1506c0ba74285`, clean. Read-only.
- Python 3.10.6: `/data/ycfeng/envs/frontier-py310/bin/python`. Python 3.12.13:
  `/data/ycfeng/envs/asyncopd-py312/bin/python`, used only for the stdlib
  replica, because `frontier` imports `plotly`, which is not installed there.
- Every command was wrapped as
  `sudo -n systemd-run --scope -q -p MemoryMax=30G [--working-directory=<source>] timeout 60s <python> ...`.
  Every command completed in at most 18 s.
- Retained inputs (read-only):
  - `/data/ycfeng/tmp/d29_immutable_cost_fields_repair/candidate/iteration-0/op_traces.jsonl`:
    body 2,889,655 B, SHA256
    `18d320438ee1fb5f077b12aaa83da8055b47d8cf256d88aedcca159a5d7b819c`;
  - `/data/ycfeng/tmp/d29_cost_fields_parent01/current_setup.py`: SHA256
    `b6c869b2b01bd118b02aaf0a9755ee6b87e25a7f00c9667cbd52d50a2cca2b80`;
  - `/data/ycfeng/tmp/d29_immutable_cost_fields_repair/probe_body.py`: SHA256
    `1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555`.

  The originals were not modified; every new script is a new scratch file.

## Script hashes (final, `evidence/final-script-hashes.txt`)

| Script | SHA256 | Role |
| --- | --- | --- |
| `composition.py` | `b7c20434...` | byte/work composition |
| `phases.py` | `8fb920e6...` | phase-only split |
| `control.py` | `ffc9b978...` | one-variable control A, wave + 5000 buffer + memory |
| `contracts.py` | `5ea23efc...` | negatives, call counts, all variants, buffer P1 |
| `actual_wave.py` | `cce1c7e2...` | real `record_ep_wave` events, actual vs P1 |
| `stdlib_contract.py` | `46e2913b...` | stdlib replica (py310/py312) |
| `validate_proposal.py` | `272fa37e...` | proposal module on real waves + failures |
| `run_probe_proposal.py` | `db5ea160...` | unchanged accepted probe body, TraceStore swapped |
| `proposal/trace_store_proposed.py` | `1a297cc5...` | scratch proposal module |
| `proposal/proposal.diff` | `856bbf70...` | 11+ / 1- |
| derived P1 flush text | `043eef82...` (fixed escaper factory), `f3b649e5...` (actual_wave) | derived from the actual `flush` source (`4517528e...`) |

`scripts.sha256` also keeps the superseded pre-fix hashes: `control.py`
`4ff6a18d...` (crashed on a result-recording bug before any timing was kept) and
`contracts.py` `ff8e58c9...` (crashed on a wrong row index before any output).

## Commands and results

| # | Command (scratch cwd unless noted) | Expectation / pass condition | Observed | Result |
| --- | --- | --- | --- | --- |
| 1 | `python composition.py composition.json` | parsed rows re-encode to the retained body | identical; string values 74.4 % of bytes, long strings 67.4 % | PASS |
| 2 | `python phases.py phases-py310` | actual flush output equals body on all 15 runs | equal; flush 23.52 ms, dumps 22.19 ms, write 2.95 ms, row dicts 0.79 ms | PASS |
| 3 | `python control.py control-py310 11 3` | A bytes equal on wave and 5000-event buffer (buffer SHA `1b4f9ee2...`, 69,414,347 B); escaper global restored | equal; wave 23.46 -> 18.14 ms; buffer 535.72 -> 375.32 ms; tracemalloc peak 382,366 -> 620,077 B | PASS |
| 4 | `python contracts.py contracts-py310-r2 11 3` | negatives equal to actual except where a variant is expected to go RED; wrong identity cache RED; counts differ | 8/9 equal for A/P1, lying subclass RED; guards equal; mutation equal; identity cache RED; escapes 57,248 -> 597 | PASS (RED-capable) |
| 5 | (source cwd) `python actual_wave.py actual-wave-py310-r2 11` | every real-wave whole JSONL equals the first actual flush | equal (whole SHA `d784f7a7...`); 26.60 -> 17.70 ms; only exact `str` leaves | PASS |
| 6 | `python stdlib_contract.py stdlib-py312 11` (py3.12.13) and `stdlib-py310-rerun{1,2} 21` | replica bytes equal to `json.dumps`; same negatives | equal; py312 23.25 -> 16.46 ms; py310 -6.09 / -5.71 ms; lying subclass RED | PASS |
| 7 | (source cwd) `python validate_proposal.py validate-proposal-py310 11` | proposal module byte-identical; late-JSON and circular failures equal, 3902 B, buffer restored | equal; 28.97 -> 19.57 ms | PASS |
| 8 | (source cwd) `python run_probe_proposal.py probe-proposal` | the unchanged accepted body passes with the proposal TraceStore; whole JSONL `1850a9b6...` and ledger `006dc7e8...` equal to the parent | all assertions pass; the values match `d29_cost_fields_parent01/run.log` | PASS |
| 9 | `git status --short` in source after 5, 7 and 8; `find . -newermt '2026-10-24 02:00'` | empty; no file content change | empty; HEAD `75404143`; only 284 new git-ignored root-owned `.pyc` files (33 `__pycache__` dirs) from the sudo Python runs | PASS for tracked source; bytecode side effect recorded |

Additional observations, not acceptance checks:

- `dumps_env_probe.py` measured `json.dumps` at 20.7 ms in bare, Frontier-import
  and ballast processes. This rules out an environment effect behind the first
  replica outlier (71.3 ms), which was not reproduced.
- `stdlib_contract_skipneg.py` showed run-to-run variation of the replica
  baseline (24.6-34.8 ms medians).

## Evidence retained here

`evidence/`: `composition.json`, `phases-py310.json`, `control-A-py310.json`,
`contracts-py310.json`, `derived_flush_P1.py.txt`, `actual-wave-P1-py310.json`,
`validate-proposal-py310.json`, `probe-body-with-proposal.json`,
`stdlib-replica-*.json`, `proposal.diff`, `final-script-hashes.txt`,
`buffer-body-hashes.txt`. Raw timings are inside the result JSON files. Large
JSONL outputs stay in scratch only.

## Limits

These are sequential CPU timings on a shared host. They are not full-CLI, GPU or
native evidence, and the py312 result comes from a stdlib replica only. The
private-API compatibility of `c_make_encoder` is verified for 3.10.6 and 3.12.13
only.
