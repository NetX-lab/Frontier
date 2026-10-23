# D29 trace encoding seam progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | All plan steps completed; verdict recorded. |

## Status

| Step | Status | Evidence |
| --- | --- | --- |
| a. composition | completed | `evidence/composition.json` |
| b. phase split + hypothesis ranking | completed | `evidence/phases-py310.json`, `report.md` section 1 |
| c. one-variable control (wave + 5000 buffer, memory) | completed | `evidence/control-A-py310.json` |
| d. negatives, call counts, variants, py312 replica | completed | `evidence/contracts-py310.json`, `evidence/stdlib-replica-*.json` |
| e. actual emitted wave | completed | `evidence/actual-wave-P1-py310.json` |
| f. scratch proposal + accepted probe body | completed | `evidence/proposal.diff`, `evidence/validate-proposal-py310.json`, `evidence/probe-body-with-proposal.json` |

## Decisions and notes

- I ranked the hypotheses from composition and phase timing before running any
  control (H1 repeated long-string escaping > H2 wrapper overhead > H3 write > H4
  row dicts; H5 callback overhead as the risk).
- The one-variable control patches only `json.encoder.encode_basestring_ascii`
  during the actual unmodified flush. Production-shaped variants are derived
  textually from the actual `flush` source, with exact replacement counts
  asserted.
- The guarded (`type(v) is str`) callback was measured and rejected: the Python
  call overhead cancels the benefit (-0.4 to -0.6 ms).
- Two scripts crashed once on harness bugs (a result-recording `None`, a wrong
  row index) before any measurement was kept. They were fixed in place in
  scratch, and both hashes are recorded.
- Runs that import Frontier need `--working-directory=<source>`, because the
  model config is resolved relative to the working directory. The source tree
  stayed clean.
- Five hash-identical 69 MB buffer copies are root-owned (written under sudo)
  and were not removed (`sudo rm` not authorized).
- The sudo Python runs wrote 284 git-ignored root-owned `.pyc` files (33
  `__pycache__` dirs) into the source worktree. Tracked source is unchanged.
  Removal is left to the parent (exact command in `report.md` Limits); future
  runs should use `python -B`.

## Handoff / next actions for the parent

1. Decide whether a 12-gross `trace_store.py` allocation using the private
   `c_make_encoder` API is acceptable, given the preconditions in `report.md`.
2. Whatever that decision is, the 12-17 ms per-wave budget needs emission work
   (~50 ms per wave, mandated deep snapshots). This task did not study it.
