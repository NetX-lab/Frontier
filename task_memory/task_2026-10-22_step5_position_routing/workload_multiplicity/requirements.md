# D29-FULL-WORKLOAD-MULTIPLICITY requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Create the read-only multiplicity/volume determination requested by the parent after main08. |

## [Original Request]

TaskID **D29-FULL-WORKLOAD-MULTIPLICITY**. Before another expensive full CLI,
determine from the actual scheduler/pricing/emission call chain how many
measured 208-operator waves and trace bytes the unchanged 6144-prefill /
1024-decode workload must execute, and establish a defensible projected CPU
floor/range from the accepted phase timings. Do not assume `88 x 1024`: inspect
the real layer grouping, multiplicity and expand-trace flags, and the
first-token/off-by-one behavior.

Constraints carried by the request:

- Source fixed at `48d34106` (code `eece87b2`) in
  `/data/ycfeng/Frontier/.worktrees/step5-preview`.
- Read-only against primary code/tests/inputs/run data. No source
  implementation, no new budget, no GPU, no network, no dependencies.
- Records only in the isolated worktree `.worktrees/d29-workload-multiplicity`
  at `48d34106`, under this `workload_multiplicity/` directory.
- Use retained main08 command/execution/ledger metadata and, where needed, only
  bounded (<= 8 MiB) tail reads; no huge trace/CSV scan.
- Do not duplicate the common-collective-lifetime investigation, the
  cost-fields diff review, or any identity-cache work already measured as
  no-gain.
- Do not propose disabling traces, shrinking the workload, raising the timeout,
  numeric/proof sharing across layers, or unsafe aliasing.

## Questions to answer

1. Is one whole 208-event wave executed once per actual routed layer, or once
   per model group?
2. How many distinct phase/wave/pricing/emission calls per decode token follow
   from the real main08 argv and its constructor flags?
3. Are persistence-side operator traces sampled, limited or collapsed, or are
   all 1024 tokens x all layers persisted?
4. What source-only or bounded retained evidence proves the multiplicity?
5. What is the projected CPU floor/range, with startup accounted separately?
6. If even zero pricing cannot fit 2400 s because of mandatory
   emission/output, which exact existing mechanism (static proof encoding,
   layer grouping, ...) would justify a future RED probe without changing
   output or accepting unsafe aliasing?

## Acceptance used here

- Every multiplicity claim is traceable to a source line or a retained,
  bounded measurement, and synthetic counts are labelled as such.
- The arithmetic shows its inputs, includes output bytes, and separates
  measured from extrapolated terms.
- No file outside this records directory is modified.
