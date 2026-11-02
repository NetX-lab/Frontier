# Metrics correction CPU test report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Recorded the first passing unit run, exact AST controls and source-only delivery limits. |

Allocation: D-V4-P0-AUDIT-METRICS-CORRECTION. The document uses the requested Hong Kong date; receipts retain actual UTC timestamps.
Environment: local /usr/bin/python3, Python 3.10.6, standard library, no conda environment or dependency installation. Pytest is absent in this interpreter; unittest is sufficient.

```bash
timeout 60s /usr/bin/python3 -B tests/unit/test_v4_trace_audit_metrics.py -v
```

Expected: eight tests pass with exact equality, preserved input/group order and no frozen source changes. Actual: eight passed, zero failures/errors/skips, exit 0; unittest 1.823s and enclosing subprocess 1.91348662879318s. This small source/AST workload required no memory-heavy inspection; no 30 GiB scope is claimed for it.

| Control | Actual | Result |
| --- | --- | --- |
| Frozen metrics comparison | Exact count/sum/union for empty, ties, overlap, nested/fractional intervals, repeated memberships and every metrics family/key type | PASS |
| Global metrics sort | Exactly one stable sort; ties retain original order | PASS |
| Sum-order negative | Original 0.0006000000000000001 ms; wrong sorted-order sum 0.0006 ms is rejected | PASS |
| Full AST closure | Five selected metrics substitutions and batch insertion; inherited stage observations and original trace/module AST restore exactly | PASS |
| Predicate negative | Changing local_num_tokens > 0 to >= 0 fails exact original AST equality | PASS |
| Pipeline and injected function | Original pipeline AST differs only by composition after _obs assignment; original __file__/CLI preserved; injected function runs without adapter globals | PASS |
| Frozen sources | All five before/after SHA256 pins match | PASS |

Exact commands, exit conventions, budgets and source hashes are in verification_receipt.json and input_source_bindings.json. Unit stdout/stderr and source_diff.patch are retained separately. The no-index Git checks return 1 because new-file differences exist; they emit no whitespace diagnostics, and direct whitespace checks also pass.
Retained decodeDP1/rank0 and prefillDP1/rank0 JSON hashes match their recorded outputs in complete_control_bindings.json. Raw trace/index bytes were not reopened; their hashes remain inherited input bindings.

Pending primary work: two fresh complete-output controls, independent Spec/Standards reviews, then the P0 rank0 pilot and remaining seven ranks only after a complete pilot; each once at 60s/actual30GiB/max2 concurrency.
No full-trace runtime, peak memory, 4.5s saving, GPU/native ownership, R5/full32, dependency, cost or reference acceptance is claimed. No full trace, platform operation, agent spawn, stage or commit occurred.
