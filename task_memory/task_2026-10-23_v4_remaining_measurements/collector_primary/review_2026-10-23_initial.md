# Initial native collector source review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Persisted independent Spec and Standards findings on d94b821a; returned all five unique issues to the delivering owner. |

Scope: baseline `993f05f5` to `d94b821ac26a3cbff8379e355994b3bb3acb0f3c`, five non-core source/test files,728changed lines. Both reviewers inspected the supplied diff, real collector/callers/tests, MAIN task requirements/plan/seam/driver design, worktree/global rules and pinned native source. Neither ran tests or platform commands.

## Spec: request changes

Reviewer `ses_e98ca2a05ffe2CYjLe8mE23lzM`, W128-V4-NATIVE-COLLECTOR-SPEC.

1. **P1 — graph installation lifecycle**, `native_serve_profile.py:288–291`. V2 `cudagraph_manager` is None after load_model and created during initialize_kv_cache (native model_runner304/585). Installing on type(None) fails and skips replay/facts, while the same-worker early return blocks retry. Install class hooks independently of the future instance or finish at the correct native phase, isolate separate installs, and test the actual None-then-initialize lifecycle.
2. **P2 — metadata ownership**, collector35. `DefaultModelState.prepare_attn` misses the preceding GPUModelRunner.prepare_attn → BlockTables.compute_slot_mappings kernel (native model_runner1480/block_table191, before1535). Cover the actual boundary and verify order through the native body on CPU.
3. **P2 — source-visible PDL arguments**, collector37–42/130–132. Wrapper inputs omit internally generated q/KV RMSNorm launch_pdl, inverse-RoPE USE_GDC/launch_pdl and the local-import CUDA head512 CuTeDSL path. Observe the actual invoked producer/launch boundaries and test them. Hardware correlation may remain unknown; omitted source-level observations are not runtime-only gaps.
4. **P2 — final step capacity**, collector98–102/381–389. Context/dispatch enrichment bypasses admission and incomplete accounting can exceed the bytes limit. Apply one admission policy to the completed retained record; explicitly drop unretainable evidence and test that an initially small step can fail enrichment admission.

The original22tests and24source-symbol lookup do not cover these failures. Inherited inventory RPC and common original-result/error preservation are visible; no separate scope creep was found. Driver remains excluded.

## Standards: request changes

Reviewer `ses_e98ca2963ffeaN1Qc0kHtYPS70`, W128-V4-NATIVE-COLLECTOR-STANDARDS.

- Duplicates Spec issues3 and4 as hard failures against plan28 and bounded retention.
- **P2 — bounded error logging**, collector65–70. The retained errors array is capped but every failure still prints its traceback. Persistent annotation failure can flood stderr and interfere with the workload. Bound detailed output or deduplicate, count suppressed failures, and preserve original returns/exceptions under repeated failures.
- No additional actionable heuristic finding. Shared registry, composition, inherited RPC and private JSON are appropriate; no extra native synchronization or cost summation was found.
- A separate explicit matrix binding is still required for H. Source-name resolution is not successful executable import resolution.

## Root disposition

All five unique findings are assigned to the same delivering owner. Necessary repair allocation is now five non-core files/987changed lines: collector550, existing test400, native worker25, command builder8, P/D environment4. This is a bounded source/CPU correction under existing root authority, with no driver or GPU expansion.

Root's attempted d94 suite rerun stopped before pytest because its preflight incorrectly treated the completion receipt's per-file counts as a `sha256` map. No root test pass is claimed for d94. Final repaired-source verification will bind actual committed file bytes, use separate suites and retain the original failure.

GPU stream-ID mapping, hardware PDL, compile/breakable survival, captured-segment ownership and CUPTI restart remain unknown independently of these source fixes.
