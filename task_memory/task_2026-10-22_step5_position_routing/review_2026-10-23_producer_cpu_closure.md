# Producer and CPU closure review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record one P1 and one P2; accept CPU consumer behavior while retaining producer join and test-portability blockers. |

TaskID: D29-PRODUCER-CPU-CLOSURE-REVIEW. Reviewer:
ses_e9afdb7c4ffejb3hbkjZCeHFKS. Frozen source84e66115, full10-path delta
and related source/native/CORE interfaces. Parent evidence:724 unique cases,
691 PASS/33 SKIP across environments; cold68 is a subset. Prior12 HTSIM results
were explicitly rebound to unchanged sources, not rerun. Reviewer did not execute
tests, shell or platform operations or edit files.

## Findings

1. P1: step5_dispatch_probe.py406–407 reuses prepared_context, whose native_facts
   includes every currently loaded vLLM Python module from runtime_identity.py190–214.
   Dispatch probe and isolated MoE profiler have different valid ambient import
   inventories. Otherwise equal actual backend/layout/padding/source points therefore
   differ under the full canonical equality required by the CORE reader and H04.
   Fix the shared prepared_context to expose deterministic compute provenance for
   joining; retain full ambient verification and its output separately as diagnostic
   evidence. Preserve actual callable/source/numerical/layout checks. Verify a real
   independent-producer handoff under differing valid extra import inventories.
2. P2: test_collective_sim_cc_backend.py115–117 unconditionally reads the temporary
   verify-step5-core-delivery-01 worktree and does not check the documented source
   hash. Ordinary checkouts fail without that tree, or silently test changed external
   bytes. Resolve the real writer from the current repository by default; any staged
   external reference must be explicitly configured and identity-checked. Do not
   copy the writer body or silently skip the coverage.

## Accepted behavior and limits

The previously absent observed-writer code now exists: actual router/dispatch/
expert reference association, post-execution context generation and exact buffer/
count transmission. It is not a prediction copy. Shared-first/routed-second order,
multiplicity and policy/layout checks match the published source. Eager evidence
does not qualify captured records, and full point context stays out of fit grouping.

CPU behavior is acceptable: both artifact and plan require explicit CPU/unadmitted
provenance and exact receipts; native cost injection remains ignored and CPU trace
labels are correct. No native/tolerance/accepted finite behavior regression found.
Complete native producer handoff remains blocked by P1, and persistent test coverage
by P2. Full integration, actual GPU/backend/graph/MRv2/G3 and qualification remain open.

## Assignment

Root assigns both findings to the original PRICING owner as D29-PRODUCER-JOIN-REPAIR
at84e66115:43 existing paths,3713/3956 gross,243 remaining. No additional paths or
implicit quota borrowing. If retaining diagnostics requires a CORE writer change,
return its exact interface/path/budget for coordinated ownership before editing.
CORE's three review repairs and registry work remain separate active assignments.
