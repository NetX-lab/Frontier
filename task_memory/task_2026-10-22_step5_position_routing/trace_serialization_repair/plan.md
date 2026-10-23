# D29 trace serialization repair plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Establish sole-writer implementation, bounded checks, and parent-owned gates. |

1. Establish isolated source identity and record the allocated scope. Read the
   producer/contract evidence and existing actual-wave scripts.
2. Capture the unchanged-source focused pytest baseline, then remove
   `TraceEvent.to_dict` and build declared fields at `flush`'s existing write site.
3. In independent bounded checks, compare real disk pytest results, captured
   598-event replay bytes, real EP-wave producer payload, contract/error behavior,
   exact AST boundaries, Git numstat, and whitespace.
4. Explicitly bind the existing complete layer60 wave loop to the current
   worktree; compare all 208 emitted operator pairs, full proofs, timing values,
   and exact trace bytes against original source and captured actual data.
5. Record source-bound measurements and verification limits; commit only the
   owned source and repair records. Parent performs independent review and then
   strict77/full-unit and complete CLI acceptance on the final frozen union.

Dependency: `identity -> baseline -> edit -> {pytest, replay/contracts/AST,
complete-wave} -> records/commit -> parent review/freeze/gates`.
No agents are spawned; independent local commands may run concurrently.

Acceptance: one source path, <=20 gross, no helper/public-return serializer,
schema/cache/config/test edits; 598-event replay is 8,353,308 bytes with the
required SHA256; 208/208 complete-wave proof matches and exact baseline trace
bytes; focused failure set unchanged; exception identity and buffer order
unchanged; observed source-based serialization reduction with no diagnostic
runtime flag. Unrelated pricing residuals do not block this focused acceptance
or become claimed repairs.
