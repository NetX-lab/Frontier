# Profile04 submission-document acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded independent root verification and complete bounded-diff review. |

Result: **PASS, documentation only**. Seven authorized files changed by 68 additions and nine deletions, 77/150 gross lines. The 229 checks and 47 new local links passed independently on Python 3.10.6; all guarded evidence, scope and document fields matched the owner's receipt. Scripts, starting snapshot and report were copied byte-identically to the root scratch directory without changing the owner's output or execution identity.

Exact root argv: `timeout 60s /usr/bin/python3 -B /data/ycfeng/tmp/glm53_dsv4/profile04_stage_docs_root_b13xw9ia/verify.py`, from `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, with `GIT_OPTIONAL_LOCKS=0` and `OPENCODE_SESSION_ID` unset. `root_receipt.json` records copied-input hashes; `verification.json` contains every check and guarded file hash.

Each exact `git diff c4b59b65 -- <authorized-path>` argv and file hash is in `diff_manifest.json`. The combined complete diff is `source.diff`: 213 lines, SHA256 `5143a67b9c6a2b1ef9fdc9d7339d22eb2e77fd0ee7d7bb1f2c59cea16c7d337d`. Scoped whitespace, original checklists/history and prediction Sections 1–8 passed the validator.

Independent phase reviewer `W128-PROFILE04-STAGE-DOCS-REVIEW` / `ses_e929ee618ffeg5nhA0ydcfm25W` reported **No findings**. Its initial combined-text reading was incomplete; a follow-up read all seven per-file artifacts without truncation and closed that gap. The reviewer did not execute checks or rehash files. Root acceptance and artifact hashes are in `acceptance.json`.

The original concurrent-root-progress checker failure remains at `/data/ycfeng/tmp/glm53_dsv4/profile04_stage_docs_W128_ses_e92a7f5c5ffe8QOCEgHUMOaLuF/first_verification_failure.txt`. No failure was relabelled as a successful run.

Root's first precommit aggregate check used Git's canonical path order instead of the archived manifest's per-file order. `precommit_order_failure.json` preserves the resulting assertion and both hashes. Recomputing the same per-file commands in manifest order fixes that harness comparison without changing any document or raw diff.

Limitations: no fresh platform read, GPU submission/control, runtime success, cost admission or fixture integration. The submitted job's dated Starting snapshot, original profile03 FAILED evidence, isolated/tiny-probe boundaries, GLM pause and all full-ticket gates remain unchanged.
