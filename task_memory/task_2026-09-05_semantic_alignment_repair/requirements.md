## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Recorded the approved semantic-alignment repair and publication requirements. |

# Requirements

- [Original Request] Open a concise English issue describing why valid vLLM attention-DP topology cannot enable `attn_dp`, and link the repair PR.
- [Original Request] Create a worktree branch from `main`, publish the semantic-alignment repair as an English PR, and keep the PR description plain and structured.
- [Original Request] Validate the repair in order: co-location -> pd-disagg -> pd-af-disagg.
- [Original Request] Preserve one Frontier Replica as one complete GPU pod, use `(replica_id, dp_id)` request ownership lanes where applicable, retain logical TP abstraction, and avoid per-TP/per-EP rank schedulers.
- [Original Request] Do not rerun groundtruth or reuse legacy H800 CSV data.
- [Original Request] Set the HTTP proxy before remote publication.
