# CPU timeout diagnosis plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Established bounded CPU diagnosis sequence. |

## Scope and dependency

`isolated worktree -> stored-cache prediction feedback loop -> {profile, repeat/discriminate} -> measured proposal -> parent review`

1. Read relevant instructions, glossary and ADR; inspect the actual command and locate prediction/cache seams without duplicating the original census.
2. Load trusted estimator artifacts read-only; exercise actual attention/FFN prediction methods with original geometry. Separate loading from measured work. Assert source exact values and proof identity, and a latency/work gate tied to the original timeout.
3. Produce one bounded cProfile plus concise call statistics. Rank at least three falsifiable hypotheses, then measure their predicted boundaries without production modifications.
4. Record commands, output truth, profile and limitations. Return the smallest proposal with saved work and unchanged semantics; parent owns implementation approval, code review and full workload acceptance.

## Verification

- Python environment: original `/data/ycfeng/envs/frontier-py310/bin/python`, `-B`, `PYTHONPATH` set to this worktree, temporary artifacts only under `/data/ycfeng/tmp`.
- Each probe uses `timeout 60s` or a smaller bound. No broad source/profile CSV materialization, new cache/lock creation, fitting or new main CLI workload.
- Repeat baseline verdict; inspect profile call/cumulative/self timings; preserve exact lookup values, source hashes, feature scale and modality.
- Gate RED is an expected diagnosis result, not a failed implementation acceptance.
