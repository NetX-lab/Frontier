# D23-ACTUAL-MAIN/PERF requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded the delegated diagnosis-only scope. |

## Original Request

[Original Request] Diagnose actual05 CPU timeout and build a seconds-scale, real measured feedback loop before a production fix. Create `/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf`, branch `feat/step5-mrv2-perf`, from `ec5de15`. Own only CPU profile/repro scripts under `tests/` and this task directory. Return measured cause, three or more falsifiable hypotheses, profile evidence, the smallest source proposal, correctness invariants, limitations, exact command and task/session identity.

[Original Request] Both original offline and online main at `26db654` timed out at 2400.13/2400.11 seconds. First scheduler events were approximately 998/1004 seconds after start; 42/41 real decoded tokens were emitted by timeout, last simulation times 87.717/87.925. Parent owns original artifact census; do not repeat it.

[Original Request] Reuse immutable `/data/ycfeng/tmp/d23_main_fixture02` with 94392 CUDA_EVENT and 184 KERNEL_ONLY attention rows and existing typed caches/bundles. Trusted project-owned pickle reads are authorized. Keep scale, features and output identities. No new broad fitting, fixture/cache writes, fabricated costs, disabled metrics, timeout increase, full 1024 CLI retry, source edits in any other worktree, further delegation, GPU/platform or external operations. Parent approves and reviews production changes and owns final workload acceptance. Large inspection must be small/streamed because systemd scope memory controls are unavailable.

## Acceptance

- A short unattended command exercises the original expensive prediction path and can report RED from measured latency/work while independently asserting exact duration/count/identity truth from the stored cache.
- Profile and discriminating evidence identify dominant CPU call/subprocess, not a guessed bottleneck.
- Diagnosis and proposal do not claim native admission or full main acceptance. Preserve the original 2400-second bound.

## Session

- Task: `D23-ACTUAL-MAIN/PERF`
- Session: `ses_e9f519775ffe3Cv0bhEzJSKNhH`
- Base: `ec5de154c19ceeae23a3652f511582858170f813`
