# D29 run-fixed input implementation progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Accept the complete real-wave evidence, close both review axes, and commit the ten-path source repair as db581ff6. |
| 2026-10-24 | Primary took over the cancelled implementation, reproduced the lifetime violations, implemented the allocated interface, and verified affected suites. |

## Completed

- The cancelled implementer left a clean worktree at `ecccf83fa6f2c24a14fa5397ef567fbb32b06bef`, the byte-repair cherry-pick. No snapshot implementation or related child process remained. Primary retained this history and completed the ten allocated paths.
- Real baseline RED: after same-size/restored-mtime input replacement, an existing backend changed from `.03` to `.04` instead of retaining `.03`; see `/data/ycfeng/tmp/d29_snapshot_red01.{log,xml}`.
- Real constructor RED: changing the file between P/D predictor construction caused both actual backends to return `.04` instead of the originally bound `.03`; see `d29_snapshot_red03.{log,xml}`. The prior `red02` attempt used the wrong `SYS_ARCH`; it is an instrumentation failure, not the behavioral RED.
- Added the manager-owned `CollectiveProfileInputs` and backend capability methods. Source-based predictor construction establishes the shared input before parent initialization. Standalone backends use the same loader. CORE prepares from fixed backend bytes and retains transactional per-query maps, while pricing retains all admission/count checks.
- Existing replacement tests now use a new backend/predictor lifetime to observe file changes. Model, row-tuple and measurement-identity replacement checks remain live.
- Four affected files: **82 PASS**, no failures/errors/skips, log time154.23s. The separate both-trainer selection and five configured-path checks are recorded in the test report and receipt; counts are not added together.
- `git diff --check` passed. The step currently changes212 gross lines, including the20-line new module; this is a component reconciliation, not final114-path union acceptance.
- Independent Standards and Spec review both returned **No findings** for the complete byte-plus-snapshot diff; see `review.md`. Both read the82-PASS and separate2-PASS logs and retained the full-wave/full-Simulator qualification boundary.
- `D29-RUN-SNAPSHOT-WAVE` completed at records7c73fdd1:208 complete event values/order, whole JSONL/ledger, snapshot isolation,64 wrong-key negatives and3902-byte error prefix PASS. One shared binding read; zero cold/warm reads;288 numerical lookups/144 fresh proofs/64 collective prices remain per wave. Warm medians0.340867→0.162715s are executor measurements, not full-CLI timings.
- Parent independently checked28 manifest hashes,41 source hashes,6 script hashes and6 raw artifact hashes, plus complete JSONL and ledger byte equality. Candidate source hashes match the reviewed overlay. Source-only commit `db581ff6ad43e1ce57361ed00fc4be0256e85a45` contains exactly the ten allocated paths:149 additions/63 deletions,212 gross/net86.

- Source db581ff6 and acceptance6c7121d8 are integrated at64c83ac4. Parent `git diff --exit-code db581ff6 HEAD -- frontier tests data/config` passed. Existing byte records were already present;10/10 Git blobs matched before skipping an empty redundant cherry-pick.

## Pending

- Perform final union accounting and freeze after the separately allocated key-only successor.
- Fresh regression/strict77/full-workflow qualification, T2 final seal and native/GPU/G3/E2E gates remain separate and incomplete. No new full CLI is justified by this component result alone.

## Operational corrections

- The initial wave delegation mistakenly named `b3178400`; primary checked Git, withdrew that pointer and sent the actual `ecccf83f` source-parent correction. Candidate qualification uses actual file hashes.
- `red01` ran Python with bytecode disabled but as root, creating root-owned pytest cache entries. `red02` as `brainpp` reported cache permission warnings. Subsequent checks use `runuser -u brainpp`, Python `-B`, and an explicit scratch pytest cache. No cleanup/chown was performed.
- A receipt-only command attempted the absent host `python` executable and exited127; rerunning with `/data/ycfeng/envs/frontier-py310/bin/python` succeeded. This did not change source or test results.
