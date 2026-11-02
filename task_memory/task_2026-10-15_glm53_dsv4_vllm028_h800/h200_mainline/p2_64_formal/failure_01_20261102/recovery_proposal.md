# P2-64 recovery proposal — approved as D-P2-64-STARTUP-RECOVERY

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | Proposed a native DP startup correction from CPU-verified failure evidence; the correction itself is unimplemented and unverified, with no new GPU submission authorized by this document. |

The first formal campaign failed before warmup. The decode head logged `EADDRINUSE`; CPU execution of the frozen native methods reproduces rank 0 switching rendezvous ports while the other three ranks keep the original port. The original port occupant is unknown. Preserved artifacts are sufficient to diagnose this reachable coordination defect, but cannot classify the original contention as external infrastructure.

YC approved this proposal on 2026-11-02: “批准，类似debug修复无需询问我，默认都批准”. The proposal below preserves the original options and pre-approval evidence. Current authority is recorded in the task requirements.

## Options and cost

| Option | Concrete action | Cost and limitation |
| --- | --- | --- |
| A — fresh unchanged generation | Keep native source, harness semantics, geometry and deadlines; change only the run/output identity, then build/check and submit one new formal generation. | Up to 64 H200 GPUh, expected about 24.3 from the prior successful campaign. The port-retry defect remains; the first failed generation spent about 34.02 observed GPUh before timing out. No reliable probability of recurrence is available. The cause has not been established as infrastructure failure, so the lead does not invoke the conditional automatic retry authority. |
| B — correct native startup coordination (recommended) | Extend the existing bound-socket/coordination-store mechanism to the current nonelastic multiprocess DP path. Verify on CPU, freeze a new source/package identity, then submit generation 02 with the unchanged serving workload and resource limits. | Source implementation and CPU validation use 0 GPUh. The subsequent formal job retains the 64 H200 / 3600 s / 64 GPUh ceiling and all-preemptible `step_main` class. The native source identity changes and must be explicitly approved and reviewed. No separate GPU diagnostic is proposed. |

## Proposed scope for option B

Primary source locations are the frozen native repository's `vllm/config/parallel.py` (selection and retry) and `vllm/v1/engine/utils.py` (multiprocess startup ownership and HELLO metadata). Implement in a new native worktree; retain commit `2cf0a6915ce544dc493a0990f2ea38d81601128a` and the failed package unchanged.

- Reuse a startup coordination store owned by the multiprocess launcher, and propagate its address through the existing DP engine handshake. This is internal startup state; do not enable elastic EP or switch the executor to Ray.
- The leader binds the actual DP listener before publishing its port. Keep that socket reserved until the native TCPStore takes ownership; followers read that published port. A local retry must never silently leave other ranks on the previous address.
- Give separate group-initialization calls distinct coordination keys. A later initialization must not read the earlier group's port. This also applies to the engine/worker call sequence and separate P/D roles.
- Preserve tensor/data/expert parallel geometry, native routing, model/kernel code, request schedule, warmup pair targeting, metrics, deadlines and auxiliary soft-gate behavior.
- Rebuild with a distinct generation-02 run tag/cloud path, updated source hashes and the already issued promotion receipt. Changing only the package directory is insufficient because `build_package.py` currently fixes the case run tag to generation 01.

This is a proposed implementation boundary, not a claim that a production fix is complete. The CPU experiment of the existing coordination path has both successful and failed trials; one shared port alone did not guarantee all Gloo initializations completed. It does not prove multiprocess launcher integration, store lifetime or repeated-call key safety. The implementation must verify these properties and successful communication explicitly; no broader elastic/Ray investigation is proposed.

## Acceptance before a second formal submission

1. The CPU baseline and occupied-port cases both complete native TCPStore/Gloo initialization for two and four ranks; all ranks agree on the active port and communicate successfully. Keep the original failing reproduction alongside the corrected result.
2. Two consecutive group initializations and simultaneous separate role groups use the correct independent identities. Verify socket/store ownership and cleanup without importing the full GPU model stack.
3. Review the final startup-only native diff and hash identity. Re-run only checks affected by the source/harness packaging changes, then build the new frozen package and pass `launch.py --check-only`.
4. Submit generation 02 through `/ds4-exec` after quota verification: 8 x 8 H200, all preemptible, `step_main`, timeout 3600 s, ceiling 64 GPUh. Preserve generation 01 and the three-job-per-case cap.
5. Complete terminal evidence, hash-identical copy, pair coverage soft gate, R4, all-request gate, warm reading and actual usage. Reuse the already completed Frontier replay; never manufacture metrics for the failed generation.

## Approval boundary

D-P2-64-FORMAL-B currently limits package changes to pair warmup, preemptible resources and receipt binding, and permits automatic new generations only for preemption or infrastructure failures. A native startup-source correction goes beyond those three approved changes. The requested decision is whether to authorize option B's startup-only native correction and the second formal generation under the existing per-job resource ceiling. No deployment or submission is made by this proposal.

The seven missing container finish clocks remain an independent accounting limitation. The exact first-generation job total stays unset; 34.01883068666667 GPUh is an observed-runtime estimate, not a complete bill or a rigorous bound.
