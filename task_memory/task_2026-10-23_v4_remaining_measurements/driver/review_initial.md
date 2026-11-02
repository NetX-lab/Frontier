# G2 driver initial independent review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded both initial reviews: Standards2P2,Spec2P2,three distinct issues; owner RED controls and repairs are in progress. |
| 2026-10-23 | All initial findings below were closed by both R1 reviews; see review_final.md. Their original open state remains the historical record. |

Component: first-formal native profiling driver and collector retention. Baseline313c8167; reviewed artifact `change.diff`, six source hashes in `source_manifest.json`,381changed lines. Reviewers are read-only and did not run tests or operate Git/platform resources.

## Standards

Reviewer: `ses_e9841fef9ffePvnX6RHfChfqKD`, task `W128-G2-DRIVER-STANDARDS`.

1. **P2, open:** `native_pd_worker.py:159–161` unconditionally reads `torch_profiler_dir` when the selected extension has a native-valid CUDA profiler configuration. That auxiliary path derivation can stop native startup. Derive the path only for the relevant torch configuration with a usable directory; retain native CUDA behavior and the driver's UNVERIFIED auxiliary-window result. Basis: `driver_design.md` soft-failure contract and AGENTS.md799.
2. **P2, open:** `native_serve_profile.py:113–118` performs startup archival before initializing the independent window budget. An archive failure leaves startup recording and exhausted capacity active, including after native automatic stop. Isolate archival with its own observation boundary, then initialize the window; strengthen the failure control to prove capacity, automatic stop and window snapshot retention. Basis: `driver_design.md` independent-retention contract.

No additional actionable standards or heuristic-smell findings. Existing CPU receipts were inspected, not rerun. The seventh-file cold-start fixture correction needs a follow-up review.

## Spec

Reviewer: `ses_e984289acffe3o6TiR1NxOFgNA`, task `W128-G2-DRIVER-SPEC`.

1. **P2, open:** startup archival failure prevents the new quota/window from being established, violating the fresh row/byte allowance promised in `driver_design.md:41`. This is independently the same mechanism as Standards finding2; both reviews require archive-failure capacity/stop controls.
2. **P2, open:** `pd_replay.py:523–537` calls ordinary `persist()` during the active window, including its idle boundary. A profiling receipt error then enters the native `observation_errors` list, making a healthy workload REFERENCE_UNSTABLE. The initial test injects only the final post-stop write failure. Isolate active and later writes carrying the profiling receipt, retain unselected checkpoint behavior, and cover persistent as well as active-window failure. Basis: `driver_design.md:24,37`.

No scope creep found in the supplied six-file artifact. Native API/iteration source and root receipts were inspected, not executed. No GPU coverage or reference admission conclusion.

## Next owner action

Root is the delivering G2 owner after the two initial design delegation failures. Both reviews are complete; the original artifact remains preserved. Reproduce all three distinct findings, apply bounded repairs, rerun affected checks and request both follow-ups on the complete seven-file artifact. No source/CPU acceptance or GPU qualification is granted by this review record.
