## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-15 | Added direct actual-replica-ID routing coverage in commit `7b6a3eb1` after the production constructor repair. |
| 2026-09-15 | Added the final CPU routing-identity repair and device-event path-contract verification to the completed acceptance record. |
| 2026-09-15 | Recorded completion of all planned work units, final Review 2 closure, and PR #33 publication; merge remains user-gated. |
| 2026-09-14 | Translated plan v2 execution order into local work units without editing the source plan. |

# Execution Plan

Source of truth: read-only `/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md`, revision v2.

## Dependency order

`0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 12 -> 8 -> 14A/B -> 7 -> 9 -> 10 -> 11 -> 13 -> 14C/D -> docs/PR publication`

## Work units

| Unit | Scope | Acceptance evidence |
| ---- | ----- | ------------------- |
| 0 | Freeze baseline/source SHAs, environment, existing failures, and Review 1 attention-assumption inventory. | Local records and baseline commands are reproducible; no source change. |
| 1 | Rename misleading internal `linear_attention` identifiers to `attention_linear_ops` while preserving upstream config literals. | Focused rename tests and import/compile checks pass. |
| 2 | Add explicit GPU platform/SKU/topology metadata and CPU discovery/error tests. | **Completed in `0989f0ed`;** unknown SKU and mismatch behavior are deterministic; no hardware claim. |
| 3 | Add `DEVICE_EVENT` timer/measurement plumbing with strict CUDA/ROCm mapping. | **Completed in `49fcd9b8`;** CPU schema/cache/family tests pass; CUDA behavior remains unchanged. |
| 4 | Add scoped hybrid resolver, CPU-only GDN features/inputs, and synthetic test fixture. | Resolver and phase/schema tests pass; fixture is clearly synthetic/test-only. |
| 5 | Migrate to unified per-layer `ExecutionTime` and ordered `StageExecutionTime`. | Existing simulator behavior and execution-time tests pass within current tolerances. |
| 6 | Add fixed GDN state memory, D57 approximation, lifecycle and unsupported-feature guards. | Memory/state/guard tests pass; non-GDN behavior is unchanged. |
| 12 | Extract the existing MoE routing-ratio helper without changing RNG/count outputs. | Old/new helper outputs match for pinned cases. |
| 8 | Add CPU-safe GDNTrainer/GDNPredictor and model-manager/config wiring. | Six task fit/save/load, identity separation, and synthetic predictor tests pass. |
| 14A/B | Run real synthetic CPU training/loading/simulation E2E and real-model structural checks. | Eight-layer order, prefill/decode/continuation, state lifecycle, and Qwen3.8 pins pass. |
| 7 | Add standard vLLM GDN producer and CLI/schema path. | CPU/input/mock tests pass; GPU tests remain explicit AMD SKIP. |
| 9 | Add generic ROCm compatibility and explicit `VLLM_ROCM` full-attention producer. | CPU/import/schema tests pass; grouped NVIDIA regression attempted if access exists. |
| 10 | Add standard vLLM/AITER MoE and MXFP4 metadata/producer path. | CPU/config/lazy-import tests pass; AMD runtime is explicit SKIP. |
| 11 | Add standalone RCCL profiler without simulator coupling. | CPU/mock schema and byte accounting pass; GPU RCCL is explicit SKIP. |
| 13 | Add independent experimental SGLang primitives/replay/routed replay/GDN trace importer. | CPU import/input/trace tests pass; no standard CSV contamination. |
| 14C/D | Fresh final Review 2, broad CPU regression/parity, wall-time check where applicable, evidence consolidation. | Zero unexplained hybrid-unsafe paths; required CPU checks pass; unavailable GPU checks labeled. |
| Docs/PR | Update focused docs and local summary, commit coherent increments, preserve attribution, push branch, and create new PR. | **Completed in PR #33;** merge remains explicitly user-gated. |

## Completion checkpoint — 2026-09-15

All work units in the dependency order are complete for the authorized CPU acceptance scope. The implementation was delivered through independently reviewable commits, with `powderluv` co-author trailers retained on the selective integration commits. The following evidence closes each unit:

| Unit | Completion evidence |
| ---- | ------------------- |
| 0 | Baseline SHA/environment/failure inventory and Review 1 call-site audit in `requirements.md`, `test_report_2026-09-14_baseline.md`, and `review.md`. |
| 1 | Naming cleanup and focused rename/import tests recorded in progress history. |
| 2 | Commit `0989f0ed`; MI355X metadata/discovery tests and explicit hardware skip. |
| 3 | Commit `49fcd9b8`; strict `DEVICE_EVENT` timer/registry tests. |
| 4 | Commit `ddd9bbb3`; hybrid/GDN semantic resolver, fixture schema, and CPU contract tests. |
| 5 | Commit `b37cb38b`; per-layer `ExecutionTime`/ordered stage migration, five-case fidelity, and wall-time observation. |
| 6 | Commit `0d1b87a4`; fixed state/memory/lifecycle/guard tests and Qwen3.8 structural capacity check. |
| 12 | Commit `5764daf7`; shared routing helper and 48/48 old-formula equivalence evidence. |
| 8 | Commit `fd5f49b1`; six GDN estimator artifacts, manifest identity, predictor and manager tests. |
| 14A/B | Commits `45e66170` and `31b11762`; synthetic hybrid training/load/dispatch/stage E2E and dense fidelity. |
| 7 | Commit `0d4efa22`; standard GDN producer/CLI/schema CPU contracts. |
| 9 | Commit `294bc1b4`; `VLLM_ROCM` producer and current-vLLM compatibility contracts. |
| 10 | Commit `3b20d702`; standard MoE/AITER/MXFP4 planning and metadata contracts. |
| 11 | Commit `a092f637`; standalone NCCL/RCCL runner and dtype-aware byte accounting contracts. |
| 13 | Commit `f51231d4` plus `31b11762`; experimental SGLang primitive/replay/routed/trace contracts. Current exact-HEAD re-review confirms `_make_replay_call()` closes the tuple-shape concern. |
| 14C/D | Commits `97c1a737`, `31b11762`, `ffb9feea`, and final pushed head `578785bb`; Review 2, persistent hybrid CPU Simulator E2E, final full CPU regression, compatibility fix, metrics ledger audit, and documentation closure. |
| Docs/PR | Commit `7937e534`, local English archive docs, push to `origin/feature-amd-sglang-gdn`, and open PR [#33](https://github.com/NetX-lab/Frontier/pull/33). |

The source plan remains read-only and outside the worktree. No merge, rebase onto `main`, branch deletion, or modification of PR #31 was performed.

## Post-Review CPU closure — 2026-09-15

The first focused-suite rerun after wiring the real hybrid production constructor exposed a genuine monolithic MoE identity issue: `Replica.id` is process-global, whereas the shared routing map used local indices. The repair passes actual monolithic cluster replica IDs through `Simulator` and the MoE Registry path, validates the map shape, and leaves direct no-ID predictor tests on local keys. The manager's complete training-path API was also brought into contract with its three derived/configured device-event paths.

Verification for this closure:

- Production constructor CPU E2E: **1 passed in 4.43s**.
- Monolithic routing identity contract: **1 passed in 2.60s**.
- Manager path contract: **1 passed in 2.73s**.
- Full focused command (timer, GDN, hybrid, predictor, MoE, and measurement-family suites): **219 passed in 13.54s**.
- Direct actual-replica-ID contract: **1 passed in 2.58s**; production constructor follow-up rerun: **1 passed in 4.60s**.
- `SKIP: AMD/MI355X hardware unavailable`; no ROCm, benchmark, or groundtruth evidence is inferred from these CPU checks.

This closes the CPU acceptance sub-step for the scoped implementation. Production-data hybrid fidelity with complete model-specific attention/MoE CSVs remains an explicit future extension.

## Verification strategy

- Before each check, state the failure it detects and the expected decision.
- Use focused tests after each work unit, broad CPU checkpoints after Unit 8 and before publication, and existing E2E/parity harnesses after Unit 5 and at final integration.
- Run `python -m compileall` or targeted import checks for touched CPU-safe modules.
- Use separate temporary output/cache roots for baseline and candidate.
- Never convert a missing AMD environment into a PASS or substitute dummy timing for GPU evidence.
