# CPU timeout diagnosis progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Created isolated worktree and located read-only prediction seam. |
| 2026-10-22 | Completed deterministic FFN RED loop, CPU/collective/trace discrimination, and diagnosis-only proposal. |

## Status

- completed: Created `feat/step5-mrv2-perf` from `ec5de15`; moved this session to the new worktree. Existing preview worktree untracked `.cortexkit/` and `.pi/` were preserved.
- completed: Read repository AGENTS, GLOSSARY, ADR 0001 and `diagnosing-bugs`. Located actual command/environment and cached attention root/count estimators. CUDA_EVENT exact lookup has 94392 keys per estimator (~71 MB artifact); KERNEL_ONLY has 184 keys (~200 KB).
- completed: Short real-cache attention loop `/data/ycfeng/tmp/d23_perf_attention02`: RED, source truth PASS, loading 1.3583s, profiled work 0.0995s. Individual actual attention query approximately 0.0167s versus conservative 0.007409s per-layer budget for 1024 decode steps under the unchanged 2400s bound.
- completed: Actual EP16 FFN replay, exact source truth assertions, two repeated non-profiled RED runs, CPU profile, cold/warm real collective discrimination, and trace metadata/persistence measurement.
- completed: Recorded measured cause, bounded proposal, original-bound latency gate, and limitations. No production files changed.
- pending: Parent approval/implementation/review and full workload acceptance; outside this subtask.

## Evidence and decisions

- Original Python: 3.10.6; executable `/data/ycfeng/envs/frontier-py310/bin/python`.
- Worktree creation: `git worktree add -b feat/step5-mrv2-perf /data/ycfeng/Frontier/.worktrees/step5-mrv2-perf ec5de15`.
- Original input scale is retained in stored estimators, not minimized to a smaller coverage table. Loading pickles is authorized and avoids initialization/fitting. Inputs remain explicitly `cpu_fixture`, not native evidence.
- Systemd memory-control scopes are unavailable per delegation; all inspections are scoped, bounded and avoid dumping large identities.
- Initial harness failure: duration pickle does not contain its paired kernel-count estimator, because production attaches it after caching. Corrected harness by reading the existing bundle link and both trusted artifacts, verifying their SHA256, and attaching the link only in memory. No cache/fixture changes.
- Phase 3 ranked hypotheses: (1) repeated full model validation/metadata/precision construction per layer and EP lane; predicts high calls into `validate_step5_config`, `build_operator_manifest`, `resolve_module_precision`. (2) coverage/registry/hash rescans; predicts high lookup/fingerprint time, scaling with exact-key count. (3) collective subprocess work; predicts real repeated collective cost and subprocess calls. (4) proof/metrics serialization; predicts recorded output boundary dominates compute query.
- Baseline attention cProfile: `describe_step5_layer` 0.083s for six calls; `validate_step5_config` 25 calls/0.041s; `resolve_module_precision` 34 calls/0.038s; no estimator prediction or dataset hashing in measured work. Setup independently re-derives features and is reported separately from per-query latency.
- FFN harness minimization exposed older bundle selection: actual03 scalar profiles lacked the actual05 aggregate `num_tokens=2` rows. Checked bundle registrations and exact keys, then bound replay to the actual05 offline READ profile path; actual03 and actual05 share attention artifacts but not FFN artifacts. This is a harness setup correction, not a production behavior issue. The first FFN attempt also included non-feature `load_distribution`/`seed` fields; corrected projection to the existing `MOE_LOAD_IMBALANCE_FEATURES` registry, matching production. No training/cache modification occurred.
- Second checkpoint after FFN harness attempts: real feedback loop remains complete; repeated attempts narrowed two setup problems, not alternate production fixes. Stop broad bundle selection and use exact actual05 bound bundle. Remaining frontier is the same scoped FFN profile and communication/proof discrimination. No scope expansion.
- Final repeat baseline `/data/ycfeng/tmp/d23_perf_red_final01` and `final02`: RED / source truth PASS. 64 EP lane queries took 0.52744/0.54212s, cache loading 1.47872/1.51473s. Average actual query 8.22655ms, metadata 5.83032ms, operator queries 2.38895ms. The unchanged 2400s bound after 1004s setup implies at most 0.96824ms per EP lane/layer over 1024 steps and 88 routed layers, even giving FFN the entire remaining budget.
- `d23_perf_ffn06/prediction.prof`: 32 `describe_step5_layer` calls took 0.45234s (73.3% of 0.61686s measured work); 192 precision resolutions, 97 full Step validations, 228 layer-contract resolutions. Large exact-key lookup and source hashes were not dominant. Fixture SHA256 runs only during setup.
- Hypothesis discrimination: actual ATTN_TP8 allreduce (8192 bytes) cold 1.88543s / hit 0.10995ms, actual EP16 all-to-all (16384 bytes) cold 2.54529s / hit 0.17243ms; predictions exactly repeat. Correct real collective replay is `d23_perf_cc03`, not the earlier EP allreduce exploratory run. No full CLI, cloning, build, original runner/cache writes or native admission.
- Actual source-bound metadata and enabled TraceStore persistence: 228 real operator events, 172466 bytes, 0.04050s unprofiled; persisted duration/name/layer tuple truth PASS. Profile identifies dataclass `asdict`/deepcopy as the largest trace-only cost, but trace work is well below the measured FFN query work. Full MetricsStore EP scheduling and final acceptance were not replayed.
- Attention-only unprofiled query is 6.59–7.36ms and standalone budget is 7.409ms: GREEN. Earlier standalone attention RED was profiler overhead; only FFN non-profiled RED is the authoritative fast feedback signal.
- Minimal proposal: predictor-bound cached metadata for each physical layer plus cached immutable selected operator contract/model binding per `(layer, operator, measurement family)`. Preserve feature validation, exact coverage, paired count queries, source proof identity, all EP lanes, physical layer/precision differences and enabled metrics. Metadata-only savings are measured at approximately 8.21s per modeled full round; output remains unmodified. Source approval belongs to parent.
- Final checkpoint: all four scoped diagnosis steps completed; D1–D7 CLEAR, S1–S6 no detected defects. Stop extending the harness now; return measured evidence/proposal. Parent's production/final-workload tasks intentionally remain pending.
- Final current-script verification: `d23_perf_red_final03` RED/source PASS, complete process3.798875s, max RSS815716KiB, loading1.483458s, measured0.543830s. Python syntax PASS for both scripts. Re-read requirements before close. `/usr/bin/time` is absent; used stdlib timing/resource wrapper without installing anything.
