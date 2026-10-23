# W128-V4-MOE-PACKAGE progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Inspected corrected root/native sources and existing package mechanisms. |
| 2026-10-23 | Clarified the permission-refused CPU-control mkdir attempt in the boundary summary. |

## Status

- completed: Read actual root corrected requirements/plan/moe_grid_design; clean owner base56e9405d and native2cf0a69 confirmed.
- completed: Found the actual balanced-decode builder `balanced_decode/prepare_package.py`, not build_package.py. Its hash/original-verification utilities and existing W4 launch helpers can be reused.
- completed: Inspected original W4 and W6 producer patterns. W4 worker skips traces; retained Q1 worker already copies them without deletion. The MoE RecordFunctionTracer exports Chrome traces and the existing wrapper creates profiler_traces. No required runtime flags or patches are missing.
- completed: Generated h800_moe_local_01 with1003 pinned inputs, snapshot56e9405d, native2cf0a69, exact four rows per method. Manifest2ce8333148675cbec721c7f0f9710f70c1b031032be971ffcf7b1e754a1377f4; two maintained files204 lines before staging.
- completed: CPU fresh-entry/hash/trace-copy controls, exact routing/backend selectors, existing tests and matrix binding; see cpu_checks.json/routing_checks.json/test_report_2026-10-23_cpu_preparation.md.
- completed: Source frozen at737b631191542202c42b32c0c4a225058021ff92;153+51=204/300 gross lines in exactly two maintained non-core files.
- completed: Generated package (1004 files including manifest/23680569 bytes), machine receipts, selected XML, source-scope receipt and final records; included in the separate records/package commit.
- pending (root-owned): Actual locked-image/native-symbol/binary checks, SDK/platform resources/mount, quota/auth, seal/submission, GPU outputs/traces and independent data acceptance.

## Boundaries

No GPU, Docker, platform, auth, network or seal calls were performed, and no successful cloud read/write occurred. One faulty CPU control attempted a local /mnt mkdir and received PermissionError; details and corrected controls are below. Existing source domains frontier/data/examples/fidelity/canonical_case are byte-identical to97803f0, enabling explicit existing77matrix binding after checking its original comparison bytes. Preparation never grants rank/time/weight/graph/serving admission.

## CPU test setup observations

Initial test invocation named a nonexistent tracer test file (exit4/no tests); actual coverage is test_profiling_timing_stats_contract.py. A direct distillm environment lacks pandas; a minimal Frontier environment lacks torch. Reusing the established combined Python path resolves imports but prepending Frontier's NumPy2 to Torch2.1 causes `_ARRAY_API not found` / `Numpy is not available`. Before rerun, inspect confirmed distillm has NumPy1.26.4; place its site-packages before Frontier's site-packages. No dependency installation or source change is needed. One unrelated online-AITER test requires absent native vLLM's modular API; verify equal baseline failure rather than patching it. Raw evidence/XML remain under /data/ycfeng/tmp/w128-v4-moe-*.

CPU suite is now172 PASS/1 FAIL/2 SKIP; pinned source56e9405d reproduces the same native-API failure and skips. Eight exact CPU entry/dry-run controls,1003 immutable hashes,474 native committed files,20 binary pins,440 installed-source pins, original527 inputs and existing77matrix binding passed in cpu_checks.json. A worker-entry control initially used Exception, which the existing runtime caught, attempted a local /mnt/codesign-exp mkdir, and received PermissionError; no successful write. BaseException plus an audit prohibition now passes before runtime/cloud execution. Do not reuse the faulty control. It did not execute any network/GPU/platform operation.

Final routing control hypothesis: actual native marlin_moe.py:333-335 iterates candidates and keeps64 even when no condition matches. Manual `next()` incorrectly raised StopIteration for12288. Read and execute the exact native AST assignment/for statements; expected aligned rows are5120/3072/128/128. Earlier CPU-only signature/key assertions used incorrect names; corrected from actual source. No runtime edit and no further harness extension.

Final routing control passed by executing the actual native AST; rank0 assignments4608/2320/16/16 and aligned rows5120/3072/128/128. torch.cuda.is_initialized() remained False. Existing shell wrapper passed bash -n; its proxy/auth/network body was not executed. Root's existing balanced_decode/root_seal.json procedure was read only as the sealing reference: source/file hashes, actual image/SDK/destination/quota checks, then a separate root seal. No additional seal helper was introduced and no seal was executed here.

Final owner acceptance: requirements.md re-read; source two files204/300 gross lines, generated builder/plan bytes match source commit737b6311, manifest1003 inputs unchanged, original527 inputs unchanged, exact4rows/method and routing selectors passed, existing77matrix explicitly source-bound. Current checkpoint retains two corrected execution deviations; no further harness expansion. Unfinished work is entirely root-owned: independent reviews, actual-image/native backend and SDK gates, personal auth/quota/local/cloud mount and new-destination checks, external seal and1H800/1800s run/data acceptance. No rank/time/weight/graph/serving admission.
