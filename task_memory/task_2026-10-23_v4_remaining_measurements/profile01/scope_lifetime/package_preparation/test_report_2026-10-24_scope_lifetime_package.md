# CPU/source package preparation test report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Retained commands, observed results and preparation-only limits. |

Working directory: `/data/ycfeng/Frontier/.worktrees/v4-scope-lifetime-package`; taskID `W128-SCOPE-LIFETIME-PACKAGE`. Interpreter `/data/ycfeng/envs/asyncopd-py312/bin/python -B`, Python3.12.13 (Clang22.1.3), Torch2.9.1+cu128; no conda activation/install. Actual locked-image Torch2.13.0+cu130 is not this local environment.
Bounded command, twice only: `sudo -n systemd-run --quiet --scope -p MemoryMax=30G timeout 60s /data/ycfeng/envs/asyncopd-py312/bin/python -B -`; exact stdin is `cpu_candidate_01.stdin.txt` and `cpu_candidate_02.stdin.txt`, argv/cwd/elapsed/closed raw hashes in their `.execution.json` files.
| Check | Passing condition | Observed result |
| --- | --- | --- |
| Candidate1 | Frozen input census and complete CPU candidate verification | FAIL/exit1/0.101831s: verifier `KeyError: package_hashes`; actual45-input manifest uses `files`; no scratch/package generation |
| Candidate2 | Five exact payloads; mirrored frozen imports; real positive limits; accepted ABI; dry-run exec argv without run token; explicit workload/resources | PASS/exit0/2.012785s; actual32212254720-byte limit; CUDAfalse;18candidate inputs/74launcher lines; only scratch config/argv paths relocated |
| Immutable inputs | Seven original bindings,13runtime modules,45sealed04 inputs,77-case existing binding unchanged | PASS:67distinct file pins before/after; no source/test suite rerun or new matrix capture |
| Generated package | One exclusive generation; exact byte copies/manifest closure; no rerun | PASS/exit0/0.071120s;18inputs/19files/184495bytes; PREPARED_NOT_SEALED/submittedfalse |
| Static equality | Official launcher equals CPU-tested launcher; config/argv equal after relocation; accepted image receipt/ABI/binary pins preserved | PASS/0.095395s; initial EACCES retained in `closure_initial_failure.json`, corrected with three exact sudo cat reads only |
| Allocated code |155/40/80individual,275total; launcher100 | PASS:140/27/78=245; launcher74. Copied/generated/diff/JSON separately accounted |
| Image gate script | CPU-only import/manifest/ABI/binary checks plus existing dry-run; no CUDA initialization calls | Compiled in candidate2; actual-image execution NOT RUN/root gate |
Generation command: `/data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-23_v4_remaining_measurements/profile01/scope_lifetime/package_preparation/build_package.py`; exact absolute argv and raw logs in `generation_01.execution.json`.
Static inventory command: `/data/ycfeng/envs/asyncopd-py312/bin/python -B -`, executing retained `closure_01.stdin.txt`; no probe/launcher execution. Exact sudo cat paths appear there. Source/package staging used three exact allocated files plus the package directory; `git diff --cached --check` PASS before commit10677995. Full source-step diff/numstat and all final commands are retained in the task directory.
Authoritative source/manifest/source_lock hashes are in `package_inventory.json` and `closure_01.json`. All retained raw hashes are computed after the corresponding writer exits; original failed logs remain unchanged.
Archive lint initially flagged one original blank stdout line at EOF; `whitespace_initial_failure.json` preserves that result. Raw diffs/logs remain byte-exact and hash-checked; maintained source/Markdown/JSON/commands are whitespace-checked separately.
Limits: CPU closure is not callback/TLS freshness, measured defaultFalse lifetime, native integration, non-perturbation/performance, GPU/source-to-replay ownership or cost/reference acceptance. No SDK/platform/auth/quota/private network/SSH/Docker/GPU/seal actions or new allocations were performed. Root must complete independent review and actual-image/SDK/storage/quota/destination/seal gates before its single approved GPU attempt.
