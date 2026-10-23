# Balanced q1 preparation test report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archived CPU package/source/entry/suite/aggregate verification and its limits. |

Task: W128-V4-BALANCED-DECODE-PACKAGE. CPU preparation PASS; root image/seal/runtime checks remain pending. Detailed assertions, commands, stdout, stderr, negative results and hashes are in `cpu_checks.json`. Commands ran from `/data/ycfeng/Frontier/.worktrees/v4-balanced-decode-run` unless otherwise specified. Python environments were existing virtual environments, not conda: Frontier Python3.10.6; CPU Torch tests Python3.10.20 / Torch2.1.2+cu121, CUDA initialization false. `CUDA_VISIBLE_DEVICES=''` was set for Torch tests.

## Exact generation and suite commands

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-23_v4_remaining_measurements/balanced_decode/prepare_package.py
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_dsv4_attention_profiling.py tests/unit/test_balanced_routing.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_balanced_decode_pytest --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_balanced_decode_pytest.xml
env PYTHONPATH=.:/data/ycfeng/envs/frontier-py310/lib/python3.10/site-packages PYTHONDONTWRITEBYTECODE=1 VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0 CUDA_VISIBLE_DEVICES='' /data/ycfeng/envs/distillm-py310/bin/python -B -m pytest tests/unit/test_dsv4_attention_profiling.py tests/unit/test_balanced_routing.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_balanced_decode_torch_pytest --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_balanced_decode_torch_pytest.xml
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_dsv4_attention_workload.py tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_balanced_decode_related_pytest --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_balanced_decode_related_pytest.xml
git diff --check
```

| Check | Passing condition | Observation | Result |
| --- | --- | --- | --- |
| Original q1 / balanced prefill | Every original manifest input unchanged | 527 / 529 inputs match; manifest hashes in package plan | PASS |
| Candidate02 inputs | All hashes match, accepted source snapshot/model/C26/strategy bound | 530 manifest inputs, 512 snapshot files, 520 tool pins, 60 native Git pins, 20 binary pins recorded | PASS CPU; actual image binaries UNVERIFIED |
| Minimal profiler/routing run | Both modules run without skips | 65 passed / 1 module skipped (Torch absent), 1.20s | INCOMPLETE; superseded |
| Existing Torch environment | Zero failures/errors/skips | 131 passed, 18.27s | PASS |
| Related workload/measured inputs | Zero failures/errors/skips | 256 passed, 78.53s | PASS |
| Exact case and constructor | Current plan and configured engine fields agree | seed6141, warmup2, measured1024, contexts6143–7166, native estimated1027/method, planner128/effective1 | PASS CPU stand-in |
| CLI versus LLM budget | Derive from actual source, do not conflate runtime observation | CLI6144; existing LLM constructor argument6144; actual native value UNKNOWN | PASS distinction |
| Launcher entry | Fresh imports and correct WORKSPACE before submit body | Both Frontier and StepMind Python interpreters reach actual launcher submit boundary, no runtime/platform preload or platform calls | PASS CPU boundary |
| Worker entry | Fresh stdlib CLI import | Actual copied worker `--help` accepts `--config` | PASS |
| Bootstrap/wrapper | Use real bootstrap launch argv/env, origin/hash checks before registration, no dependency preload | Fresh child reaches actual `_run_command` boundary using CPU-only installed package/binary stand-ins; source/selector/hash negatives rejected | PASS CPU stand-ins only |
| Existing bootstrap refusals | Eight unchanged controls pass | Success, source, base, binary, binary manifest, extra file, origin, early import | PASS8 |
| Synthetic full-window producer aggregate | Exact records and rows plus selected identity | 1024 steps/method; 51200 / 45056 calls; 6144 rows/method; balanced_topk + strategy SHA + moe_routing present; bad strategy rejected | PASS CPU fixture; no GPU timing |
| Fresh generation refusal | Existing destination not modified | Second builder invocation exits1 with `Fresh generation only` and input hashes stay identical | PASS |
| Matrix gate | Existing capture digest and all selected domains unchanged | Existing77/77 at4a1b53e6, no new capture; comparison SHA7a8728ecc0851373827d9cdc7dfb68c552299e6c54b5fae75bd8cc5429d36ad6 | PASS explicit binding |
| Scope | ≤2 maintained non-core files/300 gross lines; no forbidden edits | Builder175 + plan39 =214 lines; generated/records excluded | PASS |

The direct assertion scripts ran as `/data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY' ... PY`; they inspect only local package/source bytes and invoke the recorded fresh-process commands. `cpu_checks.json` retains the full child probe strings and argv, generated bootstrap argv/environment, negative tracebacks and synthetic aggregate argv/output. Aggregate uses the existing `tests/unit/test_dsv4_attention_profiling.py` record factories; no consumer layout fixture or admission exists.

## Observed corrections and retained limitations

1. Candidate01's unchanged launcher selected the wrong WORKSPACE because the package is one directory deeper than historical packages. Candidate02 changes only the generated `WORKSPACE` assignment to the absolute owned worktree root. Original launcher/bootstrap/worker/model bytes remain unchanged. Candidate01 remains locally retained and unsealed; its530 input hashes still match.
2. First evidence script compared tuple-bearing `plan.to_json()` against parsed JSON lists and failed. Serialized value equality passes. The next script sliced aggregate argv at index4 instead of the actual subcommand at index3; it failed in argparse, then the corrected source-derived slice passed. Neither changed product/package bytes. These are evidence-script errors, not ignored PASS results.
3. Existing `ResourceConfig` construction is reused unchanged; the job JSON requests non-preemptible execution, but root must check actual SDK defaults/platform resources. No StepMind call was made.
4. Local native Git hashes and fake installed binaries do not establish the actual locked-image contents. Root owns real image/bootstrap checks, quota/output-destination checks, external seal and launch. No local/cloud destination existence is inferred from a planned path.
5. The worker archives to `job.output_dir / phase.name`; therefore the actual tables/raw/traces would live one repeated phase-name level below the planned output root. This follows the unchanged worker and is documented in `summary.md`.
6. No GPU timing, physical TP8/DP2/EP16 parity, remote-prefill first-tail equivalence, calibrated serving total or decoder sidecar/admission is delivered. No cloud, Docker, StepMind, GPU, seal or external publishing operation occurred.
