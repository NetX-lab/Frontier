# Local runtime inventory for the allocator abort

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Hong Kong continuation: bounded interpreter, installed library and symbol inventory; recorded primary's same-version debug-hook proposal without execution. |

Component: profile04 retained prefill DP0/rank0 CPU audit. Phase: read-only abort readiness, parallel with primary's archive. Reviewer: Codex local-runtime sidecar; primary retains diagnosis, allocation and acceptance.
Paths: `W=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`; `T=W/task_memory/task_2026-10-23_v4_remaining_measurements`; `G=T/profile01/profile04/results/p0_cpu_loop/group_metrics_tail02`.
Other prefixes: `E=/data/ycfeng/envs`; `S=/data/ycfeng/tmp/glm53_dsv4`; `D=S/p0_allocator_docs_20261025`. All SHA values below are SHA256. Prefixes denote exact absolute paths, not shell variables.

## Retained evidence and boundaries

| Inspected record | SHA256 at inspection |
| --- | --- |
| `W/AGENTS.md` | `82354337fec8131194f31ad8f09d298d2ef6e3ad48282b0b4bd46e87b5a57249` |
| `T/requirements.md` | `44e502d16093edc9406eabc703fe3954deda533c888d5928e22426a4ed71a6b5` |
| `T/plan.md` | `6292fada36076e09f469a600d7d59dcf9fc05901a8bf7189f8e733d10b5d829a` |
| `T/progress.md` | `804619dab70903cf23e87357dbf1fc1ce7286da26a126e27a72b34f39a2884ef` |
| `G/primary/abort_rca_allocation.json` | `bb74d7761e93624d8d804cc17e53dab13b4af4fb30bf2d42d341e01f55c2a21c` |
| `G/primary/diagnostic_failure.json` | `0c440c674c38f696f94efae3166f97b1a51f9af1d58a3b55bd597095b83d4b0c` |
| `G/primary/abort_runtime_metadata.json` | `e469d217e270b71fd6bad15472029989d8e68cc3026ce6828176f5ae00a1eadc` |
| `G/primary/prefill_dp0_rank0_segments.stderr` | `b9c69d8104eb32979f211d26877b54fa82b846a7748465ea07b4b30ca625b42e` |
| `G/primary/prefill_dp0_rank0_segments.hb.log` (last 48 lines read) | `1dcbd13b50f9d31eb2a1ec703b7d508a0da25f4324d33c5a6ad9182569b242bd` |

Observed: unique diagnostic 1/1 FAILED SIGABRT/-6, driver 1, enclosing 40.333549 seconds, actual 32,212,254,720-byte limit. Stderr: `malloc(): unsorted double linked list corrupted`. Last kernel progress was 39.935 seconds; no kernel-done, metrics boundary or final JSON. All three metrics hypotheses remain UNMEASURED; R5 remains 24/32. No index was opened or decompressed.
The abort Python frame is frozen `audit.py:96`, the `with gzip.open(...)` context. An earlier periodic sample is in JSON encoding. Neither frame identifies the native corruption writer; no C stack is retained. Native root cause remains UNKNOWN.
Parent metadata records RLIMIT_CORE 0/-1, core_pattern `core`, core_uses_pid 1 and no core in this exact repository root. Child core limit, child library map and cores elsewhere remain UNKNOWN; this inventory did not scan for cores. Primary reported archive commit `b06dea5d`; no Git operation verified it here.

## Existing interpreters

Eight small metadata-only Python invocations completed successfully; no further Python invocation was used. Queries used `timeout 10s <exe> -I -S -B -c ...`, preventing site processing and task imports.
For queries 1–6, exact code: `import sys, sysconfig; print("version=" + sys.version.replace("\n", " ")); print("executable=" + sys.executable); print("prefix=" + sys.prefix + "; base_prefix=" + sys.base_prefix); print("stdlib=" + sysconfig.get_path("stdlib")); print("config=" + repr({k: sysconfig.get_config_var(k) for k in ("LDLIBRARY", "LIBDIR", "Py_DEBUG", "WITH_PYMALLOC", "CONFIG_ARGS")})); print("builtin_native=" + repr(tuple(m for m in sys.builtin_module_names if m in ("_json", "zlib", "_io", "gc", "_struct"))))`.
Queries 7–8 used the same version/executable/prefix/stdlib prints, then `print("config=" + repr({k: sysconfig.get_config_var(k) for k in ("LDLIBRARY", "LIBDIR", "Py_DEBUG", "WITH_PYMALLOC")}))`; no builtin-module print. `readlink -f` and `sha256sum` inspected executable identity separately.

| Queries | Invocation paths and canonical executable | Version/compiler | Executable SHA256 |
| --- | --- | --- | --- |
| 1 | `/usr/bin/python3` -> `/usr/bin/python3.10` | 3.10.6 / GCC 11.3.0 | `dbb04ace02303fa5cb3a6d1aa0e7a180d8dd23d3257cc6a0b155d0ee219432a2` |
| 2, 3 | `/home/brainpp/.local/bin/python3.10` -> `E/cpython-3.10-linux-x86_64-gnu` -> `E/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10`; also queried the canonical executable | 3.10.20 / Clang 22.1.3 | `10d0db98ee6f9297a58cad5c8edefcc9f133ca4da8f9fb0de048dbe4a6aafa2a` |
| 4 | `/usr/bin/python3.11` | 3.11.0rc1 / GCC 11.4.0; release candidate | `639fad1cea20a48dd4b80974b5e086d30b26f97a5cc523ee37756539ffd3abe7` |
| 5, 6 | `E/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`; `E/asyncopd-py312/bin/python` resolves to it | 3.12.13 / Clang 22.1.3 | `202c17d1671602a4ef1d43e9b2fdbef0769443f37bf5e51f6b603e0b2c27d9d8` |
| 7, 8 | `E/frontier-runtime-base-py310-d29-20261025/bin/python3.10`; `E/frontier-runtime-py310-d29-20261025/bin/python3.10` | 3.10.6 / GCC 11.3.0; both byte-identical to original executable | `dbb04ace02303fa5cb3a6d1aa0e7a180d8dd23d3257cc6a0b155d0ee219432a2` |

All eight queries report `WITH_PYMALLOC=1`, `Py_DEBUG=0`; build flags are not an observation of the failed child's active allocator. The system builds report libpython3.10.so/libpython3.11.so and LIBDIR `/usr/lib/x86_64-linux-gnu`; standalone builds report their prefix-specific lib directories. ELF NEEDED does not list libpython for the inspected system 3.10.6 or standalone executables.
Bounded registration checks: `task_memory/env_handbook.md` is absent in W and `/data/ycfeng/Frontier`; checked conventional conda/pyenv registries did not supply environments. Existing `E/*/pyvenv.cfg`, `E` and uv registration directories supplied paths. No recursive host search occurred.
Manifest/readlink/hash evidence also identifies `E/frontier-py310/bin/python` as original 3.10.6; `E/distillm-py310/bin/python` as the same 3.10.20 executable; and `E/{oneshot-opd-py312,openmopd-py312}/bin/python` as the same 3.12.13 executable. Their hashes match the corresponding table rows. No extra Python queries qualified these environments. Under `-S`, queried venv prefixes resolve to their base; package/import compatibility remains UNKNOWN.

## Source, library and summation evidence

| Inspected source | SHA256 |
| --- | --- |
| `S/W128-PROFILE02-DATA/audit.py` | `7ed0ea45a61d5150e2f45bbc1cdf245f2331e40a351c235228b209b064ede0c8` |
| `W/tests/comparison/calibration/forward_device_timeline.py` | `fe4c3f82500df79ce097ce762808bcaf2b78040e41cbab942ccc746a723b3a52` |
| `S/p0_audit_cpu_20261024/stage_audit.py` | `f7f1253ce9ebae5dcb3b60e2aeb97d7f17f58f34a7d68af56611918f809b9fe4` |
| `S/p0_gc_observation_20261024/observe_gc.py` | `3ae64dce5e1498418c3489d48bb74393f08c4dbbcb95231bab4e11a414a5a83c` |
| `S/p0_gc_observation_20261024/load_without_gc.py` | `6b91e25fc592d4f1340ef22e6d6e73c4dd8ed6e7ce6239f7a113d946a74edd02` |
| `W/tests/analysis/v4_trace_audit_group_metrics.py` | `4dec265e20426a20e437588a4e44828e257247c82897cc8188621f93369a5251` |
| `W/tests/analysis/v4_trace_audit_group_metrics_timing.py` | `9f456317208fa3646fd32cf24d6e92cfe7c9ca08da716b98aabf0b0aa619c426` |
| `/usr/lib/python3.10/json/encoder.py` | `721a182f3473f01f4dc9ce61d6dbc1ad1f9db0138cd93e4121e0ac9173b91a27` |
| `/usr/lib/python3.10/json/scanner.py` | `8604d9d03786d0d509abb49e9f069337278ea988c244069ae8ca2c89acc2cb08` |
| `/usr/lib/python3.10/gzip.py` | `fb131d4bbe711c410f54c7f0b3c2c64292e1279199fd04c53837db03e542a907` |
| `/usr/lib/python3.10/pydoc_data/topics.py` | `413867b6761f923ddf79ff8c916e8317d82413f8e50901ff70c8e966301aea4c` |
| `/usr/lib/python3.11/pydoc_data/topics.py` | `9c9b074163452f062f7154b12b177276f676076d9d778e051b023ec2d0949114` |
| `E/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/pydoc_data/topics.py` | `646348f24f4e5f79c2deedbfc830197938c0a136b42cac9ec33634756a09d8a5` |

The pinned reader uses gzip + json.load; the pipeline disables cyclic GC only during loading, freezes after loading and re-enables it. Installed JSON scanner/encoder source selects `_json` acceleration; gzip uses zlib. System 3.10.6's `_json` is an installed extension; standalone 3.10.20/3.12.13 metadata lists `_json` as builtin. These are reachable native components, not cause assignments or a complete child module list.
Frozen `audit.py:34` and group adapter `:32` delegate `sum_ms` to `sum(e['dur'] for e in members)` in original member order; duplicates matter. Inspected installed pydoc topics do not specify this accumulation algorithm. Standalone 3.10.20/3.12.13 embedded `sum` docstrings describe numeric summation but do not specify rounding strategy; inspected conventional `Python/bltinmodule.c` and `test/test_builtin.py` paths are absent.
Consequently, only aliases/copies of the identical 3.10.6 executable preserve its builtin implementation by byte identity; environment equivalence is untested. No distinct interpreter is verified here by installed source/docs to preserve the old sequential float sum: 3.10.20 UNKNOWN, 3.11.0rc1 UNKNOWN, 3.12.13 UNKNOWN. Do not assume 3.12 exact equality. This is readiness inventory, not a runtime recommendation or qualification.

| Installed native evidence | Identity / finding | SHA256 |
| --- | --- | --- |
| `/usr/bin/gdb` | dpkg gdb `12.1-0ubuntu1~22.04`, installed; not executed | `2a2f167b5f9c9053732f79a0f4418ed75c211e00275da6090ab64a05b81a12e0` |
| `/lib/x86_64-linux-gnu/libc.so.6` | libc6/libc6-dbg `2.35-0ubuntu3.1`; Build ID `69389d485a9793dbe873f0ea2c93e02efaa9aa3d`; contains the exact abort text | `568740b06a8afa26db4874f8cf61985ecbc6dd127f4229416fe95da8f9ec13fb` |
| `/usr/lib/debug/.build-id/69/389d485a9793dbe873f0ea2c93e02efaa9aa3d.debug` | matching debuglink, DWARF and `_int_malloc`, `malloc_printerr`, `unlink_chunk` symbols | `0e50c8edfdc06a29bdce9f787686db39e245a622fdabd1a1b9a430f30d5cf618` |
| `/lib/x86_64-linux-gnu/libz.so.1` -> libz.so.1.2.11 | zlib1g `1:1.2.11.dfsg-2ubuntu9.2`; NEEDED by original executable | `64c206f0146cc58bbddc4f22054436f4ff278f5a554aa3ce6921ddf7e9133370` |
| `/usr/lib/python3.10/lib-dynload/_json.cpython-310-x86_64-linux-gnu.so` | existing local CPython JSON extension | `c97effb6e762e78af6eb451950283ebce4f6da82968ed53d51b8f5c252e77faf` |

System python3.10/libpython packages are `3.10.6-1~22.04.2`; python3.10-dev is installed. python3.10-dbg/libpython3.10-dbg were not found. Matching Python 3.10 Build ID `9399be8ddf8047a8ae050a6d3d18e4ce488264da` and Python 3.11 Build ID `e6879846d787838e9af6da4484aac7dd4ab3cf25` debug files were absent; no CPython GDB helper was found in the checked auto-load/doc directories. Source files malloc.c/bltinmodule.c were absent at checked conventional paths. No package was installed.

Primary supplied official v3.10.6 references: `D/memory.rst` SHA `2d57812f1b5666cbb2ed0abbb6bef53b6520eb7fb305b2791b20505c95bd425e`; `D/obmalloc.c` SHA `2e5a9ce05a252f6bdf70304f682038af64ffe23c68ea8aba7702d204ea522484`; `D/receipt.json` SHA `9f5d9b8c7c3fcbd051f05e16d88d9713924e7a07a08eb4d0100a0dfa2ef28bde`. The receipt and bounded debug-hook lines were read locally; these references are not proof of installed binary contents or a bug match.
Readiness: primary's potential next single-variable observation is `PYTHONMALLOC=debug` with the same 3.10.6/frozen composition. Existing metadata supplies WITH_PYMALLOC/Py_DEBUG, and GDB plus matching libc symbols are available for native evidence if needed. No new invocation is selected here; primary must decide and freeze it. Detection location, child allocator/library state and corruption origin remain UNKNOWN.

Self-check against current plan item 5: local inventory PASS; primary archive/diagnostic decisions remain primary-owned, R5 incomplete and full R6/R7 blocked. In this sidecar window D1–D7 are CLEAR, S1–S6 show no current-record defect; Branch A retains the recorded route and stops inventory at eight Python calls. Earlier shell-only reads had zsh `path` variable errors and one extra `done`; corrected bash reads completed without task execution or environment changes. Only this report is written; zero controls, tests, benchmarks, raw-trace/index reads, task-script imports, debugger, Git, GPU/platform/network/auth/SSH or child-agent actions occurred.
