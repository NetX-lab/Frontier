# Graph resource payload CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded exact CPU commands, safe RED/GREEN, ABI/source/matrix evidence and limits. |

Task ID: `W128-GRAPH-PAYLOAD-PROBE`. CPU/source component only; no runtime qualification.

## Environment

Workdir: `/data/ycfeng/Frontier/.worktrees/v4-graph-payload-probe`.
Python: `/data/ycfeng/envs/oneshot-opd-py312/bin/python`,3.12.13, Clang22.1.3.
pytest9.1.1; Linux5.10.25-nvidia-gpu-x86_64, glibc2.35; C++ compiler Ubuntu11.3.0.
Used an existing absolute interpreter without conda activation, installation or dependency changes. `python` on PATH was unavailable (exit127); no behavioral workaround was added.

Common environment: `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode`. Raw temporary logs/XML are `/data/ycfeng/tmp/w128_graph_payload_{red,green,inherited}.{log,xml}`. Selected exact copies are retained beside this report.

## Exact CPU invocations

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode \
  FRONTIER_GRAPH_PROBE_SOURCE=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_graph_probe_attempt3_20261021/probe_diagnostic.py \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -m pytest \
  /data/ycfeng/Frontier/.worktrees/v4-graph-payload-probe/tests/unit/test_cupti_graph_resource_payload.py \
  -q -p no:cacheprovider -k resource_envelope_exact_handles_ids_and_copy \
  --basetemp=/data/ycfeng/tmp/w128_graph_payload_red \
  --junitxml=/data/ycfeng/tmp/w128_graph_payload_red.xml

env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -m pytest \
  tests/unit/test_cupti_graph_resource_payload.py -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/w128_graph_payload_green \
  --junitxml=/data/ycfeng/tmp/w128_graph_payload_green.xml
```

The unchanged original runner's `DiagnosticCopy` plugin is loaded with `runpy.run_path`, not executed as a CLI. Its existing `HERE` function-global is set to the new candidate directory. The exact pytest argument vector and runner hash are in `cpu_receipts.json`. The injection invocation was:

```python
module = runpy.run_path(str(runner))
plugin = module["DiagnosticCopy"]()
plugin.pytest_sessionstart.__func__.__globals__["HERE"] = candidate_directory
code = pytest.main(args, plugins=[plugin])
```

Original tests: `/data/ycfeng/Frontier/.worktrees/glm-full-dummy-h200/tests/{profiling/test_cuda_graph_probe_attempt1.py,unit/test_cuda_graph_scope_probe.py}`. Selected `-k` excludes only `callback_fields_are_copied_only_while_native_data_is_valid`, `callback_never_propagates_python_exceptions_across_c`, and `callback_layout_matches_compiled_cuda13_header` (five parametrized cases). Those frozen tests pass bare GraphData or demand the old exact ABI dictionary; new allocated-envelope controls replace them. No frozen test/probe/runner source was rewritten, and no test ran accidentally on the old source:71 loads redirected to candidate.

## Expectations and observed results

| Check | Passing condition | Observation | Verdict |
| --- | --- | --- | --- |
| Safe old-source RED | Allocated padded envelope yields wrong handles; fake functions only; pytest exit1 | 2 intentional failures,14 deselected,0.16s; clone descriptor-address/927, exec901/951 | PASS expected RED, not a crash reproduction |
| Actual corrected callback | Exact11/12 handles/full uint64 IDs and701/702 graph/exec IDs, native callback order, copied scalar lifetime | 16PASS,0.25s; malformed/null descriptor has no query; KeyboardInterrupt isolated; later callback succeeds | PASS |
| Subscription lifecycle | Only NODE_CLONED20/EXEC_CREATED18; ordinary occupied/enable/unsubscribe failures preserved | Included in16 new cases and inherited lifecycle/weak-reference controls | PASS |
| Inherited graph predicates | Candidate-driven original controls preserve strict9/5/4, scope DAG, unscoped kernel, direct links/negatives and actual attempt1 failure | 71PASS/5deselected,0.92s;71 candidate loads | PASS within stated exclusions |
| ABI | Header native sizes/offsets equal ctypes; unchanged GraphData fixture | C++ syntax-only assertions return0, no output; ResourceData24/align8/0,8,16; union8/stream0; GraphData56/align8 | PASS source-layout only |
| Source budget | At most80 gross lines; only approved decoding/ABI differs | 24 additions+2 deletions=26;645 identical inherited lines/32705 bytes; all other definitions and callback tail AST equal | PASS |
| CPU controls budget | At most160 lines; no compressed implementation | New unit file154; additional compiler stdin5; total159 | PASS |
| Historical preservation | Existing37 pins and12 attempt3 manifest files unchanged | All49 exact SHA256 checks match | PASS |
| Existing matrix | Exact bound complete77 comparison and unchanged simulator domains | SHA2567a8728ecc0851373827d9cdc7dfb68c552299e6c54b5fae75bd8cc5429d36ad6;77identical/no mismatches | PASS binding, no new capture |

ABI command: `c++ -std=c++11 -I/data/ycfeng/envs/oneshot-opd-py312/lib/python3.12/site-packages/nvidia/cu13/include -x c++ -fsyntax-only -`. Exact five-line stdin and compiler output are in `abi_source_binding.json`; header hash `b2f82221fb03ecb2b2a27afa2d64284babdf49fee1a3a2811258d473285e1105` matches publisher-retained and root-image bytes. No linking or native execution.

Source verification uses stdlib AST and SequenceMatcher on the exact frozen/candidate files. `source_verification.json` records all unchanged old definitions, unchanged subscription methods, the unchanged callback tail after envelope decoding and old→new line mapping. `source_delta.diff` retains the full exact change. No diagnostic I/O, query wrapper, mapper or callback-set change was inserted.

Whole staged `git diff --cached --check` returned2 solely for verbatim pytest RED log/XML trailing spaces and unified-diff blank context lines. These raw evidence bytes were intentionally preserved. Source, maintained documents and JSON are checked with those three raw artifacts excluded; no source whitespace repair or evidence rewriting was needed.

## Worker fault/source boundary

Root's retained `graph_source_evidence/image_source_receipt.json` reports worker Python3.12.3 and image `sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`. Frozen source SHA256 `f7314cfbc70a1ba347339245754e385d24239eff9f24ceb485e5f3fba379a1e0` corresponds to retained faulthandler `_read:119 → node_id:125 → _callback:197 → graphs.py:180 instantiate`. The source expression at197 is `self.ids.node_id(row["original_node"])`, now candidate219. Local3.12.13 disassembly has node CALL516/line196 and original-node CALL574/line197. Exact worker3.12.3 bytecode was not executed here; `fault_source_mapping.json` explicitly separates static binding, observations and inference.

## Limits

87 CPU cases pass;2 safe expected old-source assertions fail;5 original cases are explicitly replaced. This proves the allocated source-layout correction and preserved CPU predicates, not that the driver supplies this envelope in the actual callback or that native graph clone coverage succeeds. No real invalid-pointer CUPTI query occurred. Original exit139 cause, actual callback delivery, strictU1–U4 runtime outcome, native serving ownership and consumer admission remain unproven. Root owns reviews and actual package/image/SDK/seal/run acceptance. No paused GLM operation was resumed; old3/3 attempts remain unchanged.
