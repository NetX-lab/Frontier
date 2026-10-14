# Tool Boundaries

v2 adds no runtime helper implementation. Each skill uses an existing tested
tool only when the registry row names it as an input or validation tool. The
active v2 tree intentionally contains no helper file outside `tests/`.

## Existing Tools and Owners

| Tool | Owner | Use | bytes | sha256 | v2 rule |
| --- | --- | --- | ---: | --- | --- |
| `/data/ycfeng/frontier-calibration-old-20260831/current-default/frontier-calibration/e2e-metrics-gap/scripts/normalize_request_metrics.py` | `e2e-metrics-gap` | Request ID normalization, timing-sidecar join, formal-window filtering, and atomic normalized output | `95818` | `4fb0f4bfd1b44a25493d7bb7a2fcc73cb2fb63413d2bed13311ef6237dcfff8b` | Invoke with an explicit manifest, case, generation, source profile, and fresh output path. |
| `/data/ycfeng/frontier-calibration-old-20260831/current-default/frontier-calibration/op-supplement/scripts/profiling_dataset_tools.py` | `op-supplement` and `cpu-overhead-supplement` | Coverage diff and lossless complete-dataset merge | `99483` | `8e23cf1e0c6bd274c4230d2d8310ebab953f032042b4a436565a7a8756416c8f` | Keep the base CSV unchanged; write the complete merged CSV under the case `supplements/` directory and verify its completion marker. |
| `/data/ycfeng/frontier-calibration-old-20260831/current-default/frontier-calibration/dispatch-align-trace/scripts/generate_workload_trace.py` | `dispatch-align-trace` | Deterministic staggered online input | `2806` | `74131ecd857d614dc7dbb318758d978ebb7c8c2b32f92b669fa90fa7d274d317` | Use only for an online dispatch study. |
| `/data/ycfeng/frontier-calibration-old-20260831/current-default/frontier-calibration/dispatch-align-trace/scripts/replay_openai_workload.py` | `dispatch-align-trace` | Observe dispatch offsets and priorities | `11680` | `3dfc0c9cb4cad8f64cf2b71272ba4cda554481b461617e36922eb34049ba5960` | Preserve request IDs, token counts, and priorities. |
| `/data/ycfeng/frontier-calibration-old-20260831/current-default/frontier-calibration/dispatch-align-trace/scripts/build_frontier_trace_from_vllm_metrics.py` | `dispatch-align-trace` | Convert observed offsets to a Frontier trace | `4675` | `30435ad3841ff24424d1e2382a6cba239b577731fea3a44b29ae761c6b364c0d` | Write only the declared aligned-trace path. |

The paths point to the maintained helper files in the archived
`current-default` recovery tree. They are direct execution targets, not v2
skill directories, and they are outside the configured discovery root. v2 does
not copy or fork them, so the active tree stays prompt-first and its helper
owner remains explicit. Before every invocation, verify the recorded byte count
and SHA-256. A missing path, byte-count mismatch, or hash mismatch is an
immediate `FAIL`; do not fall back to an unlisted helper path.

## Command Receipt

`exec capture: UNKNOWN`. The available platform configuration has no proven hook
for intercepting every shell command. A v2 entry records a caller-written receipt
for each command it launches: exact command, working directory, declared
environment, UTC start/end, exit code, selected checkout, and expected output
paths. This receipt covers that entry's own calls only.

The legacy machine field `mutation_marker`, when encountered in an old receipt,
means the same thing as the v2 plain `change_marker`: an identifier tying the
approved code-change intent to the resulting receipt. New v2 documents use
`change_marker` and “code change” wording.

## Tool Failure

Treat a missing tool, non-zero exit, malformed output, stale output path, or
completion-marker mismatch as an explicit failure. Do not substitute an
unverified script or hand-edit an active profiling CSV. Record the failure in
the case progress log and stop at the owning atomic.
