# Native checker paths and worker mount

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Locked-image RED confirmed both missing checker paths; removed only redundant worker tool entries, retaining host checker bindings. |
| 2026-10-23 | Recorded the pre-execution mount hypothesis and minimal locked-image verification plan. |

## Suspected failure before package construction

Builder0832200 adds the host-only R3 and R2 paths to `config.tool_hashes`. The unchanged submitter changes cwd to the package and mounts exactly that directory through `spawn_tasks(..., code_mount_point=config['workspace'])`. The unchanged bootstrap validates every tool path before native imports. Neither checker lies within the mounted stage.

Ranked hypotheses before execution:

1. Host-only checker paths are absent in the worker namespace. Actual `bootstrap.verify_hash` should raise FileNotFoundError on a checker while all staged inputs pass.
2. A checker is visible but unreadable. The same call would raise PermissionError, rather than FileNotFoundError.
3. The file is visible with stale bytes. The same call would raise the existing Hash mismatch ValueError.

Use the locked image without GPUs/network, with only immutable native05 stage mounted read-only. Exercise the original bootstrap verifier over its original valid tool map plus the two current builder-added host paths. Compare with that same map excluding only these two additions. Preserve original files, checker rules and package bytes. This is the minimal real pre-import failure seam; constructing a model, changing C26 or adding persistent test machinery cannot improve this signal. Full new-package verification remains required afterwards.

R3 is already bound by the case manifest checker path/hash, and `successor_source()` verifies its R2 dependency's fixed SHA256. These host-side bindings must remain intact in any correction.

## Observed RED and correction

Exact command and verifier snapshot: `/data/ycfeng/tmp/glm53_dsv4/native_checker_mount_d97gefx5/{command.json,check.py}`. Docker used the locked2286e853 image, Python3.12.3, no GPU/network and only native05 stage mounted read-only. The original bootstrap verified all16 original staged tools. Each added host-only checker raised **FileNotFoundError**, confirming hypothesis1 and excluding permission/hash errors for this failure. The map without those two entries passed all original tool checks. Receipt: `evidence/native_checker_mount.json`; no package/native/C26 file was modified.

Root removed the builder's two routing-added lines that appended host-only checker paths to worker `tool_hashes`. R3 path/SHA in the case manifest and its fixed R2 dependency SHA stay unchanged; root preflight and final invocation verify these host files against the freeze receipt. No checker predicate, bootstrap check, worker mount or acceptance threshold changes. Current actual routing delta decreases from952 to950 (A432/B316/C202), within the recorded952 ceiling. Dispatch remains491 lines.

The deterministic real bootstrap check is the focused regression evidence; the new package's same-image preflight will re-exercise its complete generated tool map. No generic harness or additional permanent test module is needed for this task-local builder correction. Native06 build and full image admission remain pending.
