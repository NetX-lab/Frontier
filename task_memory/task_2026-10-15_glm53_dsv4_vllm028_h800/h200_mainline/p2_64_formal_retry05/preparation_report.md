# Generation 05 offline preparation report

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-04 | Pi completed offline preparation; lead independently verified identity replacements and check-only. |

Status: `PREPARED_UNSUBMITTED_PENDING_USER_APPROVAL`.

No GPU, platform, quota, cloud, or SSH operation ran. No submission or wrapper
launch ran. `launch.py --check-only` is the only execution of the generated
launcher. No approval receipt was created and no resource allocation is approved.

## Inputs (unchanged)

| Item | Value |
| --- | --- |
| Builder | `h200_mainline/build_package.py` (one added `CASES` entry) |
| Source package | `h200_mainline/p2_64_reference/package_01` (read only) |
| Harness | `/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness` @ `f33306d40443cfc0c44842d6ee044f6703c736e7` (clean) |
| Native | `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/p2-dp-startup-coordination` @ `a06a72a4326cba02234df6cfbdc21b78d3a15d26` (clean) |
| Promotion receipt | `h3o_operator_costs/formal_p2_20261102/promotion/receipt_p2_32_to_p2_64.json` sha256 `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596` |
| Predecessor package | `h200_mainline/p2_64_formal_retry04/package_01` (read only, manifest sha256 `d75b0277a3c08592359002d256ac8115844f50fa051efe422815cc5b2d19f081`) |

Resource shape carried over unchanged: H200 / `step_main` / 8 replicas x 8 GPUs
= 64 GPUs, wholly preemptible, `timeout_seconds` 3600, 64 GPU-hours payload
guard, three clean repetitions of 128 formal requests at 1 QPS with
6144 prompt / 1024 output tokens, 16 warmup requests per repetition, pair
warmup, P32+D32 TP8/DP4/EP32/PP1.

## Command and result

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp
/data/ycfeng/envs/openmopd-py312/bin/python -B build_package.py p2_64_formal_retry05 \
  --package package_01 --harness /data/ycfeng/Frontier/.worktrees/p2-64-startup-harness
```

Exit 0. Stdout retained verbatim in `executor_build.log` (a `.log` file whose
content is the builder JSON, byte-identical to `package_01_build.json`).
stderr empty (`executor_build.stderr.log`, 0 bytes).

Builder-internal steps that passed: package comparison against the source
package, `seal.py`, run-manifest preflight on scratch copy
(`/data/ycfeng/tmp/glm53_dsv4/h200/p2_64_formal_retry05_preflight_zell7boh`,
authorization `preflight only; no submission`), R3 Frontier reconstruction
(`h200`, TP8, attn_dp 4, EP32, PP1, world 32 for both roles), per-host launch
plan, and `launch.py --check-only`.

## Seal

- sealed entries: 47
- seal mismatches: none (`manifest['files']` digests recomputed from the package)
- `manifest.json` sha256 `23e26da6c02cc42e54134c12abfdfea298f64191e0f985a860505e374278227c`
- sealed digest multiset sha256 `76c7762364e511ed50dc45cff5d55700f7cded61624d2bf7dc8b6d020317f414`
- `submitted: false`, `native_qualification: false`
- no `launcher_started.json`, no `launcher_exit.json`, no `case/runs`

## Comparison with generation 04

`compare_to_retry04.txt`: 47 sealed entries, 6 changed, 41 unchanged.
Changed: `config.json`, `case/inputs/native_config.json`, `case/manifest.yaml`,
`evidence_readiness.json`, `overlay.json`, `source_lock.json`.

After normalizing the package path, the run/cloud identity, the purpose text and
the promotion `build_utc`, the only remaining differences are:

- `retained_files.overlay.json`: `72bef4cc...` -> `b8de9508...` (derived;
  `overlay.json` records each staged file's `source` path inside the package
  directory)
- `case/manifest.yaml`: `run_generation` and `runtime_source_overlay.sha256`
  (the same derived overlay digest)
- purpose text (generation04 lead-caused interruption -> generation05 external release_quota stop)

All input pins are equal between 04 and 05: `engine`, `expected_source_hashes`,
`harness_files`, `model_hashes`, trace digests, `timeout_seconds`, image,
harness commit, native commit, promotion receipt path and sha256, and each of
the 41 unchanged sealed files (including all four native DP startup sources,
`launch.py`, `native_pd_worker.py`, `submit_native_diagnostic.py`, the retained
harness modules and the trace files).

## Check-only

```bash
/data/ycfeng/envs/openmopd-py312/bin/python -B <package>/launch.py --check-only
```

exit 0, stdout `INPUT_HASHES_PASS_NO_SUBMISSION`
(`launch_check_only.txt`). No launcher marker file was produced.

## Remaining work (owned by the lead)

- User resource decision for a fifth GPU submission. Nothing here authorizes it.
- If approved: the approval receipt and submission path stay with the lead.
