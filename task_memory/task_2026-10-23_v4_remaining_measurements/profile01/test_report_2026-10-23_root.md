# Root native-profile package verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded initial profile01 image/hook checks, fresh-entry RED controls and preserved verification-harness errors. |

## Scope and environment

Initial package source `69f92c55`, records `62092da3`, accepted runtime `05952396`. Package path is recorded in `initial_delivery.json` once archived by its owner; its basename is `candidate_h800_p0_profile_01` under the `v4-native-profile-run` worktree's `v4_p0_native_20261021/` directory. It remains unsealed and unsubmitted because its actual entry import fails. This report grants no GPU, native-reference, operator-cost or E2E qualification.

Host: `kun-workspace-vgen2`, local NFS IP `100.96.128.195`. Host verification interpreter: `/data/ycfeng/envs/frontier-py310/bin/python` (Python3.10.6); locked-image interpreter Python3.12.3. Image digest: `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`. No conda activation. Raw evidence below is under `/data/ycfeng/tmp/glm53_dsv4/`.

## Observed results

| Check | Expected condition | Observed result | Verdict |
| --- | --- | --- | --- |
| Initial image invocation | Explicit source/binary verifier JSON | Exit0 with empty stdout: Docker stdin was not enabled | FAIL, harness invocation; no image verdict |
| Corrected locked-image check | Explicit PASS with87source/23tool/20binary pins; unchanged45files | Exact expected counts, no torch/vLLM import or CUDA initialization, package unchanged | PASS for pins only |
| Four isolated actual-wrapper probes | Inject selected extension and install hooks, correct limits/paths, no CUDA | Four separate PASS witnesses;8profile hooks and7inventory hooks, empty error lists, four role/DP paths | PASS for isolated hooks, not actual entry |
| Fresh direct worker import | Import from package root without preloading dependencies | `ModuleNotFoundError: No module named 'vllm_replay'` | RED, package defect reproduced |
| Actual copied launcher | Reach manifest-call boundary without any run write | Same error at launcher30/worker19 | RED, package defect reproduced |
| First bootstrap callback adapter | Execute fresh child through actual bootstrap environment | `TypeError: inspect_launch() takes 2 positional arguments but 3 were given` | FAIL, root adapter; no entry verdict |
| Corrected bootstrap callback | Fresh child imports worker using generated environment | Same `vllm_replay` import error | RED, package defect reproduced |

The isolated probe imports `pd_replay` before the worker, which inserts the needed dependency path. Its successful hook results therefore do not settle launcher/bootstrap entry correctness. The fresh-process controls remove that masking dependency.

## Exact commands and retained evidence

Image verification adapts only the two expected count literals in the existing `/data/ycfeng/tmp/glm53_dsv4/verify_v4_prefill_image.py` in memory, preserving its original SHA-256 `55236cfa09d3463fc26a81120260903d5542c32565904a4a342860fa9068038d`. The exact Docker argv and executed `stdin.py` are retained in each directory:

- `profile01_image_2r56_agv/{receipt.json,verdict.json,stdin.py,stdout.log,stderr.log}`: rejected empty-output attempt.
- `profile01_image02_yrvh04pj/{receipt.json,stdin.py,stdout.log,stderr.log}`: corrected `docker run -i --rm --pull=never --network=none --cap-drop=ALL --cap-add=DAC_OVERRIDE --memory=4g --cpus=2 --pids-limit=128 -v <package>:<package>:ro --entrypoint python3 <locked-image> -I -B - <package>`; stdout contains the explicit PASS witness.
- `profile01_startup_41qkb0xr/{receipt.json,stdin.py,0.receipt.json,1.receipt.json,2.receipt.json,3.receipt.json}` plus per-position logs: four Docker commands with6GiB memory and the executed CPU stand-in probe. Actual native worker constructors, model hooks, forwards and graph replays were not executed.

The direct-import command and original traceback are in `profile01_entry_red_acbrv5ib/receipt.json` and `stderr.log`:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -I -B -c \
  'import sys; sys.path.insert(0,sys.argv[1]); import native_pd_worker' <profile01-package>
```

The real launcher control was:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B \
  /data/ycfeng/tmp/glm53_dsv4/profile_package_entry_probe.py <profile01-package>
```

`profile_entry_naaqrp6l/` retains the exact copied launcher argv, test-only manifest, `fixture/sitecustomize.py`, stdout/stderr and receipt. The only added import path contains the trace hook, not a dependency module. No dependency is preloaded. Authorization is absent as an additional stop before any real manifest write if the trace hook fails; no platform call or original-package seal occurs.

The real bootstrap-child control reuses the image-verifier callback with `/data/ycfeng/tmp/glm53_dsv4/profile_bootstrap_entry.py`; exact composed `stdin.py` and Docker argv are in `profile01_bootstrap_entry_red_4wcvq03i/` (adapter error) and `profile01_bootstrap_entry_red02_7mv1c4cq/` (actual entry RED). The latter uses the generated child environment and only models the script's own package-directory entry in `sys.path`; it does not add the missing dependency directory. Both attempts preserved all45package files.

## Platform and storage preparation

Read-only personal Python census: `profile01_quota_7hgcy67x/{quota.json,census.json}` at2026-10-23T09:06:17Z, Python3.10.6/brainpp2.8.6. The existing `read_quota.py` was executed with only its output destination changed in memory; source SHA-256 `19746066439f539ff889dd154cac6f32bffd5467a63f892e8dd87f3ca784f275`. Codesign non-preemptible use124/128GPUs; zero active personal jobs. This does not reserve capacity or prohibit a single queued submission. Fresh quota is required at submission.

The personal cloud parent `/mnt/ws-jfs/ycfeng/frontier/glm53_dsv4_vllm028/t18` was readable/traversable with mode0755, and child `dsv4_h800_p0_profile_20261023` did not exist. Read-only `ResourceConfig()` inspection confirmed `preemptible=False` and RDMA defaults `mellanox.com/mlnx_rdma:1`, `rdma/mlnx_shared:8`.

## Open acceptance

Original owner is repairing only generated entry paths through the builder and preparing fresh profile02. Required: independently reviewed repair, fresh launcher and real-bootstrap entry GREEN, updated image/source/package checks, then root seal/submission and actual retained-artifact verification. All runtime unknowns in the proposal remain open.
