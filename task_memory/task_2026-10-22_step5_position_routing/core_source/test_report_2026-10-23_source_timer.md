# Source package, callable provenance and singleton timer validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record scoped source/timer evidence and limits. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Workdir `/data/ycfeng/Frontier/.worktrees/step5-position-core`. Python3.10.6 (`frontier-py310`) and Python3.12.13 (`asyncopd-py312`). Thread caps/no bytecode/cacheprovider apply to every command; no GPU/platform/auth/cloud/SSH/Docker operation.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_moe_profiling.py -k 'position_adapter or real_device_timer or masked_expert or oracle_callable or lazy_norm' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-router-timer-py312-03
```

Observed **18 PASS/11 deselected**,8.88s (two installed SWIG deprecation warnings). Expected/observed: one real CPU source tensor pair reaches native factory contract through nested context/setter and is restored after callable failure; no legacy biased-score assignment in position adapter; real singleton DeviceTimer/torch.record_function annotations/parser under CUDA_EVENT count replay and original RECORD_FUNCTION; sample count differs from kernel_count3; event duration2.5ms remains2.5ms while separate parsed duration0.3ms belongs only to RECORD_FUNCTION; call/parser errors restore method and clear singleton state; masks remove-1 before expert map/bincount; full installed-source norm/FP8/BF16 oracle module/callable root/spec/hash proof, lazy norm factory/class/bound method verified before caching. Source modules/image references are generated CPU fixtures; oracle kernel invocation is PRICING-owned and not qualified by these tests. Only CUDA-event endpoints and tracer transport are stubbed for timer evidence; real annotation generation and parser are not mocked.

First source fixture Py3.12 run had4 FAIL/11 PASS because a previous real vLLM import left installed `_version.py` in sys.modules beside synthetic root. Verifier correctly rejected mixed roots. Fix isolated the receipt fixture's native modules before loading its declared source set, rather than weakening verifier. The final18 cases pass. A direct linear attachment probe then exposed ordinary JSON whitespace conflicting with strict identity attachment. Producer attachment now canonicalizes existing per-row facts before applying the same strict identity owner; persistent actual consumer test passes and shared mismatch still fails.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_step5_fixed_probe.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-source-identity-12
```

Observed **35 PASS**,2.92s. Passing conditions: old exact fingerprint/columns omit native_facts; bound producer-specific imports/callables differ while shared fingerprint agrees; actual linear attachment preserves canonical per-row maps and rejects shared mismatch; canonical source-input parser retains D26 tail/empty owner and same executor cell serialization, rejects physical-sum mismatch and duplicate census identity; stage copier seals explicit file artifacts and forwards declared image to existing ResourceConfig launcher (platform modules replaced with CPU fixture only); missing adapter fails before stage creation; old immutable workload/config/package/shell/dry-run behavior preserved.

Actual old stage/verifier integration (not platform submission):

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
from pathlib import Path
import json
from frontier.profiling.step5.job import stage_probe
from tests.integration.step5_profiling.verify_fixed_probe import verify
root=Path('/data/ycfeng/tmp/d29-core-actual-stage-verifier-01')
root.mkdir()
dependency=root/'dependencies';dependency.mkdir()
stage=root/'stage'
manifest=stage_probe(stage,dependency_site=dependency)
result=verify(stage)
print(json.dumps(result,indent=2))
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
PY
```

Observed **PASS**,540 sealed files;18 attention rows/4 GPU-active candidate rows,16 linear rows, both CUDA_EVENT/RECORD_FUNCTION dry-runs PASS;0 platform operations,admitted=false. Manifest SHA256 `435a941acb66a82b2a2eb4568d20a71616c858bfc08eefba2edcbc819e4a6be0`. Raw retained result `/data/ycfeng/tmp/d29-core-actual-stage-verifier-01/verification.json`. Candidate count is not GPU execution. Full new position package remains unverified because baseline MoE main rejects the new path/input and selected adapter/schema dependencies are not imported.

`bash -n tests/integration/step5_profiling/run_on_worker.sh`, `git diff --check`, AST and exact29-owned changed paths/per-file ceilings: **PASS**. Source follow-up15 paths,622 gross lines beyond943 previously committed gross. Cumulative +1470/-95=1565/4404;2839 remain. Real final-image/full-source receipt, native callable execution/parity, selected trainer/preflight/schema pairs, finite plan and new actual-main fixture/run are **UNVERIFIED/BLOCKED**, not substituted by the old dry-run or synthetic receipt.
