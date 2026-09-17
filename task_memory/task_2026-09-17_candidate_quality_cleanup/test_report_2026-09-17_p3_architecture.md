## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Completed S6 exact raw-type policy reuse, import-cycle RCA/correction, 181 focused tests and 333 direct pre/post predicate/guard checks. |
| 2026-09-17 | Follow-up caller audit established required norm/model_type fields; removed both defensive getattr calls; 183 focused tests and 330 supported-input pre/post checks passed. |

# P3 S6 — Architecture-owned native profiling type policy

## Scope, decisions and status

- Baseline: `8e67fc25967cf5aaa01a87298a40b9b08ee13fc4`; the three owned production files were clean. Original comparison remains main `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2` through candidate `c288a19f59bec09529ee18d782fa57218da2c781`.
- Inspected the actual frozen diffs, existing `large_module_review.md` S6 proposal, registry declarations/resolution, both native predicate consumers, constructor normalization and existing focused tests.
- Honored the P2 source freeze: baseline inspection/tests only, then production edits after main explicitly released the freeze. No changes to model binding, S4/S5 config/schema/predictor, J1/J2 orchestration or S7 routing.
- Plan: admission baseline -> metadata/query/consumer edit -> focused raw-type/guard regression and pre/post matrix -> handoff without commit. Current status: implementation and required-field follow-up verification complete; ready for main integration. Latest inspected handoff HEAD: `dcf66a4cc73074a30f4c0146aa6380ab10bf877c`; no commit or edits to other owners' files.
- Codebase-design keeps policy in the existing owner and the norm/platform/native API checks in their existing consumers. Planning-with-files uses this authorized report; shared records remain main-owned.

## Refactoring record

| Module | Problem | Root cause / unnecessary complexity | Refactor | Reused abstraction | Verification |
| --- | --- | --- | --- | --- | --- |
| `frontier/model_architectures.py` | Native policy was not declared beside architecture metadata. | Runtime profile resolution is broader than exact native type admission, so reusing resolution itself would change behavior. | Add two immutable empty-default tuple fields and two concrete exact-membership queries; annotate existing generic/Qwen declarations only. Append fields to preserve existing positional field order. | `ModelArchitectureProfile`, `ModelArchitectureRegistry.iter_profiles()` | Existing registry/serialization tests plus direct exact-type matrix and entry extension test. |
| `frontier/profiling/linear_op/linear_op_impl.py` | Consumer repeats Qwen identity membership. | Norm choice combines local norm kind with centrally owned model type policy. | Keep the norm guard and pass the unchanged `model_type` value to the registry query. | Existing raw-type profile metadata | Norm/profile/alias matrix and unchanged output-metadata suite. |
| `frontier/profiling/moe/moe_vllm_kernel.py` | MXFP4 adapter hard-codes architecture eligibility. | Eligibility is policy; ROCm/API availability remains native adapter responsibility. | Query exact MXFP4 raw-type membership; keep guard order, exception classes/messages and all platform/API checks. | Same existing architecture registry | Type/platform/API matrix and existing MXFP4/layout tests. |

No second registry, capability dispatcher, normalization, profile recursion, cache, GPU dependency in registry metadata or experimental dependency was added. `qwen3_next` remains associated with the existing generic declaration, not reclassified as GDN. Explicit `generic` does not suppress exact Qwen native eligibility; explicit Qwen profile or architecture alias does not grant eligibility to another raw type.

Retained optionality: the model-type value is genuinely optional in the profiling constructor, so None/empty reject naturally; the field itself is always present. Norm and platform/API guards remain unchanged. Existing upstream `ModelConfig` lowercasing is untouched; "raw" here means the value supplied to the existing predicate/gate, without any new normalization. Historical missing-field checks characterized unsupported inputs and are not a production preservation requirement; see the follow-up below.

## Baseline execution

Environment: CPU, Python 3.12.3, `/data/ycfeng/tmp/quality-review-env/bin/python` (uv venv with same-interpreter system-site-packages), no conda/GPU execution. CWD: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_model_architecture_registry.py tests/unit/test_moe_mxfp4_increment10.py tests/unit/test_linear_op_profiling_output_metadata.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p3-architecture-before-20260917 --tb=short
```

Observed **80 passed in 10.22s** before production changes. Native vLLM is absent; its normal import warning was visible. CPU checks control platform/API signals and do not exercise native kernels.

Direct predicate/gate baseline: ten raw values (None, empty, other, exact Qwen3-next, exact Qwen3.5, each uppercase/padded, and an architecture-name string). Gemma crosses three profile values, two architecture-alias configurations and four norm values: **240 checks, 12 admitted**, plus three missing-field controls rejecting. MXFP4 crosses three platforms and three API states: **90 checks: 1 admitted, 81 model-type rejections, 6 platform rejections, 2 API rejections**. Exact error text/type and platform-detector call order were asserted.

Discovery-only error: initial search used `frontier/profiling/linear/linear_op_impl.py`; actual path is `frontier/profiling/linear_op/linear_op_impl.py`. No file/test failure or workaround resulted.

## Import-cycle RCA and correction

- Initial implementation passed **181 focused tests in 10.28s**, and direct comparison passed **240 norm + 3 missing-field + 90 MXFP4 checks**. A fresh registry-first import then failed with `ValueError: model architecture profile 'step3_text' references unknown operator family`, caused by `ImportError: cannot import name 'get_model_architecture_profile' from partially initialized module 'frontier.model_architectures'`.
- A fresh native-kernel import reproduced the same failure. Root cause: the new top-level registry import preceded existing profiling imports, entering `model_architectures -> profile construction -> operators.families -> config.parallel_semantics/config.__init__ -> config.model_config -> partially initialized model_architectures`.
- Running the pinned registry source in a fresh interpreter reproduced the existing registry-first cycle; pinned native-kernel initialization previously reached its normal optional-vLLM handling. Thus registry-first behavior is pre-existing debt, but the early kernel dependency was a candidate-only regression in the initial S6 patch.
- Correction: function-local import in `validate_mxfp4_runtime` resolves the canonical registry at validation time, after normal module initialization. No lazy object state, fallback, import exception suppression or new registry is introduced. New tests enter through the profiling consumer before inspecting registry metadata. This is a verified production import-order constraint, not support for incomplete mocks. Final fresh-import/focused evidence follows below.

## Final verification and exact commands

Criteria: no new candidate test failure; exact raw membership and norm guard preserved; exact MXFP4 exception class/text and type -> platform -> API precedence; profile extension needs one existing declaration; consumers never resolve a runtime profile for native admission.

Final focused run, with the new regression file first so unrelated test collection cannot initialize its dependencies for it:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_native_profiling_model_type_policy.py tests/unit/test_model_architecture_registry.py tests/unit/test_moe_mxfp4_increment10.py tests/unit/test_linear_op_profiling_output_metadata.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p3-architecture-final-20260917 --tb=short
```

**PASS: 181 passed in 10.13s.** The new file is 85 lines: ten type-policy cases (240 norm/profile/alias assertions), 90 platform/API cases, and one declarative extension test. This was the completed initial S6 run; the subsequent required-field cleanup and its final verification are recorded below.

Fresh native-module initialization, without entering its executable self-test block:

```bash
git show 8e67fc25967cf5aaa01a87298a40b9b08ee13fc4:frontier/profiling/moe/moe_vllm_kernel.py | env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/tmp/quality-review-env/bin/python -c 'import sys; exec(compile(sys.stdin.read(), "baseline_moe_vllm_kernel.py", "exec"), {"__name__": "baseline_native"}); print("Baseline native module initialization: PASS")'
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/tmp/quality-review-env/bin/python -c 'import frontier.profiling.moe.moe_vllm_kernel; print("Current native module initialization: PASS")'
```

**PASS:** both printed their PASS line after the expected optional-vLLM warning. The earlier standalone registry-first smoke failed; it is not claimed as passing or fixed.

Direct pre/post comparison was rerun after the import correction. This executes the two exact baseline function bodies from Git against the same current dependencies and inputs; AST selection isolates these gates and does not stand in for native-kernel execution. All admitted/rejected results, exception classes/text and detector call arguments are compared exactly. The command below now excludes the three missing-field inputs following the required-field audit; the 240 valid-field combinations remain unchanged:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python - <<'PY'
import ast
import subprocess
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch
from frontier.profiling.linear_op import linear_op_impl as linear
from frontier.profiling.moe import moe_vllm_kernel as kernel
from tests.unit.test_native_profiling_model_type_policy import TYPE_CASES

revision = '8e67fc25967cf5aaa01a87298a40b9b08ee13fc4'
def baseline_function(module, path, name):
    source = subprocess.check_output(['git', 'show', f'{revision}:{path}'], text=True)
    node = next(item for item in ast.parse(source).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = vars(module).copy()
    exec(compile(ast.Module(body=[node], type_ignores=[]), f'{revision}:{path}', 'exec'), namespace)
    return namespace[name]

before_norm = baseline_function(linear, 'frontier/profiling/linear_op/linear_op_impl.py', '_uses_gemma_rms_norm')
before_gate = baseline_function(kernel, 'frontier/profiling/moe/moe_vllm_kernel.py', 'validate_mxfp4_runtime')
norm_count = 0
for (raw, gemma, _), profile, architectures, norm in product(
    TYPE_CASES, (None, 'generic', 'qwen3_5_moe'),
    ((), ('Qwen3_5MoeForCausalLM',)), (None, 'rms_norm', 'layer_norm', 'RMS_NORM'),
):
    config = SimpleNamespace(model_type=raw, norm=norm,
                             model_architecture_profile=profile, architectures=architectures)
    assert before_norm(config) == linear._uses_gemma_rms_norm(config) == (gemma and norm == 'rms_norm')
    norm_count += 1
print(f'Gemma pre/post: PASS {norm_count} exact admission matches')

def outcome(function, raw):
    try:
        function(model_type=raw)
    except (ValueError, NotImplementedError) as error:
        return (type(error).__name__, str(error))
    return ('accepted', '')

gate_count = 0
for (raw, _, _), platform, api in product(TYPE_CASES, ('rocm', 'cuda', 'cpu'),
                                         (None, '0.10.x', 'functional_fused_experts')):
    detector = Mock(return_value=platform)
    before_gate.__globals__.update(accelerator_platform=detector, VLLM_API_VERSION=api)
    expected = outcome(before_gate, raw)
    expected_calls = list(detector.call_args_list)
    detector.reset_mock()
    with patch.object(kernel, 'accelerator_platform', detector), patch.object(kernel, 'VLLM_API_VERSION', api):
        assert outcome(kernel.validate_mxfp4_runtime, raw) == expected
        assert detector.call_args_list == expected_calls
    gate_count += 1
print(f'MXFP4 pre/post: PASS {gate_count} exact admission/error class/error text/platform-call matches')
PY
```

Observed after the required-field cleanup, rerunning the command above:

```text
Gemma pre/post: PASS 240 exact admission matches
MXFP4 pre/post: PASS 90 exact admission/error class/error text/platform-call matches
```

## Handoff

Changed paths only:

1. `frontier/model_architectures.py`
2. `frontier/profiling/linear_op/linear_op_impl.py`
3. `frontier/profiling/moe/moe_vllm_kernel.py`
4. `tests/unit/test_native_profiling_model_type_policy.py` (new)
5. `task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_p3_architecture.md` (new)

Production diff: **26 insertions / 4 deletions**, three files. Scoped `git diff --check` passed without output. No new registry mechanism, no S7 implementation, no model-binding edits, no commit. Main's S5/J1/J2 edits were preserved.

Owned pending work and semantic-choice blockers: none. Main next steps: review/integrate this patch, then run the broader phase/final gates with other workers' changes. Native GPU numerical parity is not established by these CPU gate tests. The pre-existing registry-first import cycle remains a documented out-of-scope constraint; supported native consumer import passes.

## Follow-up — Required ModelConfig fields

- Main requested direct-access review after S6 handoff. At HEAD `69eb7be8883ff8f400035d1cd050f9e9728fd337`, `frontier/profiling/common/model_config.py:113` unconditionally assigns `self.norm = str(norm)` and line 132 assigns `self.model_type = str(model_type).lower() if model_type is not None else None`. Thus optionality belongs to the model-type value, not attribute existence.
- All four production call sites are in `linear_op_impl.py`: `_build_untimed_norm`, `CausalSelfAttention.__init__` QK norm, and `GPTBlock.__init__` input/post-attention norm. They receive the same normally constructed profiling `ModelConfig`; `_build_untimed_norm` and GPTBlock already read `config.norm` directly. MTP and `non_kv_cache_overhead.runtime_estimator` pass that constructed config through unchanged.
- The two test call sites are in the new native-policy regression; both lightweight doubles explicitly supply `norm` and `model_type`. No fixture change or compatibility branch is required.
- Removed only the two `getattr(..., None)` calls in `_uses_gemma_rms_norm`. None-valued model types still reject, exact membership and short-circuit norm guard remain unchanged, and MXFP4 guards/error strings/import-cycle correction are untouched.
- The three historical missing-field controls were unsupported partial objects, not evidence that production fields were optional. Their former silent-False behavior is deliberately not retained under the user's constructor-invariant requirement. No test fixtures changed, no P4 edits.

Final focused command (same Python/environment as above; includes the two embedded-MTP plan tests to cover another production config consumer):

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_native_profiling_model_type_policy.py tests/unit/test_model_architecture_registry.py tests/unit/test_moe_mxfp4_increment10.py tests/unit/test_linear_op_profiling_output_metadata.py tests/unit/test_profiling_plan_target_embedded_mtp.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p3-architecture-required-fields-20260917 --tb=short
```

**PASS: 183 passed in 10.32s.** The direct baseline comparison above also passed again: **240 exact Gemma admission matches and 90 exact MXFP4 admission/error class/error text/platform-call matches**. Expected optional-vLLM warning remained visible. No production/test edits followed these checks. The actual import-cycle RCA and the necessary function-local import are unchanged. This follow-up modified only `linear_op_impl.py` and this report; the other S6 files retain the already-reviewed implementation. No owned pending work or new semantic-choice blocker remains; await main integration before the separately scoped P4 work.
