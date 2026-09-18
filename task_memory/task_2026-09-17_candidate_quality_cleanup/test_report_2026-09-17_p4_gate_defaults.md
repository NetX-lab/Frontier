## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Centralized four GDN gate defaults and verified raw/resolved values, error wrapping and layer identities against unchanged pre-refactor source. |

# P4 GDN gate default ownership

## Result and scope

Completed, ready for main review/commit; no sidecar commit. Baseline started at `b88fcd90d29fa46f69acd95f3192aac64ac5b2b7`; main advanced to `540f23150d12670e268bd0ef74b9af30a00bd88a` during this bounded work. `git diff b88fcd90..HEAD --` for the three owned production files is empty: their committed pre-refactor source remained unchanged throughout characterization. After evidence uses that source plus this patch.

Owned files:

- `frontier/attention/gdn/config.py`: use the existing `GatedDeltaNetConfig.output_gate_type` class default in the generic shape resolver.
- `frontier/config/model_config.py`: reuse that default in the runtime dataclass and HF loader.
- `frontier/profiling/common/model_config.py`: reuse it in the profiling constructor.
- `tests/unit/test_gdn_gate_defaults.py`: dedicated 43-case characterization/regression coverage.
- This report only; main's inventory/refactoring records and other workers' files were preserved.

| Module | Problem | Root cause | Why unnecessarily complex | Refactor | Reused existing abstraction | Verification |
| --- | --- | --- | --- | --- | --- | --- |
| GDN shape resolver and runtime/profiling config | Four consumers repeat the literal default `silu`. | The default is already owned by the frozen GDN dataclass but adapters redeclare it. | A default-policy change could drift across constructor, loader and resolver paths. | Replace exactly four default expressions with the existing dataclass class attribute; leave conversions and validation untouched. | `GatedDeltaNetConfig.output_gate_type`, already imported by both model config modules. | 31 existing + 43 new pre-change PASS; 74 combined after PASS; 54 complete output snapshots byte-identical. |

No new constant, registry, resolver, import or module dependency. A source search confirms the only remaining GDN gate default literal is its authoritative declaration. Literal `silu` in alias normalization and numerical activation selection is an explicit operation identity, not a duplicated default.

## Preserved behavior: concrete before/after outputs

All entries below are observed in both snapshots, not inferred from process exit:

| Input/path | Raw field | Resolved result / rejection |
| --- | --- | --- |
| Omitted gate, runtime/profiling/HF | `"silu"` | GDN gate `"silu"` |
| Explicit `"swish"` | `"swish"` | GDN gate `"silu"` |
| Explicit `" SWISH "` | `" SWISH "` | GDN gate `"silu"` |
| Explicit `"sigmoid"` | `"sigmoid"` | GDN gate `"sigmoid"` |
| Explicit null, hybrid constructors/HF loaders | Construction rejected | Profile structural `ValueError`; cause is `GDN output_gate_type must be silu/swish or sigmoid, got 'None'` |
| Explicit empty string, hybrid constructors/HF loaders | Construction rejected | Same profile structural error; cause ends in `got ''` |
| Explicit null, non-GDN direct runtime constructor | `None` | No GDN shape; all layers remain full attention |
| Explicit null, non-GDN profiling constructor or either HF loader | `"None"` | No GDN shape; all layers remain full attention |
| Complete external shape, null gate | N/A | Direct resolver gate `ValueError`, without profile wrapping |
| No shape fields, null gate | N/A | `resolve_gdn_shape` returns `None` |
| Partial shape fields | N/A | `GDN shape configuration is incomplete` |

Retain genuine missing-field fallback for the external shape resolver. Retain `str()` conversion, explicit-null rejection, alias normalization, lazy topology ownership and constructor validation order. In particular, do not substitute `value or default` or normalize the raw field early.

Physical-layer assertions remain: the small hybrid constructor has 3 GDN + 1 full-attention layers; the existing Qwen HF fixture has 69 GDN + 23 full-attention layers. Snapshots compare every layer's global ID/family/variant, not just counts.

## Verification and initial characterization correction

Environment: `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, existing virtual environment (no conda environment activated). Working directory:

```text
/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn
```

| Check | Result | Evidence |
| --- | --- | --- |
| Existing baseline, before production edits | **31 passed in 7.79s** | `/data/ycfeng/tmp/quality-p4-gate-existing-before.log` |
| Initial new characterization | **8 failed, 35 passed in 2.95s** | `/data/ycfeng/tmp/quality-p4-gate-new-before.log` |
| Corrected characterization, still before production edits | **43 passed in 2.97s** | `/data/ycfeng/tmp/quality-p4-gate-new-before-corrected.log` |
| Combined after regression | **74 passed in 7.91s** | `/data/ycfeng/tmp/quality-p4-gate-after.log` |
| Exact output comparison | **PASS**, 259,793 bytes per file | `/data/ycfeng/tmp/quality-p4-gate-before.json`, `/data/ycfeng/tmp/quality-p4-gate-after.json` |
| Whitespace check | **PASS** | `git diff --check` |

RCA of initial failures: the new test incorrectly expected the low-level gate error directly at construction. Existing `frontier/model_architectures.py:398`, `StructuralRequirement.validate`, catches the predicate's `ValueError` and raises the profile message **from** that error. Both runtime and profiling constructors invoke this existing structural requirement. The new tests now assert the outer profile error and exact underlying cause. No production edits had been made when all 43 corrected tests passed; no existing expectation or approved output was changed. This is a characterization mistake, not a candidate bug or semantic decision.

The exact comparison includes **54 cases**: 24 direct constructor outcomes, 24 HF loader outcomes (hybrid and non-GDN), and 6 generic resolver outcomes. All **10 error outcomes** preserve error text, cause and construction-versus-topology stage. Accepted cases preserve raw values, normalized shape dataclasses and complete ordered layer identities.

### Focused commands

The common environment prefix was:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python
```

Each invocation below used that prefix, `set -o pipefail`, and stdout piped through the stated `tee` command:

```bash
# Existing baseline
-m pytest tests/unit/test_gdn_semantic_core.py tests/unit/test_mla_model_config_contracts.py -q -p no:cacheprovider --tb=short --basetemp=/data/ycfeng/tmp/quality-p4-gate-existing-before-tmp | tee /data/ycfeng/tmp/quality-p4-gate-existing-before.log
# Initial characterization; retained failure evidence
-m pytest tests/unit/test_gdn_gate_defaults.py -q -p no:cacheprovider --tb=short --basetemp=/data/ycfeng/tmp/quality-p4-gate-new-before-tmp | tee /data/ycfeng/tmp/quality-p4-gate-new-before.log
# Corrected pre-change characterization
-m pytest tests/unit/test_gdn_gate_defaults.py -q -p no:cacheprovider --tb=short --basetemp=/data/ycfeng/tmp/quality-p4-gate-new-before-corrected-tmp | tee /data/ycfeng/tmp/quality-p4-gate-new-before-corrected.log
# After cleanup
-m pytest tests/unit/test_gdn_gate_defaults.py tests/unit/test_gdn_semantic_core.py tests/unit/test_mla_model_config_contracts.py -q -p no:cacheprovider --tb=short --basetemp=/data/ycfeng/tmp/quality-p4-gate-after-tmp | tee /data/ycfeng/tmp/quality-p4-gate-after.log
```

### Exact snapshot command

The following body was run before production edits and after them. Invocation was `env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python - <<'PY'`, redirecting stdout to the corresponding before/after JSON path. It uses the same dedicated fixture data, real constructors and public model loaders. Exception capture is diagnostic output only, not production fallback.

```python
import importlib.util
import itertools
import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
spec = importlib.util.spec_from_file_location('gate_tests', 'tests/unit/test_gdn_gate_defaults.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
logging.disable(logging.CRITICAL)
rows = []
def observe(factory):
    stage = 'construction'
    try:
        config = factory()
        stage = 'topology'
        shape = config.get_gdn_config()
        return {'raw': config.gdn_output_gate_type, 'shape': asdict(shape) if shape else None,
                'layers': [asdict(layer) for layer in config.get_layer_attention_specs()]}
    except ValueError as exc:
        return {'stage': stage, 'error': str(exc), 'cause': str(exc.__cause__)}
for config_type, hybrid, case in itertools.product((t.BaseModelConfig, t.ModelConfig), (False, True), t.GATE_CASES):
    overrides, raw, normalized = case.values
    values = t._model_values(hybrid) | overrides
    if config_type is t.ModelConfig:
        values['name'] = 'gate-contract'
    rows.append({'path': 'constructor', 'type': config_type.__name__, 'hybrid': hybrid, 'case': case.id,
                 'output': observe(lambda: config_type(**values))})
source = json.loads(Path('data/config/models/Qwen3.8-2.4T-A95B-Quark-MXFP4.json').read_text())
root = Path(tempfile.mkdtemp(prefix='quality-p4-gate-snapshot-', dir='/data/ycfeng/tmp'))
directory = root / 'data/config/models'
directory.mkdir(parents=True)
os.chdir(root)
for config_type, hybrid, case in itertools.product((t.BaseModelConfig, t.ModelConfig), (False, True), t.GATE_CASES):
    overrides, raw, normalized = case.values
    values = dict(source)
    if not hybrid:
        values.pop('model_architecture_profile')
        values.update(model_type='qwen3_next', architectures=['Qwen3NextForCausalLM'])
    if overrides:
        values['output_gate_type'] = overrides['gdn_output_gate_type']
    (directory / 'gate-contract.json').write_text(json.dumps(values))
    loader = t.BaseModelConfig.create_from_name if config_type is t.BaseModelConfig else t.ModelConfig.from_model_name
    rows.append({'path': 'hf', 'type': config_type.__name__, 'hybrid': hybrid, 'case': case.id,
                 'output': observe(lambda: loader('gate-contract'))})
for case in t.GATE_CASES:
    overrides, raw, normalized = case.values
    try:
        output = asdict(t.resolve_gdn_shape(t.SimpleNamespace(**(t._model_values(True) | overrides))))
    except ValueError as exc:
        output = {'error': str(exc), 'cause': str(exc.__cause__)}
    rows.append({'path': 'resolver', 'case': case.id, 'output': output})
print(json.dumps(rows, indent=2, sort_keys=True))
```

```bash
cmp /data/ycfeng/tmp/quality-p4-gate-before.json /data/ycfeng/tmp/quality-p4-gate-after.json
git diff --check
```

These are CPU configuration/schema contracts; no native GPU or timing result is claimed. No further optional cleanup or testing is required for this substep. Pending sidecar work: none. New semantic blockers: none. Main owns commit and integrated source freeze.
