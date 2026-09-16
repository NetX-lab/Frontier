## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Inspected W06 path callers, reproduced baseline N03, and independently verified 23 path-precedence cases against the developing implementation. |

# W06 independent path review and regression evidence

## Scope and ownership

- Target Component/Phase: W06, canonical profiling input paths and measurement-family selection.
- Reviewer Agent Identity: `/root/w03_contract_review`.
- Inspected Artifacts: `measurement_input_paths.py`; manager public/private path APIs; sklearn predictor `_initialize_file_paths`, `_get_input_files`, and event-family selector; standalone training CLI and `AttentionTrainer`/`BaseTrainer` constructors.
- Remediation/Verification Code Actions Taken: added only `tests/unit/test_measurement_path_precedence.py` and this report. Root implemented production changes. No commits by this reviewer.
- Environment: `/usr/bin/python`, Python 3.12.3; no conda activation. CPU-only execution; no GPU, worker, or remote commands.

## Confirmed baseline causes and callers

At baseline commit `41777755`, `SklearnExecutionTimePredictor._initialize_file_paths()` handles a nonempty supplied dictionary by assigning absent eager/network/PP fields to empty strings. It then tests only whether `_compute_input_file_device_event` is empty. If empty, it resolves the DEVICE_EVENT family from config and assigns **all three** compute/attention/MoE fields. Consequently, an explicit attention or MoE value can be discarded solely because compute was omitted.

`ExecutionTimePredictionModelManager.get_training_file_paths()` independently repeats linear alias selection, filename-suffix derivation, template substitution, kernel-only paths, and PP-specific outputs. Its sibling `_resolve_measurement_input_files_for_config()` already delegates to the canonical resolver. This is a concrete duplicated algorithm, not proof that all preexisting DEVICE_EVENT outputs were incorrect.

The two public tuple boundaries intentionally have different order:

- Predictor: `(compute, attention, moe, all_reduce, send_recv, cpu_overhead)`.
- Manager: `(compute, attention, all_reduce, send_recv, cpu_overhead, moe)`.

The manager public dictionary carries 17 fields, including four PP inputs and kernel-only CPU overhead. A six-field-only replacement would lose supported inputs. Tests retain these distinctions.

Baseline canonical kernel-only CPU selection uses `configured or eager`; therefore explicit empty configuration is replaced. Root clarified the W06 contract: an explicit empty string stays empty, while absent/`None` optional configuration inherits eager. During implementation verification, the first root revision handled empty correctly but left `None` as empty; one independent test exposed this. Root corrected it before the final 23-case pass.

Both manager and predictor baseline event-family selectors catch invalid/missing lookup metadata and select CUDA. Their selectors also classify any non-ROCm supplied platform as CUDA. Tests distinguish validated `a100`/CUDA and `mi355x`/ROCm metadata from explicitly unknown SKU or platform values; the latter must raise.

Relevant source callers:

1. Predictor constructors call `_initialize_file_paths()` with the optional training dictionary; `_get_input_files()` supplies direct config resolution.
2. `ExecutionTimePredictionModelManager.get_training_context()` obtains `file_paths` through the public `get_training_file_paths()` API.
3. The random-forest predictor is a `__new__` factory selecting the concrete sklearn class. Directly allocating its factory with `object.__new__` does not create a predictor. The tests use the existing measurement-selector suite's concrete sklearn probe pattern, while keeping **real** `RandomForrestExecutionTimePredictorConfig` and validated `ReplicaConfig` instances.
4. Standalone `frontier.training.cli.train_attention()` accepts explicit `layer_dataset_path` and `compute_dataset_path`, checks their existence, and passes them to `create_attention_trainer_from_model_config()`. `AttentionTrainer` retains those paths and `BaseTrainer` retains the explicit measurement family. No independent template/derivation algorithm exists at this CLI boundary. The tests verify canonical resolved paths survive actual CLI argument parsing and trainer handoff unchanged, with only trainer fitting replaced by a capture stub.

## Regression inventory

`tests/unit/test_measurement_path_precedence.py` has 23 cases covering:

- Missing DEVICE_EVENT compute with explicitly supplied attention/MoE paths, through canonical and historical dictionary aliases.
- CUDA_EVENT, DEVICE_EVENT, and KERNEL_ONLY agreement across canonical resolution, manager public dictionary, and both historical tuple orders.
- Partial dictionaries preserve configured eager/network/CPU/PP fields; complete dictionaries preserve explicitly overridden PP paths.
- Empty compute/attention/MoE configs do not invent DEVICE_EVENT filenames.
- Legacy `mlp_input_file` selection, extensionless suffix derivation, and device/model/network substitution.
- Kernel-only CPU `None`, explicit empty, and configured path behavior.
- Explicit empty DEVICE_EVENT overrides and primary-key precedence over historical aliases, including an empty primary value.
- Unknown explicit device SKU and unknown explicit metadata platform fail in both manager and predictor.
- Valid CUDA and ROCm device metadata select their own event family.
- Three-family standalone attention CLI handoff preserves resolved compute/attention paths and the requested measurement type.

## Execution and observed evidence

Exact final test command from the worktree root:

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR python -m pytest tests/unit/test_measurement_path_precedence.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w06-path-precedence
```

Observed final result: **23 passed in 3.09 seconds**. `git diff --check -- tests/unit/test_measurement_path_precedence.py` also passed.

The first attempt had a test-fixture construction error: allocating the random-forest factory directly produced `AttributeError` for sklearn methods. This was corrected in the test fixture, and it is not classified as a production defect or RED baseline. The next meaningful run against developing production code produced **17 passed, 1 failed**, specifically the kernel-only CPU `None` case described above. Additional family/alias checks and the root correction yielded the final passing result.

The new test suite was first executed after root had begun production changes. Therefore this report does **not** claim a whole-suite RED run against unchanged baseline. N03 was separately reproduced by compiling the exact old method AST from commit `41777755` and invoking it on the real config fixture with the current canonical default resolver. The current resolver's default paths are unchanged for this case.

Exact baseline reproduction command:

```bash
PYTHONPATH=$PWD python - <<'PY'
import ast
import runpy
import subprocess
from frontier.types import MeasurementType

namespace = runpy.run_path('tests/unit/test_measurement_path_precedence.py')
config, replica, predictor, manager = namespace['paths'].__wrapped__()
source = subprocess.check_output([
    'git', 'show',
    '41777755:frontier/execution_time_predictor/sklearn_execution_time_predictor.py',
], text=True)
tree = ast.parse(source)
method = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name == '_initialize_file_paths')
module = ast.Module(body=[method], type_ignores=[])
scope = {'Dict': dict, 'MeasurementType': MeasurementType}
exec(compile(ast.fix_missing_locations(module),
             'baseline_41777755_initialize_file_paths', 'exec'), scope)
overrides = {
    'attention_device_event_input_file': 'overrides/attention_device_event.csv',
    'moe_device_event_input_file': 'overrides/moe_device_event.csv',
}
scope['_initialize_file_paths'](predictor, overrides)
print(predictor._attention_input_file_device_event)
print(predictor._moe_input_file_device_event)
assert predictor._attention_input_file_device_event != overrides['attention_device_event_input_file']
assert predictor._moe_input_file_device_event != overrides['moe_device_event_input_file']
PY
```

Observed overwritten paths:

```text
./data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention_device_event.csv
./data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/moe_device_event.csv
```

Supplied values were `overrides/attention_device_event.csv` and `overrides/moe_device_event.csv`; both were lost. This reproduces N03 directly without mutating production files or inventing a mock resolver.

## Limits and remaining work

These tests verify CPU path/configuration behavior and CLI handoff. They do not train models, inspect profiling CSV metadata, validate native accelerator execution, or establish performance. Other W06 integration gates and W09 trainer-contract work remain with root. No unresolved failures remain in the 23-case owned suite.
