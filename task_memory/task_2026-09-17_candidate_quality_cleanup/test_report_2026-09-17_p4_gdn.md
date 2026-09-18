## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Completed S1 supplied-model cleanup: 51 before / 53 after PASS; eight campaign pairs, 48 byte-identical task artifacts and 192 equal prediction dictionaries. |

# P4 S1 — GDN supplied-model and training-state contracts

## Scope and status

- Original diff: main `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2` through candidate `c288a19f59bec09529ee18d782fa57218da2c781`; both production modules were added by that candidate.
- Pre-edit P4 baseline: `de06cf9a369f3758f2eba5e0e66155bf37e55889`. Owned paths were clean. Read-only preparation respected the P3 freeze; implementation began only after explicit release.
- Ownership: `frontier/execution_time_predictor/gdn_predictor.py`, `frontier/training/gdn_trainer.py`, `tests/unit/test_gdn_training_predictor_increment8.py`, and this report. No manager/path/config/MoETrainer edits, additional fixture files, commits or remote actions.
- Plan: constructor/caller audit -> focused baseline -> direct supplied-model access and fixture repair -> focused regression and frozen-source artifact comparison -> handoff. Implementation and owned verification complete; ready for main review/integration without a commit.
- Codebase-design informed reuse of the existing model interface without a new protocol/wrapper; planning-with-files uses this assigned report instead of editing main-owned shared records.

## Refactoring record

| Module | Problem | Root cause / unnecessary complexity | Refactor | Reused abstraction | Verification |
| --- | --- | --- | --- | --- | --- |
| `gdn_predictor.py::_validate_requested_identity` | Missing supplied-model members silently reduced identity checking. | Genuine optionality of the whole model was conflated with optional existence of its methods/hidden size. | Directly call all three model methods and read `embedding_dim`; keep the absent-model and absent-GDN-shape branches. | Existing runtime/profiling model interface and `GatedDeltaNetConfig` | Focused artifact/hybrid tests; complete manifest, lookup, estimator metadata and prediction comparisons. |
| `gdn_trainer.py::__init__` | Getter discovery obscured selector precedence. | Supplied models already implement both getters. | Use direct methods only for omitted profile/quant selectors. | Existing model identity methods | Compare dataset-only, supplied-model and explicit-override training scopes. |
| `gdn_trainer.py::_load_dataset` | Dataset retained in unused `self.df`. | Local training flow already owns and returns the frame; GDN and BaseTrainer have no reader of the attribute. | Remove only initialization and assignment of GDNTrainer.df. | Existing BaseTrainer local-frame interface | One-row/multirow training, cache reuse and unchanged artifacts. MoETrainer.df has real readers and remains untouched. |
| `test_gdn_training_predictor_increment8.py` | Two duplicate partial model doubles omitted hidden size and GDN shape. | Existing reflection let manager tests bypass valid identity comparisons. | One minimal shared model double implements required methods and provides the fixture's real normalized GDN shape. Add two optional-shape/hidden-size checks. | `GatedDeltaNetConfig` | Manager load and changed-dataset rejection retain their original expected values. |

## Constructor and consumer evidence

- Runtime `BaseModelConfig` declares required `embedding_dim` at `frontier/config/model_config.py:292`, and methods `get_quant_signature`/`get_gdn_config`/`get_model_architecture_profile` at lines 480/514/587.
- Profiling `ModelConfig` unconditionally assigns `embedding_dim`; its equivalent methods are at `frontier/profiling/common/model_config.py:352`, 285 and 254. Both model types can legitimately return no GDN shape.
- The production shared-manager GDN loader supplies its bound model after a positive GDN-layer check. The GDN CLI deliberately omits the model and selects a dataset scope. These are different valid entry paths, not reasons for member reflection.
- GDN and BaseTrainer consume the DataFrame returned by `_load_dataset`; the only GDN `self.df` occurrences were the two removed writes. MoETrainer's retained frame has actual feature-selection readers.

### Manager fixture handoff

Both manager tests now use `_FixtureModelConfig` with `embedding_dim=256`; methods `get_num_gdn_layers()`, `get_name()`, `get_model_architecture_profile()`, `get_quant_signature()`, and `get_gdn_config()`. The latter returns `GatedDeltaNetConfig(conv_kernel_size=4, key_head_dim=32, value_head_dim=32, num_key_heads=2, num_value_heads=4)`, matching the CSV identity.

The existing replica double supplies `model_config`, `device="cpu"`, `attn_tensor_parallel_size=1`; predictor config supplies `gdn_input_file`; the locally constructed manager supplies `_cache_dir` and `_gdn_predictors`. These cover direct access in the inspected manager GDN path. No default-path assertion or other manager behavior was weakened.

## Retained optionality and boundaries

- Entire `model_config=None`: dataset-only training/loading remains supported.
- `get_gdn_config() is None`: omit GDN shape expectations, still check supplied profile, quantization and hidden size.
- Explicit profile/quant selectors retain precedence. Trainer does not call their getters when selectors are provided; predictor derives model expectations first and then applies explicit overrides, as before.
- Manifest/estimator boundary validation, numeric rejection, exact lookup versus extrapolation, missing-cache handling and atomic publication are unchanged. Serialized input is not a constructor-owned invariant and still requires validation.
- Identity initialization before `_load_dataset` is unchanged. This task does not redesign the trainer lifecycle.

## Execution and evidence

Environment: CPU; Python 3.12.3 at `/data/ycfeng/tmp/quality-review-env/bin/python`, uv venv with same-interpreter system-site-packages; no conda activation or native GPU execution. Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.

Baseline command, before owned production/test edits:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_training_predictor_increment8.py tests/unit/test_gdn_artifact_boundary.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p4-gdn-before-20260917 --tb=short
```

**PASS: 51 passed in 17.64s.** This includes actual CPU fit/persistence/load, corrupt-artifact rejection, cache reuse, publication interruption and synthetic hybrid runtime tests. It is not native-device fidelity evidence.

Post-change command:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_training_predictor_increment8.py tests/unit/test_gdn_artifact_boundary.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p4-gdn-after-20260917 --tb=short
```

**PASS: 53 passed in 19.19s.** The two additional cases cover supplied models with/without GDN shape and reject a hidden-size mismatch in both. No existing assertion was loosened. Discovery-only errors during preparation came from guessed nonexistent `model_trainer.py` and unmatched test-name globs; corrected to the actual CLI and discovered test paths. No production workaround or test failure resulted.

### Frozen-source artifact comparison command

The following command executes the exact pre-edit trainer/predictor modules from Git with the same installed dependencies. It trains separate baseline/current outputs, compares full manifest and six final artifact bytes, checks normalized lookup/bounds/identity and estimator class/parameters, and cross-loads both artifact sets through both predictors. It covers one-row and multirow scopes, dataset-only, real supplied runtime model, a supported absent GDN shape, and conflicting explicit-selector precedence. No comparison ignores identity fields or changes numerical tolerances.

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python - <<'PY'
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd

from frontier.attention.gdn.features import GDNBatchFeatures, GDN_FEATURE_COLUMNS
from frontier.execution_time_predictor.gdn_predictor import GDNPredictor
from frontier.training.gdn_trainer import GDNTrainer
from tests.unit.test_gdn_hybrid_e2e_increment14ab import _qwen35_fixture_config
from tests.unit.test_gdn_training_predictor_increment8 import FIXTURE, _FixtureModelConfig

revision = "de06cf9a369f3758f2eba5e0e66155bf37e55889"
def load_baseline(name, path):
    module = ModuleType(name)
    sys.modules[name] = module
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
    exec(compile(source, f"{revision}:{path}", "exec"), module.__dict__)
    return module

BeforeTrainer = load_baseline("before_gdn_trainer", "frontier/training/gdn_trainer.py").GDNTrainer
BeforePredictor = load_baseline("before_gdn_predictor", "frontier/execution_time_predictor/gdn_predictor.py").GDNPredictor
root = Path(tempfile.mkdtemp(prefix="quality-p4-gdn-artifacts-", dir="/data/ycfeng/tmp"))
original = pd.read_csv(FIXTURE, keep_default_na=False)
rows = []
for _, template in original.iterrows():
    for multiplier in (1, 2, 3, 4):
        row = template.copy()
        columns = (("batch_num_tokens", "batch_num_prefill_tokens", "max_query_len")
                   if row["batch_num_prefill_tokens"] else
                   ("batch_size", "batch_num_tokens", "batch_num_decode_tokens"))
        for column in columns:
            row[column] *= multiplier
        rows.append(row)
multirow = root / "multirow.csv"
pd.DataFrame(rows).to_csv(multirow, index=False)
queries = [GDNBatchFeatures(1, size, size, 0.0, 0, "prefill") for size in (16, 24, 512)]
queries += [GDNBatchFeatures(size, size, 1, 0.0, size, "decode") for size in (4, 6, 64)]
query_frame = pd.DataFrame([q.as_vector() for q in queries], columns=GDN_FEATURE_COLUMNS)
counts = dict(campaign_pairs=0, task_artifact_pairs=0, cross_loads=0, prediction_dicts=0)
logging.disable(logging.WARNING)
for dataset in (FIXTURE, multirow):
    for mode in ("dataset_only", "real_model", "no_gdn_shape", "explicit_override"):
        selectors = {}
        if mode == "real_model":
            selectors["model_config"] = _qwen35_fixture_config()
        elif mode != "dataset_only":
            model = _FixtureModelConfig()
            if mode == "no_gdn_shape":
                model.get_gdn_config = lambda: None
            else:
                model.get_model_architecture_profile = lambda: SimpleNamespace(profile_id="other")
                model.get_quant_signature = lambda: "other"
                selectors.update(model_architecture_profile="qwen3_5_moe", quant_signature="none")
            selectors["model_config"] = model
        outputs = []
        trained = []
        for label, Trainer in (("before", BeforeTrainer), ("after", GDNTrainer)):
            output = root / dataset.stem / mode / label
            trainer = Trainer(str(dataset), str(output), num_estimators=[2], max_depth=[2],
                              min_samples_split=[2], k_fold_cv_splits=2,
                              num_training_job_threads=1, **selectors)
            trained.append(trainer.train())
            outputs.append(output)
        assert (outputs[0] / "gdn_manifest.json").read_bytes() == (outputs[1] / "gdn_manifest.json").read_bytes()
        assert trained[0].keys() == trained[1].keys()
        for task, before in trained[0].items():
            after = trained[1][task]
            assert type(before) is type(after)
            assert before.get_params() == after.get_params()
            np.testing.assert_array_equal(before.predict(query_frame), after.predict(query_frame))
            assert (outputs[0] / f"{task}.pkl").read_bytes() == (outputs[1] / f"{task}.pkl").read_bytes()
            counts["task_artifact_pairs"] += 1
        reference = BeforePredictor.from_directory(outputs[0], dataset_path=dataset, **selectors)
        expected = [reference.predict_operator_times(query) for query in queries]
        for output in outputs:
            for Predictor in (BeforePredictor, GDNPredictor):
                loaded = Predictor.from_directory(output, dataset_path=dataset, **selectors)
                assert loaded.identity == reference.identity
                for key, before in reference._models.items():
                    after = loaded._models[key]
                    assert before.feature_names == after.feature_names
                    assert before.exact_lookup == after.exact_lookup
                    assert before.feature_bounds == after.feature_bounds
                assert [loaded.predict_operator_times(query) for query in queries] == expected
                counts["cross_loads"] += 1
                counts["prediction_dicts"] += len(queries)
        counts["campaign_pairs"] += 1
logging.disable(logging.NOTSET)
print("Artifact comparison PASS", json.dumps(counts, sort_keys=True))
print("Evidence directory:", root)
PY
```

Observed result:

```text
Artifact comparison PASS {"campaign_pairs": 8, "cross_loads": 32, "prediction_dicts": 192, "task_artifact_pairs": 48}
Evidence directory: /data/ycfeng/tmp/quality-p4-gdn-artifacts-zigqyxs4
```

**PASS:** all eight complete manifest pairs and all 48 final estimator artifact pairs are byte-identical. All 32 baseline/current cross-loads preserve the complete identity, normalized exact lookup and bounds; all 192 prediction dictionaries match exactly. Estimator classes, parameters and direct numeric arrays also match for all 48 task pairs. Generated artifacts remain in the reported temporary evidence directory. The comparison intentionally reduces repeated INFO/expected extrapolation WARNING output; exceptions and assertion failures remain fatal. The focused suite independently verifies the extrapolation warning.

## Final handoff

Owned changed paths:

1. `frontier/execution_time_predictor/gdn_predictor.py`
2. `frontier/training/gdn_trainer.py`
3. `tests/unit/test_gdn_training_predictor_increment8.py`
4. `task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_p4_gdn.md`

The production edit is confined to supplied-model access and two unused DataFrame writes. No alternate registry, model wrapper, artifact schema, timing normalization or compatibility path was introduced. Existing externally loaded artifact validation remains intact. The accurate lightweight model double now satisfies the inspected manager GDN interface; main may directly access the fields listed above without restoring reflection. No separate artifact fixture was necessary.

Owned pending work and new semantic-choice blockers: none. Main's next steps are review/integration, then the broader P4 regression with its independent manager/path/config work. No native-device numerical or performance claim is made by this CPU artifact comparison.
