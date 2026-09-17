## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified bounded P4 S8 and typed-registry required-field cleanup with focused tests and exact before/after snapshots. |

# P4 S8 and typed registry contracts

## Result and scope

- Status: completed and ready for main to inspect/commit; this sidecar made no commit.
- Baseline source: `de06cf9a369f3758f2eba5e0e66155bf37e55889`, following the released P3 source freeze. The original reviewed additions are from `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2...c288a19f59bec09529ee18d782fa57218da2c781`.
- Production ownership: only `frontier/config/config.py` and `frontier/operators/typed_contracts.py`.
- Dedicated test: `tests/unit/test_config_owned_contracts.py`. No existing tests were edited.
- Documentation ownership: only this report. Main's manager/path files, Nash's GDN predictor/trainer/training fixture, and shared progress/refactoring records were not edited.
- GDN gate default ownership is a subsequent substep and was not changed here.

## Refactoring record

| Module | Problem | Root cause | Why unnecessarily complex | Refactor | Reused abstraction | Verification |
| --- | --- | --- | --- | --- | --- | --- |
| `config/config.py`, `ReplicaConfig.__post_init__` | Cross-node guard defaults missing `world_size` and node capacity to 1. | Reflection ignores values already established by normal construction. | Every role branch sets `world_size`; node construction precedes the guard and `BaseNodeSKUConfig` requires `num_devices_per_node`. | Direct access to both fields, preserving integer conversion and guard position. | Existing ReplicaConfig construction and node SKU contract. | 20 normally constructed hybrid replicas across five roles, two node capacities and two TP widths; five cross-node failures unchanged. |
| `config/config.py`, `SimulationConfig._validate_gdn_runtime_guards` | Nested reflection treats required speculative config/`enabled` as optional. | Local guard duplicates defensive treatment despite ReplicaConfig's default factory and earlier direct access. | Supported replicas already own a valid `SpeculativeDecodingConfig`. | Direct `replica_config.speculative_decoding_config.enabled`; retain `bool()`. | Existing speculative dataclass and runtime guard. | Nine guard-routing cases retain exact ordered calls, disabled/enabled speculative fields, optional role handling and identity deduplication. |
| `operators/typed_contracts.py`, `validate_typed_operator_metadata` | Registry iteration skips profiles without `attention_linear_ops`. | Required registry fields are handled like unknown external input. | `ModelArchitectureProfile` requires this field and accesses `.sharded_ops` during its own validation. Registry iteration returns these profiles. | Direct internal field access; update the same two operator sets in the same order. | Existing architecture registry and attention linear-op declaration. | Existing typed metadata/registry tests plus 66 full metadata outcomes, including nine unchanged rejections and both implicit/selected profile contexts. |

The existing large-module analysis in `large_module_review.md`, "Aggregate configuration", applies: keep dataclass registration, flattened CLI, role inheritance and validation ordering together for this bounded cleanup. No config split or new policy/helper was introduced.

## Retained behavior and test limits

- Keep prefix-cache capability reflection: `OrcaSchedulerConfig` does not declare `enable_prefix_caching`; VllmV1 does.
- Keep optional role enumeration and object-identity deduplication. Configured inactive roles are not interchangeable with materialized active clusters.
- Keep external `architecture_profile` validation and its exact `TypeError("architecture_profile must expose attention_linear_ops")`.
- Keep unknown/conflicting owner validation, model-config boundary handling, MTP ownership and all metadata field checks.
- Keep `int()` and `bool()` conversion, guard ordering and error messages. No constructor default, schema, scheduling or numerical change.
- New tests construct real ReplicaConfig, ClusterConfig, SimulationConfig, node configs and model configs. Guard spies call the original implementation; they do not bypass admission.
- The role-routing test changes the architecture selector and optional role references on an already constructed SimulationConfig to isolate this method's configured-role domain. It does not claim that each synthetic role mixture is a full admitted deployment. Existing PD-AF and PDD tests exercise real construction/CLI behavior.
- Snapshot tests are CPU contract evidence, not a new Simulator fidelity, native-device or performance measurement. Main owns phase integration.

## Execution and observed results

Working directory for all commands:

```text
/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn
```

Environment: `/data/ycfeng/tmp/quality-review-env/bin/python`, Python **3.12.3**, existing virtual environment (not a newly activated conda environment). All temporary outputs remain under `/data/ycfeng/tmp`.

| Check | Source | Observed result | Raw evidence |
| --- | --- | --- | --- |
| Existing focused baseline | `de06cf9a`, before production edits | **135 passed in 38.26s** | `/data/ycfeng/tmp/quality-p4-config-baseline.log` |
| New constructor/metadata tests before cleanup | Same production source, new test only | **30 passed in 5.30s** | `/data/ycfeng/tmp/quality-p4-owned-before.log` |
| Combined focused regression | Same source plus the two-file cleanup | **165 passed in 38.68s** | `/data/ycfeng/tmp/quality-p4-config-after.log` |
| Full snapshot comparison | Before vs after cleanup | **Byte-identical**, 605,937 bytes each | `/data/ycfeng/tmp/quality-p4-config-before.json`, `/data/ycfeng/tmp/quality-p4-config-after.json` |
| Patch whitespace | Current diff | **PASS**, no output | `git diff --check` |

Snapshot inventory: **20** replica cases, **9** simulation guard cases, **47** ordered guard calls in total, **66** metadata outcomes from three real model profiling plans. Metadata outcomes comprise **57** accepted dictionaries and **9** identical exceptions. Replica cases include **5** cross-node rejections with the unchanged message. Guard snapshots include full model dataclass payloads, including resolved physical-layer identities, and all supplied guard kwargs.

No test failure occurred in this substep. Initial discovery included nonexistent guessed filenames; `rg --files` identified the actual files. The shell has no `apply_patch` PATH entry, so edits used the previously verified Codex apply-patch entry point, not file-overwrite scripts.

### Focused test commands

Use this common environment prefix for each Python command:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. \
  TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 \
  WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/ycfeng/tmp/quality-review-env/bin/python
```

The existing baseline command was:

```bash
set -o pipefail
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest \
  tests/unit/test_gdn_runtime_guards.py \
  tests/unit/test_pdaf_config_contract.py \
  tests/unit/test_pd_transfer_types_and_configs.py \
  tests/unit/test_model_architecture_registry.py \
  tests/unit/test_profiling_governance_minimal_red.py::test_typed_metadata_rejects_wrong_family_tp_and_padded_width \
  tests/unit/test_profiling_governance_minimal_red.py::test_typed_metadata_requires_registry_owned_operator_and_family \
  tests/unit/test_profiling_governance_minimal_red.py::test_typed_architecture_operator_requires_profile_context \
  -q -p no:cacheprovider --tb=short \
  --basetemp=/data/ycfeng/tmp/quality-p4-config-baseline-tmp \
  | tee /data/ycfeng/tmp/quality-p4-config-baseline.log
```

The new pre-change characterization command used the same environment prefix with:

```text
-m pytest tests/unit/test_config_owned_contracts.py -q -p no:cacheprovider --tb=short --basetemp=/data/ycfeng/tmp/quality-p4-owned-before-tmp
```

Its stdout was piped through `tee /data/ycfeng/tmp/quality-p4-owned-before.log` under `set -o pipefail`. The post-change command was the complete existing-baseline command with `tests/unit/test_config_owned_contracts.py` added as its first test path, `--basetemp=/data/ycfeng/tmp/quality-p4-config-after-tmp`, and `tee /data/ycfeng/tmp/quality-p4-config-after.log`.

### Exact output characterization

The following inline Python body was run once before either production edit and once after, using the common environment prefix followed by `- <<'PY'`. Stdout was redirected respectively to the two snapshot paths above. The dedicated test functions assert the expected guard values and actual cross-node failures; the spies retain the complete observed arguments. Logging and incidental console output are suppressed before the final JSON, not filtered from the compared payload.

```python
import contextlib
import importlib.util
import io
import itertools
import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock
import pytest

spec = importlib.util.spec_from_file_location("owned_contracts", "tests/unit/test_config_owned_contracts.py")
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
logging.disable(logging.CRITICAL)
spies = []
def spy_factory(*args, **kwargs):
    spy = Mock(*args, **kwargs)
    spies.append(spy)
    return spy
tests.Mock = spy_factory
snapshot = {"replicas": [], "simulation_guards": [], "metadata": []}
def calls():
    return [
        {"model": asdict(item.args[0]), "kwargs": item.kwargs}
        for spy in spies for item in spy.call_args_list
    ]
with contextlib.redirect_stdout(io.StringIO()):
    for role, node, tp in itertools.product(
        (None, "prefill", "decode", "decode_attn", "decode_ffn"),
        (("a100_pairwise_nvlink", 4), ("a100_dgx", 8)), (4, 8)
    ):
        spies.clear()
        with pytest.MonkeyPatch.context() as patch:
            tests.test_replica_gdn_guard_uses_initialized_topology(patch, role, *node, tp)
        snapshot["replicas"].append({"input": [role, *node, tp], "calls": calls()})
    root = Path(tempfile.mkdtemp(prefix="quality-p4-config-snapshot-", dir="/data/ycfeng/tmp"))
    for index, (arch, prefix) in enumerate(itertools.product(
        ("co-location", "pd-disaggregation", "pd-af-disaggregation"), (None, False, True)
    )):
        spies.clear()
        with pytest.MonkeyPatch.context() as patch:
            tests.test_simulation_guard_keeps_optional_roles_and_scheduler_capabilities(
                patch, root / str(index), arch, prefix
            )
        snapshot["simulation_guards"].append({"input": [arch, prefix], "calls": calls()})
    tests.test_typed_metadata_retains_external_profile_validation()
    from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
    for model_name in ("llama2_7b_dense_example", "step-moe-noquant", "Qwen3.8-2.4T-A95B-Quark-MXFP4"):
        model = tests.BaseModelConfig.create_from_name(model_name)
        plan = build_profiling_plan(model_config=model, tp_size=2, attn_tp=[2], ffn_tp=[2], moe_tp=[1], is_moe=model.is_moe)
        for operator, metadata in plan["typed_operator_contracts"].items():
            for selected in (False, True):
                context = {"model_config": model} if selected else {}
                try:
                    outcome = tests.validate_typed_operator_metadata(metadata, operator_name=operator, expected_metadata=metadata, **context)
                except (ValueError, TypeError) as exc:
                    outcome = {"exception": type(exc).__name__, "message": str(exc)}
                snapshot["metadata"].append({"model": model_name, "operator": operator, "selected_profile": selected, "outcome": outcome})
print(json.dumps(snapshot, sort_keys=True, indent=2))
```

Comparison command, observed no differences:

```bash
cmp /data/ycfeng/tmp/quality-p4-config-before.json /data/ycfeng/tmp/quality-p4-config-after.json
git diff --check
```

The snapshot-only exception capture records the actual admission result; it is not production fallback logic. Both accepted metadata and error types/messages are compared without exclusions. Console/log timestamps and generated output directory names are not part of these guard/metadata contracts and never enter the snapshot.

## Handoff

1. Main can review/commit the two production files, dedicated test and this report as one bounded result.
2. Wait for main's continuation signal before the GDN default-ownership substep. Omitted/null/swish behavior and validation timing remain unchanged by this patch.

Open defects or semantic decisions in this completed substep: **none**. Pending sidecar implementation in this substep: **none**. No new native/performance claim.

## Integrated follow-up: isolate process-global test state

The first P4 integrated run exposed three subsequent GDN constructor failures because the new dense SimulationConfig cases left IS_MOE=False in the test process. Production correctly rejects a later conflicting MoE model. A two-node command reproduced the order dependency; no production regression was found.

Using the common environment above:

```text
-m pytest -q -p no:cacheprovider 'tests/unit/test_config_owned_contracts.py::test_simulation_guard_keeps_optional_roles_and_scheduler_capabilities[None-co-location]' tests/unit/test_gdn_hybrid_e2e_increment14ab.py::test_hybrid_gdn_real_simulator_cpu_e2e --tb=short
```

Before: **1 passed / 1 failed**, 6.71 s; `/data/ycfeng/tmp/quality-p4-state-leak-red.log`, exact error `IS_MOE already initialized to False, cannot change to True`. Added autouse setup/teardown using the existing test reset function only in the new config test module. Then ran both complete modules with the same flags: **36 PASS**, 13.79 s; `/data/ycfeng/tmp/quality-p4-state-leak-green.log`. All guard/error assertions remain unchanged. The original 165-test run did not detect this test-order leak; final broader evidence supersedes that limited isolation claim.
