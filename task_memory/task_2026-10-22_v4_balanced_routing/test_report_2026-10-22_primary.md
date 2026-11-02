# Primary balanced strategy verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Added installed-image CPU evidence, complete config reconstruction and primary218-test verification on186dfef. |
| 2026-10-22 | Verified immutable6ca6777 strategy and tests independently of active integration worktrees. |

## Scope and outcome

Fixed source commit6ca6777b5eabb0685ed755db8e88deae2103d6c6: two files,211added lines. Source SHA256 `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`; test SHA256 `33bc288c8024ac8259ecbcff0d44fd891834d29b5bdee6bdf8681728893106d3`. Both hashes were asserted before pytest. Independent fixed-slice reviewer returned APPROVE and reported no P1/P2 findings; reviewer read evidence but did not rerun tests.

Observed primary result:56PASS,0failures/errors/skips,2.26s. Actual registry/SP-context source plus real Frontier counts cover assignment conservation, integer remainders, dtypes and metadata rejection. This is CPU/source acceptance, not native backend selection, GPU capture or worker activation.

## Execution

An initial attempt in the native-binding worktree stopped before collection with FileNotFoundError: the strategy commit had not yet been integrated there. No test ran, and no code behavior was diagnosed from that error. Original log: `/data/ycfeng/tmp/glm53_dsv4/root_balanced_strategy_20261022.log`.

Root created an immutable detached checkout instead:

```bash
git worktree add --detach /data/ycfeng/Frontier/.worktrees/v4-balanced-verify-6ca6777 6ca6777b5eabb0685ed755db8e88deae2103d6c6
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/v4-balanced-verify-6ca6777:/usr/local/lib/python3.10/dist-packages PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
import hashlib
from pathlib import Path
import pytest
expected={'frontier/profiling/moe/balanced_routing.py':'6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea','tests/unit/test_balanced_routing.py':'33bc288c8024ac8259ecbcff0d44fd891834d29b5bdee6bdf8681728893106d3'}
for name,digest in expected.items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
print('FIXED_6ca6777_SOURCE_AND_TEST_HASHES_PASS',flush=True)
raise SystemExit(pytest.main(['tests/unit/test_balanced_routing.py','-q','-p','no:cacheprovider','--junitxml=/data/ycfeng/tmp/glm53_dsv4/root_balanced_strategy_fixed_20261022.xml']))
PY
```

Working directory is the detached checkout. Interpreter is the existing Python3.10.6 environment; existing system Torch is supplied through PYTHONPATH, with no dependency installation. The XML/log are `/data/ycfeng/tmp/glm53_dsv4/root_balanced_strategy_fixed_20261022.{xml,log}`. Required pass condition was all56 controls passing with matching immutable hashes and no skips. Source acceptance passed; shared wiring and both runtime packages remain pending.

## Installed-image CPU registry

Exact Docker command and stdin program: `/data/ycfeng/tmp/glm53_dsv4/balanced_registry_image03_dug53fkf/command.json`; output `internal.log`, terminal `receipt.json`, assertion marker `pass.txt`. It used the unchanged accepted C26 bootstrap through its launch callback and the locked image, with no GPU device exposed. Python3.12.3/Torch2.13.0+cu130; policy SHA remains6562ec1c... . Expected6144*6=36864 assignments over256 experts, exactly144 each; observed min=max144. Native registry instance/type and dtype controls passed; CUDA initialized=false; exit0.

Attempt01 lost its failure after the110s client timeout (container exit1 at172s). Attempt02 retained the failing whole-environment equality after vLLM imports. Pinned env_override.py explicitly initializes environment on import; before attempt03 the hypothesis isolated those dependency imports before testing registration. The five dependency-modified keys were PYTORCH_NVML_BASED_CUDA_CHECK, TILELANG_CLEANUP_TEMP_FILES, TORCHINDUCTOR_CACHE_DIR, TORCHINDUCTOR_COMPILE_THREADS and TRITON_CACHE_AUTOTUNING. Registration then preserved the whole initialized environment. No policy/native/source or run-budget change was made. Retained warnings about vllm._C and absent CUDA runtime mean this check proves CPU registry/tensor behavior only, not GPU binary/backend/capture readiness.

## Real shared callers on186dfef

Primary command, working directory `/data/ycfeng/Frontier/.worktrees/v4-balanced-callers`:

```bash
env PYTHONPATH=.:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_balanced_routing.py tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_gating_constructor_boundary.py tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_shared_prediction_model_manager_mixed_layer_moe.py tests/unit/test_model_architecture_registry.py tests/unit/test_profiling_governance_minimal_red.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/glm53_dsv4/root_balanced_callers_186dfef_tests --junitxml=/data/ycfeng/tmp/glm53_dsv4/root_balanced_callers_186dfef.xml > /data/ycfeng/tmp/glm53_dsv4/root_balanced_callers_186dfef.log 2>&1
```

Expected: every required control passes, including real V4 mixed-order cold/warm/public-role lookup and PDD/PD-AF/copy propagation. Observed218PASS/0failure/0error/0skip in18.482s; Python3.10.6. The controlled test predictions2 and7 are test values, not GPU measurements. Full shared delta is12 files/432 changed lines. Independent review and final fixed-source strict77 remain separate outstanding gates at this entry.

## Actual complete Frontier input

`balanced_frontier_input/frontier.json` is the complete775-field existing flat SimulationConfig input, SHA256747a08e5da6759f8189d29b2c8419ab48eb215ac2db4a2596af4f1d15ef34897. It was generated from the real retained P0 command through create_from_cli_args().to_dict(), then reconstructed through create_flat_dataclass(SimulationConfig)(**json).reconstruct_original_dataclass(). The initial root postcondition used the wrong attribute path; the later check used cfg.cluster_config.replica_config and preserved identical input bytes.

On fixed186dfef, get_cluster_configs_for_disaggregation() produced both actual roles with balanced/balanced_topk, TP8/DP2/MoE-TP1/EP16/PP1, H800, E256/K6. Source/input/role evidence: `balanced_frontier_input/{command.json,receipt.json,roles_186dfef.json,roles_186dfef.log}`. The unchanged canonical trace SHA is2d557d7d923b71fbf62bc7e552d1fb547a8ec88c1e54660c0c76cc2de3c49ec2; source68f891b. Native base block256 was checked in native05 initialization/cache evidence; it is distinct from its64/4-token group blocks. CPU overhead is explicitly disabled. No Simulator/predictor or GPU run occurred. Timing inputs, capacity and full native graph equivalence remain unqualified, with existing closed guards preserved; this input establishes real runtime/workload binding, not a numerical prediction.
