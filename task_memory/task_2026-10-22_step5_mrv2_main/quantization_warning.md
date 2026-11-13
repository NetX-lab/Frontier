# D23 mixed-precision warning investigation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record source audit and independently executed factor-injection control. |

## Finding

The observed BF16-to-FP8 approximation warning is emitted during generic
profiling metadata registration. Its0.5 factor does not adjust this D23 typed
Step execution path. Keep the original warning and precision labels; no source
behavior or timing correction is needed for this finding.

Read-only investigator: `ses_e9f9f9d64ffeFbf7PTFJbCwBCR`,
task `D23-QUANTIZATION-WARNING`. Primary independently verified the factor
consumer locations and ran the control below on merged source `8e1d0a1`.

## Source and observed evidence

- Actual04 offline `run.log:2018-2023` contains six registration warnings for
  `attn_pre_proj`, `attn_post_proj`, `moe_grouped_gemm`, before the unrelated
  root-ledger KeyError. No completed D cost publication exists in this run.
- `quantization_manager.py:783-813,846-851` stores approximation0.5 and
  speedup2.0 for the mismatched generic metadata, then emits the warning.
- All production calls to `adjust_compute_time` are in
  `sklearn_execution_time_predictor.py:8573-8614`, the legacy dense layer path.
  The Step MoE/PDD factory chooses the disaggregation predictor and its
  MoE/dense-layer builders instead. Those builders do not call that adjustment.
- Step attention publishes typed `step5_layer_e2e`, with source quantization
  identity retained; legacy model projection slots are zero. Typed routed FFN
  resolves precision from Step layer metadata and uses the corresponding exact
  model key. Generic registration metadata does not supply those durations.
- The generic factor can affect the legacy scalar dense path. This conclusion
  is scoped to the supported D23 Step MoE/PDD graph, not all Frontier models.

## Primary control: PASS

The probe reads the exact retained kernel-only **compute** tables:24 linear
rows and12 MoE rows. The earlier investigator's184-row description referred to
the separate attention table and is corrected here. Both compute hashes match
actual04; see [precision_factor_receipt.json](precision_factor_receipt.json).

| Typed layer | Source/baseline duration | With0.5 factors | Count | Precision |
| --- | ---: | ---: | ---: | --- |
| 3 | 122.9ms | 122.9ms | 7 | FP8 |
| 88 | 184.35ms | 184.35ms | 7 | BF16 |

Positive control: the existing legacy adjustment maps10ms to5ms with the same
registered-factor state. Both typed results remain exact lookups, equal to the
source CSV means/counts/precision, and labelled `cpu_fixture/KERNEL_ONLY`.
This injects the observed factor dictionaries; it does not retest warning
registration or alter production configuration. It does not establish native
accuracy or complete the blocked actual CLI.

Exact primary command, cwd=`/data/ycfeng/Frontier/.worktrees/step5-preview`:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY' > /data/ycfeng/tmp/d23_precision_parent01.log 2>&1
import csv, hashlib, json, platform, subprocess
from pathlib import Path
from types import SimpleNamespace
from frontier.config.model_config import BaseModelConfig
from frontier.config.quantization_manager import QuantizationManager
from frontier.config.global_vars import set_quantization_manager
from frontier.moe_ep_workload import build_contiguous_expert_ownership, generate_moe_routing_ratios, materialize_layer_ep_workload
from frontier.types import ClusterType
from tests.unit.test_step5_measured_ffn import _config, _replica
from tests.unit.test_step5_measured_ffn_replay import _predictor
from tests.unit.test_step5_measured_training import _manager
fixture=Path('/data/ycfeng/tmp/d23_main_fixture02/fixture_compute_kernel_only')
output=Path('/data/ycfeng/tmp/d23_precision_parent01')
output.mkdir(exist_ok=False)
model=BaseModelConfig.create_from_name('stepfun-ai/Step-5-Preview')
replica=_replica(model); replica.moe_expert_parallel_size=16
manager=_manager(output/'cache')
manager._cluster_configs={ClusterType.MONOLITHIC:SimpleNamespace(replica_config=replica)}
manager._train_ffn_models_for_cluster(ClusterType.MONOLITHIC,replica,_config(),str(fixture/'linear_op.csv'),str(fixture/'moe.csv'),True,set())
predictor=_predictor(model,manager)
predictor._replica_config.moe_expert_parallel_size=predictor._moe_ep_size=16
qm=QuantizationManager(); set_quantization_manager(qm)
assert not qm._operation_speedup_factors and not qm._operation_approximation_factors
rows=list(csv.DictReader((fixture/'moe.csv').open()))
def measure():
 result={}
 for layer in (3,88):
  w=materialize_layer_ep_workload(routing_ratios=generate_moe_routing_ratios(total_expert_num=352,distribution_type='balanced',seed=42,layer_id=layer),target_replica_id=0,global_layer_id=layer,routing_token_count=6145,router_topk=8,total_expert_num=352,moe_expert_parallel_size=16,expert_to_ep=build_contiguous_expert_ownership(352,16))
  with predictor._step5_measured_layer(layer):
   duration=predictor._get_grouped_gemm_time(w.lane(w.participant_ep_ids[0]),batch=SimpleNamespace(get_effective_total_tokens_rounded=lambda _:6145))
   cost=predictor._step5_ffn_costs['moe_grouped_gemm']
  source=[r for r in rows if int(float(r['num_tokens']))==6145 and int(r['moe_layer_ids'].split('-')[0])<=layer<=int(r['moe_layer_ids'].split('-')[-1])]
  assert source and cost['exact_lookup'] and cost['evidence_kind']=='cpu_fixture'
  assert all(duration==float(r['time_stats.moe_grouped_gemm.mean']) and cost['kernel_count']==float(r['time_stats.moe_grouped_gemm.kernel_count']) and cost['precision']==r['moe_routed_precision'].upper() for r in source)
  result[str(layer)]={k:cost[k] for k in ('duration_ms','kernel_count','precision','exact_lookup','evidence_kind','measurement_type')}
 return result
baseline=measure()
ops=('attn_pre_proj','attn_post_proj','moe_grouped_gemm')
qm._operation_approximation_factors.update(dict.fromkeys(ops,0.5))
qm._operation_speedup_factors.update(dict.fromkeys(ops,2.0))
control=qm.adjust_compute_time('moe_grouped_gemm',10.0)
assert control==5.0
after=measure()
assert after==baseline
result={'status':'PASS','scope':'CPU typed grouped-GEMM lookup with injected observed factor state; not a native measurement or warning-registration test','python':platform.python_version(),'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'fixture_hashes':{str(fixture/n):hashlib.sha256((fixture/n).read_bytes()).hexdigest() for n in ('linear_op.csv','moe.csv')},'baseline':baseline,'after_factors':after,'approximation_factors':qm._operation_approximation_factors,'speedup_factors':qm._operation_speedup_factors,'legacy_control_10ms':control}
(output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
```

Python3.10.6; exit0. A rerun must choose a fresh output directory and log path.
Original primary receipt/log remain under `/data/ycfeng/tmp/d23_precision_parent01*`.
