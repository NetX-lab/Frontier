# Actual13 descriptor CPU check command body

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Archive the executed local-only Python command body. |

Invocation is recorded in the test report. No platform/cloud/native GPU operation occurs.

```python
import json
from pathlib import Path
from frontier.profiling.step5.inputs import MRv2DummyDescriptor, Step5ProfileInput
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.step5 import load_step5_attention_rows, STEP5_FEATURE_COLUMNS, step5_row_features
from test_step5_mrv2_inputs import producer_row
from test_step5_measured_training import _replica, _scheduler, _manager, _train, _write
from frontier.types import MeasurementType
source = Path('/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-21_step5_priority_runs/canonical_g3/native_judgment')
groups = json.loads((source/'mrv2_descriptors.json').read_text())['dummy_descriptors_tp0']
model = BaseModelConfig.create_from_name('stepfun-ai/Step-5-Preview')
root = Path('/data/ycfeng/tmp/d23_inputs_observed13')
root.mkdir(exist_ok=True)
results, rows_by_method = [], {}
for index, group in enumerate(groups):
    raw = group['descriptor']
    descriptor = MRv2DummyDescriptor(raw['num_tokens'], raw['num_reqs'], raw['cg_mode'].lower(), raw['max_query_len'], group['evidence'][0], raw['uniform_token_count'])
    assert list(descriptor.query_lens) == group['query_lengths']
    assert group['metadata']['layer_count'] == group['metadata']['layers_with_metadata'] == 92
    assert group['metadata']['block_table_shapes'] == [[group['num_reqs'],448]]*4
    assert group['metadata']['slot_mapping_shape'] == [4,group['num_tokens']]
    for layer in (0,3):
        cell = Step5ProfileInput.from_mrv2_dummy(layer, descriptor, block_table_columns=448)
        row = producer_row(model, cell, duration=.5, count=17)
        assert json.loads(row['dummy_descriptor'])['max_query_len'] == raw['max_query_len']
        method = MeasurementType.KERNEL_ONLY if descriptor.graph_mode == 'none' else MeasurementType.CUDA_EVENT
        rows_by_method.setdefault(method,[]).append(row)
    results.append({'rank':group['rank'],'descriptor':raw,'queries':group['query_lengths'],'status':'PASS','evidence':group['evidence'][0]})
for method, rows in rows_by_method.items():
    folder = root/method.value
    folder.mkdir(exist_ok=True)
    path = _write(folder,rows,method)
    loaded = load_step5_attention_rows(path,_replica(model),_scheduler(),method)
    cold = _train(_manager(folder/'cache',method),model,path)
    warm_manager = _manager(folder/'cache',method)
    warm_manager._create_estimator_and_params = lambda *_: (_ for _ in ()).throw(AssertionError('warm refit'))
    warm = _train(warm_manager,model,path)
    for row in rows:
        key = tuple(step5_row_features(row)[name] for name in STEP5_FEATURE_COLUMNS)
        assert cold['step5_layer_e2e']._frontier_exact_lookup[key] == .5
        assert cold['step5_layer_e2e_kernel_count']._frontier_exact_lookup[key] == 17
    assert warm['step5_layer_e2e']._frontier_model_hash == cold['step5_layer_e2e']._frontier_model_hash
    assert cold['step5_layer_e2e']._frontier_step5_domains['native_workload_code'] == (1.,)
report = {'actual_dummy_groups':len(results),'PASS':len(results),'producer_rows':sum(map(len,rows_by_method.values())),'full_raw_null_groups':sum(r['descriptor']['cg_mode']=='FULL' and r['descriptor']['max_query_len'] is None for r in results),'results':results,'limits':'Local descriptor parsing and CPU fixture fit/cache only; no native timing/profile/G3 admission.'}
Path('task_memory/task_2026-10-22_step5_mrv2_inputs/observed_descriptors.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
```
