"""Verify the witnessed EP-dispatch count mismatch through the current gap code.

Only pure functions are AST-extracted. A one-member communication family is a
scoped fixture; its count is read from the actual spec default and checked
against the actual V4 registration's omission. This is not a simulator replay.
"""
import ast
import csv
import json
from pathlib import Path
from types import SimpleNamespace

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
NAME='dsv4_ep_dispatch_allgather'
tree=ast.parse((REPO/'frontier/operators/spec.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CommOperatorSpec')
field=next(n for n in cls.body if isinstance(n,ast.AnnAssign) and n.target.id=='kernel_count')
default=ast.literal_eval(field.value)
family=ast.parse((REPO/'frontier/operators/families.py').read_text())
registration=next(n for n in ast.walk(family) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='CommOperatorSpec' and any(k.arg=='name' and isinstance(k.value,ast.Name) and k.value.id=='name' for k in n.keywords) and any(k.arg=='sp_rows_builder' and isinstance(k.value,ast.Name) and k.value.id=='_ep_sp_rows' for k in n.keywords) and any(k.arg=='num_devices_builder' and isinstance(k.value,ast.Name) and k.value.id=='_moe_ep_devices' for k in n.keywords))
assert not any(k.arg=='kernel_count' for k in registration.keywords)
rows=list(csv.DictReader(open(REPO/'data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/kernel_gap.csv')))
gaps={r['execution_mode']:float(r['kernel_gap_us'])/1000 for r in rows}
comm=SimpleNamespace(name=NAME,kernel_count=default)
ns=dict(COMM_FAMILY=SimpleNamespace(operators=[comm]), operator_segment_class=lambda mode,name:'eager')
runtime=ast.parse((REPO/'frontier/execution_time_predictor/dsv4_runtime.py').read_text())
function=next(n for n in runtime.body if isinstance(n,ast.FunctionDef) and n.name=='_operator_gaps')
exec(compile(ast.Module(body=[function],type_ignores=[]),'dsv4_runtime.py','exec'),ns)
layer=SimpleNamespace(op_times={NAME:1},_dsv4_scope_gap_ms={},_dsv4_kernel_counts={NAME:3})
predictor=SimpleNamespace(_kernel_gap_ms=gaps)
before=ns['_operator_gaps'](predictor,layer,'NONE')[NAME]
comm.kernel_count=3
after=ns['_operator_gaps'](predictor,layer,'NONE')[NAME]
assert before==gaps['eager'] and after==3*gaps['eager']
result=dict(current_registry_kernel_count=default,witnessed_kernels_per_dispatch=3,
    layers=43,missing_gaps_per_forward=86,gap_eager_ms=gaps['eager'],
    direct_device_delta_ms=43*(after-before), before_per_layer_ms=before,after_per_layer_ms=after,
    expected_effect_limit='Both P-step NONE forwards gain 0.046836 ms in direct device pricing. P2-64 FIFO and decode READ phase can change TTFT discontinuously; TTFT mean is not estimated without an authorized replay.')
json.dump(result,open(OUT/'gap_verify.json','w'),indent=2)
print(json.dumps(result,indent=2))
