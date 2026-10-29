"""Archive selected frozen results without duplicating raw suites or caches."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASK = Path('/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/native_cache_key')
OUTPUT = TASK/'evidence/frozen_regressions'
OUTPUT.mkdir(exist_ok=False)
selected = {
    'unit_execution.json':'frozen-unit/execution.json',
    'matrix_execution.json':'frozen-matrix/execution.json',
    'unit_comparison.json':'frozen_unit_comparison.json',
    'unit_comparison.log':'frozen_unit_comparison.log',
    'matrix_comparison.json':'frozen-matrix/comparison.json',
    'matrix_compare.log':'frozen-matrix/compare.log',
    'unit_launcher.log':'frozen_unit_launcher.log',
    'matrix_launcher.log':'frozen_matrix_launcher.log',
    'host_audit_failure.json':'frozen-matrix/postrun_receipt.json',
    'host_audit_failure.log':'frozen_matrix_audit.log',
    'cache_only_mount_launch.json':'frozen-audit-02/launch.json',
    'cache_only_mount_failure.log':'frozen-audit-02/audit.log',
    'correct_namespace_launch.json':'frozen-audit-03/launch.json',
    'correct_namespace_audit.json':'frozen-audit-03/postrun_receipt.json',
    'correct_namespace_audit.log':'frozen-audit-03/audit.log',
    'environment_comparison.json':'frozen-audit-03/environment_comparison.json',
    'correct_namespace_comparison.json':'frozen-audit-03/comparison.json',
}
bindings = {}
for name, relative in selected.items():
    source = ROOT/relative
    raw = source.read_bytes()
    (OUTPUT/name).write_bytes(raw)
    bindings[name] = dict(source=str(source),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
scripts = {}
for name in ('compare_frozen_unit.py','audit_frozen_matrix.py','audit_frozen_matrix02.py',
             'audit_frozen_matrix03.py','run_frozen_audit.py','frozen_checks.py','archive_frozen.py'):
    raw = (ROOT/name).read_bytes()
    (OUTPUT/name).write_bytes(raw)
    scripts[name] = dict(source=str(ROOT/name),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
manifest = dict(files=bindings,retained_scripts=scripts,
    status='PASS_FROZEN_BEHAVIOR_PRESERVATION_ONLY',
    source='376983d2ecc441d1e0a8cb8b943e65a2dcb217b5',
    script_scope='Current retained script bytes; exact executed commands and mount adaptations are in launch/execution receipts.',
    limits='Independent clean accounting/integration/native/whole-CLI acceptance remain separate. Raw unit XML/logs and caches are referenced, not copied.')
(OUTPUT/'artifact_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name, binding in {**bindings,**scripts}.items():
    assert hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() == binding['sha256']
assert json.loads((OUTPUT/'correct_namespace_audit.json').read_bytes())['verdict'] == 'PASS'
assert json.loads((OUTPUT/'host_audit_failure.json').read_bytes())['verdict'] == 'FAIL'
print(json.dumps(dict(status=manifest['status'],file_count=len(bindings)+len(scripts),
                     bytes=sum(item['bytes'] for item in [*bindings.values(),*scripts.values()]),
                     destination=str(OUTPUT)),indent=2))
