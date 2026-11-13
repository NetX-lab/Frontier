"""Prepare isolated diagnostic copies around the unchanged accepted wave."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = Path('/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-08')
ACCEPTED = HERE.parent / 'runtime_model_view/repair_evidence'
ROOT = Path(sys.argv[1]).resolve()
ROOT.mkdir(exist_ok=False)
base = json.loads((ACCEPTED / 'freeze.json').read_text())
head = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
assert head == 'dbe19f531ff68d9edafc480418e8927c40eb1c60'
hashes = {}
for path, expected in base['sources']['candidate']['hashes'].items():
    raw = (SOURCE / path).read_bytes()
    hashes[path] = hashlib.sha256(raw).hexdigest()
    assert hashes[path] == expected, path
sources = {label: dict(root=str(SOURCE), head=head, hashes=hashes,
    submodule_pin=base['sources']['candidate']['submodule_pin']) for label in ('baseline', 'candidate')}
base.update(task_id='D29-RUNTIME-JSON-REUSE', session_id=None, sources=sources,
    source_kind='unchanged committed source with explicit candidate-only method replacements')
for field, expected_field in [('original_setup', 'setup_sha256'), ('protected_body', 'protected_body_sha256')]:
    assert hashlib.sha256(Path(base[field]).read_bytes()).hexdigest() == base[expected_field]
raw = Path(base['collective_original']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == base['collective_sha256']
paths = ['frontier/execution_time_predictor/prediction_model_registry.py',
         'frontier/execution_time_predictor/moe_routing_workload.py']
for path in paths:
    destination = ROOT / 'patches' / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((SOURCE / path).read_bytes())
driver = (ACCEPTED / 'run.py').read_text()
anchor = 'samples = []\npayloads = []'
assert driver.count(anchor) == 1
driver = driver.replace(anchor, '''from install_control import install
diagnostic = install(SOURCE, ROOT, LABEL)
view_control.update(diagnostic_control=diagnostic,
    production_method_files_verified=(LABEL == 'baseline'))
samples = []
payloads = []''')
driver = driver.replace("'task_id':'D29-NAMED-MODEL-QUERY'", "'task_id':'D29-RUNTIME-JSON-REUSE'")
for variant in ('registry', 'point', 'combined'):
    case = ROOT / variant
    case.mkdir()
    (case / 'control.json').write_text(json.dumps({'variant': variant, 'patch_root': str(ROOT / 'patches')}) + '\n')
    (case / 'freeze.json').write_text(json.dumps(base, indent=2) + '\n')
    (case / 'run.py').write_text(driver)
    (case / 'fixture_wiring.py').write_bytes((ACCEPTED / 'fixture_wiring.py').read_bytes())
    (case / 'install_control.py').write_bytes((HERE / 'install_control.py').read_bytes())
    for label in sources:
        target = case / label / 'collective-fixture/h800/collective_local.json'
        target.parent.mkdir(parents=True)
        target.write_bytes(raw)
(ROOT / 'cache').mkdir()
receipt = {'task_id': base['task_id'], 'source': sources, 'protected_body_sha256': base['protected_body_sha256'],
           'original_driver_sha256': hashlib.sha256((ACCEPTED / 'run.py').read_bytes()).hexdigest(),
           'derived_driver_sha256': hashlib.sha256(driver.encode()).hexdigest(), 'variants': ['registry', 'point', 'combined']}
(ROOT / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'source': head, 'source_bindings': len(hashes), 'root': str(ROOT),
                  'protected_body_sha256': base['protected_body_sha256']}))
