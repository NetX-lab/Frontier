"""Prepare one source-bound query-filter control using the accepted wave."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PARENT = Path('/data/ycfeng/tmp/d29_plan_key_parent01_wave')
frozen = json.loads((PARENT / 'freeze.json').read_text())
source = frozen['sources']['candidate']
source_root = Path(source['root'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip() == source['head']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=source_root, text=True)
for name, expected in source['hashes'].items():
    assert hashlib.sha256((source_root / name).read_bytes()).hexdigest() == expected
frozen['sources'] = {arm: dict(source) for arm in ('baseline', 'candidate')}
frozen['session_id'] = 'primary-D29-RUNTIME-MODEL-VIEW'
raw = Path(frozen['collective_original']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == frozen['collective_sha256']
for arm in frozen['sources']:
    target = ROOT / arm / 'collective-fixture/h800/collective_local.json'
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
(ROOT / 'fixture_wiring.py').write_bytes((PARENT / 'fixture_wiring.py').read_bytes())
driver = (PARENT / 'run.py').read_text()
changes = {
    "binding_s = time.perf_counter() - start": "binding_s = time.perf_counter() - start\nfrom narrow_view import install\nview_control = install(setup_namespace, LABEL, ROOT)",
    "'predict_collective_plan', '_step5_position_queries', 'cache_key', 'loads', 'dumps')": "'predict_collective_plan', '_step5_position_queries', 'cache_key', 'loads', 'dumps',\n    '_models_view_for_family', '_resolve_cluster_model_contract', '_get_family_model')",
    "(8 if LABEL == 'candidate' else 200)": '8',
    "(0 if LABEL == 'candidate' else 576)": '0',
    "(0 if LABEL == 'candidate' else 192)": '0',
    "(0 if LABEL == 'candidate' else 64)": '0',
    "'task_id':'D29-PLAN-KEY-REUSE'": "'task_id':'D29-RUNTIME-MODEL-VIEW','view_control':view_control",
}
for old, new in changes.items():
    assert driver.count(old) == 1, old
    driver = driver.replace(old, new)
(ROOT / 'run.py').write_text(driver)
(ROOT / 'cache').mkdir()
(ROOT / 'freeze.json').write_text(json.dumps(frozen, indent=2) + '\n')
print(json.dumps({'source': source, 'driver_parent_sha256': hashlib.sha256((PARENT / 'run.py').read_bytes()).hexdigest(),
    'driver_control_sha256': hashlib.sha256(driver.encode()).hexdigest(),
    'protected_body_sha256': frozen['protected_body_sha256']}, indent=2))
