"""Reuse the accepted verifier with relocated output and explicit test provenance."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
OWNER = Path('/data/ycfeng/tmp/d29_plan_key_reuse/wave01')
source = (OWNER / 'verify.py').read_text()
assert source.count("TESTS = ROOT.parent / '01'") == 1
source = source.replace("TESTS = ROOT.parent / '01'", "TESTS = Path('/data/ycfeng/tmp/d29_plan_key_reuse/01')")
namespace = {'__file__': str(ROOT / 'verify.py'), '__name__': '__main__'}
exec(compile(source, str(OWNER / 'verify.py'), 'exec'), namespace)
report = namespace['report']


def statuses(path):
    return {(case.attrib['classname'], case.attrib['name']):
            next((label for tag, label in [('failure', 'FAIL'), ('error', 'ERROR'), ('skipped', 'SKIP')]
                  if case.find(tag) is not None), 'PASS')
            for case in ET.parse(path).findall('.//testcase')}


xml = Path('/data/ycfeng/tmp/d29_plan_key_parent01.xml')
parent = statuses(xml)
assert parent == statuses(OWNER.parent / '01/baseline-tests.xml')
assert list(parent.values()).count('PASS') == 124 and list(parent.values()).count('SKIP') == 1
report['task_id'] = 'D29-PLAN-KEY-PARENT'
report['tests_provenance'] = 'tests contains retained owner RED/GREEN/affected/baseline; parent_tests is fresh primary execution.'
report['parent_tests'] = {'passed': 124, 'skipped': 1, 'failed': 0, 'errors': 0,
    'same_test_ids_and_statuses_as_baseline': True, 'xml': str(xml),
    'xml_sha256': hashlib.sha256(xml.read_bytes()).hexdigest()}
report['verification_adaptation'] = 'Accepted verifier unchanged except owner test path; wave output root relocated via __file__.'
report['parent_timing_limit'] = 'Baseline/candidate replay processes ran concurrently for parity; owner sequential timing remains the comparison evidence.'
(ROOT / 'parent_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
