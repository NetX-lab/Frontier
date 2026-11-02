"""Run only the allocated new unit module once and retain copied evidence."""

import copy
import hashlib
import importlib.util
import json
import platform
import sys
import time
import unittest
from pathlib import Path

OWNER = Path(__file__).resolve().parent
REPO = Path.cwd()
started = time.monotonic()
allocation = json.loads((OWNER.parent / 'primary/allocation.json').read_text())
test_path = REPO / allocation['source_paths'][1]
spec = importlib.util.spec_from_file_location(test_path.stem, test_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
expected_tests = suite.countTestCases()
with (OWNER / 'unit.log').open('x') as stream:
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
elapsed = time.monotonic() - started
evidence = copy.deepcopy(module.EVIDENCE)
serialized = json.dumps(evidence, indent=2)
consistent = all(row['expected'] == row['actual'] and row['status'] == 'PASS'
                 for row in json.loads(serialized))
status = 'PASS' if result.wasSuccessful() and consistent and result.testsRun == expected_tests else 'FAIL'
with (OWNER / 'ast_closure.json').open('x') as stream:
    json.dump(copy.deepcopy(module.AST_EVIDENCE), stream, indent=2)
    stream.write('\n')
paths = [REPO / relative for relative in allocation['source_paths']]
paths += [*module.PINS, Path(__file__), OWNER / 'source_check.py', OWNER / 'unit.log', OWNER / 'ast_closure.json']
receipt = dict(command=f'timeout 60s /usr/bin/python3 -B {Path(__file__).relative_to(REPO)}',
               cwd=str(REPO), executable=sys.executable, python=sys.version, conda='not used',
               platform=platform.platform(), elapsed_seconds=elapsed, status=status,
               expected_tests=expected_tests, tests_run=result.testsRun,
               failures=[dict(test=case.id(), detail=detail) for case, detail in result.failures],
               errors=[dict(test=case.id(), detail=detail) for case, detail in result.errors],
               skipped=[dict(test=case.id(), reason=reason) for case, reason in result.skipped],
               evidence_count=len(evidence),
               serialized_evidence_consistent=consistent, evidence=json.loads(serialized),
               source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
               owner_unit_invocations_used=1, owner_unit_invocation_ceiling=allocation['owner_unit_invocation_ceiling'],
               complete_pipeline_invocations=0, complete_trace_controls=0, affected_invocations=0,
               gpu_operations=0)
with (OWNER / 'unit_result.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps({key: receipt[key] for key in
                 ('status', 'tests_run', 'evidence_count', 'elapsed_seconds',
                  'serialized_evidence_consistent', 'owner_unit_invocations_used')}, sort_keys=True))
sys.exit(0 if status == 'PASS' else 1)
