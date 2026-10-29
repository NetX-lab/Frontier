"""Reuse the retained evidence auditor for the unchanged strict77 gate."""
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('retained_auditor', '/data/ycfeng/tmp/step5_workflow_fidelity_20261022/audit_evidence.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.ROOT = ROOT / 'strict77_final04'
audit.BASELINE = Path('/data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate')
audit.CANDIDATE = audit.ROOT / 'candidate'
audit.REPO = Path('/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05')
audit.REVISION = 'e41abc073500d61f224eda4fca1123e74e40544d'
audit.BASELINE_REVISION = '713e6a81708e87b86cf91fd89f3b1f954bdcbead'
exit_code = audit.main()
from tests.e2e.refactor_fidelity.run_matrix import _package_versions
baseline_versions = json.loads((audit.BASELINE/'manifest.json').read_text())['package_versions']
candidate_versions = json.loads((audit.CANDIDATE/'manifest.json').read_text())['package_versions']
actual_versions = _package_versions('/data/ycfeng/envs/frontier-py310/bin/python')
version_check = {'baseline':baseline_versions, 'candidate':candidate_versions, 'actual':actual_versions,
                 'equal':baseline_versions == candidate_versions == actual_versions}
(audit.ROOT/'environment_comparison.json').write_text(json.dumps(version_check,indent=2)+'\n')
print(json.dumps(version_check,indent=2))
raise SystemExit(exit_code or int(not version_check['equal']))
