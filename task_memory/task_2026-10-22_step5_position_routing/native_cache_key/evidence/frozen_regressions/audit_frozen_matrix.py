"""Bind the unchanged retained strict77 auditor to the clean native-key run."""
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('retained_auditor','/data/ycfeng/tmp/step5_workflow_fidelity_20261022/audit_evidence.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.ROOT = ROOT/'frozen-matrix'
audit.BASELINE = Path('/data/ycfeng/tmp/d29_workflow_fidelity_20261024_final07/candidate')
audit.CANDIDATE = audit.ROOT/'candidate'
audit.REPO = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-native-cache-key')
candidate = json.loads((audit.CANDIDATE/'manifest.json').read_bytes())
baseline = json.loads((audit.BASELINE/'manifest.json').read_bytes())
audit.REVISION = subprocess.check_output(['git','-C',str(audit.REPO),'rev-parse','376983d2^{commit}'],text=True,timeout=10).strip()
audit.BASELINE_REVISION = 'dbe19f531ff68d9edafc480418e8927c40eb1c60'
code = audit.main()
from tests.e2e.refactor_fidelity.run_matrix import _package_versions
actual = _package_versions('/data/ycfeng/envs/frontier-py310/bin/python')
versions = dict(baseline=baseline['package_versions'],candidate=candidate['package_versions'],actual=actual)
versions['equal'] = versions['baseline'] == versions['candidate'] == actual
(audit.ROOT/'environment_comparison.json').write_text(json.dumps(versions,indent=2)+'\n')
print(json.dumps(versions,indent=2))
raise SystemExit(code or int(not versions['equal']))
