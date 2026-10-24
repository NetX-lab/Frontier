"""Bind the unchanged retained strict77 auditor to the final ledger source."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('retained_auditor', '/data/ycfeng/tmp/step5_workflow_fidelity_20261022/audit_evidence.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.ROOT = ROOT/'strict77_audit'
audit.BASELINE = Path('/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/frozen-matrix/candidate')
audit.CANDIDATE = ROOT/'strict77_output/candidate'
audit.REPO = Path('/data/ycfeng/Frontier/.worktrees/d29-ledger-final-1f6b4314')
audit.REVISION = '1461573de388b72450189bc0c1b88b4ac32491f0'
audit.BASELINE_REVISION = '376983d2ecc441d1e0a8cb8b943e65a2dcb217b5'
code = audit.main()
from tests.e2e.refactor_fidelity.run_matrix import _package_versions
baseline = json.loads((audit.BASELINE/'manifest.json').read_bytes())
candidate = json.loads((audit.CANDIDATE/'manifest.json').read_bytes())
actual = _package_versions('/data/ycfeng/envs/frontier-py310/bin/python')
versions = dict(baseline=baseline['package_versions'], candidate=candidate['package_versions'], actual=actual)
versions['equal'] = versions['baseline'] == versions['candidate'] == actual
(audit.ROOT/'environment_comparison.json').write_text(json.dumps(versions, indent=2)+'\n')
print(json.dumps(versions, indent=2))
raise SystemExit(code or int(not versions['equal']))
