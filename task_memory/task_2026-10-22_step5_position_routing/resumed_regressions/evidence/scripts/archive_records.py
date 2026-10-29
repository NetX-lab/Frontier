"""Retain exact new evidence without overwriting existing files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = Path('/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_regressions/evidence')
parser = argparse.ArgumentParser()
parser.add_argument('scope', choices=['short','strict77','unit','scripts'])
scope = parser.parse_args().scope
sources = []
if scope=='short':
    for suite in ['supplement101','metadata196','catalogue53']:
        sources += [(ROOT/suite/name, Path(suite)/name) for name in ['execution.json','results.xml','command1.log']]
    sources.append((ROOT/'metadata196/business_artifact_receipt.json',Path('metadata196/business_artifact_receipt.json')))
    for source in sorted((ROOT/'metadata196/basetemp').glob('test_precision_csv_requires_ac[0-9]/op_precision_metadata.csv')):
        sources.append((source,Path('metadata196/csv')/source.parent.name/source.name))
elif scope=='strict77':
    sources += [(ROOT/'strict77'/name,Path('strict77')/name) for name in ['execution.json','command1.log','command2.log']]
    for name in ['comparison.json','postrun_receipt.json','final04_receipt.json','environment_comparison.json','audit.log']:
        sources.append((ROOT/'strict77_final04'/name,Path('strict77')/name))
    for name in ['manifest.json','results.jsonl']:
        sources.append((ROOT/'strict77_final04/candidate'/name,Path('strict77/candidate')/name))
elif scope=='unit':
    sources += [(ROOT/'unit'/name,Path('unit')/name) for name in ['execution.json','results.xml','command1.log']]
    sources.append((ROOT/'unit_final04_comparison.json',Path('unit_final04_comparison.json')))
else:
    sources += [(ROOT/name,Path('scripts')/name) for name in ['run_suite.py','audit_strict77.py','run_strict_audit.py','compare_unit.py','archive_records.py']]
inventory = []
for source,relative in sources:
    target = DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    content = source.read_bytes()
    with target.open('xb') as stream:
        stream.write(content)
    inventory.append({'source':str(source),'retained':str(target),'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
with (DEST/f'{scope}_archive_manifest.json').open('x') as stream:
    stream.write(json.dumps(inventory,indent=2)+'\n')
print(json.dumps(inventory,indent=2))
