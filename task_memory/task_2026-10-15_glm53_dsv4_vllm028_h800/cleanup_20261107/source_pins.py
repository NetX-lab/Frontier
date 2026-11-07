#!/usr/bin/env python3
"""Read pinned native/harness revisions without invoking or changing Git."""
import json,sys
from pathlib import Path
from safety import R,now
TARGETS={'/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/p2-dp-startup-coordination':'a06a72a4326cba02234df6cfbdc21b78d3a15d26','/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness':'f33306d40443cfc0c44842d6ee044f6703c736e7'}
def main():
 name=sys.argv[1]
 if name=='source_pins_before.json' and (R/'deletion_receipt.json').exists():raise RuntimeError('Cannot take a before pin snapshot after deletion started')
 rows={}
 for root,wanted in TARGETS.items():
  marker=Path(root)/'.git';g=Path(marker.read_text().split(': ',1)[1].strip()) if marker.is_file() else marker
  if not g.is_absolute():g=(marker.parent/g).resolve()
  common=(g/(g/'commondir').read_text().strip()).resolve() if (g/'commondir').exists() else g
  head=(g/'HEAD').read_text().strip();actual=head
  if head.startswith('ref: '):
   ref=head[5:];actual=None
   for p in [g/ref,common/ref]:
    if p.exists():actual=p.read_text().strip();break
   if actual is None and (common/'packed-refs').exists():
    for l in (common/'packed-refs').read_text().splitlines():
     if l.endswith(' '+ref):actual=l.split()[0];break
  rows[root]={'expected':wanted,'observed':actual,'matches':actual==wanted}
 (R/name).write_text(json.dumps({'observed_utc':now(),'pins':rows},indent=2)+'\n')
 print(name,json.dumps(rows))
if __name__=='__main__':main()
