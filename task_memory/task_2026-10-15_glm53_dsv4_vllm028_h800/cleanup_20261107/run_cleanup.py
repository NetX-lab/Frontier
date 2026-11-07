#!/usr/bin/env python3
"""Run required checks sequentially, stop on failure, preserve exact commands/logs."""
import json,os,subprocess,sys
from safety import R,df,now

def main():
 os.environ['PYTHONDONTWRITEBYTECODE']='1';logdir=R/'logs';logdir.mkdir(exist_ok=True);commands=[]
 def run(script,args=(),limit=300):
  argv=['timeout',str(limit),'python3','-u',str(R/script),*args];print('RUN',argv,flush=True);started=now()
  p=subprocess.run(argv,capture_output=True,text=True)
  label=script.removesuffix('.py')+('_'+args[0].removesuffix('.json') if args else '')
  (logdir/(label+'.stdout.log')).write_text(p.stdout);(logdir/(label+'.stderr.log')).write_text(p.stderr)
  commands.append({'argv':argv,'cwd':os.getcwd(),'environment':{'PYTHONDONTWRITEBYTECODE':'1'},'started_utc':started,'finished_utc':now(),'returncode':p.returncode,'stdout_log':str(logdir/(label+'.stdout.log')),'stderr_log':str(logdir/(label+'.stderr.log'))})
  (R/'execution_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  print(p.stdout[-2000:],flush=True)
  if p.returncode:print(p.stderr[-3000:],flush=True);raise RuntimeError('required command failed '+script)
 run('prepare_deletion.py',limit=300)
 run('protect_inputs.py',['protected_hash_before.json'],limit=600)
 b=json.loads((R/'protected_hash_before.json').read_text())
 if b['errors']:raise RuntimeError('protected before snapshot has errors; no deletion')
 (R/'df_before.json').write_text(json.dumps(df(),indent=2)+'\n')
 run('delete_entries.py',limit=300)
 run('protect_inputs.py',['protected_hash_after.json'],limit=600)
 run('verify_cleanup.py',limit=300)
 print('Completed validation, deletion and readback.',flush=True)
if __name__=='__main__':main()
