import hashlib, json, os, resource, subprocess, sys
from pathlib import Path
s=Path(__file__).resolve().parent
r=Path("/data/ycfeng/Frontier/.worktrees/b29-ingress-delays")
base=Path("/data/ycfeng/Frontier/.worktrees/b29-base")
resource.setrlimit(resource.RLIMIT_AS,(31457280*1024,31457280*1024))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",WANDB_DISABLED="true",VIDUR_DISABLE_WANDB="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TMPDIR=str(s/"tmp"),FRONTIER_TMP_ROOT=str(s/"tmp"),PYTHONPATH=f"{base}:{base}/tests/unit:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages")
candidate={}
for path in (s/"tmp/focused_r3").glob("test_explicit_zero_delays_pres*/explicit/facts.json"):
    facts=json.loads(path.read_text())
    arch="co-location" if facts["transfer_start"] is None else "pd-disaggregation"
    candidate[arch]=Path(facts["metrics_dir"])
assert len(candidate)==2,candidate
results={}
for architecture,directory in candidate.items():
    runroot=s/f"base_zero_audit_{architecture}"
    command=["timeout","120",sys.executable,str(r/"tests/unit/test_serving_path_delays.py"),str(runroot),architecture,"{}"]
    with (s/f"base_zero_audit_{architecture}.log").open("w") as output:
        result=subprocess.run(command,cwd=s,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=130)
    assert result.returncode==0,(architecture,result.returncode)
    facts=json.loads((runroot/"facts.json").read_text())
    files={}
    for name in ("request_metrics.csv","system_metrics.json"):
        data=(Path(facts["metrics_dir"])/name).read_bytes()
        assert data==(directory/name).read_bytes(),(architecture,name)
        files[name]=dict(status="PASS",sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    results[architecture]=dict(command=command,cwd=str(s),env={name:env[name] for name in ("PYTHONPATH","TMPDIR","FRONTIER_TMP_ROOT","PYTHONDONTWRITEBYTECODE","WANDB_DISABLED","VIDUR_DISABLE_WANDB","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},memory_limit_kib=31457280,files=files,candidate_metrics_dir=str(directory))
(s/"zero_base_identity_audit.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps(results,indent=2))
