# W128-GRAPH-PAYLOAD-DATA commands
## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded local-only audit, executable, complete-script archive, deterministic replay and input digests. |

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Local executable: `/data/ycfeng/envs/oneshot-opd-py312/bin/python`, Python3.12.13, Clang22.1.3; standard library only, no conda activation/install/GPU imports. `-B` prevents bytecode writes.
Executed creation command; `--verify` exclusively creates the artifact and refuses to overwrite an existing file:
```bash
/data/ycfeng/envs/oneshot-opd-py312/bin/python -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_graph_payload_data_audit.py --verify
```
Read-only rerun; it starts two independent fresh processes, compares complete outputs, and prints their SHA256:
```bash
/data/ycfeng/envs/oneshot-opd-py312/bin/python -B -c 'import subprocess,hashlib,sys; p="/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_graph_payload_data_audit.py"; runs=[subprocess.run([sys.executable,"-B",p],check=True,capture_output=True).stdout for _ in range(2)]; assert runs[0]==runs[1]; print(hashlib.sha256(runs[0]).hexdigest())'
```
Complete136-line script is archived verbatim in `data_audit.json#/reproducibility/full_script`, with its absolute path and hash. It reads the full probe source as bytes but never imports it or calls any producer decision function. Its closure algorithm derives DAG reachability; kernel association uses explicit full callback IDs and launch correlation only. Trace annotation names select the prescribed launch, not kernel ownership.
Script SHA256: `34b25b87148ad404d1f1361b6c1a4e060b1f00feb2fe76c6f51261195cc8bb04`. Both canonical process outputs: `1a56bb00bd7fa8bde329f2c9e9933e97fc237b2d70232659edc1021281235905`.
`data_audit.json` SHA256: `3b7c57a32d090b4494a4e35cc81e3173974bde117159a82e6a3f92b40d9c100c`. Each of31 inputs is bound by exact absolute path, byte count and SHA256 under `/input_bindings`; immutable package closure is8 files plus manifest. Clock values are excluded from stage output, not used for identity; fixed trace timestamps remain byte-identical.
Primary/package/report source SHA256: `a9d6a24e0aff13ef61088d75b5f693b599f42e8c50c2702342fbb84d3754d1a0`; manifest: `ec816706edeb0930756ad7077684a2872b93c7728ae8851264e3e2592409c146`. Source revision1aa1e4a5 and merge4add627d are receipt-bound, not independently established through Git. Future reruns also rehash maintained task/instruction inputs, so later edits correctly change the overall digest.
Final local guard reads the three deliverables, compares all four raw files against their copy receipt, checks31 PASS results, six negative rejections, identical process digests, full-script equality, script<=180lines and combined Markdown<=130lines. No platform/SSH/network/Docker/Git/admission/native/core/collector operation was performed.
Exact executed guard (before this command was archived):
```bash
/data/ycfeng/envs/oneshot-opd-py312/bin/python -B -c 'import hashlib,json;from pathlib import Path; g=Path("task_memory/task_2026-10-23_v4_remaining_measurements/profile01/graph_probe/results"); a=g/"data_audit.json"; j=json.loads(a.read_text()); md={p.name:len(p.read_text().splitlines()) for p in g.glob("*.md")}; s=Path(j["reproducibility"]["script_path"]); raw=Path("/data/ycfeng/tmp/glm53_dsv4/graph_payload_results_tl_k7ef9/artifacts"); receipt=json.loads((raw.parent/"receipt.json").read_text()); assert set(p.name for p in g.iterdir())=={"data_audit.json","command.md","test_report_2026-10-24_data.md"}; assert sum(md.values())<=130 and len(s.read_text().splitlines())<=180; assert j["reproducibility"]["full_script"]==s.read_text(); assert len(j["checks"])==31 and all(v=="PASS" for v in j["checks"].values()); assert len(j["negative_U4_fixtures"])==6 and all(v=="PASS_REJECTED" for v in j["negative_U4_fixtures"].values()); assert len(set(j["reproducibility"]["result_sha256"]))==1; assert all(hashlib.sha256((raw/f["path"]).read_bytes()).hexdigest()==f["sha256"] and (raw/f["path"]).stat().st_size==f["bytes"] for f in receipt["files"]); assert hashlib.sha256(a.read_bytes()).hexdigest()=="3b7c57a32d090b4494a4e35cc81e3173974bde117159a82e6a3f92b40d9c100c"; print(json.dumps({"status":"PASS","checks":31,"raw_files_unchanged":4,"markdown_lines":md,"combined_markdown_lines":sum(md.values()),"script_lines":len(s.read_text().splitlines()),"data_audit_sha256":hashlib.sha256(a.read_bytes()).hexdigest()}))'
```
Observed guard: PASS; initial Markdown89lines before archiving this command, final94lines; script136lines, four raw files unchanged, data audit digest unchanged.
