# EP Lane Ledger Scope and Allocation Verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Parent-verify original/current blobs and zero-sum 135-gross allocation; no source acceptance. |

## Result

**PASS — allocation only.** Both newly classified existing files have zero
historical cumulative gross. The new allocation funds four paths at 135 gross,
with 135 removed from unchanged CORE donors. All owner/global ceilings remain
unchanged. Accepted code usage is still source 9,566 / global 9,580.

## Environment and actual checks

Worktree: `/data/ycfeng/Frontier/.worktrees/step5-preview`; full source reference
`b02772872d0c99ceb5ebb31a8e041926f12f7676`; comparison original
`1aee0af11cd58691e64603b99eca0ac459a5b350`. Python3.10.6 at
`/data/ycfeng/envs/frontier-py310/bin/python`. Both exact parent commands below
exited 0 without fallback. This records configured limits, not a measured
cgroup/RSS peak.

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- /data/ycfeng/envs/frontier-py310/bin/python -B -c 'import csv,hashlib,json,subprocess; from pathlib import Path; root=Path("/data/ycfeng/Frontier/.worktrees/step5-preview"); baseline="1aee0af11cd58691e64603b99eca0ac459a5b350"; names=("frontier/metrics/metrics_store.py","tests/performance/step5_actual06_iteration.py"); result=[]; git=lambda *args:subprocess.check_output(["git",*args],cwd=root); current=git("rev-parse","HEAD").decode().strip(); rows=list(csv.DictReader((root/"task_memory/task_2026-10-22_step5_position_routing/native_cache_key/evidence/frozen_scope/frozen_scope_paths.tsv").open(),delimiter="\t")); totals={owner:{"cap":sum(int(r["cap"]) for r in rows if r["owner"]==owner),"actual":sum(int(r["actual"]) for r in rows if r["owner"]==owner)} for owner in sorted({r["owner"] for r in rows})};
for name in names:
 old=git("show",baseline+":"+name); new=git("show",current+":"+name); working=(root/name).read_bytes(); assert old==new==working,name; delta=git("diff","--no-renames","--numstat",baseline,current,"--",name); assert not delta,delta; result.append({"path":name,"baseline_blob":git("rev-parse",baseline+":"+name).decode().strip(),"current_blob":git("rev-parse",current+":"+name).decode().strip(),"bytes":len(working),"lines":len(working.splitlines()),"sha256":hashlib.sha256(working).hexdigest(),"historical_gross":0});
print(json.dumps({"status":"PASS_TWO_PATH_BASELINE_AND_ACCEPTED_TOTALS","baseline":baseline,"current":current,"paths":result,"allocated_paths":len(rows),"owner_totals":totals,"limits":"60s and configured MemoryMax=30G; no simulator/model/trace/CSV work."},indent=2))' > /data/ycfeng/tmp/d29_ledger_scope_basis_parent_20261024.json
```

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- /data/ycfeng/envs/frontier-py310/bin/python -B -c 'import csv,json; from pathlib import Path; root=Path("task_memory/task_2026-10-22_step5_position_routing"); amend=json.loads((root/"repairs/ep_lane_ledger_scope_amendment.json").read_text()); rows={r["path"]:dict(r) for r in csv.DictReader((root/"native_cache_key/evidence/frozen_scope/frozen_scope_paths.tsv").open(),delimiter="\t")};
for a in amend["path_ceiling_adjustments"]:
 r=rows[a["path"]]; assert int(r["cap"])==a["before"],a; assert a["before"]+a["delta"]==a["after"],a; assert a["after"]>=int(r["actual"]),a; r["cap"]=a["after"];
for r in amend["new_paths"]:
 assert r["path"] not in rows,r; rows[r["path"]]={"owner":r["owner"],"cap":r["ceiling"],"actual":r["baseline_actual"]};
totals={owner:{"paths":sum(r["owner"]==owner for r in rows.values()),"cap":sum(int(r["cap"]) for r in rows.values() if r["owner"]==owner),"actual":sum(int(r["actual"]) for r in rows.values() if r["owner"]==owner)} for owner in amend["owner_gross_ceilings"]}; assert len(rows)==amend["effective_unique_paths"]==116; assert sum(amend["path_step_ceilings"].values())==amend["repair_delta_ceiling"]==135; assert sum(r["ceiling"] for r in amend["new_paths"])+sum(r["delta"] for r in amend["path_ceiling_adjustments"])==0; assert all(totals[o]["cap"]==cap for o,cap in amend["owner_gross_ceilings"].items()); assert all(totals[o]["paths"]==n for o,n in amend["owner_paths"].items()); print(json.dumps({"status":"PASS_135_GROSS_ZERO_SUM_ALLOCATION","effective_paths":len(rows),"owner_totals":totals,"accepted_source_usage":sum(t["actual"] for t in totals.values()),"accepted_global_usage":9580,"source_allocation":sum(t["cap"] for t in totals.values()),"production_changed":False},indent=2))' > /data/ycfeng/tmp/d29_ledger_allocation_parent_20261024.json
```

The first executed Python command calls these exact Git subcommands for each of
`frontier/metrics/metrics_store.py` and `tests/performance/step5_actual06_iteration.py`:

```text
git rev-parse HEAD
git show 1aee0af11cd58691e64603b99eca0ac459a5b350:<path>
git show b02772872d0c99ceb5ebb31a8e041926f12f7676:<path>
git diff --no-renames --numstat 1aee0af11cd58691e64603b99eca0ac459a5b350 b02772872d0c99ceb5ebb31a8e041926f12f7676 -- <path>
git rev-parse 1aee0af11cd58691e64603b99eca0ac459a5b350:<path>
git rev-parse b02772872d0c99ceb5ebb31a8e041926f12f7676:<path>
```

It compares each Git blob and current working bytes and sums the original 114-row
frozen TSV. The second executed Python command loads
`repairs/ep_lane_ledger_scope_amendment.json`, validates every adjustment's original
cap/delta/final cap and actual floor, adds the two zero-history paths, and checks
the exact path/owner/source sums. It does not rerun an application or tests.

## Expected versus observed

| Check | Expected | Observed |
| --- | --- | --- |
| MetricsStore original/current/working blob | Equal; gross0 | Equal OID7a1d753b;250,545B;5,803lines;gross0 |
| Performance probe original/current/working blob | Equal; gross0 | Equal OID90afa290;33,486B;614lines;gross0 |
| Added cap + adjustment deltas | 0 | 50+85−135 =0 |
| Step caps | 135 | 38+20+65+12 =135 |
| Effective allocation | 116paths | 116paths;CORE51/PRICING46/native19 |
| Owner caps | 4404/3956/1582 | Unchanged4404/3956/1582 |
| Source cap | 9942 | 9942 |
| Actual usage | Unchanged9566source/9580global | Unchanged;no production edit |

Selected raw receipts:
- `evidence/scope_basis_parent.json`: SHA256 `31d588956c8908a3b4bd5df41dd13040f8134e89392f2e655de7953d758f0dca`.
- `evidence/allocation_parent.json`: SHA256 `0c4c22c410fcb35dd523031844f37141d4b8892dc367ac21aaa19653bbbaaad3`.

## Limits

The read-only subagent's authentication failure and timeout-only fallback are
explicitly retained in `design.md`; no applied30GiB cap is claimed for that
subagent. Primary results independently settle the small blob/allowance question.
This is not clean successor source acceptance, byte-equivalence, bounded-RSS,
runtime, cloud-readiness or full-CLI/native/G3/E2E qualification.
