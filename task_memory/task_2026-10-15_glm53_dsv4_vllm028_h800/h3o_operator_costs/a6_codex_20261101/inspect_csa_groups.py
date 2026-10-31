from pathlib import Path
import collections,csv,hashlib,json,resource,statistics
resource.setrlimit(resource.RLIMIT_AS,(30*1024**3,30*1024**3))
task=Path("/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800")
root=task/"h3o_operator_costs/a6_codex_20261101"
table=Path("/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv")
scopes=["dsv4_q_projection","dsv4_index_fused_query_prepare","dsv4_compressor_fused_compress_store","dsv4_index_compressor_fused_compress_store"]
with table.open() as stream:
    rows=list(csv.DictReader(stream))
groups=collections.defaultdict(dict)
for row in rows:
    if row["attention_variant"]=="csa":
        assert row["plans_tile_schedule"] not in groups[row["case_id"]]
        groups[row["case_id"]][row["plans_tile_schedule"]]=row
assert len(groups)==1421 and all(set(pair)=={"True","False"} for pair in groups.values())
result={"table_path":str(table),"table_sha256":hashlib.sha256(table.read_bytes()).hexdigest(),"paired_workloads":len(groups),"scope_comparisons":{}}
for scope in scopes:
    differences=[]; unequal_counts=0
    for case,pair in groups.items():
        first,reuse=pair["True"],pair["False"]
        col=f"time_stats.{scope}.median"
        if not first.get(col) or not reuse.get(col):continue
        a,b=float(first[col]),float(reuse[col]); assert min(a,b)>0
        differences.append(abs(a-b)/min(a,b))
        unequal_counts+=first[f"time_stats.{scope}.kernel_count"]!=reuse[f"time_stats.{scope}.kernel_count"]
    result["scope_comparisons"][scope]={"pairs":len(differences),"median_absolute_difference_over_min":statistics.median(differences),"above_10_percent":sum(x>.1 for x in differences),"unequal_kernel_counts":unequal_counts}
raw=Path("/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw")
cases={"g1-e0-0263-full","g1-e1-0729-first_tail"}
selected=[]; inputs=[]
for name in ["dsv4_raw_e0.jsonl","dsv4_raw_e1.jsonl"]:
    source=raw/name; digest=hashlib.sha256()
    with source.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not any(case.encode() in line for case in cases):continue
            record=json.loads(line)
            if record["kind"]=="call" and record.get("scope") in scopes:
                layer=record["global_layer_id"]
                ids=set(json.loads(groups[record["case_id"]]["True"]["global_layer_ids"]))|set(json.loads(groups[record["case_id"]]["False"]["global_layer_ids"]))
                if layer in ids:selected.append(record)
    inputs.append({"path":str(source),"sha256":digest.hexdigest()})
checks=[]
for case in sorted(cases):
    for planner,row in groups[case].items():
        layer_ids=json.loads(row["global_layer_ids"])
        for scope in scopes:
            calls=[x for x in selected if x["case_id"]==case and x["scope"]==scope and x["global_layer_id"] in layer_ids]
            assert len(calls)==int(row[f"time_stats.{scope}.count"])
            observed=statistics.median(x["time_ms"] for x in calls)
            expected=float(row[f"time_stats.{scope}.median"])
            assert abs(observed-expected)<1e-12
            checks.append({"case_id":case,"plans_tile_schedule":planner,"layer_ids":layer_ids,"num_steps_sampled":int(row["num_steps_sampled"]),"scope":scope,"raw_call_count":len(calls),"measured_us":1000*expected,"reconstructed_us":1000*observed})
result.update(raw_inputs=inputs,raw_calls=selected,reconstruction=checks,status="PASS_SELECTED_RAW_MEDIANS_RECONSTRUCT")
(root/"csa_measurement_groups.json").write_text(json.dumps(result,indent=2)+"\n")
print(result["status"],len(checks),"scope groups",len(selected),"raw calls")
