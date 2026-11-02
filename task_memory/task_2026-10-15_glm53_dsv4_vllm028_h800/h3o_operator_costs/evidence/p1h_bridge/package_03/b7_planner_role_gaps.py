import pandas as pd, numpy as np
p="/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_review/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv"
df=pd.read_csv(p)
key=["padded_rows","prefill_seq_lens","prefill_kv_cache_sizes","decode_kv_cache_sizes"]
df[key[1:]]=df[key[1:]].fillna("")
for v in ["csa","hca"]:
  d=df[df.attention_variant==v]
  scopes=[c[len("gap_stats."):-len(".median")] for c in d.columns if c.startswith("gap_stats.") and c.endswith(".median")]
  for mode in ["FULL","PIECEWISE"]:
    for ph in ["decode","first_tail","dummy"]:
      dd=d[(d.graph_runtime_mode==mode)&(d.phase==ph)]
      if dd.empty: continue
      a=dd[dd.plans_tile_schedule==True].set_index(key); b=dd[dd.plans_tile_schedule==False].set_index(key)
      common=a.index.intersection(b.index)
      print(v,mode,ph,len(common))
      for s in scopes:
        g=f"gap_stats.{s}.median"; t=f"time_stats.{s}.median"
        dg=(a.loc[common,g]-b.loc[common,g]).dropna()
        if len(dg)==0: continue
        dt=((a.loc[common,t]-b.loc[common,t])/b.loc[common,t]).dropna()
        print("   ",s.replace("dsv4_",""), len(dg), "gap_diff_us med",round(dg.median()*1e3,2),"p90abs",round(np.percentile(np.abs(dg),90)*1e3,2),"gap_us",round(b.loc[common,g].median()*1e3,2),"cost_rel_med%",round(dt.median()*100,1))
