import csv, collections, statistics
def load(p):
    with open(p) as f: return list(csv.DictReader(f))
orig=load('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv')
os_=load('original_shared/attention_graph_replay_kernel_only.csv')
ms=load('median3_shared/attention_graph_replay_kernel_only.csv')
ma=load('median3_all/attention_graph_replay_kernel_only.csv')
k=lambda r:(r['case_id'],r['attention_variant'],r['plans_tile_schedule'])
O={k(r):r for r in orig}
changed=collections.Counter(); ratio=collections.defaultdict(list)
for a,b in zip(os_,ms):
    assert k(a)==k(b) and a==O[k(a)]
    for c in a:
        if a[c]!=b[c]:
            assert c.startswith('time_stats.') and c.endswith('.median'), c
            changed[c.split('.')[1]]+=1; ratio[c.split('.')[1]].append(float(b[c])/float(a[c]))
shared={k(r) for r in os_}
msd={k(r):r for r in ms}
assert len(orig)==len(ma)
for r,s in zip(orig,ma):
    assert k(r)==k(s)
    assert (r==s) if k(r) not in shared else (msd[k(s)]==s)
print('changed cells', sum(changed.values()), 'shared cases', len({r['case_id'] for r in os_}), 'all rows', len(ma))
for s in ['dsv4_q_projection','dsv4_index_fused_query_prepare','dsv4_compressor_fused_compress_store','dsv4_index_compressor_fused_compress_store']:
    v=sorted(ratio[s]); n=len(v)
    print(s, changed[s], 'median3/original p10 %.3f p50 %.3f p90 %.3f'%(v[n//10],v[n//2],v[9*n//10]))
