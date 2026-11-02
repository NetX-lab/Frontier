import json, collections
for case in ['p2_32','p2_64','p0']:
    j=json.load(open(f'{case}.json'))
    print('==',case)
    lc=j['lifecycle']
    for role in ['decode','prefill']:
        bs=set(); kv=[10**9,-1]; zmax=0; wmax=(0,0); tails=collections.Counter(); fw=0
        for k,v in lc.items():
            if not k.startswith(role): continue
            fw+=v['forwards']
            for b,f,n,kr,mr in v['b_first_tails']:
                bs.add(b); kv[0]=min(kv[0],kr[0]); kv[1]=max(kv[1],kr[1]); tails[(b,f)]+=n
            zmax=max(zmax,max(z for z,_ in v['zeroed_blocks']))
            for r,e,n in v['block_table_writes']:
                wmax=max(wmax,(r,e))
        multi=sorted((bf,n) for bf,n in tails.items() if bf[1]>1)
        print(f' {role}: lifecycle forwards {fw}; b {min(bs)}-{max(bs)}; kv {kv}; zero max {zmax}; bt write max {wmax}; multi-tail {multi}')
    # attention serving and dummy coordinates and coverage
    att=j['attention']
    cov=collections.defaultdict(lambda:[0,0,0])  # classes, forwards, uncovered classes
    unc=collections.defaultdict(set)
    for a in att:
        key=(a['role'],a['kind'],a['mode'])
        c=cov[key]; c[0]+=1; c[1]+=a['forwards']
        covered=any(isinstance(v,list) and v[0]>0 for v in a["coverage"].values())
        if not covered:
            c[2]+=1
            unc[key].add((a['W'],a['R'],a['own_reqs'],a['has_prefill']))
    for key,c in sorted(cov.items()):
        print(' attn',key,'lane classes',c[0],'forwards',c[1],'uncovered (no same-mode row)',c[2],'distinct (W,R,own,prefill)',len(unc[key]))
    print(' moe', json.dumps(j['moe_rank_classes'])[:1500])
    print(' gate', json.dumps(j['gate_rows']))
