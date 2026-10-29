import json, collections
def rng(xs): xs=sorted(xs); return f"{xs[0]}-{xs[-1]}" if xs else "-"
for case in ['p2_32','p2_64','p0']:
    j=json.load(open(f'{case}.json'))
    print('==',case)
    lc=j['lifecycle']
    comp=collections.defaultdict(lambda:[0,0]); tails=collections.Counter()
    for k,v in lc.items():
        if not k.startswith('decode'): continue
        for b,n4,n128,n in v['compression_group_writes']['counts']:
            c=comp[b]; c[0]=max(c[0],n4); c[1]=max(c[1],n128)
        for b,f,n,kr,mr in v['b_first_tails']:
            if f: tails[f]+=n
    print(' decode max (n4,n128) per b:', dict(sorted(comp.items())))
    print(' decode forwards by first-tail count:', dict(tails))
    att=j['attention']
    for role in ['decode','prefill']:
        for kind in ['serving','dummy']:
            for mode in ['FULL','PIECEWISE','NONE']:
                sel=[a for a in att if a['role']==role and a['kind']==kind and a['mode']==mode]
                if not sel: continue
                byW=collections.defaultdict(set); fw=collections.Counter()
                for a in sel:
                    byW[a['W']].add((a['own_reqs'], a['has_prefill'])); fw[a['W']]+=a['forwards']
                desc='; '.join(f"W{W}: own {rng({o for o,_ in s})}" + (f", first tail at own {rng({o for o,p in s if p})}" if any(p for _,p in s) else '') + f" ({fw[W]} fw)" for W,s in sorted(byW.items()))
                cov=collections.Counter()
                for a in sel:
                    cov['covered' if any(isinstance(v,list) and v[0]>0 for v in a['coverage'].values()) else 'uncovered']+=a['forwards']
                print(f' attn {role} {kind} {mode}: {desc}; forwards {dict(cov)}')
    pts=collections.defaultdict(set)
    for c in j['collectives']:
        pts[(c['collective'],c['group']+str(c['size']),'graph' if c['graph'] else 'eager')].add((json.dumps(c['point']),c['coverage']))
    for k,v in sorted(pts.items()):
        print('  coll',k, sorted(v))
