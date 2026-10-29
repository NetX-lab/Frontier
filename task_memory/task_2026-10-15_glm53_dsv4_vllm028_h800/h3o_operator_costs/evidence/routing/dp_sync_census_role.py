"""Count native decode forwards whose lane-local graph widths differ before DP sync.

Reads retained native serving counters (rank0 of each decode node) and compares
the routed row count T under lane-local widths with T under the DP-synced widths.
"""
import bisect, collections, glob, json, sys

TP = 8
ROLE = sys.argv.pop(1)
for base in sys.argv[1:]:
    files = sorted(glob.glob(base + "/nodes/*/serving/rank0.json"))
    decode = [f for f in files if json.load(open(f)).get("role") == ROLE]
    d = json.load(open(decode[0]))
    caps = d["inventory"]["effective_graph"]["capture_sizes"]
    total = differ = 0
    t_gap = collections.Counter()
    kinds = collections.Counter()
    for r in d["counters"]["rows"]:
        if r.get("kind") not in ("serving", "dummy"):
            continue
        de = r.get("dispatch_evidence") or {}
        disp = de.get("dispatch") or {}
        mode = (disp.get("descriptor") or {}).get("cg_mode")
        local, synced = de["dp_post_all_reduce"][0], disp["num_tokens_across_dp"]
        n = r.get("successful_forwards") or 0
        total += n
        desired = de["dp_post_all_reduce"][1]
        # Frontier keeps each real lane's own mode and width (no DP sync).
        lane = [caps[bisect.bisect_left(caps, x)] if want else x
                for x, want in zip(local, desired)]
        t_lane = sum(-(-w // TP) * TP for w in lane)
        t_native = sum(-(-w // TP) * TP for w in synced)
        if t_lane != t_native:
            differ += n
            kinds["mode (synced NONE, a lane wants a graph)" if mode == "NONE" else "width (graph, unequal capture sizes)"] += n
            t_gap[(mode, tuple(desired), tuple(local), tuple(lane), tuple(synced))] += n
    print(base.rstrip("/").split("/")[-1], ROLE, "node", decode[0].split("/nodes/")[1].split("/")[0])
    print(f"  forwards={total} T_differs={differ} ({100*differ/max(total,1):.1f}%)")
    print("  by cause:", dict(kinds))
    for k, v in t_gap.most_common(8):
        print("  ", v, "mode,desired,local,lane_width,native_synced =", k)
