"""Check the proposed DP-sync rule against every retained native forward.

Rule (vLLM V1 DP coordination as Frontier would apply it to real lanes):
synced mode = minimum of the lanes' desired modes (NONE=0 < PIECEWISE=1 < FULL=2);
synced widths = capture(max local) on every lane when the synced mode is a graph,
else each lane's own local width. Compared with the recorded descriptor mode and
num_tokens_across_dp of every serving and dummy row, on all decode and prefill nodes.
"""
import bisect, collections, glob, json, sys

NAMES = {0: "NONE", 1: "PIECEWISE", 2: "FULL"}
for base in sys.argv[1:]:
    files = sorted(glob.glob(base + "/nodes/*/serving/rank0.json"))
    for role in ("prefill", "decode"):
        node_files = [f for f in files if json.load(open(f)).get("role") == role]
        rows = forwards = bad = 0
        modes = collections.Counter()
        for f in node_files:
            d = json.load(open(f))
            caps = d["inventory"]["effective_graph"]["capture_sizes"]
            for r in d["counters"]["rows"]:
                if r.get("kind") not in ("serving", "dummy"):
                    continue
                de = r["dispatch_evidence"]
                disp = de["dispatch"]
                local, desired = de["dp_post_all_reduce"][0], de["dp_post_all_reduce"][1]
                mode = min(desired)
                synced = ([caps[bisect.bisect_left(caps, max(local))]] * len(local)
                          if mode else list(local))
                ok = (NAMES[mode] == disp["descriptor"]["cg_mode"]
                      and synced == list(disp["num_tokens_across_dp"]))
                n = r.get("successful_forwards") or 0
                rows += 1
                forwards += n
                modes[NAMES[mode]] += n
                if not ok:
                    bad += n
                    print("  MISMATCH", f.split("/nodes/")[1], desired, local,
                          disp["descriptor"]["cg_mode"], disp["num_tokens_across_dp"])
        print(base.rstrip("/").split("/")[-1 if not base.endswith("artifacts") else -2],
              role, f"nodes={len(node_files)} rows={rows} forwards={forwards} mismatched={bad}",
              "synced modes:", dict(modes))
