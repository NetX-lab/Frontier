#!/usr/bin/env python3
"""Stage-0 C1: the workload coordinates of one native V4 case and their coverage by existing data.

  python c1_workload.py --case NAME --reference DIR --lifecycle DIR --output PREFIX
      --gemm-stream-threshold N --shared-expert-stream-threshold N
      [--attention NAME=CSV ...] [--collectives CSV] [--moe-scope JSON]

Run from the repository root; the model config is read from data/config/models/.

--reference is the artifacts directory of a canonical run. Rank 0 of every node
records the forwards of its DP lane in the serving counters: dispatched
descriptor, own request and token counts, and the synchronized widths.
--lifecycle is the artifacts directory of a lifecycle run of the same case.
Its scheduler records give the exact membership of every real forward, so the
contexts, first tails, newly zeroed KV blocks and block-table writes are exact.
The two runs are different executions of the same workload; part C compares them.

Existing data is passed per device; H800 and H200 data never qualify each other.
Writes PREFIX.txt (the report) and PREFIX.json (the coordinates).
"""

import argparse
import collections
import glob
import json
import math
import os

import pandas as pd

ACTIVATION_BYTES = {"bfloat16": 2}
ROUTING_WEIGHT_BYTES = 4  # float32 top-k weights
ROUTING_ID_BYTES = 4  # int32 top-k ids
GRAPH_MODES = ("FULL", "PIECEWISE")
CSV_GRAPH_MODE = {"NONE": "eager", "FULL": "full", "PIECEWISE": "piecewise"}
# V4 collective -> (curve op, curve group); TP collectives run inside one lane.
CURVE_OF = {
    "embedding_allreduce": ("all_reduce", "tp"),
    "sp_attention_allgather": ("all_gather", "tp"),
    "sp_attention_reduce_scatter": ("reduce_scatter", "tp"),
    "sp_final_mhc_allgather": ("all_gather", "tp"),
    "logits_allgather": ("all_gather", "tp"),
    "ep_activation_allgather": ("all_gatherv", "ep"),
    "ep_weights_allgather": ("all_gatherv", "ep"),
    "ep_ids_allgather": ("all_gatherv", "ep"),
    "ep_output_reduce_scatter": ("reduce_scatterv", "ep"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--case", required=True)
parser.add_argument("--reference", required=True)
parser.add_argument("--lifecycle", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--gemm-stream-threshold", type=int, required=True)
parser.add_argument("--shared-expert-stream-threshold", type=int, required=True)
parser.add_argument("--attention", action="append", default=[])
parser.add_argument("--collectives")
parser.add_argument("--moe-scope")
args = parser.parse_args()

lines = []


def out(text=""):
    lines.append(text)


def load(path):
    with open(path) as handle:
        return json.load(handle)


def hist(counter):
    return " ".join(f"{k}:{v}" for k, v in sorted(counter.items()))


engine = load(os.path.join(args.reference, "run", "replay_summary.json"))["engine_config"]
model = load(os.path.join("data/config/models", engine["model_name"].replace("/", "__") + ".json"))
H, topk, experts = model["hidden_size"], model["num_experts_per_tok"], model["n_routed_experts"]
hc_mult, vocab, index_topk = model["hc_mult"], model["vocab_size"], model["index_topk"]
layers = model["num_hidden_layers"]
act = ACTIVATION_BYTES[model["torch_dtype"]]
indexer_ratios = {r for r in model["compress_ratios"][:layers] if r not in (0, 128)}
(indexer_ratio,) = indexer_ratios
compress_ratios = sorted({r for r in model["compress_ratios"][:layers] if r})
calls = {"embedding_allreduce": 1, "sp_final_mhc_allgather": 1, "logits_allgather": 1}
calls.update({name: layers for name in CURVE_OF if name not in calls})

# ---------------------------------------------------------------- part A: reference counters
roles = {}
lane_classes = collections.defaultdict(collections.Counter)
global_vectors = collections.defaultdict(collections.Counter)
startup = []
for path in sorted(glob.glob(os.path.join(args.reference, "nodes/*/serving/rank0.json"))):
    data = load(path)
    inventory, groups = data["inventory"], data["inventory"]["groups"]
    role, dp_rank = data["role"], groups["dp"]["rank_in_group"]
    topology = {"tp": groups["tp"]["world_size"], "dp": groups["dp"]["world_size"],
                "ep": groups["ep"]["world_size"],
                "block_sizes": [g["kv_cache_spec"]["block_size"] for g in inventory["cache"]["kv_cache_groups"]],
                "capture_sizes": inventory["effective_graph"]["capture_sizes"],
                "device": inventory["device"].get("name") if isinstance(inventory["device"], dict) else inventory["device"]}
    if roles.setdefault(role, topology) != topology:
        raise SystemExit(f"{path}: topology differs inside role {role}")
    for row in data["counters"]["rows"]:
        kind = row.get("kind")
        if kind not in ("serving", "dummy"):
            continue
        evidence = row["dispatch_evidence"]
        dispatch = evidence["dispatch"]
        descriptor, inputs = dispatch["descriptor"], dispatch["inputs"]
        classifier = (evidence.get("classifier") or {}).get("inputs") or {}
        forwards = row.get("successful_forwards") or 0
        mode, width = descriptor["cg_mode"], descriptor["num_tokens"]
        across = tuple(dispatch["num_tokens_across_dp"])
        if kind == "dummy" and inputs["num_tokens"] > 1:
            startup.append((role, dp_rank, mode, width, descriptor["num_reqs"], inputs["num_tokens"],
                            inputs["num_reqs"], forwards))
            continue
        attention_reqs = descriptor["num_reqs"] if mode == "FULL" else inputs["num_reqs"]
        key = (kind, mode, width, attention_reqs, inputs["num_tokens"], inputs["num_reqs"],
               bool(classifier.get("has_prefill")), across)
        lane_classes[(role, dp_rank)][key] += forwards
        if dp_rank == 0:
            global_vectors[role][(mode, across)] += forwards

out(f"# C1 workload coordinates: {args.case}")
out()
out(f"reference: {args.reference}")
out(f"lifecycle: {args.lifecycle}")
out(f"model: {engine['model_name']}, hidden {H}, top-k {topk} of {experts} experts, hc_mult {hc_mult}, "
    f"vocab {vocab}, {layers} layers, indexer compress ratio {indexer_ratio}, index_topk {index_topk}")
out(f"thresholds: outer-projection fork when attention rows <= {args.gemm_stream_threshold}; "
    f"shared expert on its stream when MoE input rows <= {args.shared_expert_stream_threshold}")
for role, topology in sorted(roles.items()):
    out(f"role {role}: device {topology['device']}, TP{topology['tp']} DP{topology['dp']} EP{topology['ep']}, "
        f"block sizes {topology['block_sizes']}, capture sizes {topology['capture_sizes'][0]}.."
        f"{topology['capture_sizes'][-1]} ({len(topology['capture_sizes'])})")

out()
out("## A1. Lane forward classes (serving counters, rank 0 of each lane)")
out("key: kind mode W(descriptor tokens) R(attention requests: descriptor requests for FULL, own requests "
    "otherwise) own_tokens own_reqs has_prefill | forwards | widths across DP")
for lane in sorted(lane_classes):
    classes = lane_classes[lane]
    by_kind = collections.Counter()
    for key, n in classes.items():
        by_kind[(key[0], key[1])] += n
    out(f"### {lane[0]} dp{lane[1]}: {sum(classes.values())} forwards; " +
        ", ".join(f"{k} {m} {n}" for (k, m), n in sorted(by_kind.items())))
    merged = collections.Counter()
    for key, n in classes.items():
        merged[key[:7]] += n
    for key, n in sorted(merged.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][5])):
        widths = sorted({k[7] for k in classes if k[:7] == key})
        shown = widths if len(widths) <= 4 else widths[:4] + [f"... {len(widths)} vectors"]
        out(f"  {' '.join(str(v) for v in key)} | {n} | {shown}")

out()
out("## A2. Startup warmup dummies (own tokens > 1), excluded from serving coverage")
for item in startup:
    out(f"  {item[0]} dp{item[1]}: {item[2]} W{item[3]} R{item[4]} own {item[5]} tokens / {item[6]} reqs, "
        f"{item[7]} forwards")

# ---------------------------------------------------------------- global vectors, MoE, collectives
moe_classes = collections.defaultdict(collections.Counter)
gate_rows = collections.defaultdict(collections.Counter)
collective_points = collections.defaultdict(collections.Counter)
global_records = []
out()
out("## A3. Global forwards (DP rank 0 of each role): N (synchronized widths) -> L = ceil(N/TP) -> M = TP x sum(L)")
for role, vectors in sorted(global_vectors.items()):
    topology = roles[role]
    tp, ep = topology["tp"], topology["ep"]
    local_experts = experts // ep
    m_counter = collections.Counter()
    for (mode, across), forwards in sorted(vectors.items()):
        rows = [math.ceil(n / tp) for n in across]
        gathered = tp * sum(rows)
        m_counter[(mode, gathered)] += forwards
        global_records.append({"role": role, "mode": mode, "N": list(across), "L": rows, "M": gathered,
                               "forwards": forwards})
        graph = mode in GRAPH_MODES
        quotient, remainder = divmod(topk * gathered, experts)
        ranks = collections.Counter()
        for rank in range(ep):
            high = min(max(remainder - rank * local_experts, 0), local_experts)
            ranks[(high * (quotient + 1) + (local_experts - high) * quotient, high, quotient)] += 1
        moe_classes[(role, gathered)].update({cls: forwards * count for cls, count in ranks.items()})
        for d, local_rows in enumerate(rows):
            gate_rows[(role, mode)][local_rows] += forwards
            lane = {"embedding_allreduce": across[d] * H * act,
                    "sp_attention_allgather": local_rows * H * act,
                    "sp_attention_reduce_scatter": tp * local_rows * H * act,
                    "sp_final_mhc_allgather": local_rows * hc_mult * H * act}
            for name, size in lane.items():
                collective_points[(name, "tp", tp, graph)][size] += forwards * calls[name]
        ep_inputs = {"ep_activation_allgather": [r * H * act for r in rows],
                     "ep_weights_allgather": [r * topk * ROUTING_WEIGHT_BYTES for r in rows],
                     "ep_ids_allgather": [r * topk * ROUTING_ID_BYTES for r in rows]}
        for name, per_lane in ep_inputs.items():
            collective_points[(name, "ep", ep, graph)][tuple(per_lane)] += forwards * calls[name]
        collective_points[("ep_output_reduce_scatter", "ep", ep, graph)][(gathered * H * act,) +
                                                                         tuple(r * H * act for r in rows)] += \
            forwards * calls["ep_output_reduce_scatter"]
    out(f"### {role}: {sum(vectors.values())} global forwards, {len(vectors)} distinct (mode, N)")
    for (mode, gathered), n in sorted(m_counter.items()):
        examples = sorted({tuple(r['N']) for r in global_records if r['role'] == role and r['mode'] == mode
                           and r['M'] == gathered})
        shown = examples if len(examples) <= 3 else examples[:3] + [f"... {len(examples)} vectors"]
        out(f"  {mode} M{gathered}: {n} forwards, N {shown}")

# logits gather on real lanes only: own requests of each serving forward. It is eager in every mode:
# compute_logits runs in sample() after the graph replay (V2 model_runner.py:1333-1340, 1707-1764).
for (role, dp_rank), classes in lane_classes.items():
    tp = roles[role]["tp"]
    for key, n in classes.items():
        if key[0] == "serving":
            collective_points[("logits_allgather", "tp", tp, False)][
                key[5] * (vocab // tp) * act] += n * calls["logits_allgather"]

out()
out("## A4. Routed MoE rank classes (balanced_topk: expert e gets floor(6M/256) + [e < 6M mod 256] rows)")
out("class: rows routed to the rank, local experts with one extra row, base rows per expert | EP ranks x forwards")
moe_scope = load(args.moe_scope)["scope"] if args.moe_scope else None
moe_records = []
for (role, gathered), classes in sorted(moe_classes.items()):
    ep = roles[role]["ep"]
    local_experts = experts // ep
    quotient, remainder = divmod(topk * gathered, experts)
    rank0 = (min(remainder, local_experts) * (quotient + 1) + (local_experts - min(remainder, local_experts)) * quotient,
             min(remainder, local_experts), quotient)
    for cls, weight in sorted(classes.items()):
        covered = bool(moe_scope and moe_scope["producer_ep"] == ep and moe_scope["ep_rank"] == 0
                       and gathered in moe_scope["num_tokens"] and cls == rank0)
        moe_records.append({"role": role, "M": gathered, "ep": ep, "local_experts": local_experts,
                            "rows": cls[0], "experts_plus_one": cls[1], "base_rows": cls[2],
                            "rank_forwards": weight, "isolated_eager_candidate": covered})
    out(f"  {role} EP{ep} M{gathered}: " + "; ".join(
        f"({c[0]},{c[1]},{c[2]}) x{w}" + (" [rank-0 isolated eager candidate]" if r["isolated_eager_candidate"] else "")
        for (c, w), r in zip(sorted(classes.items()), moe_records[-len(classes):])))
out("gate and top-k rows per rank (L), by mode:")
for (role, mode), counter in sorted(gate_rows.items()):
    out(f"  {role} {mode}: {hist(counter)}")

# ---------------------------------------------------------------- collective coverage
curves = collections.defaultdict(list)
if args.collectives:
    for row in pd.read_csv(args.collectives).itertuples():
        curves[(row.op, row.group, int(row.group_size), str(row.cuda_graph) == "True")].append(int(row.max_input_bytes))


def curve_verdict(name, group, size, graph, point):
    points = curves.get((CURVE_OF[name][0], group, size, graph))
    if not points:
        return "no curve"
    uniform = len(set(point[1:] if name == "ep_output_reduce_scatter" else point)) == 1 if isinstance(point, tuple) \
        else True
    probe = (point[0] if name == "ep_output_reduce_scatter" else max(point)) if isinstance(point, tuple) else point
    where = ("exact point" if probe in points else "inside range" if min(points) <= probe <= max(points)
             else "below range" if probe < min(points) else "above range")
    return where + ("" if uniform else ", unequal vector")


out()
out("## A5. Collective payloads (input bytes per rank; EP: ordered per-lane inputs, each lane = TP ranks)")
out(f"curves: {args.collectives or 'none for this device'}")
collective_records = []
for (name, group, size, graph), points in sorted(collective_points.items()):
    verdicts = collections.Counter()
    for point, n in points.items():
        verdict = curve_verdict(name, group, size, graph, point)
        verdicts[verdict] += n
        collective_records.append({"collective": name, "group": group, "size": size, "graph": graph,
                                   "point": list(point) if isinstance(point, tuple) else point,
                                   "calls": n, "coverage": verdict})
    shown = sorted(points, key=lambda p: -points[p])[:6]
    out(f"  {name} {group}{size} {'graph' if graph else 'eager'}: {len(points)} distinct, calls {sum(points.values())}; "
        f"coverage {dict(verdicts)}; top {shown}")

# ---------------------------------------------------------------- part B: lifecycle
lifecycle = {}
for path in sorted(glob.glob(os.path.join(args.lifecycle, "nodes/*/scheduler_lifecycle/*.jsonl"))):
    limit, running = {}, {}
    lane = None
    stats = {"steps": 0, "cached_mismatch": 0, "token_mismatch": 0, "bf": collections.Counter(),
             "kv": {}, "max_seq": {}, "zero": collections.Counter(), "zero_groups": collections.Counter(),
             "writes": collections.Counter(), "query": collections.Counter(), "phase": collections.Counter(),
             "compressed": collections.Counter()}
    with open(path) as handle:
        for text in handle:
            record = json.loads(text)
            event = record["event"]
            if event == "start":
                lane = (record["role"], record["dp_rank"])
                block_sizes = roles[record["role"]]["block_sizes"]
                continue
            if event == "add":
                limit[record["request_id"]] = record["prompt_tokens"] + record["max_tokens"] - 1
                continue
            if event != "schedule" or not record.get("tokens"):
                continue
            stats["steps"] += 1
            stats["cached_mismatch"] += record.get("cached", 0) != len(running)
            batch = [(rid, kv, kv + 1, False) for rid, kv in running.items()]
            batch += [(rid, computed, computed + n, True) for rid, computed, n in record.get("new", [])]
            stats["token_mismatch"] += sum(new - old for _, old, new, _ in batch) != record["tokens"]
            for rid, _, new, _ in batch:
                running[rid] = new
            first_tails = sum(is_new for *_, is_new in batch)
            key = (len(batch), first_tails)
            stats["bf"][key] += 1
            cached = [old for _, old, _, _ in batch]
            low, high = stats["kv"].get(key, (min(cached), max(cached)))
            stats["kv"][key] = (min(low, min(cached)), max(high, max(cached)))
            longest = max(new for _, _, new, _ in batch)
            low, high = stats["max_seq"].get(key, (longest, longest))
            stats["max_seq"][key] = (min(low, longest), max(high, longest))
            per_group = [sum(math.ceil(new / b) - math.ceil(old / b) for _, old, new, _ in batch) for b in block_sizes]
            stats["zero"][sum(per_group)] += 1
            stats["zero_groups"][tuple(per_group)] += 1
            rows = sum(1 for _, old, new, is_new in batch
                       if is_new or any(math.ceil(new / b) > math.ceil(old / b) for b in block_sizes))
            entries = sum(math.ceil(new / b) - (0 if is_new else math.ceil(old / b))
                          for _, old, new, is_new in batch for b in block_sizes)
            stats["writes"][(rows, entries)] += 1
            stats["query"][tuple(sorted({new - old for _, old, new, _ in batch}))] += 1
            stats["compressed"][(len(batch),) + tuple(sum(new // r > old // r for _, old, new, _ in batch)
                                                        for r in compress_ratios)] += 1
            stats["phase"][tuple(sorted({("warmup" if "-warmup-" in rid else "formal") for rid, *_ in batch}))] += 1
            for rid in [rid for rid, kv in running.items() if kv >= limit[rid]]:
                del running[rid]
    lifecycle[lane] = stats

out()
out("## B. Lifecycle forwards per lane (real forwards only; dummies are not scheduler steps)")
out("kv = cached tokens before the step (the attention CSV convention); max_seq = longest sequence after it")
for lane, stats in sorted(lifecycle.items()):
    out(f"### {lane[0]} dp{lane[1]}: {stats['steps']} forwards; cached-count mismatches {stats['cached_mismatch']}, "
        f"token-count mismatches {stats['token_mismatch']}; phases {dict(stats['phase'])}")
    b_hist = collections.Counter()
    for (b, _), n in stats["bf"].items():
        b_hist[b] += n
    out(f"  b (requests = logits rows): {hist(b_hist)}")
    out("  (b, first tails): forwards, kv range, max_seq range")
    for key, n in sorted(stats["bf"].items()):
        out(f"    {key}: {n}, kv {list(stats['kv'][key])}, max_seq {list(stats['max_seq'][key])}")
    out(f"  query lengths present: {hist(stats['query'])}")
    out(f"  newly zeroed blocks per forward: {hist(stats['zero'])}")
    out(f"  per group {roles[lane[0]]['block_sizes']}: {hist(stats['zero_groups'])}")
    out(f"  block-table writes (rows, entries): {hist(stats['writes'])}")
    out(f"  (b, requests completing a compression group of {compress_ratios} tokens): {hist(stats['compressed'])}")

# ---------------------------------------------------------------- part C: lifecycle against reference
out()
out("## C. Real-forward request counts: lifecycle run against reference run (different executions)")
for lane in sorted(set(lifecycle) | {l for l in lane_classes}):
    ref = collections.Counter()
    ref_prefill = 0
    for key, n in lane_classes.get(lane, {}).items():
        if key[0] == "serving":
            ref[key[5]] += n
            ref_prefill += n if key[6] else 0
    life = collections.Counter()
    life_tails = 0
    for (b, f), n in lifecycle.get(lane, {"bf": {}})["bf"].items():
        life[b] += n
        life_tails += n if f else 0
    out(f"  {lane[0]} dp{lane[1]}: reference {sum(ref.values())} forwards ({ref_prefill} with has_prefill), "
        f"lifecycle {sum(life.values())} ({life_tails} with a new request)")
    out(f"    reference b: {hist(ref)}")
    out(f"    lifecycle b: {hist(life)}")

# ---------------------------------------------------------------- part D: attention grid and coverage
existing = {}
for item in args.attention:
    name, path = item.split("=", 1)
    frame = pd.read_csv(path, usecols=["phase", "graph_mode", "batch_size", "num_prefill_seqs", "num_decode_seqs",
                                       "prefill_seq_lens", "prefill_kv_cache_sizes", "decode_kv_cache_sizes"])
    for column in ("prefill_seq_lens", "prefill_kv_cache_sizes", "decode_kv_cache_sizes"):
        frame[column] = frame[column].map(json.loads)
    existing[name] = frame


def attention_rows(frame, graph_mode, shape):
    selected = frame[frame.graph_mode == graph_mode]
    if shape["kind"] == "decode":
        selected = selected[(selected.phase == "decode") & (selected.num_decode_seqs == shape["reqs"])
                            & (selected.num_prefill_seqs == 0)]
        low, high = shape["kv"]
        return int(selected.decode_kv_cache_sizes.map(lambda kv: all(low <= v <= high for v in kv)).sum())
    selected = selected[(selected.phase == "prefill") & (selected.num_prefill_seqs == shape["reqs"])
                        & (selected.num_decode_seqs == 0)]
    return int(selected.apply(lambda r: r.prefill_seq_lens == shape["query"] and r.prefill_kv_cache_sizes == shape["kv"],
                              axis=1).sum())


out()
out("## D. Attention coordinates per lane and their coverage")
out("real forwards: the reference classes with the lifecycle kv range of the same request count "
    "(decode-shaped: every query is 1 token; the first tail is a new request with kv = prompt - 1, so it is "
    "decode-shaped too, with has_prefill set). FULL pads to R = W requests with sequence length 0. "
    "Dummies: R requests splitting W tokens, sequence length = query length, kv 0.")
out("coverage counts: rows of each existing file in the same mode / eager rows of the same shape")
attention_records = []
for lane in sorted(lane_classes):
    role = lane[0]
    tp = roles[role]["tp"]
    merged = collections.Counter()
    for key, n in lane_classes[lane].items():
        merged[key[:7]] += n
    out(f"### {role} dp{lane[1]}")
    stats = lifecycle.get(lane)
    for (kind, mode, width, reqs, own_tokens, own_reqs, has_prefill), n in sorted(merged.items()):
        if kind == "serving" and own_tokens == own_reqs:
            keys = [k for k in stats["bf"] if k[0] == own_reqs and (k[1] > 0) == has_prefill]
            if not keys:
                kv, max_seq = None, None
            else:
                kv = [min(stats["kv"][k][0] for k in keys), max(stats["kv"][k][1] for k in keys)]
                max_seq = [min(stats["max_seq"][k][0] for k in keys), max(stats["max_seq"][k][1] for k in keys)]
            shape = {"kind": "decode", "reqs": own_reqs, "kv": kv, "query": [1] * own_reqs}
        elif kind == "serving":
            shape = {"kind": "prefill", "reqs": own_reqs, "query": [own_tokens // own_reqs] * own_reqs,
                     "kv": [0] * own_reqs}
            max_seq = [own_tokens // own_reqs] * 2
        else:
            base, extra = divmod(width, reqs)
            query = [base] * (reqs - extra) + [base + 1] * extra
            shape = ({"kind": "decode", "reqs": reqs, "kv": [0, 0], "query": query} if max(query) == 1 else
                     {"kind": "prefill", "reqs": reqs, "query": query, "kv": [0] * reqs})
            max_seq = [max(query)] * 2
        features = {"outer_fork": width <= args.gemm_stream_threshold,
                    "shared_expert_stream": math.ceil(width / tp) <= args.shared_expert_stream_threshold,
                    "short_context": mode == "NONE" and max_seq is not None
                    and max_seq[1] // indexer_ratio <= index_topk,
                    "prepare_prefill_inputs": has_prefill}
        coverage = {}
        for name, frame in existing.items():
            if shape["kv"] is None:
                coverage[name] = "no lifecycle kv"
                continue
            same_mode = attention_rows(frame, CSV_GRAPH_MODE[mode], shape)
            eager = same_mode if mode == "NONE" else attention_rows(frame, "eager", shape)
            coverage[name] = [same_mode, eager]
        attention_records.append({"role": role, "dp": lane[1], "kind": kind, "mode": mode, "W": width, "R": reqs,
                                  "own_tokens": own_tokens, "own_reqs": own_reqs, "has_prefill": has_prefill,
                                  "forwards": n, "shape": shape, "max_seq": max_seq, "features": features,
                                  "coverage": coverage})
        flags = ",".join(k for k, v in features.items() if v) or "-"
        out(f"  {kind} {mode} W{width} R{reqs} own {own_tokens}/{own_reqs} prefill={has_prefill}: {n} fw; "
            f"{shape['kind']} kv {shape['kv'] if shape['kind'] == 'decode' else 0} query "
            f"{shape['query'][:2]}{'...' if len(shape['query']) > 2 else ''}; max_seq {max_seq}; [{flags}]; "
            f"coverage {coverage}")

with open(args.output + ".txt", "w") as handle:
    handle.write("\n".join(lines) + "\n")
lifecycle_json = {f"{lane[0]}/dp{lane[1]}": {
    "forwards": stats["steps"], "cached_mismatch": stats["cached_mismatch"],
    "token_mismatch": stats["token_mismatch"],
    "b_first_tails": [[*k, n, list(stats["kv"][k]), list(stats["max_seq"][k])] for k, n in sorted(stats["bf"].items())],
    "zeroed_blocks": sorted(stats["zero"].items()),
    "zeroed_blocks_per_group": [[list(k), n] for k, n in sorted(stats["zero_groups"].items())],
    "block_table_writes": [[*k, n] for k, n in sorted(stats["writes"].items())],
    "compression_group_writes": {"ratios": compress_ratios,
                                 "counts": [[*k, n] for k, n in sorted(stats["compressed"].items())]}}
    for lane, stats in sorted(lifecycle.items())}
with open(args.output + ".json", "w") as handle:
    json.dump({"case": args.case, "reference": args.reference, "lifecycle": args.lifecycle,
               "roles": roles, "startup_dummies": startup, "global_forwards": global_records,
               "moe_rank_classes": moe_records,
               "gate_rows": [[role, mode, sorted(c.items())] for (role, mode), c in sorted(gate_rows.items())],
               "collectives": collective_records, "lifecycle": lifecycle_json,
               "attention": attention_records}, handle, separators=(",", ":"))
    handle.write("\n")
print(f"wrote {args.output}.txt ({len(lines)} lines) and {args.output}.json")
