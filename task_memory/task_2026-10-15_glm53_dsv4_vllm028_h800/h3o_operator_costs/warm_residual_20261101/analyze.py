"""Read existing CPU/native artifacts; write auditable warm TTFT span summaries.

Usage: python analyze.py OUTPUT_DIRECTORY
Native clean rows establish totals only. Diagnostic rows establish finer
spans, whose differences from clean totals remain explicitly unallocated.
"""
import collections
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

from fifo_model import mean_wait, phase_wait, simulate

TASK = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
TMP = Path("/data/ycfeng/tmp/glm53_dsv4")
FIRST = {6, 17, 27, 35, 36}


def rows(path):
    with open(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def avg(items):
    items = list(items)
    return statistics.mean(items)


def summarize(records):
    return {key: avg(r[key] for r in records) for key in records[0] if isinstance(records[0][key], (float, int))}


def frontier(case, generation):
    root = TMP / (f"h3o_b/stage4/a10/{case}_warm_k19b29" if generation == "B29"
                  else f"a6_pass_replays_20261101/runs/{generation}_{case}")
    metric = root / f"metrics/deepseek_ai_deepseek_v4_flash/online_serving/a10_{case}_operator_path"
    conf = json.load(open(metric / "config.json"))
    ingress = conf["request_ingress_delay_ms"]
    handoff = conf["kv_cache_transfer_config"]["request_handoff_delay_ms"]
    assert ingress == 10.3 and handoff == 11.4
    assert conf["cluster_config"]["replica_scheduler_config"]["max_forwards_in_flight"] == 19
    assert conf["kv_cache_transfer_config"]["nixl_read_completion"] == "engine_step"
    done = {r["request_id"]: r for r in rows(metric / "metrics_ground_truth.jsonl") if r["event_type"] == "request_completion"}
    kv = {r["request_ids"][0]: r for r in rows(metric / "frontier_transfer_ledger.jsonl") if r["transfer_kind"] == "kv_cache"}
    pre, first, dummy = {}, {}, []
    with gzip.open(root / "forwards.jsonl.gz", "rt") as stream:
        for line in stream:
            f = json.loads(line)
            if f["role"] == "prefill":
                if f["step_class"] == "prefill":
                    for rid in f["request_ids"]:
                        pre[rid] = f
                elif f["dp"] == 0 and f["mode"] == "FULL":
                    dummy.append(f["composed_ms"])
            else:
                for rid in f["request_ids"]:
                    if abs(f["end_s"] - done[rid]["first_decode_token_completed_at"]) < 1e-7:
                        first[rid] = f
    assert len(done) == len(pre) == len(first) == len(kv) == 128
    records = []
    for rid, d in sorted(done.items()):
        p, k, f = pre[rid], kv[rid], first[rid]
        r = dict(request=rid, ttft=(d["first_decode_token_completed_at"] - d["arrived_at"]) * 1000,
                 p_side=(d["prefill_completed_at"] - d["arrived_at"]) * 1000,
                 client_dispatch=0.6, prefill_ingress=ingress - 0.6,
                 prefill_queue=(p["start_s"] - d["arrived_at"]) * 1000 - ingress,
                 prefill_step=(p["end_s"] - p["start_s"]) * 1000,
                 prefill_response=2.3, decode_ingress=handoff - 2.3,
                 kv=(k["end_ts_s"] - k["start_ts_s"]) * 1000,
                 first_decode=(d["first_decode_token_completed_at"] - k["end_ts_s"]) * 1000,
                 first_decode_queue=(f["start_s"] - k["end_ts_s"]) * 1000,
                 first_decode_compute=(f["end_s"] - f["start_s"]) * 1000,
                 egress=0.0, queued_ahead=p["queued_ahead_at_formation"])
        assert abs(sum(r[x] for x in ("client_dispatch", "prefill_ingress", "prefill_queue", "prefill_step", "prefill_response", "decode_ingress", "kv", "first_decode", "egress")) - r["ttft"]) < 1e-5
        records.append(r)
    g, p = avg(dummy), avg(r["prefill_step"] for r in records)
    means = summarize(records)
    means["first_decode_without_request0"] = avg(r["first_decode"] for r in records if r["request"] != 0)
    means["matched_warm_ttft"] = avg(r["ttft"] for repeat in range(3) for r in records if not (case == "p2_64" and repeat == 0 and r["request"] in FIRST))
    fifo = dict(g=g, p=p, wait=mean_wait(simulate(g, p)), observed_wait=means["prefill_queue"],
                phase_wait=phase_wait(g, p), ahead_hist=dict(collections.Counter(r["queued_ahead"] for r in records)))
    fifo["wait_error_ms"] = fifo["wait"] - fifo["observed_wait"]
    fifo["phase_ttft"] = means["ttft"] - means["prefill_queue"] + fifo["phase_wait"]
    fifo["sensitivity"] = {str(delta): dict(finite=mean_wait(simulate(g, p + delta)) - fifo["wait"],
                                           phase=phase_wait(g, p + delta) - fifo["phase_wait"]) for delta in (1, 2.7, 7.5, 9.7, 10, 12)}
    fifo["native_dummy_23.81"] = dict(finite_wait=mean_wait(simulate(23.81, p)), phase_wait=phase_wait(23.81, p))
    json.dump(records, open(OUT / f"{case}_{generation}_rows.json", "w"), indent=2)
    return dict(root=str(root), means=means, fifo=fifo)


def clean(case):
    root = TASK / f"h200_mainline/diagnostic/inputs/{case}r/run"
    proxy = {r["request_id"]: r for r in rows(root / "proxy_requests.jsonl")}
    out = []
    for c in rows(root / "client_requests.jsonl"):
        if "formal" not in c["request_id"]:
            continue
        key = c["request_id"]
        i = int(key.split("-formal-r")[1].split("-")[0])
        p = proxy[key]
        spans = dict(request=key, client_dispatch=(p["proxy_arrival_monotonic"] - c["dispatch_monotonic"]) * 1000,
                     p_side=(p["prefill_response_monotonic"] - p["proxy_arrival_monotonic"]) * 1000,
                     handoff=(p["decode_first_byte_monotonic"] - p["prefill_response_monotonic"]) * 1000,
                     client_egress=(c["first_token_monotonic"] - p["decode_first_byte_monotonic"]) * 1000,
                     ttft=c["ttft_ms"])
        spans["excluded_first_contact"] = case == "p2_64" and "repeat01" in key and i in FIRST
        out.append(spans)
    assert len(out) == 384
    kept = [r for r in out if not r["excluded_first_contact"]]
    result = dict(root=str(root), all_n=len(out), warm_n=len(kept), warm=summarize(kept),
                  per_repeat={rep: summarize([r for r in kept if rep in r["request"]]) for rep in ("repeat01", "repeat02", "repeat03")})
    json.dump(out, open(OUT / f"{case}_clean_rows.json", "w"), indent=2)
    return result


def diagnostic(path, excludes=None):
    r = json.load(open(path))["rows"]
    if excludes:
        first = set(json.load(open(excludes)))
        r = [x for x in r if x["request"] not in first]
    means = {p: avg(x["parts"][p] for x in r) for p in r[0]["parts"]}
    return dict(path=str(path), n=len(r), parts=means, ttft=avg(x["ttft_ms"] for x in r))


result = {case: dict(native=clean(case), frontier={g: frontier(case, g) for g in ("B29", "pass1_all", "pass2_all")}) for case in ("p2_64", "p2_32")}
result["native_diagnostics"] = dict(
    wave64=diagnostic(TASK / "h3o_operator_costs/a10/remainder/remainder_out.json", TASK / "h3o_operator_costs/a10/remainder/first_contact.json"),
    lifecycle64=diagnostic(TASK / "h3o_operator_costs/a10/remainder/lifecycle/remainder_out.json", TASK / "h3o_operator_costs/a10/remainder/lifecycle/first_contact.json"),
    lifecycle32=diagnostic(OUT / "native32_remainder.json"))
json.dump(result, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(result, indent=2))
