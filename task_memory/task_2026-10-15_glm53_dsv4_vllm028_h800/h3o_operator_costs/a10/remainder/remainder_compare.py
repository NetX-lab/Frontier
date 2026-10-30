"""Native against Frontier P2-64 TTFT after the prefill side, by part (lead scratch).

    python remainder_compare.py <frontier_remainder.json> <native remainder_out.json> <first_contact.json> [...]

Native arguments come in pairs: remainder_out.json and the formal requests that
open a (prefill lane, decode lane) pair (pairs.py). Means over the formal
requests that do not open a pair; native parts are merged into the spans that
Frontier models.
"""
import json
import statistics
import sys

SPANS = {
    "P first token -> D add (ingress)": ["p_update->prefill_response", "prefill_response->d_add"],
    "D add -> finished_recving": ["d_add->d_recv_done"],
    "finished_recving -> first token": ["d_recv_done->d_schedule", "d_schedule->d_update"],
    "first token -> client": ["d_update->decode_first_byte", "decode_first_byte->client_first"],
}
FRONTIER = {
    "P first token -> D add (ingress)": [],
    "D add -> finished_recving": ["prefill_done->kv_start", "kv_start->kv_end"],
    "finished_recving -> first token": ["kv_end->first_decode"],
    "first token -> client": [],
}


def span_means(rows, spans):
    return {name: statistics.mean(sum(r["parts"][p] for p in parts) for r in rows) for name, parts in spans.items()}


frontier = json.load(open(sys.argv[1]))["rows"]
columns = {"Frontier": (span_means(frontier, FRONTIER), statistics.mean(r["remainder_ms"] for r in frontier),
                        statistics.mean(r["ttft_ms"] for r in frontier), len(frontier))}
for i in range(2, len(sys.argv), 2):
    first = set(json.load(open(sys.argv[i + 1])))
    rows = [r for r in json.load(open(sys.argv[i]))["rows"] if r["request"] not in first]
    columns[sys.argv[i]] = (span_means(rows, SPANS), statistics.mean(r["remainder_ms"] for r in rows),
                            statistics.mean(r["ttft_ms"] for r in rows), len(rows))
for name, (spans, remainder, ttft, n) in columns.items():
    print(f"{name}: n {n} ttft {ttft:.1f} remainder {remainder:.1f}")
    for span, value in spans.items():
        print(f"    {span:36s} {value:7.1f}")
