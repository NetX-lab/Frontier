"""Stream diagnostic logs and verify worker-local batch and formal request joins."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def identity(row):
    return tuple(row[key] for key in ("dp_rank", "tp_rank", "pp_rank"))


def analyze(run, output, mode):
    clients = list(rows(run / "client.jsonl"))
    expected = {f"pf4096_dc1024:{i}" for i in range(100)}
    expected |= {f"warmup:pf4096_dc1024:r{r}:{i}" for r in range(3) for i in range(100)}
    assert len(clients) == 400 and {r["request_id"] for r in clients} == expected
    assert all(r["prompt_tokens"] == 4096 and r["completion_tokens_observed"] == 1024 for r in clients)
    formal = {r["response_id"] + "-0": r["request_id"] for r in clients if not r["request_id"].startswith("warmup:")}
    workers, prefill_workers, results = set(), {}, []
    batch_fields = ("batch_size", "batch_num_tokens", "batch_num_prefill_tokens",
                    "batch_num_decode_tokens")
    for path in sorted(run.glob("server.batch.dp*.tp*.pp*.jsonl")):
        batches = {}
        worker = None
        for row in rows(path):
            worker = identity(row) if worker is None else worker
            assert identity(row) == worker and row["batch_id"] not in batches
            assert len(row["request_ids"]) == len(row["request_num_tokens"]) == row["batch_size"]
            assert row["batch_num_prefill_tokens"] == sum(
                tokens for tokens in row["request_num_tokens"] if tokens != 1)
            assert row["batch_num_decode_tokens"] == row["request_num_tokens"].count(1)
            batches[row["batch_id"]] = row
            for request, tokens in zip(row["request_ids"], row["request_num_tokens"]):
                if request in formal and tokens > 1:
                    assert tokens == 4096
                    prefill_workers.setdefault(formal[request], Counter())[worker] += 1
        assert worker is not None and worker not in workers
        workers.add(worker)
        selected = {key for key, row in batches.items() if set(row["request_ids"]) & formal.keys()}
        assert len(selected) >= 3
        first = min(selected)
        seen_batches, scopes, phases = set(), Counter(), Counter()
        last_batch, seen_keys, count = -1, set(), 0
        detail = path.with_name(path.name.replace("server.batch.", f"server.{mode}."))
        for row in rows(detail):
            assert identity(row) == worker
            key = row["batch_id"]
            assert key in batches and key >= last_batch
            assert all(row[field] == batches[key][field] for field in batch_fields)
            if mode == "ops":
                assert row["batch_request_num_tokens"] == batches[key]["request_num_tokens"]
            if key != last_batch:
                seen_keys.clear()
                last_batch = key
            local = (row["op_name"], row["scope_seq"]) if mode == "ops" else row["layer_name"]
            assert local not in seen_keys
            seen_keys.add(local)
            if mode == "ops":
                assert row["timing_mode"] == "cuda_event" and row["aggregation_mode"] == "per_scope"
                assert math.isfinite(row["cuda_time_ms"]) and row["cuda_time_ms"] >= 0
            else:
                assert row["ep_size"] == 8 and row["ep_rank"] == worker[0] * 4 + worker[1]
                assert sum(row["per_expert_tokens"].values()) == row["total_routed_tokens"]
            if key in selected:
                if key not in seen_batches:
                    phase = "mixed" if row["batch_num_prefill_tokens"] and row["batch_num_decode_tokens"] else ("prefill" if row["batch_num_prefill_tokens"] else "decode")
                    phases[phase] += 1
                seen_batches.add(key)
                scopes[row["op_name"] if mode == "ops" else row["layer_name"]] += 1
            count += 1
        assert first in seen_batches and seen_batches == selected
        results.append({"identity": worker, "batches": len(batches), "detail_rows": count,
                        "formal_batches": len(selected), "first_formal_batch": first,
                        "formal_scopes": scopes, "local_formal_batch_phases": phases})
    assert workers == {(dp, tp, 0) for dp in range(2) for tp in range(4)}
    assert set(prefill_workers) == {f"pf4096_dc1024:{i}" for i in range(100)}
    for coverage in prefill_workers.values():
        lanes = {worker[0] for worker in coverage}
        assert len(lanes) == 1
        dp = next(iter(lanes))
        assert coverage == Counter({(dp, tp, 0): 1 for tp in range(4)})
    report = {"status": "PASS", "source": str(run), "mode": mode, "workers": results,
              "formal_requests": 100,
              "limits": "Worker-local identity and batch metadata validation only. Phases describe local scheduled requests; routing num_tokens and local expert counts describe post-dispatch inputs. DP dummy rounds are unlogged, so cross-DP collective-round joins and EP-global phases are unverified. No Frontier operator parity or E2E claim."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["ops", "routing"], required=True)
    args = parser.parse_args()
    analyze(args.run, args.output, args.mode)
