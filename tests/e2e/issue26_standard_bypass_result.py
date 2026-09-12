"""Direct standard replay validation for the approved post-MoE AR ablation."""
import argparse
import json
import statistics
from pathlib import Path


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def validate(root, source="/data/ycfeng/tmp/issue26-vllm-post-moe-bypass-20260910",
             compare_reference=True, warmups=10):
    result = {"client_checks": {}}
    for mode in ("clean/runtime/clean", "batch/runtime/batch"):
        clients = rows(root / mode / "client.jsonl")
        expected = {f"warmup:pf4096_dc1024:r{r}:{i}" for r in range(warmups) for i in range(100)}
        expected |= {f"pf4096_dc1024:{i}" for i in range(100)}
        assert len(clients) == (warmups + 1) * 100 and {r["request_id"] for r in clients} == expected
        assert all(r["prompt_tokens"] == 4096 and r["completion_tokens_observed"] == 1024 for r in clients)
        logs = rows(root / mode / "client.log")
        phases = [r for r in logs if "replay" in r]
        assert len(phases) == warmups + 1
        for index, phase in enumerate(phases):
            assert phase["replay"] == index and phase["completed_requests"] == 100
            prefix = (f"warmup:pf4096_dc1024:r{index}:"
                      if index < warmups else "pf4096_dc1024:")
            group = [r for r in clients if r["request_id"].startswith(prefix)]
            assert len(group) == 100
            assert min(r["dispatch_monotonic_s"] for r in group) >= phase["phase_start_monotonic_s"]
            assert max(r["dispatch_monotonic_s"] for r in group) < phase["phase_end_monotonic_s"]
            if index:
                assert phases[index-1]["phase_end_monotonic_s"] <= phase["phase_start_monotonic_s"]
                assert max(r["client_completion_time_ns"] for r in prior) <= min(r["request_arrival_wall_time_ns"] for r in group)
            prior = group
        assert logs[-1] == {"formal_requests": 100, "formal_unique_ids": 100}
        result["client_checks"][mode] = {"client_rows": len(clients), "warmup_rows": warmups * 100, "formal_rows": 100, "drain": "PASS", "phase_seconds": [p["phase_end_monotonic_s"]-p["phase_start_monotonic_s"] for p in phases]}
    run = root / "batch/runtime/batch"
    formal = [r for r in clients if r["request_id"] == "pf4096_dc1024:0"]
    assert len(formal) == 1
    request = formal[0]["response_id"] + "-0"
    assert request == "cmpl-pf4096_dc1024:0-0"
    selected = []
    for path in sorted(run.glob("server.batch.dp0.tp*.pp0.jsonl")):
        with path.open() as stream:
            matches = []
            for line in stream:
                row = json.loads(line)
                if any(rid == request and tokens == 4096 for rid, tokens in zip(
                        row["request_ids"], row["request_num_tokens"])):
                    matches.append(row)
        assert len(matches) == 1, (path, len(matches))
        row = matches[0]
        assert row["dp_rank"] == 0 and row["pp_rank"] == 0
        assert row["request_ids"] == [request] and row["request_num_tokens"] == [4096]
        assert row["batch_size"] == 1 and row["batch_num_tokens"] == row["batch_num_prefill_tokens"] == 4096
        assert row["batch_num_decode_tokens"] == 0 and not row["op_profile_selected"]
        selected.append(row)
    assert len(selected) == 4 and {r["tp_rank"] for r in selected} == set(range(4))
    assert len({r["batch_id"] for r in selected}) == 1
    values = sorted(r["batch_execution_time_ms"] for r in selected)
    result["new"] = {"rows": selected, "median_ms": statistics.median(values), "p90_ms": .3*values[2]+.7*values[3], "rank_max_ms": values[3], "spread_ms": values[3]-values[0]}
    manifest = json.loads((run/"mode_manifest.json").read_text())
    env = manifest["environment"]
    assert env["VLLM_MOE_UNIFORM_ROUTING"] == "1"
    assert env["PYTHONPATH"] == source
    assert not env.get("VLLM_FRONTIER_MOE_BOUNDARY_LOG_PATH")
    assert not env.get("VLLM_FRONTIER_DIAG_MOE_AR_MODE")
    assert not env.get("VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH")
    assert env.get("VLLM_FRONTIER_CUDA_PROFILER_CAPTURE", "0") == "0"
    assert env.get("VLLM_FRONTIER_TORCH_PROFILER_CAPTURE", "0") == "0"
    assert manifest["warmup_rounds"] == warmups and manifest["formal_requests"] == 100
    if compare_reference:
        reference = json.loads((root.parent/"h800-standard-replay-176000-repro-01/reproduction_summary.json").read_text())["new"]
        result["normal_reference"] = reference
        result["delta"] = {key: {"ms": result["new"][key]-reference[key], "percent": (result["new"][key]/reference[key]-1)*100} for key in ("median_ms", "p90_ms", "rank_max_ms", "spread_ms")}
    result["status"] = "PASS_STANDARD_IDENTITY_AND_DRAIN"
    result["limit"] = "One standard arm; P90 is over four ranks. An ablation delta requires an explicit paired reference and does not measure pure collective duration."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="/data/ycfeng/tmp/issue26-vllm-post-moe-bypass-20260910")
    parser.add_argument("--no-historical-reference", action="store_false", dest="compare_reference")
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()
    if args.warmups < 10:
        raise ValueError("--warmups must be at least 10")
    result = validate(args.run, args.source, args.compare_reference, args.warmups)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("new", "normal_reference")}, indent=2))
    print(json.dumps({key: value for key, value in result["new"].items() if key != "rows"}, indent=2))
