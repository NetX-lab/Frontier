"""Inspect retained native G0-B results without granting formal admission."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def inspect(package, raw):
    config = json.loads((package / "config.json").read_text())
    platform = json.loads((package / "platform_terminal_receipt.json").read_text())
    owners = config["replica"]
    tp = config["llm"]["tensor_parallel_size"]
    ranks = owners * tp
    checks = {
        "platform_succeeded": platform["status"]["phase"] == "Succeeded",
        "personal_creator": platform["creator"] == "i-fengyicheng",
        "container_count": len(platform["replicas"]) == owners,
        "container_exit_zero": all(
            c["state"].get("terminated", {}).get("exitCode") == 0
            and c["restartCount"] == 0
            for replica in platform["replicas"] for c in replica["containers"]
        ),
    }
    files = sorted(raw.glob("nodes/*/result.json"))
    checks["result_count"] = len(files) == owners
    observed_owners, observed_ranks, records, evidence = [], [], [], []
    for file in files:
        content = file.read_bytes()
        result = json.loads(content)
        owner = result["request_owner"]
        observed_owners.append(owner)
        checks[f"owner_{owner}_config"] = result["configuration"] == config
        checks[f"owner_{owner}_native"] = (
            result["passed"] and not result["observation_errors"]
            and result["stage"] == "shutdown_returned"
            and all(result["checks"].values())
            and (file.parent / "drained.json").is_file()
        )
        responses = result["responses"]
        checks[f"owner_{owner}_counts"] = (
            [r["prompt_tokens"] for r in responses] == config["prompt_lengths"][owner]
            and all(len(r["token_ids"]) == config["decode_tokens"]
                    and r["logprobs_finite"] for r in responses)
        )
        checks[f"owner_{owner}_inventory"] = len(result["inventory"]) == tp
        for entry in result["inventory"]:
            groups = entry["groups"]
            rank = groups["world"]["rank"]
            observed_ranks.append(rank)
            checks[f"rank_{rank}_groups"] = (
                groups["world"]["ranks"] == list(range(ranks))
                and groups["ep"]["ranks"] == list(range(ranks))
                and groups["tp"]["world_size"] == tp
                and groups["dp"]["world_size"] == owners
                and groups["tp"]["ranks"] == list(range(owner * tp, (owner + 1) * tp))
                and groups["dp"]["ranks"] == list(range(rank % tp, ranks, tp))
                and groups["tp"]["rank_in_group"] == rank % tp
                and groups["dp"]["rank_in_group"] == owner == rank // tp
                and all(group["rank"] == rank for group in groups.values())
                and entry["request_owner"] == owner
                and entry["host"] == file.parent.name
            )
            variants = Counter(json.dumps(v, sort_keys=True)
                               for v in entry["initialization"]["backends"].values())
            v2 = entry.get("v2_runtime")
            records.append({
                "host": entry["host"], "owner": owner, "rank": rank,
                "runner": entry["model_runner"],
                "effective_graph": entry["effective_graph"],
                "resolved": entry["initialization"]["resolved"],
                "blocks": entry["cache"]["num_blocks"],
                "cache_group_count": len(entry["cache"]["kv_cache_groups"]),
                "memory": entry["memory"],
                "backend_variants": dict(variants),
                "v2_capture_count": len(v2["captured_descriptors"]) if v2 else None,
                "v2_replay_coverage": v2["replay_coverage"] if v2 else None,
            })
        evidence.append({"path": str(file), "sha256": hashlib.sha256(content).hexdigest()})
    checks["all_owners_once"] = sorted(observed_owners) == list(range(owners))
    checks["all_ranks_once"] = sorted(observed_ranks) == list(range(ranks))
    return {
        "scope": "Retained native G0-B business and runtime inventory inspection",
        "job": platform["job"], "passed": all(checks.values()), "checks": checks,
        "records": records, "evidence": evidence,
        "formal_admission": False,
        "limitations": [
            "Capture/configuration is not direct evidence of serving replay returns.",
            "Allocator high-water values are not independent stage peaks.",
            "This is a four-request qualification run, not the formal three-round case.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = inspect(args.package, args.raw)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]),
                      "failed": [k for k, v in result["checks"].items() if not v],
                      "formal_admission": result["formal_admission"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
