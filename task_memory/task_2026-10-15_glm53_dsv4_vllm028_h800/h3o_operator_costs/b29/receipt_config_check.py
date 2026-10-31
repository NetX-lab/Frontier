"""Compare actual B29 baseline receipts/config dumps to the B28 base snapshots."""
import json
from pathlib import Path
import sys

NEW_FLAT = ("request_ingress_delay_ms", "analytical_kv_cache_transfer_config_request_handoff_delay_ms")

def config(path, run_root):
    raw = json.loads(path.read_text().replace(str(run_root), "<RUN_ROOT>"))
    def walk(value):
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()
                    if key not in ("request_ingress_delay_ms", "request_handoff_delay_ms")}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value
    return walk(raw)

def main(scratch, output):
    results = {}
    for case in ("p2_32", "p2_64"):
        root = scratch / f"{case}_baseline_native_env"
        receipt = json.loads((root / "receipt.json").read_text())
        base_root = scratch / f"b28_config_{case}_base"
        base_config = next(base_root.rglob("config.json"))
        actual_config = Path(receipt["metrics_dir"]) / "config.json"
        actual = json.loads(actual_config.read_text())
        assert actual["request_ingress_delay_ms"] == 0.0
        assert actual["kv_cache_transfer_config"]["request_handoff_delay_ms"] == 0.0
        assert config(base_config, base_root) == config(actual_config, root)
        base = json.loads((scratch / f"b28_snapshot_{case}_base.json").read_text()
                          .replace(str(base_root), "<RUN_ROOT>"))["changes"]
        changes = json.loads(json.dumps({key: receipt[key] for key in ("added_fields", "override_diff")})
                             .replace(str(root), "<RUN_ROOT>"))
        for field in NEW_FLAT:
            assert changes["added_fields"].pop(field) == 0.0
            assert field not in changes["override_diff"]
        assert changes == base
        results[case] = dict(status="PASS", receipt_status=receipt["status"],
                             config=str(actual_config), base_config=str(base_config),
                             normalization="run roots; remove only two new zero fields",
                             checked="actual nested config and complete receipt added_fields/override_diff")
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
