#!/usr/bin/env python3
"""Cross-check the two generation08 copies of the same cloud run.

The comparison consumed the SSH-route copy because the planned CPU copy job sat in
ImagePullBackOff (issues.md I-014). That job later pulled the image and wrote the
planned copy, so the native bytes can be checked against the originally planned route.

Each copy carries two files written by its own copy tool, which record the route and
its timing and therefore differ by design:

  cloud_inventory.json   records the mount view of the cloud root
  download_receipt.json  records the route, its timestamps and its script hash

Everything else is job output and must be identical. Read-only; writes one JSON report.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RUN_TAG = "dsv4-h200-p2-64-formal-20261107-08"
DOWNLOAD_ROOT = Path("/data/ycfeng/tmp/glm53_dsv4/h200/downloads")
COPIES = {
    "platform_cpu_rjob": DOWNLOAD_ROOT / f"formal_p2_64_20261107_08/{RUN_TAG}",
    "ssh_readonly": DOWNLOAD_ROOT / f"formal_p2_64_20261107_08_ssh/{RUN_TAG}",
}
COPY_TOOL_FILES = {"cloud_inventory.json", "download_receipt.json"}
CONSUMED_BY_COMPARISON = "ssh_readonly"


def inventory(root: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        out[rel] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return out


def joint_sha256(files: dict[str, dict[str, object]]) -> str:
    lines = [f"{rel}:{meta['sha256']}" for rel, meta in sorted(files.items())]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def main() -> int:
    missing = [name for name, root in COPIES.items() if not root.is_dir()]
    if missing:
        print(f"COPY_CROSSCHECK_BLOCKED missing={missing}", file=sys.stderr)
        return 2

    full = {name: inventory(root) for name, root in COPIES.items()}
    payload = {
        name: {rel: meta for rel, meta in files.items() if rel not in COPY_TOOL_FILES}
        for name, files in full.items()
    }

    left, right = COPIES
    only_left = sorted(set(payload[left]) - set(payload[right]))
    only_right = sorted(set(payload[right]) - set(payload[left]))
    differing = sorted(
        rel for rel in set(payload[left]) & set(payload[right])
        if payload[left][rel] != payload[right][rel]
    )
    identical = not (only_left or only_right or differing)

    report = {
        "status": "PAYLOAD_IDENTICAL" if identical else "PAYLOAD_DIFFERS",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": RUN_TAG,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "consumed_by_comparison": CONSUMED_BY_COMPARISON,
        "copies": {
            name: {
                "root": str(COPIES[name]),
                "files_total": len(full[name]),
                "payload_files": len(payload[name]),
                "payload_bytes": sum(int(m["bytes"]) for m in payload[name].values()),
                "payload_joint_sha256": joint_sha256(payload[name]),
                "copy_tool_files": {
                    rel: full[name][rel] for rel in sorted(COPY_TOOL_FILES) if rel in full[name]
                },
            }
            for name in COPIES
        },
        "payload_diff": {
            f"only_in_{left}": only_left,
            f"only_in_{right}": only_right,
            "hash_differing": differing,
        },
        "excluded_by_design": sorted(COPY_TOOL_FILES),
    }

    out_path = Path(__file__).with_name("copy_route_crosscheck.json")
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"{report['status']} report={out_path}")
    for name in COPIES:
        copy = report["copies"][name]
        print(
            f"  {name}: payload {copy['payload_files']} files {copy['payload_bytes']} bytes"
            f" joint {copy['payload_joint_sha256'][:16]}"
        )
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
