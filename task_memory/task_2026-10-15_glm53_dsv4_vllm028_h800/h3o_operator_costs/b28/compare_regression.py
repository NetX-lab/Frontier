"""Save node-level outcomes and raw/normalized failure messages for B28's pair."""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

RECORDS = Path(__file__).resolve().parent
ROOT = RECORDS.parents[3]
BASE = ROOT.parent / "b28-base"
SCRATCH = Path("/data/ycfeng/tmp/codex-implement/d684ea95-b7ff-4a6d-8fec-63b038641a41/b28")


def normalize(text):
    # Preserve raw messages in the receipt; remove only per-run identifiers.
    for root in (ROOT, BASE):
        text = text.replace(str(root), "<root>")
    text = re.sub(re.escape(str(SCRATCH)) + r"/tmp/[^/\s]+", "<basetemp>", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "<address>", text)
    text = re.sub(r"batch id = \d+", "batch id = <batch>", text)
    text = re.sub(r"(KeyError: \('(?:decode|prefill|monolithic)', )\d+", r"\1<batch>", text)
    text = re.sub(r"(\('(?:decode|prefill|monolithic)', )\d+(, 0\))", r"\1<batch>\2", text)
    return text


def load(label):
    tree = ET.parse(SCRATCH / f"{label}.xml")
    nodes = {}
    for case in tree.iter("testcase"):
        classname, name = case.attrib.get("classname", ""), case.attrib["name"]
        if classname.startswith("tests.unit."):
            parts = classname.split(".")
            node = "/".join(parts[:3]) + ".py::" + "::".join([*parts[3:], name])
        else:
            node = classname + "::" + name
        status, message, detail = "passed", None, None
        for kind in ("failure", "error", "skipped"):
            element = case.find(kind)
            if element is not None:
                status = {"failure": "failed", "error": "error", "skipped": "skipped"}[kind]
                message, detail = element.attrib.get("message", ""), element.text or ""
                break
        assert node not in nodes, (label, node)
        nodes[node] = dict(status=status, message=message, detail=detail,
                           normalized_message=normalize(message) if message else message)
    return nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_label")
    parser.add_argument("candidate_label")
    parser.add_argument("--output", default="regression_comparison.json")
    args = parser.parse_args()
    base, candidate = load(args.base_label), load(args.candidate_label)
    common = base.keys() & candidate.keys()
    failures = []
    for node in sorted(common):
        old, new = base[node], candidate[node]
        if old["status"] in ("failed", "error") or new["status"] in ("failed", "error"):
            failures.append(dict(node=node, base=old, candidate=new,
                                 raw_message_equal=old["message"] == new["message"],
                                 normalized_message_equal=old["normalized_message"] == new["normalized_message"]))
    record = dict(base_label=args.base_label, candidate_label=args.candidate_label,
                  base_counts=dict(Counter(row["status"] for row in base.values())),
                  candidate_counts=dict(Counter(row["status"] for row in candidate.values())),
                  added=[dict(node=node, outcome=candidate[node]) for node in sorted(candidate.keys() - base.keys())],
                  removed=[dict(node=node, outcome=base[node]) for node in sorted(base.keys() - candidate.keys())],
                  outcome_changes=[dict(node=node, base=base[node], candidate=candidate[node])
                                   for node in sorted(common) if base[node]["status"] != candidate[node]["status"]],
                  common_failure_details=failures,
                  candidate_only_failures=[dict(node=node, **row) for node, row in sorted(candidate.items())
                                           if row["status"] in ("failed", "error")
                                           and (node not in base or base[node]["status"] != row["status"])],
                  normalization="root and scratch paths, memory addresses and generated batch IDs only; raw messages retained")
    (RECORDS / args.output).write_text(json.dumps(record, indent=2) + "\n")
    short = {key: value for key, value in record.items() if key not in ("common_failure_details", "added", "removed")}
    short.update(added_nodes=len(record["added"]), removed_nodes=len(record["removed"]),
                 shared_failures=len(failures), differing_raw_messages=sum(not f["raw_message_equal"] for f in failures),
                 differing_normalized_messages=sum(not f["normalized_message_equal"] for f in failures))
    print(json.dumps(short, indent=2))


if __name__ == "__main__":
    main()
