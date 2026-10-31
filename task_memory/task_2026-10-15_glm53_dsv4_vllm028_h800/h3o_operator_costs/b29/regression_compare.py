"""Compare B29/base pytest outcomes, retaining failures rather than excluding them."""
from collections import Counter
import json
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET

ROOTS = ("/data/ycfeng/Frontier/.worktrees/b29-base", "/data/ycfeng/Frontier/.worktrees/b29-ingress-delays")


def normalize(text, scratch):
    for root in ROOTS:
        text = text.replace(root, "<ROOT>")
    text = re.sub(r"(?:\.\./)+Frontier/\.worktrees/(?:b29-base|b29-ingress-delays)", "<ROOT>", text)
    text = text.replace(str(scratch), "<SCRATCH>")
    # Test-created temp names and log wall-clock stamps are run identifiers.
    text = re.sub(r"(?:regression_base_continue|regression_base_final|regression_candidate_final|regression_candidate)(?:_cwd)?", "<REGRESSION>", text)
    text = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<WALL_CLOCK>", text)
    # Object identities differ across interpreter processes; preserve exception values.
    text = re.sub(r"object at 0x[0-9a-fA-F]+", "object at <ADDRESS>", text)
    # Pytest truncates these redundant repr lines before roots are normalized.
    # Full stderr and return code remain above CompletedProcess's repr.
    text = re.sub(r"^.*CompletedProcess\(args=.*$", "<COMPLETED_PROCESS_REPR>", text, flags=re.MULTILINE)
    text = re.sub(r"^tmp_path_factory = TempPathFactory\(.*$", "tmp_path_factory = <TEMP_FACTORY_REPR>", text, flags=re.MULTILINE)
    # The same config log moved from base line 534 to candidate line 551.
    text = re.sub(r"config.py:(?:534|551)\]", "config.py:<OFFLINE_PDAF_LOG>]", text)
    text = re.sub(r"^AssertionError: =+$", "AssertionError: <BANNER>", text, flags=re.MULTILINE)
    # Long example diagnostics can clip the banner and the initial label.
    # Restore only the inspected header; the entire PYTHONPATH value remains.
    text = text.replace("AssertionError: <BANNER>\n  PYTHONPATH:", "AssertionError: PYTHONPATH:")
    text = re.sub(r"^AssertionError: (?:THONPATH|HONPATH):", "AssertionError: PYTHONPATH:", text)
    # Unittest shortens this environment dict before printing; the sole raw
    # difference is the length of the base/candidate PYTHONPATH root.
    text = re.sub(r"/data/ycfeng/Frontier/\.worktrees/b29-(?:bas|ing)\[(?:490|530) chars\]", "<ROOT>[PATH chars]", text)
    text = re.sub(r"Diff is (?:656|696) characters long", "Diff is <ROOT_LENGTH_DEPENDENT> characters long", text)
    return text


def node_id(node):
    parts = node.attrib["classname"].split(".")
    if not node.attrib["classname"]:
        return node.attrib["name"].replace(".", "/") + ".py"
    i = next(i for i, part in enumerate(parts) if part.startswith("test_"))
    return "/".join(parts[:i+1]) + ".py::" + "::".join(parts[i+1:] + [node.attrib["name"]])


def outcomes(path, scratch):
    result = {}
    for node in ET.parse(path).iter("testcase"):
        children = [child for child in node if child.tag in ("failure", "error", "skipped")]
        assert len(children) <= 1
        child = children[0] if children else None
        status = {"failure": "FAIL", "error": "ERROR", "skipped": "SKIP"}[child.tag] if child is not None else "PASS"
        message = child.attrib.get("message", "") if child is not None else ""
        if child is not None and child.tag == "error":
            message += "\n" + (child.text or "")
        result[node_id(node)] = dict(status=status, message=message,
                                    normalized_message=normalize(message, scratch),
                                    detail=child.text if child is not None else None)
    return result


def main(scratch, output):
    finalized = (scratch / "regression_base_final.xml").exists() and (scratch / "regression_candidate_final.xml").exists()
    base_label, candidate_label = (("regression_base_final", "regression_candidate_final") if finalized
                                   else ("regression_base_continue", "regression_candidate"))
    base = outcomes(scratch / f"{base_label}.xml", scratch)
    candidate = outcomes(scratch / f"{candidate_label}.xml", scratch)
    focused = outcomes(scratch / "focused_r3.xml", scratch)
    # During the initial run, the new module ran separately; final selection
    # includes it directly. Never replace a final-run result with a focused one.
    for key, value in focused.items():
        if key.startswith("tests/unit/test_serving_path_delays.py::"):
            candidate.setdefault(key, value)
    common = sorted(set(base) & set(candidate))
    added, removed = sorted(set(candidate) - set(base)), sorted(set(base) - set(candidate))
    differences = [{"node": key, "base": base[key], "candidate": candidate[key]} for key in common
                   if (base[key]["status"], base[key]["normalized_message"]) !=
                      (candidate[key]["status"], candidate[key]["normalized_message"])]
    expected = {key for key in focused if key.startswith("tests/unit/test_serving_path_delays.py::") or any(
        f"::{name}" in key for name in ("test_serving_delay_options_and_receipt_diff", "test_field_duplicates_and_option_conflicts_fail",
                                      "test_field_variants_record_backend_and_override_diff", "test_field_override_is_written_in_receipt"))}
    assert set(added) == expected, ("unexpected added nodes", set(added) ^ expected)
    assert not removed, removed
    result = dict(base_label=base_label, candidate_label=candidate_label, base_counts=dict(Counter(row["status"] for row in base.values())),
                  candidate_counts=dict(Counter(row["status"] for row in candidate.values())), common_nodes=len(common),
                  added_nodes=added, removed_nodes=removed, differences=differences,
                  common_nonpasses={key: {"base": base[key], "candidate": candidate[key]} for key in common if base[key]["status"] != "PASS"},
                  normalizations="worktree roots; scratch/run identifiers; wall clocks; object addresses; redundant truncated CompletedProcess and TempPathFactory reprs; the moved offline-PDAF config log line; clipped banners/PYTHONPATH labels; unittest truncated PYTHONPATH root lengths. Raw messages/details retained; exception and assertion values preserved.")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("common_nonpasses", "differences")}, indent=2))
    print("node/message differences", len(differences))
    assert not differences, [row["node"] for row in differences]


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
