"""Classify the fidelity matrix differences of main a69a5213 (B29) against candidate_b6c28858."""

import collections
import json
import sys

EXPECTED = {
    ("config.json", "/request_ingress_delay_ms: missing in baseline"),
    ("config.json", "/kv_cache_transfer_config/request_handoff_delay_ms: missing in baseline"),
}

report = json.load(open(sys.argv[1]))
other = collections.Counter()
expected_per_case = collections.Counter()
for case in report["mismatched_cases"]:
    for difference in case["differences"]:
        key = (difference["artifact"], difference["detail"])
        if key in EXPECTED:
            expected_per_case[case["case_id"]] += 1
        else:
            other[(case["case_id"],) + key] += 1
print("cases compared", report["cases_compared"], "mismatched", len(report["mismatched_cases"]),
      "identical", len(report["identical_cases"]))
print("cases whose only differences are the two new config keys:",
      sum(1 for n in expected_per_case.values() if n == 2))
print("other differences:", sum(other.values()))
for key, n in sorted(other.items()):
    print("  ", n, key)
print("candidate-only failures", report["candidate_only_failures"],
      "cache names only in baseline", report["predictor_cache_files_only_in_baseline"],
      "only in candidate", report["predictor_cache_files_only_in_candidate"])
