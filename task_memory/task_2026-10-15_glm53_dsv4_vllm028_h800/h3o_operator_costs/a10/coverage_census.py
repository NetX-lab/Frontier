"""Diagnostic census of the V4 attention workloads that an A10 replay requests.

Usage: coverage_census.py <case> <output root>   # p2_32 or p2_64

The census process admits every measured workload group so the replay runs past
B19's joint admission, and records each distinct attention request with the
measured group ranges of the models it uses. ``check`` lists the requests that
fall outside those ranges. Requests after the first such request see fitted
values outside measured data, so the census lists the runtime domain, not
A10 timings. Not for simulator use.

Usage: coverage_census.py check <output root>
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

from frontier.execution_time_predictor import dsv4_predictor
from frontier.execution_time_predictor.dsv4_predictor import (
    DeepSeekV4AttentionPredictor, dsv4_feature_columns, dsv4_model_key, dsv4_model_name,
)

RECORDS = "attention_requests.jsonl.gz"
RANGES = "measured_group_ranges.json"
MISSES = "coverage_misses.json"
FLUSH_EVERY = 1000


class _AdmitAll(dict):
    def get(self, key, default=None):
        return {}


class Census:
    def __init__(self, root: Path):
        self.root = root
        self.counts = Counter()
        self.ranges = {}
        self.unflushed = 0

    def install(self):
        census = self
        original_init = DeepSeekV4AttentionPredictor.__init__
        original_scope_times = DeepSeekV4AttentionPredictor.scope_times

        def init(predictor, models):
            original_init(predictor, models)
            for name, model in predictor._models.items():
                if not isinstance(model._frontier_dsv4_workload_group_ranges, _AdmitAll):
                    census.ranges[name] = {str(rows): ranges for rows, ranges
                                           in model._frontier_dsv4_workload_group_ranges.items()}
                    model._frontier_dsv4_workload_group_ranges = _AdmitAll()

        def scope_times(predictor, features, *, variant, graph_runtime_mode, plans_tile_schedule, scopes):
            record = (variant, graph_runtime_mode, features.phase, bool(plans_tile_schedule),
                      tuple(sorted(scopes)), tuple(sorted(features.values.items())))
            census.counts[record] += 1
            if census.counts[record] == 1:
                census.unflushed += 1
                if census.unflushed >= FLUSH_EVERY:
                    census.write()
            return original_scope_times(predictor, features, variant=variant,
                                        graph_runtime_mode=graph_runtime_mode,
                                        plans_tile_schedule=plans_tile_schedule, scopes=scopes)

        DeepSeekV4AttentionPredictor.__init__ = init
        DeepSeekV4AttentionPredictor.scope_times = scope_times

    def write(self):
        with gzip.open(self.root / RECORDS, "wt") as stream:
            for (variant, mode, phase, planner, scopes, values), count in self.counts.items():
                stream.write(json.dumps(dict(variant=variant, graph_runtime_mode=mode, phase=phase,
                                             plans_tile_schedule=planner, scopes=scopes,
                                             values=dict(values), count=count)) + "\n")
        (self.root / RANGES).write_text(json.dumps(self.ranges, sort_keys=True) + "\n")
        self.unflushed = 0


def check(root: Path):
    ranges = json.loads((root / RANGES).read_text())
    misses = Counter()
    examples = {}
    total = Counter()
    with gzip.open(root / RECORDS, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            values = record["values"]
            for scope in record["scopes"]:
                key = dsv4_model_key(scope, record["variant"], record["graph_runtime_mode"],
                                     record["phase"], record["plans_tile_schedule"])
                name = dsv4_model_name(key, dsv4_predictor.TARGET_TIME)
                group = ranges[name].get(str(float(values["padded_rows"])))
                total[name] += record["count"]
                if group is None:
                    reasons = [f"padded_rows={values['padded_rows']} not measured"]
                else:
                    reasons = [f"{column}={values[column]} outside [{lower}, {upper}]"
                               for column, (lower, upper) in group.items()
                               if not lower <= values[column] <= upper]
                for reason in reasons:
                    miss = (name, values["padded_rows"], reason.split("=")[0])
                    misses[miss] += record["count"]
                    examples.setdefault(miss, {"reason": reason, "values": values,
                                               "columns": dsv4_feature_columns(scope, record["phase"])})
    rows = [dict(model=name, padded_rows=rows, column=column, forwards=count, calls_of_model=total[name],
                 example=examples[(name, rows, column)])
            for (name, rows, column), count in sorted(misses.items())]
    (root / MISSES).write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{len(rows)} (model, padded_rows, column) misses written to {root / MISSES}")


def main(case: str, root: Path):
    import a10_driver

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    census = Census(root)
    census.install()
    try:
        a10_driver.main(case, root, observe=False,
                        fields={"metrics_config_run_id": f"a10_{case}_coverage_census"})
    finally:
        census.write()


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(Path(sys.argv[2]))
    else:
        main(sys.argv[1], Path(sys.argv[2]))
