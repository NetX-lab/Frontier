"""Capture first-stage prediction branches, with estimator fitting opt-in only."""

import argparse
from collections import Counter
from functools import wraps
import json
import linecache
from pathlib import Path
import subprocess
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class FirstStageReached(Exception):
    """Stop the diagnostic before the first stage-completion handler."""


def configure_fit_policy(allow_fit):
    """Enforce the process-local fit policy and record completed fit calls."""
    policy = {"allowed": allow_fit, "successful_calls_in_process": 0}
    original_fit = RandomForestRegressor.fit

    @wraps(original_fit)
    def audited_fit(estimator, *args, **kwargs):
        if not allow_fit:
            raise RuntimeError("This audit requires existing caches; estimator fitting is forbidden.")
        result = original_fit(estimator, *args, **kwargs)
        policy["successful_calls_in_process"] += 1
        return result

    RandomForestRegressor.fit = audited_fit
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fit", action="store_true",
                        help="Allow RF fitting for an explicitly prepared fresh-cache run.")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    settings["metrics_config_output_dir"] = str(args.output / "metrics")
    queries, frames, endpoint = [], {}, {}

    fit_policy = configure_fit_policy(args.allow_fit)

    def record(frame, event, result):
        filename = frame.f_code.co_filename
        if "/execution_time_predictor/" not in filename:
            return
        name = frame.f_code.co_name
        local = frame.f_locals
        if event == "call" and name in {"_train_single_model", "_train_model"}:
            frames[local["model_name"]] = {
                "df": local["df"].copy(),
                "features": list(local["feature_cols"]),
                "target": local["target_col"],
                "context": local.get("training_context"),
            }
        if event != "return" or name not in {
            "_get_prediction_for_features", "_get_on_demand_prediction"
        } or result is None:
            return
        model = local.get("model")
        key = local.get("feature_key")
        exact = local.get("exact_lookup") or {}
        source_line = linecache.getline(filename, frame.f_lineno).strip()
        if "normalized_exact_value" in source_line:
            branch = "measured_exact" if key in exact else "materialized_table"
        elif "cached" in source_line:
            branch = "runtime_cache"
        elif source_line == "return prediction":
            branch = "estimator_predict"
        else:
            branch = "delegated"
        queries.append({
            "model": local["model_name"],
            "method": name,
            "features": local.get("normalized_features", local["features"]),
            "feature_key": key,
            "branch": branch,
            "prediction_ms": float(result),
            "estimator": type(model).__name__ if model is not None else None,
            "return_file": filename,
            "return_line": frame.f_lineno,
            "return_source": source_line,
        })

    from frontier.events.batch_stage_end_event import BatchStageEndEvent

    def stop_at_first_stage(event, *_args):
        endpoint.update(time_s=event.time, batch_id=event._batch.id,
                        request_ids=[r.id for r in event._batch.requests])
        raise FirstStageReached()

    BatchStageEndEvent.handle_event = stop_at_first_stage
    sys.argv = [sys.argv[0]]
    for key, value in settings.items():
        if isinstance(value, bool):
            sys.argv.append(("--" if value else "--no-") + key)
        else:
            sys.argv.extend(["--" + key, str(value)])
    from frontier.main import main as run

    sys.setprofile(record)
    try:
        run()
    except FirstStageReached:
        pass
    finally:
        sys.setprofile(None)
    if not endpoint or not queries:
        raise AssertionError("The first stage or its prediction queries were not observed.")

    counts = Counter(json.dumps(q, sort_keys=True) for q in queries)
    unique = []
    for serialized, count in counts.items():
        query = json.loads(serialized)
        query["call_count"] = count
        training = frames.get(query["model"])
        if training:
            df = training["df"]
            mask = pd.Series(True, index=df.index)
            for key, value in query["features"].items():
                mask &= df[key] == value
            rows = df.loc[mask & df[training["target"]].notna()].copy()
            rows.insert(0, "dataframe_index", rows.index)
            query["matching_filtered_rows"] = json.loads(
                rows.to_json(orient="records", double_precision=15)
            )
            query["filtered_row_count"] = len(df)
            query["target_column"] = training["target"]
            query["training_context"] = training["context"]
        unique.append(query)
    receipt = {
        "status": "PASS_BOUNDED_PREDICTOR_QUERY_AUDIT",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version,
        "config_source": str(args.config.resolve()),
        "settings": settings,
        "estimator_fitting": (
            "allowed; inspect actual fit calls and configured cache provenance"
            if args.allow_fit else "forbidden; existing caches reused for attribution only"
        ),
        "rf_fit_policy": fit_policy,
        "stop_boundary": endpoint,
        "queries": unique,
        "total_query_calls": len(queries),
        "training_frame_models": sorted(frames),
    }
    (args.output / "query_receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps({"status": receipt["status"], "unique_queries": len(unique),
                      "calls": len(queries), "endpoint": endpoint}))


if __name__ == "__main__":
    main()
