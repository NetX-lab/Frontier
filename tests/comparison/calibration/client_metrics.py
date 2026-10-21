"""Client output-token endpoints for supported canonical native artifact formats."""

import json
import math
from pathlib import Path

import pandas as pd

from tests.comparison.calibration.e2e_metrics_gap import require_written_after
from tests.comparison.calibration.pd_metrics import read_jsonl

TTFT_SEMANTIC = "request_issue_to_first_output_token"
NATIVE_CLIENT_FORMATS = {
    "vllm028": dict(directory="run", summary="replay_summary.json", repetitions="canonical_repetitions",
                    status=0, stream="stream_done", output="output_token_count", adapter="0.28.0",
                    provenance="git"),
    "step5": dict(directory="replay", summary="canonical_summary.json", repetitions=None,
                  status="CLIENT_WORKLOAD_VERIFIED", stream="stream_completed", output="generated_token_count",
                  adapter="step5", provenance="image"),
}


def endpoint_row(identity, arrival, first, last, prompt, output, **diagnostics):
    if (not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (arrival, first, last))
            or not 0 <= arrival <= first <= last or first == 0 or output <= 1):
        raise ValueError(f"Missing or invalid client output-token endpoints: {identity}")
    return dict(request_id=identity, arrival_s=arrival, completion_s=last,
                ttft_ms=(first - arrival) * 1000, tpot_ms=(last - first) * 1000 / (output - 1),
                request_e2e_time_ms=(last - arrival) * 1000,
                request_num_prefill_tokens=prompt, request_num_decode_tokens=output,
                ttft_semantic=TTFT_SEMANTIC, **diagnostics)


def native_client_rows(run_dir: Path, workload: tuple, request_ids: dict, source_format="vllm028") -> dict:
    """Keep repeat-qualified transport IDs, joining only by the stable template ID."""
    workload, formal_rows, warmup_rows = workload
    if request_ids["rows"] != formal_rows:
        raise ValueError("Pinned canonical formal identity/arrival mismatch")
    source = NATIVE_CLIENT_FORMATS[source_format]
    run = run_dir / source["directory"]
    manifest = run_dir / "run_manifest.json"
    if source["provenance"] == "git":
        metadata = json.loads(manifest.read_text())
        if (metadata.get("mode") != "clean" or metadata.get("groundtruth_tree_dirty") is not False
                or not metadata.get("groundtruth_commit")):
            raise ValueError("Client metrics require recorded clean source provenance")
        for name in ("client_requests.jsonl", source["summary"]):
            require_written_after(run / name, manifest)
    summary = json.loads((run / source["summary"]).read_text())
    repetitions = summary[source["repetitions"]] if source["repetitions"] else summary
    if (source["provenance"] == "git" and (
            summary.get("vllm_version") != source["adapter"] or summary.get("mode") != "clean")
            or summary.get("status") != source["status"] or not repetitions.get("client_workload_verified")
            or repetitions.get("observation_errors")):
        raise ValueError("Incomplete or unstable native client repetitions")
    if source["provenance"] == "image" and summary.get("clock") != (
            "prefill-head Linux monotonic; proxy and client on the same host"):
        raise ValueError("Unbound native client clock")
    records = read_jsonl(run / "client_requests.jsonl")
    repeats = workload["repetitions"]["repeat_indices"]
    if [r["repeat"] for r in repetitions["repetitions"]] != repeats:
        raise ValueError("Native repetition population mismatch")
    if source["provenance"] == "image" and not all(
            r.get("client_workload_verified") is True for r in repetitions["repetitions"]):
        raise ValueError("Unverified individual native repetition")
    formal = {r["request_id"]: r for r in request_ids["rows"] if r["role"] == "formal"}
    rule = workload["repetitions"]["runtime_request_id_rule"]
    if len({r["request_id"] for r in records}) != len(records):
        raise ValueError("Duplicate native transport IDs")
    boundaries = repetitions["idle_boundaries"]
    expected_boundaries = ["before_repeat01_warmup"] + [
        f"after_repeat{repeat:02d}_{phase}" for repeat in repeats for phase in ("warmup", "formal")]
    if ([b["boundary"] for b in boundaries] != expected_boundaries
            or not all(b["verified"] for b in boundaries)):
        raise ValueError("Missing independently warmed idle boundaries")
    result, previous_end = {}, -math.inf
    for repeat in repeats:
        selected = [r for r in records if r["repeat"] == repeat]
        rows = []
        for phase, templates in (("warmup", warmup_rows), ("formal", formal_rows)):
            arrivals = {r["request_id"]: r["arrived_at"] for r in templates}
            phase_rows = [r for r in selected if r["segment"] == phase and r["role"] == phase]
            if len(phase_rows) != len(templates):
                raise ValueError("Native warmup/formal population mismatch")
            phase_clock = None
            if source["provenance"] == "image":
                phase_clock = next(item for item in repetitions["repetitions"] if item["repeat"] == repeat)["phases"][phase]
                if (phase_clock.get("planned_requests") != len(templates)
                        or phase_clock.get("recorded_requests") != len(templates)
                        or phase_clock.get("failed_request_ids") != []):
                    raise ValueError("Native phase summary population mismatch")
            for r in phase_rows:
                template = r["template_request_id"]
                expected = workload["lengths"]
                if (r["request_id"] != rule.format(request_id=template, repeat=repeat)
                        or template not in arrivals or r.get("arrived_at") != arrivals[template]
                        or r["status"] != "completed" or r["http_status"] != 200
                        or r.get("error") or r.get("stream_error") or r["validation_errors"]
                        or not r.get(source["stream"])
                        or r["finish_reason"] != "length"
                        or r["num_prefill_tokens"] != expected["prompt_tokens_per_request"]
                        or r["num_decode_tokens"] != expected["output_tokens_per_request"]
                        or r.get(source["output"]) != r["num_decode_tokens"]
                        or any(r["usage"].get(key) != value for key, value in dict(
                            prompt_tokens=r["num_prefill_tokens"], completion_tokens=r["num_decode_tokens"],
                            total_tokens=r["num_prefill_tokens"] + r["num_decode_tokens"]).items())):
                    raise ValueError(f"Invalid native client identity, scheduled arrival or tokens: {r['request_id']}")
                row = endpoint_row(template, r["dispatch_monotonic"], r.get("first_token_monotonic"),
                                   r.get("last_token_monotonic"), r["usage"]["prompt_tokens"],
                                   r[source["output"]], transport_request_id=r["request_id"], repeat=repeat)
                if source["provenance"] == "image":
                    origin = phase_clock["origin_monotonic"]
                    response = r.get("response_monotonic")
                    if (not isinstance(origin, (int, float)) or not math.isfinite(origin)
                            or r["dispatch_monotonic"] < origin + r["arrived_at"]
                            or not isinstance(response, (int, float)) or not math.isfinite(response)
                            or response < r["last_token_monotonic"]
                            or r.get("accepted_prompt_token_count") != r["num_prefill_tokens"]
                            or not r.get("prompt_sha256")
                            or r.get("accepted_prompt_sha256") != r["prompt_sha256"]):
                        raise ValueError("Step prompt identity, phase clock or response timestamp mismatch")
                for metric_source, target in (("ttft_ms", "ttft_ms"), ("tpot_ms", "tpot_ms"),
                                       ("e2e_ms", "request_e2e_time_ms")):
                    if not math.isclose(r[metric_source], row[target], abs_tol=1e-3, rel_tol=0):
                        raise ValueError("Client metric provenance disagrees with token timestamps")
                if phase == "formal":
                    rows.append(row)
            start = min(r["dispatch_monotonic"] for r in phase_rows)
            end = max(r["last_token_monotonic"] for r in phase_rows)
            if start <= previous_end:
                raise ValueError("Overlapping native warmup/formal windows")
            previous_end = end
        if {r["request_id"] for r in rows} != set(formal):
            raise ValueError("Native formal template identity mismatch")
        result[f"repeat{repeat:02d}"] = pd.DataFrame(rows)
    if len(records) != len(repeats) * (len(formal) + workload["warmup"]["num_requests_per_repetition"]):
        raise ValueError("Unclassified native records")
    return result


def frontier_client_rows(run_dir: Path, trace_rows: pd.DataFrame) -> pd.DataFrame:
    [path] = (run_dir / "metrics").rglob("metrics_ground_truth.jsonl")
    require_written_after(path, path.with_name("config.json"))
    mapping = {str(r["frontier_request_id"]): (r["request_id"], r["arrived_at"]) for r in trace_rows.to_dict("records")}
    rows = []
    for r in read_jsonl(path):
        if r["event_type"] != "request_completion":
            continue
        identity, arrival = mapping.get(str(r["request_id"]), (f"unmapped:{r['request_id']}", None))
        if r["arrived_at"] != arrival:
            raise ValueError(f"Simulator arrival disagrees with pinned trace: {identity}")
        rows.append(endpoint_row(identity, r["arrived_at"], r.get("first_decode_token_completed_at"),
                                 r["completed_at"], r["num_prefill_tokens"], r["num_decode_tokens"],
                                 internal_ttft_ms=r["ttft_ms"]))
    if not rows:
        raise ValueError("No completed simulator client records")
    return pd.DataFrame(rows)
