# Canonical client metrics

`run_gate.py normalize` and `gate` support the canonical client gate without
changing the legacy internal-TTFT gates. Select the source format through
`cases.<case>.native_client_source` (`vllm028` by default, or `step5`). The
source-format registry lives in `calibration/client_metrics.py`; model names do
not select reader behavior.

## Step command

From the repository root:

```bash
export PYTHONPATH=$PWD
PY=/data/ycfeng/envs/frontier-py310/bin/python
CELL=step5_canonical/fixed6144_1024_req128_qps1
MATRIX=tests/comparison/e2e_gate/cases/step5_canonical.json
$PY tests/comparison/e2e_gate/run_gate.py normalize --matrix "$MATRIX" \
  --cell "$CELL" --artifacts-root "$PWD" \
  --vllm-run-dir "$NATIVE_RETURNED_EVIDENCE" \
  --frontier-run "measured=$FRONTIER_RUN" --output-dir "$NEW_ROWS"
$PY tests/comparison/e2e_gate/run_gate.py gate --matrix "$MATRIX" \
  --cell "$CELL" --rows "$NEW_ROWS" --output "$NEW_READING"
```

The checked-in matrix intentionally has no run-check pin, routing `UNSET`, and
coverage `UNVERIFIED`. It computes diagnostics, never qualifies synthetic or
unadmitted ground truth. For a formal reading, the case owner freezes a new
task-local matrix using the same schema, adds a SHA-256 pin under `cell.run_check`
to the actual `analysis/run_check_status.groundtruth_clean.json`, and records the
real routing/coverage disposition. Never edit a published matrix or fabricate
a passing receipt. `--artifacts-root` resolves relative pins, while absolute pins
are also supported by the existing CLI.

## Actual input formats

| Meaning | vLLM 0.28 source | Step source |
| --- | --- | --- |
| Client rows | `run/client_requests.jsonl` | `replay/client_requests.jsonl` |
| Summary | `run/replay_summary.json` | `replay/canonical_summary.json` |
| Repetitions | `canonical_repetitions.repetitions` | `repetitions` |
| Success | integer `status=0`, `mode=clean`, `vllm_version=0.28.0` | `status=CLIENT_WORKLOAD_VERIFIED` |
| Complete stream / exact native count | `stream_done`, `output_token_count` | `stream_completed`, `generated_token_count` |
| Identity | original clean `run_manifest.json` plus bound run check | original staged `config.json`, worker receipts and actual runtime evidence plus bound run check |

Step's adapter label is `step5`; its installed package version is
`0.1.dev22912+g8224bb82f`, not vLLM 0.28.0. The summary has no native Git-clean
claim. Copy the original staged `config.json` byte-for-byte into the returned
evidence root beside `replay/` and `nodes/`; do not author a replacement manifest.

Step rows must include all three repeats, every warmup and formal request, stable
template and transport IDs, scheduled arrivals, status/validation fields, accepted
prompt count/hash, final native usage counts, and actual dispatch/first-token/
last-token/response timestamps. The summary must contain the original same-host
monotonic clock declaration, all independently verified idle boundaries, and
per-phase origin/planned/recorded/failed-request fields. Qualification-only runs
with one repeat or concurrent diagnostic requests do not satisfy this contract.

Frontier input is `metrics/**/metrics_ground_truth.jsonl` plus the adjacent original
`config.json`. Only request-completion truth is read. Template IDs map through the
pinned common `request_ids.json`; `first_decode_token_completed_at` is the first
output boundary. The later legacy `decode_first_token_completed_at` and internal
prefill `ttft_ms` are never substitutes. The original internal value is retained
as `internal_ttft_ms` in the normalized rows.

## Step admission receipt

The existing `groundtruth-run-check` owner, not the metric normalizer, judges
runtime/source/image provenance and native qualification. The pinned receipt
retains `entry=groundtruth-run-check`, `mode=groundtruth_clean`, matching `case_id`,
and `runs.<input-directory-basename>` with:

- `status=PASS`, with nonempty passing `checks`, including `clean_runtime`,
  `runtime_identity`, `native_qualification`, and `artifact_integrity`.
- `run_directory`, relative to the receipt's case root, resolving to the exact
  directory passed to `--vllm-run-dir`. A same-name copy elsewhere is not bound.
- `runtime_identity`: the case's `native_runtime` declaration (adapter, actual
  installed vLLM version, immutable image), plus `harness_commit` and
  `source_hashes` from the original staged config. This does not assert a known
  full native build Git SHA when that SHA is unknown.
- `artifacts`: existing `{path, sha256}` pins relative to the case root, including
  the original config, client population, summary, every `nodes/*/result.json`,
  and the immutable actual receipts used by the checker.
- `runtime_evidence`: `{platform_image, installed_sources, native_qualification}`,
  each an existing artifact pin from that list. These name actual observed image,
  installed-source inventory and native qualification evidence, not regenerated
  claims. The existing checker owns interpreting their native schemas.

Every artifact hash is checked; actual worker versions, finished status and full
role/lane population must agree with the original config and case declaration.
Missing checker/runtime evidence leaves admission `UNVERIFIED`, and all four
gate metrics `INSUFFICIENT_EVIDENCE`. A successful replay or a
`CLIENT_WORKLOAD_VERIFIED` summary alone does not establish native acceptance.

## Metrics and repetitions

Each formal repeat retains all 128 requests. Client TTFT is actual dispatch to
first output, E2E is dispatch to last output, and TPOT is the mean of each
request's `(E2E-TTFT)/1023`. Total THPT is `917504/window_s`, where the window is
first actual formal dispatch to last formal output. Output-only `131072/window_s`
and request `128/window_s` throughput are nongating diagnostics.

Each native repeat is compared independently with the same deterministic Frontier
replay, using four independent 10% limits. A bad repeat fails the case even when
pooled means pass. The existing `metrics` compatibility field and
`pooled_diagnostic` retain the pooled reading: latency means pool all requests;
THPT is total tokens over the sum of repeat windows, **not** the arithmetic mean
of per-repeat THPT. The verdict uses independent repetitions only. Neither pooled
THPT nor the native summary's arithmetic throughput mean can hide a bad repeat.
Client metric `absolute_error` is magnitude; `signed_delta` retains Frontier minus
native. Legacy gate fields and thresholds are unchanged.
