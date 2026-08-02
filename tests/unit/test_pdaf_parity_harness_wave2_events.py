import csv
from copy import deepcopy
from pathlib import Path
from typing import Callable

import pytest

from tests.e2e.pd_af_parity import harness


ROLES = ("PREFILL", "DECODE_ATTN", "DECODE_FFN")


def _record(
    event_class: str = "RequestArrivalEvent",
    *,
    event_id: int = 1,
    event_time: float = 1.0,
    **overrides: object,
) -> dict[str, object]:
    fields: dict[str, object]
    if event_class == "RequestArrivalEvent":
        fields = {"request_id": 0, "cluster_type": "PREFILL"}
    elif event_class == "M2NTransferStartEvent":
        fields = {
            "source_cluster_type": "DECODE_ATTN",
            "target_cluster_type": "DECODE_FFN",
            "source_replica_id": 3,
            "replica_id": 99,
            "dp_id": 0,
            "layer_id": 7,
            "afd_stage_idx": 2,
            "is_attn_to_ffn": True,
            "activation_size_bytes": 4096,
            "request_ids": [0, 1],
            "request_decode_steps": [4, 5],
            "request_layer_ids": [7, 7],
            "transfer_time_ms": 0.25,
            "num_tokens": 2,
            "batch_size": 2,
        }
    elif event_class == "M2NTransferEndEvent":
        fields = {
            "source_cluster_type": "DECODE_ATTN",
            "target_cluster_type": "DECODE_FFN",
            "source_replica_id": 3,
            "replica_id": 99,
            "dp_id": 0,
            "layer_id": 7,
            "is_attn_to_ffn": True,
            "activation_size_bytes": 4096,
            "request_ids": [0, 1],
            "request_decode_steps": [4, 5],
            "request_layer_ids": [7, 7],
            "pipeline_stage": 2,
            "transfer_time_ms": 0.25,
            "transfer_start_time": 0.5,
            "transfer_end_time": 0.50025,
            "num_tokens": 2,
            "batch_size": 2,
        }
    elif event_class == "KVCacheTransferStartEvent":
        fields = {
            "source_cluster_type": "PREFILL",
            "target_cluster_type": "DECODE_ATTN",
            "source_replica_id": 2,
            "kv_cache_size_bytes": 8192,
            "request_ids": [0, 1],
            "transfer_time_ms": 0.5,
            "num_tokens": 2,
            "batch_size": 2,
        }
    elif event_class == "BatchStageEndEvent":
        fields = {
            "replica_id": 3,
            "stage_id": 1,
            "dp_id": 0,
            "is_last_stage": False,
            "request_ids": [0, 1],
            "request_decode_steps": [4, 5],
            "request_layer_ids": [7, 7],
            "layer_id": 7,
            "num_tokens": 2,
            "batch_size": 2,
            "batch_stage_execution_time": 0.125,
        }
    elif event_class == "PrefillSyncEvent":
        fields = {
            "replica_id": 3,
            "stage_id": 1,
            "dp_id": 0,
            "sync_stage": "pre_moe",
            "layer_id": 7,
            "stage_execution_time": 0.125,
            "request_ids": [0, 1],
        }
    elif event_class == "EPAllToAllDispatchReadyEvent":
        fields = {"replica_id": 3, "stage_id": 1, "ep_id": 0}
    elif event_class == "GlobalScheduleEvent":
        fields = {
            "cluster_set": ["PREFILL", "DECODE_ATTN", "DECODE_FFN"],
            "request_mapping": [
                {"cluster_type": "PREFILL", "request_id": 0},
                {"cluster_type": "PREFILL", "request_id": 1},
            ],
        }
    else:
        fields = {"request_id": 0, "cluster_type": "PREFILL"}
    fields.update(overrides)
    return {
        "event_class": event_class,
        "event_id": event_id,
        "event_time": event_time,
        "fields": fields,
    }


def _detail_value(value: object) -> str:
    return str(value)


def _event_lines(record: dict[str, object], role: str) -> list[str]:
    event_class = str(record["event_class"])
    event_id = str(record["event_id"])
    event_time = record["event_time"]
    fields = dict(record["fields"])
    raw_event_type = fields.pop("raw_event_type", event_class)
    wallclock = str(fields.pop("wallclock", "2026-07-11 00:00:00.000"))
    duration = fields.pop("host_duration_ms", 0.123)
    batch_id = fields.pop("batch_id", 10)
    batch_global_id = fields.pop("batch_global_id", 20)
    new_events_generated = fields.pop("new_events_generated", 1)
    common = [
        f"event_time: {event_time}",
        f"cluster: {role}",
        f"event_type: {raw_event_type}",
        f"event_id: {event_id}",
        f"target_cluster: {role}",
    ]
    complete = list(common)
    complete.extend(f"{key}: {_detail_value(value)}" for key, value in fields.items())
    complete.extend(
        [
            f"time: {event_time}",
            f"batch_id: {batch_id}",
            f"batch_global_id: {batch_global_id}",
            f"new_events_generated: {new_events_generated}",
            f"cluster_time: {event_time}",
        ]
    )
    return [
        f"[{wallclock}] START {event_class} | ID: {event_id} | " + " | ".join(common),
        f"[{wallclock}] COMPLETE {event_class} | ID: {event_id} | "
        f"Duration: {duration}ms | " + " | ".join(complete),
    ]


def _write_role_log(
    root: Path,
    role: str,
    records: list[dict[str, object]],
    *,
    suffix: str = "20260711_000000",
    header_role: str | None = None,
    extra_lines: list[str] | None = None,
) -> Path:
    log_dir = root / "logs" / "cluster_events"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{role.lower()}_{suffix}.log"
    lines = [
        "=== VIDUR CLUSTER EVENT LOG ===",
        f"Cluster Type: {header_role or role}",
        "Start Time: 2026-07-11 00:00:00",
        "Log Level: INFO",
        f"Log File: {path}",
        "==================================================",
        "",
    ]
    for record in records:
        lines.extend(_event_lines(record, role))
    if extra_lines:
        lines.extend(extra_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_tree(
    root: Path,
    role_records: dict[str, list[dict[str, object]]] | None = None,
) -> None:
    records = role_records or {}
    for index, role in enumerate(ROLES):
        _write_role_log(
            root,
            role,
            records.get(
                role,
                [_record(event_id=index + 1, event_time=float(index), request_id=index)],
            ),
        )


def _compare_api() -> Callable[[str, str], object]:
    api = getattr(harness, "compare_event_logs", None)
    assert callable(api), "Required Wave 2 API is missing: compare_event_logs"
    return api


def _load_api() -> Callable[[str], object]:
    api = getattr(harness, "load_event_records", None)
    assert callable(api), "Required Wave 2 API is missing: load_event_records"
    return api


def _compare(
    tmp_path: Path,
    main_records: dict[str, list[dict[str, object]]],
    ref_records: dict[str, list[dict[str, object]]],
) -> object:
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    _write_tree(main_dir, main_records)
    _write_tree(ref_dir, ref_records)
    return _compare_api()(str(main_dir), str(ref_dir))


@pytest.mark.parametrize(
    "case_name",
    [
        "raw_ids",
        "raw_enum",
        "same_time_order",
        "request_order",
        "batch_ids",
        "millisecond_tolerance",
        "second_tolerance",
    ],
)
def test_event_semantic_normalization_passes(
    tmp_path: Path, case_name: str
) -> None:
    main = [_record("M2NTransferStartEvent", event_id=41)]
    ref = deepcopy(main)
    if case_name == "raw_ids":
        ref[0]["event_id"] = 99
    elif case_name == "raw_enum":
        main[0]["fields"]["raw_event_type"] = "EventType.M2N_TRANSFER_START"  # type: ignore[index]
        ref[0]["fields"]["raw_event_type"] = "m2n_transfer_start"  # type: ignore[index]
    elif case_name == "same_time_order":
        main = [_record(event_id=i + 1, event_time=1.0, request_id=i) for i in range(3)]
        ref = list(reversed(deepcopy(main)))
    elif case_name == "request_order":
        scheduling_main = [_record("GlobalScheduleEvent")]
        scheduling_ref = deepcopy(scheduling_main)
        scheduling_ref[0]["fields"]["cluster_set"] = [  # type: ignore[index]
            "DECODE_FFN",
            "DECODE_ATTN",
            "PREFILL",
        ]
        scheduling_ref[0]["fields"]["request_mapping"] = list(  # type: ignore[index]
            reversed(scheduling_ref[0]["fields"]["request_mapping"])  # type: ignore[index]
        )
        scheduling_result = _compare(
            tmp_path / "scheduling_order",
            {"PREFILL": scheduling_main},
            {"PREFILL": scheduling_ref},
        )
        assert scheduling_result.passed is True
        main = [_record("BatchStageEndEvent")]
        ref = deepcopy(main)
        ref[0]["fields"]["request_ids"] = [1, 0]  # type: ignore[index]
        ref[0]["fields"]["request_decode_steps"] = [5, 4]  # type: ignore[index]
        ref[0]["fields"]["request_layer_ids"] = [7, 7]  # type: ignore[index]
    elif case_name == "batch_ids":
        main[0]["fields"]["batch_id"] = 1  # type: ignore[index]
        main[0]["fields"]["batch_global_id"] = 2  # type: ignore[index]
        main[0]["fields"]["replica_id"] = 99  # type: ignore[index]
        ref[0]["fields"]["batch_id"] = 7  # type: ignore[index]
        ref[0]["fields"]["batch_global_id"] = 9  # type: ignore[index]
        ref[0]["fields"]["replica_id"] = 42  # type: ignore[index]
    elif case_name == "millisecond_tolerance":
        ref[0]["fields"]["transfer_time_ms"] = 0.25 + 5e-10  # type: ignore[index]
    elif case_name == "second_tolerance":
        main = [_record("M2NTransferEndEvent")]
        ref = deepcopy(main)
        ref[0]["fields"]["transfer_start_time"] = 0.5 + 5e-13  # type: ignore[index]

    result = _compare(
        tmp_path,
        {"DECODE_ATTN": main},
        {"DECODE_ATTN": ref},
    )

    assert result.passed is True
    assert result.first_divergence_event_index is None


@pytest.mark.parametrize("group_size", [8, 12])
def test_event_same_time_matching_has_no_size_cap(
    tmp_path: Path, group_size: int
) -> None:
    main = [
        _record(event_id=index + 1, event_time=1.0, request_id=index)
        for index in range(group_size)
    ]
    ref = list(reversed(deepcopy(main)))

    result = _compare(
        tmp_path,
        {"PREFILL": main},
        {"PREFILL": ref},
    )

    assert result.passed is True
    assert result.total_events_main == result.total_events_ref


@pytest.mark.parametrize(
    "case_name,expected_category",
    [
        ("multiplicity", "MULTIPLICITY_MISMATCH"),
        ("layer", "SEMANTIC_MISMATCH"),
        ("direction", "SEMANTIC_MISMATCH"),
        ("lanes", "SEMANTIC_MISMATCH"),
        ("source_replica", "SEMANTIC_MISMATCH"),
        ("request_set", "SEMANTIC_MISMATCH"),
        ("decode_step", "SEMANTIC_MISMATCH"),
        ("payload", "SEMANTIC_MISMATCH"),
        ("sync_stage", "SEMANTIC_MISMATCH"),
        ("time_group", "TIME_GROUP_MISMATCH"),
        ("new_events", "SEMANTIC_MISMATCH"),
        ("request_mapping", "SEMANTIC_MISMATCH"),
        ("role_shift", "ROLE_SHIFT_MISMATCH"),
        ("numeric_tolerance", "FIELD_VALUE_MISMATCH"),
        ("field_presence", "FIELD_PRESENCE_MISMATCH"),
    ],
)
def test_event_semantic_divergence_fails(
    tmp_path: Path, case_name: str, expected_category: str
) -> None:
    role = "DECODE_ATTN"
    main = [_record("M2NTransferStartEvent")]
    ref = deepcopy(main)
    if case_name == "multiplicity":
        main.append(_record("M2NTransferStartEvent", event_id=2, source_replica_id=4))
    elif case_name == "layer":
        ref[0]["fields"]["layer_id"] = 8  # type: ignore[index]
    elif case_name == "direction":
        ref[0]["fields"]["is_attn_to_ffn"] = False  # type: ignore[index]
    elif case_name == "lanes":
        for field in ("replica_id", "dp_id"):
            lane_main = [_record("BatchStageEndEvent")]
            candidate = deepcopy(lane_main)
            candidate[0]["fields"][field] = 11  # type: ignore[index]
            lane_result = _compare(
                tmp_path / field,
                {role: lane_main},
                {role: candidate},
            )
            assert lane_result.passed is False
        for field in ("afd_stage_idx",):
            candidate = deepcopy(ref)
            candidate[0]["fields"][field] = 11  # type: ignore[index]
            lane_result = _compare(
                tmp_path / field,
                {role: main},
                {role: candidate},
            )
            assert lane_result.passed is False
        main = [_record("EPAllToAllDispatchReadyEvent")]
        ref = deepcopy(main)
        ref[0]["fields"]["ep_id"] = 1  # type: ignore[index]
    elif case_name == "source_replica":
        ref[0]["fields"]["source_replica_id"] = 4  # type: ignore[index]
    elif case_name == "request_set":
        ref[0]["fields"]["request_ids"] = [0, 2]  # type: ignore[index]
    elif case_name == "decode_step":
        ref[0]["fields"]["request_decode_steps"] = [4, 6]  # type: ignore[index]
    elif case_name == "payload":
        for field in ("activation_size_bytes", "num_tokens"):
            candidate = deepcopy(ref)
            candidate[0]["fields"][field] = 8192  # type: ignore[index]
            payload_result = _compare(
                tmp_path / field,
                {role: main},
                {role: candidate},
            )
            assert payload_result.passed is False
        ref[0]["fields"]["activation_size_bytes"] = 8192  # type: ignore[index]
    elif case_name == "sync_stage":
        main = [_record("PrefillSyncEvent")]
        ref = deepcopy(main)
        ref[0]["fields"]["sync_stage"] = "post_moe"  # type: ignore[index]
    elif case_name == "time_group":
        ref[0]["event_time"] = 1.5
    elif case_name == "new_events":
        ref[0]["fields"]["new_events_generated"] = 2  # type: ignore[index]
    elif case_name == "request_mapping":
        main = [_record("GlobalScheduleEvent")]
        ref = deepcopy(main)
        ref[0]["fields"]["request_mapping"] = [  # type: ignore[index]
            {"cluster_type": "PREFILL", "request_id": 0}
        ]
    elif case_name == "role_shift":
        main = [_record("KVCacheTransferStartEvent", event_time=1.0)]
        ref = deepcopy(main)
        prefill_filler = _record(event_id=101, event_time=0.0, request_id=101)
        attn_filler = _record(event_id=102, event_time=2.0, request_id=102)
        result = _compare(
            tmp_path,
            {
                "PREFILL": [prefill_filler],
                "DECODE_ATTN": [*main, attn_filler],
            },
            {
                "PREFILL": [prefill_filler, *ref],
                "DECODE_ATTN": [attn_filler],
            },
        )
        assert result.passed is False
        assert result.mismatches[0].category in {
            "ROLE_SHIFT_MISMATCH",
            "SEMANTIC_MISMATCH",
        }
        return
    elif case_name == "numeric_tolerance":
        ref[0]["fields"]["transfer_time_ms"] = 0.25 + 5e-7  # type: ignore[index]
        second_main = [_record("M2NTransferEndEvent")]
        second_ref = deepcopy(second_main)
        second_ref[0]["fields"]["transfer_start_time"] = 0.5 + 5e-10  # type: ignore[index]
        second_result = _compare(
            tmp_path / "seconds",
            {role: second_main},
            {role: second_ref},
        )
        assert second_result.passed is False
    elif case_name == "field_presence":
        del ref[0]["fields"]["num_tokens"]  # type: ignore[index]

    result = _compare(tmp_path, {role: main}, {role: ref})

    assert result.passed is False
    assert result.first_divergence_event_index is not None
    assert result.mismatches[0].category == expected_category


def _make_invalid_tree(tmp_path: Path, case_name: str) -> Path:
    root = tmp_path / case_name
    if case_name == "missing_role":
        _write_role_log(root, "PREFILL", [_record(request_id=0)])
        _write_role_log(root, "DECODE_ATTN", [_record(request_id=1)])
        return root
    _write_tree(root)
    log_dir = root / "logs" / "cluster_events"
    prefill = next(log_dir.glob("prefill_*.log"))
    if case_name == "duplicate_role":
        _write_role_log(root, "PREFILL", [_record()], suffix="20260711_000001")
    elif case_name == "wrong_header":
        prefill.write_text(
            prefill.read_text(encoding="utf-8").replace(
                "Cluster Type: PREFILL", "Cluster Type: DECODE_ATTN"
            ),
            encoding="utf-8",
        )
    elif case_name == "no_canonical_records":
        for summary_only in (False, True):
            variant = root / ("summary" if summary_only else "header")
            _write_tree(variant)
            path = next((variant / "logs" / "cluster_events").glob("prefill_*.log"))
            content = "\n".join(path.read_text(encoding="utf-8").splitlines()[:7]) + "\n"
            if summary_only:
                content += "=== EVENT PROCESSING SUMMARY ===\nTotal Events Processed: 0\n"
            path.write_text(content, encoding="utf-8")
        return root / "header"
    elif case_name == "malformed_separator":
        prefill.write_text(
            prefill.read_text(encoding="utf-8").replace(
                " | request_id: 0", " | malformed-detail"
            ),
            encoding="utf-8",
        )
    elif case_name == "duplicate_detail_key":
        prefill.write_text(
            prefill.read_text(encoding="utf-8").replace(
                " | request_id: 0", " | request_id: 0 | request_id: 0"
            ),
            encoding="utf-8",
        )
    elif case_name in {"invalid_integer", "nonfinite_float", "invalid_bool", "invalid_list", "required_field"}:
        event = _record("M2NTransferStartEvent")
        if case_name == "invalid_integer":
            event["fields"]["layer_id"] = "abc"  # type: ignore[index]
        elif case_name == "nonfinite_float":
            event["fields"]["transfer_time_ms"] = float("inf")  # type: ignore[index]
        elif case_name == "invalid_bool":
            event["fields"]["is_attn_to_ffn"] = "yes"  # type: ignore[index]
        elif case_name == "invalid_list":
            event["fields"]["request_ids"] = {0, 1}  # type: ignore[index]
        else:
            del event["fields"]["layer_id"]  # type: ignore[index]
        _write_role_log(root, "PREFILL", [event])
    elif case_name == "error_record":
        with prefill.open("a", encoding="utf-8") as stream:
            stream.write(
                "[2026-07-11 00:00:00.000] ERROR RequestArrivalEvent | "
                "ID: 8 | Error: boom\n"
            )
    elif case_name == "ordering_violation":
        _write_role_log(
            root,
            "PREFILL",
            [_record(event_id=1, event_time=2.0), _record(event_id=2, event_time=1.0)],
        )
    elif case_name == "unknown_class":
        for class_name in ("UnsupportedEvent", "PrefixCacheFetchEndEvent"):
            variant = root / class_name
            _write_tree(variant)
            _write_role_log(variant, "PREFILL", [_record(class_name)])
        return root / "UnsupportedEvent"
    elif case_name == "unknown_line":
        with prefill.open("a", encoding="utf-8") as stream:
            stream.write("this line is not part of the event log grammar\n")
    elif case_name == "request_cardinality":
        for suffix, request_ids, decode_steps in (
            ("duplicate", [0, 0], [4, 5]),
            ("length", [0, 1], [4]),
        ):
            variant = root / suffix
            _write_tree(variant)
            _write_role_log(
                variant,
                "PREFILL",
                [
                    _record(
                        "BatchStageEndEvent",
                        request_ids=request_ids,
                        request_decode_steps=decode_steps,
                    )
                ],
            )
        return root / "duplicate"
    elif case_name == "pair_integrity":
        duplicate = root.parent / "pair_duplicate"
        _write_tree(duplicate)
        duplicate_log = next(
            (duplicate / "logs" / "cluster_events").glob("prefill_*.log")
        )
        duplicate_lines = duplicate_log.read_text(encoding="utf-8").splitlines()
        complete_line = next(line for line in duplicate_lines if " COMPLETE " in line)
        duplicate_log.write_text(
            "\n".join([*duplicate_lines, complete_line]) + "\n",
            encoding="utf-8",
        )
        lines = prefill.read_text(encoding="utf-8").splitlines()
        prefill.write_text(
            "\n".join(line for line in lines if " COMPLETE " not in line) + "\n",
            encoding="utf-8",
        )
    return root


@pytest.mark.parametrize(
    "case_name,error_pattern",
    [
        ("missing_role", "(?i)missing|no event log"),
        ("duplicate_role", "(?i)duplicate|ambiguous"),
        ("wrong_header", "(?i)header|cluster type"),
        ("no_canonical_records", "(?i)no canonical event records"),
        ("malformed_separator", "(?i)malformed"),
        ("duplicate_detail_key", "(?i)duplicate detail key"),
        ("invalid_integer", "(?i)integer|int"),
        ("nonfinite_float", "(?i)finite"),
        ("invalid_bool", "(?i)bool"),
        ("invalid_list", "(?i)list"),
        ("error_record", "(?i)ERROR record"),
        ("ordering_violation", "(?i)ordering"),
        ("required_field", "(?i)required field"),
        ("unknown_class", "(?i)unknown event class"),
        ("unknown_line", "(?i)unrecognized|unknown line"),
        ("request_cardinality", "(?i)duplicate request|positional|length"),
        ("pair_integrity", "(?i)START|COMPLETE|pair"),
    ],
)
def test_event_log_input_validation_fails_fast(
    tmp_path: Path, case_name: str, error_pattern: str
) -> None:
    root = _make_invalid_tree(tmp_path, case_name)
    roots = [root]
    if case_name == "no_canonical_records":
        roots.append(root.parent / "summary")
    elif case_name == "unknown_class":
        roots.append(root.parent / "PrefixCacheFetchEndEvent")
    elif case_name == "request_cardinality":
        roots.append(root.parent / "length")
    elif case_name == "pair_integrity":
        roots.append(root.parent / "pair_duplicate")

    for invalid_root in roots:
        with pytest.raises(harness.ParityInputError, match=error_pattern):
            _load_api()(str(invalid_root))


def _write_l1_metrics(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "request_metrics.csv"
    fields = ["Request Id", *harness.DISCRETE_FIELDS]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "Request Id": 0,
                "request_num_prefill_tokens": 8,
                "request_num_decode_tokens": 2,
                "request_num_tokens": 10,
                "request_num_restarts": 0,
                "request_thinking_round_count": 0,
            }
        )


def _case() -> harness.ParityCaseConfig:
    return harness.ParityCaseConfig(
        case_id="wave2-events",
        model="test-model",
        mode="offline",
        scale_gpu=24,
        prefill_tokens=8,
        decode_tokens=2,
        num_requests=1,
        layer=harness.ParityLayer.L1_DUMMY,
    )


def _divergent_report(tmp_path: Path) -> harness.ParityReport:
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    _write_l1_metrics(main_dir)
    _write_l1_metrics(ref_dir)
    _write_tree(main_dir, {"PREFILL": [_record(request_id=0)]})
    _write_tree(ref_dir, {"PREFILL": [_record(request_id=1)]})
    return harness.generate_report(_case(), str(main_dir), str(ref_dir))


def test_event_report_sets_first_divergence_index(tmp_path: Path) -> None:
    report = _divergent_report(tmp_path)

    assert report.first_divergence_event_index is not None


def test_event_report_markdown_shows_first_divergence_index(tmp_path: Path) -> None:
    markdown = harness.report_to_markdown(_divergent_report(tmp_path))

    assert "First divergence event index" in markdown


def test_equal_request_csv_cannot_hide_event_divergence(tmp_path: Path) -> None:
    report = _divergent_report(tmp_path)

    assert all(item.passed for item in report.request_comparisons)
    assert report.overall_pass is False


def test_event_mismatches_count_time_groups_not_fields(tmp_path: Path) -> None:
    report = _divergent_report(tmp_path)

    assert report.event_comparison.total_mismatches == 1
    assert report.event_comparison.mismatches[0].category == "SEMANTIC_MISMATCH"
    assert report.total_mismatches == 1


def test_event_report_markdown_shows_all_role_statuses(tmp_path: Path) -> None:
    markdown = harness.report_to_markdown(_divergent_report(tmp_path))

    assert "Event Comparison" in markdown
    assert all(role in markdown for role in ROLES)


def test_event_report_markdown_shows_both_divergent_records(tmp_path: Path) -> None:
    markdown = harness.report_to_markdown(_divergent_report(tmp_path))

    assert "Main normalized record" in markdown
    assert "Reference normalized record" in markdown
    assert "request_id" in markdown
