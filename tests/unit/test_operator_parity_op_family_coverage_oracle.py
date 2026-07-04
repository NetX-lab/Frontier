from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import tests.e2e.operator_parity.op_family_coverage_oracle as coverage_oracle
from tests.e2e.operator_parity.op_family_coverage_oracle import (
    FAMILY_ATTENTION,
    FAMILY_COMM,
    run_coverage_oracle,
)


class _FakeOperator:
    def __init__(self, name: str, profiling_name: str | None = None) -> None:
        self.name = name
        self._profiling_name = profiling_name or name

    def profiling_name(self) -> str:
        return self._profiling_name


class _FakeFamily:
    def __init__(
        self,
        *,
        family_id: str,
        resource_class,
        operators: tuple[_FakeOperator, ...],
    ) -> None:
        self.family_id = family_id
        self.resource_class = resource_class
        self._operators = operators

    def e2e_trace_ops(self) -> tuple[_FakeOperator, ...]:
        return self._operators

    @property
    def operators(self) -> tuple[_FakeOperator, ...]:
        return self._operators


def _write_config(path: Path, payload: dict[str, object]) -> None:
    defaults: dict[str, object] = {
        "model_type": "unit_test_model",
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "hidden_size": 128,
        "head_dim": 16,
        "num_experts": 0,
        "share_expert_dim": 0,
        "use_mla": False,
    }
    defaults.update(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(defaults), encoding="utf-8")


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trace_index",
                "type",
                "name",
                "ts_start",
                "duration_ms",
                "cluster",
                "replica_id",
                "batch_id",
                "request_id",
                "layer_id",
                "target_cluster",
                "meta_json",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            payload = {"trace_index": str(index), "duration_ms": "1.0"}
            payload.update({key: str(value) for key, value in row.items()})
            writer.writerow(payload)


def _case(
    tmp_path: Path,
    *,
    name: str,
    model_name: str,
    sys_arch: str = "co-location",
) -> dict[str, object]:
    trace_dir = tmp_path / "outputs" / name
    return {
        "name": name,
        "case_manifest": {
            "model_name": model_name,
            "config_path": str(tmp_path / "models" / f"{model_name}.json"),
            "sys_arch": sys_arch,
        },
        "reference_dir": str(trace_dir),
        "candidate_dir": str(trace_dir),
    }


def test_coverage_oracle_passes_dense_colocation_and_pdd(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "models" / "dense.json",
        {"model_type": "llama", "num_experts": 0},
    )
    dense = _case(tmp_path, name="dense_colocation", model_name="dense")
    pdd = _case(tmp_path, name="dense_pdd", model_name="dense", sys_arch="pd-disaggregation")
    _write_trace(
        Path(str(dense["reference_dir"])) / "op_traces.csv",
        [
            {"name": "attn_prefill", "duration_ms": "2.0"},
            {"name": "mlp_up_proj", "duration_ms": "3.0"},
            {"name": "input_layernorm", "duration_ms": "1.0"},
        ],
    )
    _write_trace(
        Path(str(pdd["reference_dir"])) / "op_traces.csv",
        [
            {"name": "attn_prefill", "duration_ms": "2.0"},
            {"name": "mlp_up_proj", "duration_ms": "3.0"},
            {"name": "input_layernorm", "duration_ms": "1.0"},
            {"name": "kv_cache_transfer", "duration_ms": "0.4"},
        ],
    )

    report = run_coverage_oracle([dense, pdd], side="reference")

    assert report["status"] == "PASS"
    assert report["checked_case_count"] == 2
    assert report["exempt_case_count"] == 0
    checked_by_arch = {case["sys_arch"]: case for case in report["cases"]}
    assert set(checked_by_arch) == {"co-location", "pd-disaggregation"}
    assert checked_by_arch["co-location"]["status"] == "PASS"
    assert checked_by_arch["pd-disaggregation"]["status"] == "PASS"
    assert checked_by_arch["co-location"]["expected_families"] == [
        "ATTENTION",
        "FFN",
        "MEMORY",
    ]
    assert checked_by_arch["pd-disaggregation"]["expected_families"] == [
        "ATTENTION",
        "COMM",
        "FFN",
        "MEMORY",
    ]
    assert checked_by_arch["co-location"]["missing_families"] == []
    assert checked_by_arch["pd-disaggregation"]["missing_families"] == []


def test_coverage_oracle_fails_when_every_case_is_exempt(tmp_path: Path) -> None:
    case = _case(
        tmp_path,
        name="unsupported_disaggregated",
        model_name="dense",
        sys_arch="pd-af-disaggregation",
    )

    report = run_coverage_oracle([case], side="reference")

    assert report["status"] == "FAIL_NO_CHECKED_CASES"
    assert report["case_count"] == 1
    assert report["checked_case_count"] == 0
    assert report["exempt_case_count"] == 1
    assert report["failed_case_count"] == 0
    assert report["cases"][0]["status"] == "EXEMPT"


def test_coverage_oracle_accepts_pdd_kv_cache_transfer_as_comm_family(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path / "models" / "dense.json",
        {"model_type": "llama", "num_experts": 0},
    )
    case = _case(
        tmp_path,
        name="dense_pdd",
        model_name="dense",
        sys_arch="pd-disaggregation",
    )
    _write_trace(
        Path(str(case["reference_dir"])) / "op_traces.csv",
        [
            {"name": "attn_prefill", "duration_ms": "2.0"},
            {"name": "mlp_up_proj", "duration_ms": "3.0"},
            {"name": "input_layernorm", "duration_ms": "1.0"},
            {"name": "kv_cache_transfer", "duration_ms": "0.4"},
        ],
    )

    report = run_coverage_oracle([case], side="reference")

    assert report["status"] == "PASS"
    checked = report["cases"][0]
    assert checked["expected_families"] == ["ATTENTION", "COMM", "FFN", "MEMORY"]
    assert FAMILY_COMM in checked["observed_nonzero_families"]
    assert checked["unknown_nonzero_observed_ops"] == []


def test_coverage_oracle_fails_pdd_when_kv_transfer_comm_family_is_missing(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path / "models" / "dense.json",
        {"model_type": "llama", "num_experts": 0},
    )
    case = _case(
        tmp_path,
        name="dense_pdd",
        model_name="dense",
        sys_arch="pd-disaggregation",
    )
    _write_trace(
        Path(str(case["reference_dir"])) / "op_traces.csv",
        [
            {"name": "attn_prefill", "duration_ms": "2.0"},
            {"name": "mlp_up_proj", "duration_ms": "3.0"},
            {"name": "input_layernorm", "duration_ms": "1.0"},
        ],
    )

    report = run_coverage_oracle([case], side="reference")

    assert report["status"] == "FAIL"
    failed = report["cases"][0]
    assert failed["missing_families"] == ["COMM"]
    assert failed["unknown_nonzero_observed_ops"] == []


def test_coverage_oracle_fails_when_expected_family_has_only_zero_time_rows(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path / "models" / "moe.json",
        {"model_type": "phimoe", "num_experts": 16},
    )
    case = _case(tmp_path, name="moe_colocation", model_name="moe")
    _write_trace(
        Path(str(case["reference_dir"])) / "op_traces.csv",
        [
            {"name": "attn_prefill", "duration_ms": "2.0"},
            {"name": "input_layernorm", "duration_ms": "1.0"},
            {"name": "moe_grouped_gemm", "duration_ms": "0.0"},
        ],
    )

    report = run_coverage_oracle([case], side="reference")

    assert report["status"] == "FAIL"
    failed = report["cases"][0]
    assert failed["missing_families"] == ["MOE"]
    assert failed["zero_only_families"] == ["MOE"]


def test_coverage_oracle_fails_on_unknown_nonzero_colocation_op(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path / "models" / "dense.json",
        {"model_type": "llama", "num_experts": 0},
    )
    case = _case(tmp_path, name="dense_colocation", model_name="dense")
    _write_trace(
        Path(str(case["reference_dir"])) / "op_traces.csv",
        [
            {"name": "attn_prefill", "duration_ms": "2.0"},
            {"name": "mlp_up_proj", "duration_ms": "3.0"},
            {"name": "input_layernorm", "duration_ms": "1.0"},
            {"name": "legacy_non_registry_attention_op", "duration_ms": "4.0"},
        ],
    )

    report = run_coverage_oracle([case], side="reference")

    assert report["status"] == "FAIL"
    failed = report["cases"][0]
    assert failed["missing_families"] == []
    assert failed["unknown_nonzero_observed_ops"] == [
        "legacy_non_registry_attention_op"
    ]


def test_coverage_oracle_fails_on_unknown_nonzero_pdd_op(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path / "models" / "dense.json",
        {"model_type": "llama", "num_experts": 0},
    )
    case = _case(
        tmp_path,
        name="dense_pdd",
        model_name="dense",
        sys_arch="pd-disaggregation",
    )
    _write_trace(
        Path(str(case["reference_dir"])) / "op_traces.csv",
        [
            {"name": "attn_prefill", "duration_ms": "2.0"},
            {"name": "mlp_up_proj", "duration_ms": "3.0"},
            {"name": "input_layernorm", "duration_ms": "1.0"},
            {"name": "kv_cache_transfer", "duration_ms": "0.4"},
            {"name": "legacy_non_registry_pdd_op", "duration_ms": "4.0"},
        ],
    )

    report = run_coverage_oracle([case], side="reference")

    assert report["status"] == "FAIL"
    assert report["checked_case_count"] == 1
    assert report["exempt_case_count"] == 0
    failed = report["cases"][0]
    assert failed["sys_arch"] == "pd-disaggregation"
    assert failed["missing_families"] == []
    assert failed["unknown_nonzero_observed_ops"] == [
        "legacy_non_registry_pdd_op"
    ]


def test_coverage_oracle_fails_pdd_when_expected_family_is_missing(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path / "models" / "dense.json",
        {"model_type": "llama", "num_experts": 0},
    )
    case = _case(
        tmp_path,
        name="dense_pdd",
        model_name="dense",
        sys_arch="pd-disaggregation",
    )
    _write_trace(
        Path(str(case["reference_dir"])) / "op_traces.csv",
        [
            {"name": "attn_prefill", "duration_ms": "2.0"},
            {"name": "input_layernorm", "duration_ms": "1.0"},
            {"name": "kv_cache_transfer", "duration_ms": "0.4"},
        ],
    )

    report = run_coverage_oracle([case], side="reference")

    assert report["status"] == "FAIL"
    assert report["checked_case_count"] == 1
    assert report["exempt_case_count"] == 0
    failed = report["cases"][0]
    assert failed["sys_arch"] == "pd-disaggregation"
    assert failed["missing_families"] == ["FFN"]
    assert failed["zero_only_families"] == []


def test_expected_families_are_derived_from_operator_manifest(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_operator_manifest(config):
        captured["model_type"] = config.model_type
        return SimpleNamespace(
            family_bindings=(
                SimpleNamespace(family_id="dense_attention"),
                SimpleNamespace(family_id="memory"),
                SimpleNamespace(family_id="share_expert"),
            )
        )

    monkeypatch.setattr(
        coverage_oracle,
        "build_operator_manifest",
        fake_build_operator_manifest,
        raising=False,
    )

    families = coverage_oracle.expected_families_for_model(
        {
            "model_type": "unit_manifest_model",
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_experts": 0,
        }
    )

    assert captured["model_type"] == "unit_manifest_model"
    assert families == ("ATTENTION", "MEMORY", "SHARE_EXPERT")


def test_observed_mapping_includes_manifest_family_operator_names(monkeypatch) -> None:
    manifest_family = _FakeFamily(
        family_id="manifest_only_family",
        resource_class=coverage_oracle.ResourceClass.COMP,
        operators=(
            _FakeOperator(
                "unit_manifest_only_trace_op",
                "unit_manifest_only_profile_op",
            ),
        ),
    )

    def fake_build_operator_manifest(config):
        return SimpleNamespace(
            family_bindings=(
                SimpleNamespace(
                    family_id="manifest_only_family",
                    family=manifest_family,
                ),
            )
        )

    monkeypatch.setattr(
        coverage_oracle,
        "build_operator_manifest",
        fake_build_operator_manifest,
        raising=False,
    )

    mapping = coverage_oracle._operator_family_by_name_for_model(
        {
            "model_type": "unit_manifest_model",
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "hidden_size": 128,
        }
    )

    assert mapping["unit_manifest_only_trace_op"] == "MANIFEST_ONLY_FAMILY"
    assert mapping["unit_manifest_only_profile_op"] == "MANIFEST_ONLY_FAMILY"


def test_observed_mapping_includes_registry_comm_operator_names(monkeypatch) -> None:
    registry_family = _FakeFamily(
        family_id="comm",
        resource_class=coverage_oracle.ResourceClass.COMM,
        operators=(
            _FakeOperator(
                "unit_registry_comm_trace_op",
                "unit_registry_comm_profile_op",
            ),
        ),
    )

    monkeypatch.setattr(
        coverage_oracle,
        "iter_operator_families",
        lambda: (registry_family,),
    )

    mapping = coverage_oracle._operator_family_by_name_for_model(
        {
            "model_type": "unit_registry_model",
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "hidden_size": 128,
        }
    )

    assert mapping["unit_registry_comm_trace_op"] == FAMILY_COMM
    assert mapping["unit_registry_comm_profile_op"] == FAMILY_COMM


def test_observed_mapping_includes_architecture_linear_attention_ops(monkeypatch) -> None:
    fake_profile = SimpleNamespace(
        linear_attention=SimpleNamespace(
            sharded_ops=("unit_arch_sharded_attention_op",),
            replicated_ops=("unit_arch_replicated_attention_op",),
        ),
        validate_structural_requirements=lambda _config: None,
    )

    monkeypatch.setattr(
        coverage_oracle,
        "get_model_architecture_profile",
        lambda _config: fake_profile,
    )

    mapping = coverage_oracle._operator_family_by_name_for_model(
        {
            "model_type": "unit_architecture_model",
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "hidden_size": 128,
        }
    )

    assert mapping["unit_arch_sharded_attention_op"] == FAMILY_ATTENTION
    assert mapping["unit_arch_replicated_attention_op"] == FAMILY_ATTENTION


def test_coverage_oracle_recognizes_comm_family_operator_names(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path / "models" / "step_moe.json",
        {
            "model_type": "step3_text",
            "model_arch": "step3_text",
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "hidden_size": 128,
            "head_dim": 16,
            "num_experts": 16,
            "share_expert_dim": 4096,
            "use_mfa": True,
            "share_q_dim": 64,
        },
    )
    case = _case(tmp_path, name="step_moe_colocation", model_name="step_moe")
    _write_trace(
        Path(str(case["reference_dir"])) / "op_traces.csv",
        [
            {"name": "attn_prefill", "duration_ms": "2.0"},
            {"name": "input_layernorm", "duration_ms": "1.0"},
            {"name": "moe_grouped_gemm", "duration_ms": "3.0"},
            {"name": "share_expert_up_proj", "duration_ms": "4.0"},
            {"name": "attn_tensor_parallel_allreduce", "duration_ms": "0.5"},
            {"name": "moe_tensor_parallel_allgather", "duration_ms": "0.6"},
            {"name": "moe_tensor_parallel_allreduce", "duration_ms": "0.7"},
            {"name": "share_expert_tensor_parallel_allreduce", "duration_ms": "0.8"},
        ],
    )

    report = run_coverage_oracle([case], side="reference")

    assert report["status"] == "PASS"
    checked = report["cases"][0]
    assert checked["unknown_observed_ops"] == []
    assert FAMILY_COMM in checked["observed_nonzero_families"]


def test_coverage_oracle_from_roots_forwards_workload_profiles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_golden_cases(**kwargs):
        captured["workload_profiles"] = kwargs["workload_profiles"]
        return [], {"actual": 0, "workload_profiles": {"long_single": {}}}

    monkeypatch.setattr(coverage_oracle, "build_golden_cases", fake_build_golden_cases)

    report = coverage_oracle.run_coverage_oracle_from_roots(
        config_root=tmp_path / "models",
        profile_root=tmp_path / "profiles",
        reference_root=tmp_path / "reference",
        candidate_root=tmp_path / "candidate",
        side="candidate",
        workload_profiles=("long_single",),
    )

    assert captured["workload_profiles"] == ("long_single",)
    assert report["status"] == "FAIL_NO_CHECKED_CASES"
    assert report["case_count"] == 0
    assert report["checked_case_count"] == 0
    assert report["case_manifest"]["workload_profiles"] == {"long_single": {}}
