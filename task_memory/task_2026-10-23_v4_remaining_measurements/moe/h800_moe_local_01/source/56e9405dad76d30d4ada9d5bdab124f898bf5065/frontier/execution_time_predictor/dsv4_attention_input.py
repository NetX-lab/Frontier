"""Read externally admitted isolated prefill layouts beside a selected V4 CSV."""

import hashlib
import json
from pathlib import Path

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError


SCHEMA = "dsv4_attention_input.v1"
QUALIFICATION = "isolated_prefill_input_only"


def _require(condition, message):
    if not condition:
        raise DeepSeekV4CoverageError(f"V4 isolated attention input rejected: {message}")


def _read_reference(directory, reference):
    raw = (directory / reference["path"]).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "evidence digest")
    _require(bool(raw), "empty evidence")
    return raw


def load_isolated_prefill_layout(csv_path, layer_id, measurement, phase):
    """Return a reviewed layout, or None only when no sidecar is present.

    The external reviewer qualifies the representation against its evidence.
    This reader verifies those bindings, not GPU coverage or serving additivity.
    """
    csv_path = Path(csv_path)
    path = csv_path.parent / "dsv4_attention.json"
    if not path.exists() and not path.is_symlink():
        return None
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        admission = json.loads(path.with_suffix(".admission.json").read_bytes())
        for document in (payload, admission):
            _require(document["schema_version"] == SCHEMA, "schema")
            _require(document["qualification"] == QUALIFICATION
                     and document["serving_admission"] is False, "isolated-only qualification")
        _require(phase == "prefill", "unsupported phase")
        _require(admission["verdict"] == "PASS"
                 and admission["sidecar_sha256"] == hashlib.sha256(raw).hexdigest(),
                 "external admission")
        _require(isinstance(admission["reviewed_by"], str) and admission["reviewed_by"].strip(),
                 "external reviewer")
        _read_reference(path.parent, admission["review"])
        evidence = payload["evidence"]
        _require(admission["evidence"] == evidence, "admission evidence bindings")
        _require(set(evidence) == {"acceptance", "manifest", "plan", "model", "job", "raw", "summary"},
                 "complete source lineage")
        documents = {name: _read_reference(path.parent, ref) for name, ref in evidence.items()}
        acceptance, manifest, summary = (json.loads(documents[name])
                                         for name in ("acceptance", "manifest", "summary"))
        _require(acceptance["status"] == "ACCEPTED_ISOLATED_DATA"
                 and acceptance["serving_admission"] is False, "original acceptance")
        _require(acceptance["manifest_sha256"] == evidence["manifest"]["sha256"], "original manifest")
        _require(manifest["files"]["plan.json"] == evidence["plan"]["sha256"]
                 and manifest["files"]["model/config.json"] == evidence["model"]["sha256"],
                 "plan/model lineage")
        _require(manifest["files"][evidence["job"]["manifest_path"]] == evidence["job"]["sha256"],
                 "job lineage")
        retained = {Path(name).name: pin["sha256"] for name, pin in acceptance["retained_files"].items()}
        _require(retained["dsv4_raw_e0.jsonl"] == evidence["raw"]["sha256"]
                 and retained["dsv4_aggregate_summary.json"] == evidence["summary"]["sha256"],
                 "original raw/summary lineage")
        identity = payload["identity"]
        _require(identity == {name: summary["identity"][name] for name in identity}
                 and set(identity) == {"source_revision", "runtime_fingerprint", "model_config_sha256"},
                 "summary identity")
        _require(identity["source_revision"] == acceptance["source_revision"] == manifest["source_revision"]
                 and identity["model_config_sha256"] == evidence["model"]["sha256"], "source/model identity")
        table = payload["tables"][csv_path.name]
        _require(table["measurement_type"] == measurement.value, "selected timing method")
        _require(table["file"]["path"] == csv_path.name, "selected table path")
        _require(table["file"]["sha256"] == retained[csv_path.name], "original table lineage")
        _read_reference(path.parent, table["file"])
        layout = table["layouts"][str(layer_id)]
        _require(isinstance(layout, dict) and all(layout[name] == value for name, value in identity.items())
                 and layout["input_file_sha256"] == table["file"]["sha256"], "layout identity")
        return dict(layout)
    except DeepSeekV4CoverageError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise DeepSeekV4CoverageError(f"V4 isolated attention input rejected in {path}: {exc}") from exc
