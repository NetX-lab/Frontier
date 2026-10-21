"""The P0 table binds retained EP16 rows without admitting an unmeasured SP vector."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from frontier.cc_backend import MeasuredCCBackend, MeasuredCCBackendConfig, MeasuredCCOutOfRangeError
from frontier.types import ClusterType

ROOT = Path(__file__).resolve().parents[2]
NETWORK = ROOT / "data/profiling/network/h800_dgx"
GENERATION = NETWORK / "glm_p0_ep16_20261021a"


def test_case_generation_preserves_original_rows_and_receipt():
    base = NETWORK / "summary.csv"
    receipt = json.loads((GENERATION / "merge_receipt.json").read_text())
    provenance = json.loads((GENERATION / "provenance.json").read_text())
    table = GENERATION / "summary.csv"
    rows = list(csv.DictReader(table.open()))
    assert hashlib.sha256(base.read_bytes()).hexdigest() == provenance["original_summary_sha256"]
    assert rows[:971] == list(csv.DictReader(base.open()))
    assert len(rows) == 1187
    assert {(r["group"], r["group_size"]) for r in rows[971:]} == {("ep", "16")}
    assert hashlib.sha256(table.read_bytes()).hexdigest() == provenance["output_sha256"]
    assert receipt["tables"]["summary.csv"]["sha256"] == provenance["output_sha256"]
    assert [r["output_rows"] for r in receipt["tables"]["summary.csv"]["sources"]] == [[0, 971], [971, 1187]]
    assert (GENERATION / "COMPLETE").read_text() == "status=0\n"


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
def test_ep16_dispatch_binds_but_exact_6144_idle_combine_remains_typed(role):
    backend = MeasuredCCBackend(
        MeasuredCCBackendConfig(summary_csv_path=str(GENERATION / "summary.csv")),
        role, "h800", "h800_dgx", 16,
    )
    dispatch = backend.predict_allgather(9_486_336, 16, comm_domain="EP")
    assert math.isfinite(dispatch) and dispatch > 0
    with pytest.raises(MeasuredCCOutOfRangeError, match="unequal SP rows \\(768, 1\\)") as caught:
        backend.predict_unequal_reduce_scatter(75_595_776, 16, (768, 1), "EP")
    assert caught.value.curve_key == ("reduce_scatterv", "ep", 16)
    assert caught.value.requested_bytes == 75_595_776
    assert caught.value.covered_range == (196608, 201326592)
