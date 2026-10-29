"""Supplement filtering retains engine/step/call/interval identities and gaps."""
import importlib.util
import json
from pathlib import Path

import pandas as pd

import pytest

from frontier.profiling.attention.v4_attention_windows import C1, repeat_windows
from test_dsv4_predictor import aggregate, balanced_routing, load, model, replica
from test_dsv4_replay_rows import plan

ROOT = Path(__file__).resolve().parents[2]
H3O = ROOT / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs"


@pytest.fixture
def promotion():
    spec = importlib.util.spec_from_file_location("supplement_promotion_test", H3O / "h200_tables/promote_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repeat_geometry_is_excluded_even_with_new_case_ids(tmp_path, promotion):
    repeat = repeat_windows()[0]
    cached = next(c for c in range(C1["cached_high"] - C1["context_tolerance"], C1["cached_high"] + 1)
                  if (c + 1) // C1["ratios"][0] > c // C1["ratios"][0]
                  and (c + 1) // C1["ratios"][1] == c // C1["ratios"][1])
    engine = dict(kind="engine", engine_index=0)
    step = dict(kind="step", engine_index=0, case_id="renamed-repeat", step_index=4,
                phase="decode", requests=[dict(prompt_len=C1["cached_low"],
                                                num_computed_tokens=cached, num_scheduled_tokens=1)],
                cudagraph_dispatch=dict(cg_mode=repeat.mode, num_tokens=repeat.width, num_reqs=1))
    nonrepeat = dict(step, case_id="new-case", requests=[dict(prompt_len=C1["cached_low"],
                     num_computed_tokens=C1["cached_low"], num_scheduled_tokens=1)])
    records = [engine, step, dict(kind="call", case_id=step["case_id"]),
               dict(kind="interval", case_id=step["case_id"]), nonrepeat,
               dict(kind="call", case_id=nonrepeat["case_id"]), dict(kind="interval", case_id=nonrepeat["case_id"])]
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "dsv4_raw_e0.jsonl"
    original = "".join(json.dumps(r) + "\n" for r in records)
    path.write_text(original)
    files, excluded = promotion.copy_supplement_raw(raw, tmp_path / "filtered")
    assert files == [path] and excluded == ["renamed-repeat"]
    assert path.read_text() == original
    filtered = [json.loads(line) for line in (tmp_path / "filtered" / path.name).read_text().splitlines()]
    assert filtered == [records[0], *records[4:]]
    with pytest.raises(ValueError, match="No supplement"):
        promotion.copy_supplement_raw(tmp_path / "absent", tmp_path / "unused")


def test_synthetic_supplement_merge_loads_with_different_package_origin(
        tmp_path, promotion, plan, model, replica, balanced_routing):
    base_dir, supplement_dir = (tmp_path / name for name in ("base", "supplement"))
    base_dir.mkdir()
    supplement_dir.mkdir()
    supplement_routing = dict(balanced_routing, origin=str(supplement_dir / "balanced_routing.py"))
    base = aggregate(base_dir, plan, [dict(case="base")], routing=balanced_routing)
    supplement = aggregate(supplement_dir, plan, [dict(case="supplement", tokens=4)],
                           routing=supplement_routing)
    filtered = tmp_path / "filtered"
    raw_files, excluded = promotion.copy_supplement_raw(supplement_dir / "raw", filtered)
    assert not excluded
    raw_file, = raw_files
    assert (filtered / raw_file.name).read_bytes() == raw_file.read_bytes()
    engine = json.loads((filtered / raw_file.name).read_text().splitlines()[0])
    assert engine["moe_routing"]["origin"] == supplement_routing["origin"]
    assert engine["moe_routing"]["origin"] != balanced_routing["origin"]
    merged = tmp_path / base.name
    replica.moe_gating_routing_runtime_path = "balanced_topk"
    base_rows, supplement_rows = load(base, model, replica), load(supplement, model, replica)
    assert promotion.union_csv([base, supplement], merged) == len(base_rows) + len(supplement_rows)
    loaded = load(merged, model, replica)
    assert loaded.runtime_fingerprint.nunique() == 1
    assert loaded.moe_routing.nunique() == 2
    expected = pd.concat([base_rows, supplement_rows], ignore_index=True)
    pd.testing.assert_frame_equal(loaded, expected)
