"""Supplement filtering retains engine/step/call/interval identities and gaps."""
import importlib.util
import json
from pathlib import Path

import pytest

from frontier.profiling.attention.v4_attention_windows import C1, repeat_windows

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
