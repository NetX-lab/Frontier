"""Native workload construction for short extends and concurrent prefills."""

import random
import sys
from types import SimpleNamespace

import pytest

from frontier.profiling.attention import dsa_vllm_runner as runner
from frontier.profiling.attention.dsa_plan import (
    DsaMeasurementLimits, PHASE_PREFILL, build_dsa_profile_plan, load_hf_config,
)


@pytest.fixture(scope="module")
def plan():
    return build_dsa_profile_plan(
        model_name="zai-org/GLM-5.3",
        hf_config=load_hf_config("data/config/models/zai-org__GLM-5.3.json"),
        target_tensor_parallel_size=8, num_layers=8,
        limits=DsaMeasurementLimits(
            max_context_len=33024, max_kv_tokens_per_lane=205632,
            max_num_seqs=1024, chunk_budgets=(256, 512, 1024, 2048, 4096, 8192),
            block_size=64, decode_warmup_steps=2, decode_measured_steps=8,
            prefill_warmup_repeats=1, prefill_measured_repeats=3,
        ),
    )


def test_cached_extends_cover_query_and_context_boundaries(plan):
    cases = [case for case in plan.cases if "-extend-q" in case.case_id]
    assert {2, 4, 8, 16, 32, 64, 128, 256} == {
        int(case.case_id.split("-q")[1].split("-")[0]) for case in cases
    }
    for case in cases:
        context = int(case.case_id.split("-c")[1])
        query = case.prompt_len - context
        budget = plan.engine(case.engine_index).chunk_budget
        assert context % budget == 0
        assert 2 <= query <= 256
        assert case.prompt_len + case.max_tokens <= plan.limits.max_context_len
    assert any(case.prompt_len == 32768 + 256 for case in cases) is False
    assert any(case.prompt_len == 32768 + 128 for case in cases)


def test_concurrent_short_prompts_fit_one_native_step(plan):
    cases = [case for case in plan.cases if "-extend-b" in case.case_id]
    assert max(case.batch_size for case in cases) == 1024
    for case in cases:
        assert case.batch_size * case.prompt_len <= plan.engine(case.engine_index).chunk_budget
        assert case.batch_size * (case.prompt_len + 1) <= plan.limits.max_kv_tokens_per_lane
        assert runner._case_window(case, plan.engine(case.engine_index)) == (1, 3)


def test_cached_multi_prefills_use_a_separate_native_chunk_cap(plan):
    engine = plan.engines[6]
    assert engine.prefill_query_len == 512
    assert all(engine.prefill_query_len == 0 for engine in plan.engines[:6])
    cases = plan.cases_for_engine(engine.engine_index)
    assert {case.batch_size for case in cases} == {2, 4, 8, 16}
    for case in cases:
        assert case.batch_size * engine.prefill_query_len <= engine.chunk_budget
        assert case.prompt_len % engine.prefill_query_len == 0
        assert case.batch_size * (case.prompt_len + 1) <= plan.limits.max_kv_tokens_per_lane
        chunks = case.prompt_len // engine.prefill_query_len
        assert runner._case_window(case, engine) == (chunks, 3 * chunks)
    large = plan.engines[7]
    assert large.prefill_query_len == 4096
    assert [(case.batch_size, case.prompt_len) for case in plan.cases_for_engine(7)] == [(2, 32768)]


def test_prefill_boundary_and_last_legal_decode_context_are_covered(plan):
    prefills = [case for case in plan.cases if case.phase == PHASE_PREFILL]
    assert any(case.prompt_len == 257 and case.batch_size == 1 for case in prefills)
    assert any(case.prompt_len == 257 and case.batch_size == 31 for case in prefills)
    edge = next(case for case in plan.cases if case.case_id == "e5-decode-b1-p33013")
    assert edge.prompt_len + edge.max_tokens == plan.limits.max_context_len
    assert edge.max_tokens == edge.warmup + edge.measured + 1


def test_prefill_runner_enqueues_the_whole_batch_before_stepping(plan, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    case = next(case for case in plan.cases if "-extend-b2-q2" in case.case_id)
    pending, batches = [], []

    def step():
        batches.append(tuple(pending))
        pending.clear()

    engine = SimpleNamespace(
        add_request=lambda rid, prompt, params: pending.append((rid, len(prompt["prompt_token_ids"]))),
        has_unfinished_requests=lambda: bool(pending), step=step,
    )

    def rpc(function, args):
        if function is runner.finish_case:
            return [{"recorded": case.measured, "steps": [], "calls": []}]

    llm = SimpleNamespace(llm_engine=engine, collective_rpc=rpc)
    runner.run_case(llm, plan, case, "CUDA_EVENT", random.Random(1234), 100, tmp_path, False)
    assert len(batches) == case.warmup + case.measured
    assert all(len(batch) == 2 and {length for _, length in batch} == {2} for batch in batches)
    assert len({rid for batch in batches for rid, _ in batch}) == 2 * len(batches)


def test_build_engine_preserves_original_arguments_and_adds_native_chunk_cap(plan, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **kw: kw))
    original = runner.build_engine(plan, 0, "model", 1234, 0.85)
    cached = runner.build_engine(plan, 6, "model", 1234, 0.85)
    assert original["long_prefill_token_threshold"] == 0
    assert cached["long_prefill_token_threshold"] == 512
    assert cached["max_num_batched_tokens"] == 8192
    for args in (original, cached):
        assert args["tensor_parallel_size"] == 1
        assert args["load_format"] == "dummy"
        assert args["enforce_eager"] is True
        assert args["enable_prefix_caching"] is False
