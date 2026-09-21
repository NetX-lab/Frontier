"""Deterministic case table for the refactor fidelity matrix.

The matrix answers one question: does a behavior-preserving refactor change any
simulator output?  Every case therefore runs a checked-in example wrapper, so
the matrix exercises the same configuration surface the release documents, and
a case cannot silently drift away from a supported recipe.

Cases are written out explicitly instead of being generated from a full cross
product.  A reviewer can read the tables below and see exactly which
configuration each case pins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


COLOCATION_OFFLINE_DENSE = "examples/architecture/co-location/offline/dense_model_basic.sh"
COLOCATION_OFFLINE_MOE = "examples/architecture/co-location/offline/moe_model_basic.sh"
COLOCATION_OFFLINE_THINKING = "examples/architecture/co-location/offline/thinking_mode_basic.sh"
COLOCATION_OFFLINE_SPEC_DEC = "examples/architecture/co-location/offline/moe_spec_dec.sh"
COLOCATION_OFFLINE_PREFIX_CACHE = "examples/architecture/co-location/offline/moe_prefix_caching.sh"
COLOCATION_ONLINE_DENSE = "examples/architecture/co-location/online/dense_model_basic_online.sh"
COLOCATION_ONLINE_MOE = "examples/architecture/co-location/online/moe_model_basic_online.sh"
COLOCATION_ONLINE_THINKING = "examples/architecture/co-location/online/thinking_mode_basic_online.sh"
COLOCATION_ONLINE_SPEC_DEC = "examples/architecture/co-location/online/moe_spec_dec_online.sh"
COLOCATION_ONLINE_PREFIX_CACHE = "examples/architecture/co-location/online/moe_prefix_caching_online.sh"

PDD_OFFLINE_DENSE = "examples/architecture/pdd/offline/dense_model_basic.sh"
PDD_OFFLINE_MOE = "examples/architecture/pdd/offline/moe_model_basic.sh"
PDD_OFFLINE_THINKING = "examples/architecture/pdd/offline/thinking_mode_basic.sh"
PDD_OFFLINE_SPEC_DEC = "examples/architecture/pdd/offline/moe_spec_dec.sh"
PDD_OFFLINE_PREFIX_CACHE = "examples/architecture/pdd/offline/moe_prefix_caching.sh"
PDD_ONLINE_DENSE = "examples/architecture/pdd/online/dense_model_basic_online.sh"
PDD_ONLINE_MOE = "examples/architecture/pdd/online/moe_model_basic_online.sh"

PDAF_OFFLINE_DENSE = "examples/architecture/pd-af-disagg/offline/dense_model_basic.sh"
PDAF_OFFLINE_MOE = "examples/architecture/pd-af-disagg/offline/moe_model_basic.sh"
PDAF_OFFLINE_MOE_EP = "examples/architecture/pd-af-disagg/offline/moe_model_ep.sh"
PDAF_OFFLINE_DENSE_CUDA_GRAPH = "examples/architecture/pd-af-disagg/offline/dense_cuda_graph.sh"
PDAF_OFFLINE_MOE_CUDA_GRAPH = "examples/architecture/pd-af-disagg/offline/moe_cuda_graph.sh"
PDAF_ONLINE_DENSE = "examples/architecture/pd-af-disagg/online/dense_model_basic_online.sh"
PDAF_ONLINE_MOE = "examples/architecture/pd-af-disagg/online/moe_model_basic_online.sh"
PDAF_ONLINE_MOE_EP = "examples/architecture/pd-af-disagg/online/moe_model_ep_online.sh"
PDAF_ONLINE_MOE_CUDA_GRAPH = "examples/architecture/pd-af-disagg/online/moe_cuda_graph_online.sh"

PROFILING_SMOKE_DENSE_CSV = "examples/profiling/smoke_simulator_dense_csv.sh"
PROFILING_SMOKE_MOE_CSV = "examples/profiling/smoke_simulator_moe_csv.sh"


@dataclass(frozen=True)
class FidelityCase:
    """One reproducible simulator run.

    ``env`` overrides the wrapper's uppercase variables.  ``extra_args`` is
    appended after the wrapper's ``--`` separator and therefore reaches
    ``frontier.main`` directly, which is how the matrix covers flags a wrapper
    does not expose.
    """

    case_id: str
    group: str
    script: str
    purpose: str
    env: Mapping[str, str] = field(default_factory=dict)
    extra_args: Sequence[str] = field(default_factory=tuple)
    #: Non-dummy cases train predictors into the label-wide cache directory.
    #: They are run serially and before the parallel cases so that the cache is
    #: populated once per label rather than raced by several writers.
    uses_trained_predictor: bool = False

    def as_record(self) -> dict:
        return {
            "case_id": self.case_id,
            "group": self.group,
            "script": self.script,
            "purpose": self.purpose,
            "env": dict(self.env),
            "extra_args": list(self.extra_args),
            "uses_trained_predictor": self.uses_trained_predictor,
        }


def _colocation_dense_offline() -> list[FidelityCase]:
    """Co-location offline dense: request population, parallelism, runtime toggles."""

    rows: list[tuple[str, str, dict[str, str], tuple[str, ...]]] = [
        (
            "coloc_dense_offline_small",
            "smallest deterministic dense run",
            {"NUM_REQUESTS": "4", "PREFILL_TOKENS": "128", "DECODE_TOKENS": "16",
             "ATTN_TP": "1", "DECODE_CUDA_GRAPH_MODE": "none"},
            (),
        ),
        (
            "coloc_dense_offline_default",
            "the published example defaults",
            {},
            (),
        ),
        (
            "coloc_dense_offline_long_prefill",
            "3584-token prefill through chunked prefill, just inside the 4096 context",
            {"NUM_REQUESTS": "8", "PREFILL_TOKENS": "3584", "DECODE_TOKENS": "16",
             "MAX_TOKENS_IN_BATCH": "2048", "LONG_PREFILL_TOKEN_THRESHOLD": "512"},
            (),
        ),
        (
            "coloc_dense_offline_no_chunked_prefill",
            "chunked prefill disabled",
            {"NUM_REQUESTS": "8", "PREFILL_TOKENS": "1024", "DECODE_TOKENS": "32",
             "ENABLE_CHUNKED_PREFILL": "false", "LONG_PREFILL_TOKEN_THRESHOLD": "0",
             "DECODE_CUDA_GRAPH_MODE": "none"},
            (),
        ),
        (
            "coloc_dense_offline_pp2",
            "two pipeline stages",
            {"NUM_REQUESTS": "8", "PREFILL_TOKENS": "512", "DECODE_TOKENS": "32",
             "PP": "2", "DECODE_CUDA_GRAPH_MODE": "none"},
            (),
        ),
        (
            "coloc_dense_offline_tp1",
            "no attention tensor parallelism",
            {"NUM_REQUESTS": "8", "PREFILL_TOKENS": "512", "DECODE_TOKENS": "32",
             "ATTN_TP": "1", "DECODE_CUDA_GRAPH_MODE": "none"},
            (),
        ),
        (
            "coloc_dense_offline_two_replicas",
            "cluster scheduler distributes over two replicas",
            {"NUM_REQUESTS": "16", "PREFILL_TOKENS": "256", "DECODE_TOKENS": "32",
             "NUM_REPLICAS": "2"},
            (),
        ),
        (
            "coloc_dense_offline_many_requests",
            "64 short requests",
            {"NUM_REQUESTS": "64", "PREFILL_TOKENS": "128", "DECODE_TOKENS": "16"},
            (),
        ),
        (
            "coloc_dense_offline_attn_dp2",
            "two attention DP lanes",
            {"NUM_REQUESTS": "8", "PREFILL_TOKENS": "512", "DECODE_TOKENS": "32",
             "ATTN_TP": "2", "DECODE_CUDA_GRAPH_MODE": "none"},
            ("--replica_config_attn_dp", "2"),
        ),
        (
            "coloc_dense_offline_cuda_graph_piecewise",
            "piecewise decode CUDA graph mode",
            {"NUM_REQUESTS": "8", "PREFILL_TOKENS": "512", "DECODE_TOKENS": "32",
             "DECODE_CUDA_GRAPH_MODE": "piecewise"},
            (),
        ),
        (
            "coloc_dense_offline_sarathi",
            "Sarathi replica scheduler",
            {"NUM_REQUESTS": "8", "PREFILL_TOKENS": "512", "DECODE_TOKENS": "32",
             "REPLICA_SCHEDULER": "sarathi", "DECODE_CUDA_GRAPH_MODE": "none"},
            (),
        ),
        (
            "coloc_dense_offline_sglang",
            "SGLang-style prefill-first scheduler",
            {"NUM_REQUESTS": "8", "PREFILL_TOKENS": "512", "DECODE_TOKENS": "32",
             "REPLICA_SCHEDULER": "sglang", "DECODE_CUDA_GRAPH_MODE": "none"},
            (),
        ),
        (
            "coloc_dense_offline_dummy_time_quarter",
            "different dummy operator latency",
            {"NUM_REQUESTS": "8", "PREFILL_TOKENS": "512", "DECODE_TOKENS": "32",
             "DUMMY_EXEC_TIME_MS": "0.25"},
            (),
        ),
    ]
    return [
        FidelityCase(case_id, "colocation_offline_dense", COLOCATION_OFFLINE_DENSE,
                     purpose, env, extra)
        for case_id, purpose, env, extra in rows
    ]


def _colocation_moe_offline() -> list[FidelityCase]:
    """Co-location offline MoE: EP/TP domains, routing distributions, topk."""

    rows: list[tuple[str, str, dict[str, str], tuple[str, ...]]] = [
        ("coloc_moe_offline_default", "the published MoE example defaults", {}, ()),
        (
            "coloc_moe_offline_ep1",
            "single expert-parallel domain (attn_tp*attn_dp == moe_tp*moe_ep)",
            {"MOE_EP": "1", "MOE_TP": "1", "ATTN_TP": "1"},
            (),
        ),
        (
            "coloc_moe_offline_moe_tp1",
            "expert parallelism without intra-expert TP",
            {"MOE_TP": "1", "MOE_EP": "2", "ATTN_TP": "2"},
            (),
        ),
        (
            "coloc_moe_offline_routing_random",
            "random expert-load distribution",
            {"MOE_ROUTING_DISTRIBUTION_TYPE": "random"},
            (),
        ),
        (
            "coloc_moe_offline_routing_skewed",
            "skewed expert-load distribution",
            {"MOE_ROUTING_DISTRIBUTION_TYPE": "skewed"},
            (),
        ),
        (
            "coloc_moe_offline_routing_zipf",
            "zipf expert-load distribution",
            {"MOE_ROUTING_DISTRIBUTION_TYPE": "zipf"},
            (),
        ),
        (
            "coloc_moe_offline_topk4",
            "router top-k of four",
            {"ROUTER_TOPK": "4"},
            (),
        ),
        (
            "coloc_moe_offline_long_prefill",
            "2048-token prefill with MoE layers",
            {"NUM_REQUESTS": "4", "PREFILL_TOKENS": "2048", "DECODE_TOKENS": "16",
             "MAX_TOKENS_IN_BATCH": "2048", "LONG_PREFILL_TOKEN_THRESHOLD": "256"},
            (),
        ),
        (
            "coloc_moe_offline_many_requests",
            "32 short MoE requests",
            {"NUM_REQUESTS": "32", "PREFILL_TOKENS": "128", "DECODE_TOKENS": "16"},
            (),
        ),
        (
            "coloc_moe_offline_cuda_graph_none",
            "MoE without decode CUDA graph modeling",
            {"DECODE_CUDA_GRAPH_MODE": "none"},
            (),
        ),
        (
            # The MoE wrapper enforces ATTN_TP == MOE_TP * MOE_EP, so it cannot
            # express an attn_dp > 1 shared domain; the dense matrix covers DP
            # lanes and this case widens expert parallelism instead.
            "coloc_moe_offline_ep4",
            "four expert-parallel domains over eight experts",
            {"ATTN_TP": "4", "MOE_TP": "1", "MOE_EP": "4",
             "DECODE_CUDA_GRAPH_MODE": "none"},
            (),
        ),
    ]
    return [
        FidelityCase(case_id, "colocation_offline_moe", COLOCATION_OFFLINE_MOE,
                     purpose, env, extra)
        for case_id, purpose, env, extra in rows
    ]


def _colocation_online() -> list[FidelityCase]:
    """Co-location online arrivals across three arrival rates."""

    cases: list[FidelityCase] = []
    for qps in ("0.5", "2.0", "8.0"):
        tag = qps.replace(".", "p")
        cases.append(FidelityCase(
            case_id=f"coloc_dense_online_qps{tag}",
            group="colocation_online",
            script=COLOCATION_ONLINE_DENSE,
            purpose=f"dense online arrivals at {qps} qps",
            env={"NUM_REQUESTS": "16", "PREFILL_TOKENS": "512",
                 "DECODE_TOKENS": "64", "QPS": qps},
        ))
        cases.append(FidelityCase(
            case_id=f"coloc_moe_online_qps{tag}",
            group="colocation_online",
            script=COLOCATION_ONLINE_MOE,
            purpose=f"MoE online arrivals at {qps} qps",
            env={"NUM_REQUESTS": "16", "PREFILL_TOKENS": "256",
                 "DECODE_TOKENS": "32", "QPS": qps},
        ))
    return cases


def _colocation_features() -> list[FidelityCase]:
    """Thinking mode, speculative decoding, and prefix caching, offline and online."""

    rows = [
        ("coloc_thinking_offline", COLOCATION_OFFLINE_THINKING,
         "thinking mode with tool-call rounds", {}),
        ("coloc_thinking_online", COLOCATION_ONLINE_THINKING,
         "thinking mode with online arrivals", {}),
        ("coloc_spec_dec_offline", COLOCATION_OFFLINE_SPEC_DEC,
         "speculative decoding on a MoE model", {}),
        ("coloc_spec_dec_online", COLOCATION_ONLINE_SPEC_DEC,
         "speculative decoding with online arrivals", {}),
        ("coloc_prefix_cache_offline", COLOCATION_OFFLINE_PREFIX_CACHE,
         "prefix caching over the shared-session fixture", {}),
        ("coloc_prefix_cache_online", COLOCATION_ONLINE_PREFIX_CACHE,
         "prefix caching with online arrivals", {}),
        ("coloc_spec_dec_offline_ntokens4", COLOCATION_OFFLINE_SPEC_DEC,
         "four speculative tokens per iteration",
         {"NUM_SPECULATIVE_TOKENS": "4", "COMMITTED_TOKENS_PER_ITERATION": "4"}),
    ]
    return [
        FidelityCase(case_id, "colocation_features", script, purpose, env)
        for case_id, script, purpose, env in rows
    ]


def _pdd() -> list[FidelityCase]:
    """Sequential prefill/decode disaggregation."""

    rows = [
        ("pdd_dense_offline_default", PDD_OFFLINE_DENSE,
         "the published sequential PDD dense defaults", {}),
        ("pdd_dense_offline_two_prefill_replicas", PDD_OFFLINE_DENSE,
         "two prefill replicas feeding one decode replica",
         {"PREFILL_REPLICAS": "2", "NUM_REQUESTS": "16", "PREFILL_TOKENS": "256"}),
        ("pdd_dense_offline_two_decode_replicas", PDD_OFFLINE_DENSE,
         "one prefill replica feeding two decode replicas",
         {"DECODE_REPLICAS": "2", "NUM_REQUESTS": "16", "DECODE_TOKENS": "32"}),
        ("pdd_dense_offline_tp2", PDD_OFFLINE_DENSE,
         "tensor parallelism on both roles",
         {"PREFILL_ATTN_TP": "2", "DECODE_ATTN_TP": "2", "NUM_REQUESTS": "8"}),
        ("pdd_dense_offline_no_chunked_prefill", PDD_OFFLINE_DENSE,
         "chunked prefill disabled",
         {"ENABLE_CHUNKED_PREFILL": "false", "LONG_PREFILL_TOKEN_THRESHOLD": "0"}),
        ("pdd_moe_offline_default", PDD_OFFLINE_MOE,
         "the published sequential PDD MoE defaults", {}),
        ("pdd_moe_offline_ep1", PDD_OFFLINE_MOE,
         "PDD MoE with a single expert-parallel domain",
         {"PREFILL_MOE_EP": "1", "DECODE_MOE_EP": "1",
          "PREFILL_ATTN_TP": "1", "DECODE_ATTN_TP": "1"}),
        ("pdd_moe_offline_routing_skewed", PDD_OFFLINE_MOE,
         "PDD MoE with a skewed expert-load distribution",
         {"MOE_ROUTING_DISTRIBUTION_TYPE": "skewed"}),
        ("pdd_thinking_offline", PDD_OFFLINE_THINKING,
         "PDD thinking mode with KV handoffs", {}),
        ("pdd_spec_dec_offline", PDD_OFFLINE_SPEC_DEC,
         "PDD speculative decoding", {}),
        ("pdd_prefix_cache_offline", PDD_OFFLINE_PREFIX_CACHE,
         "PDD prefix caching over the shared-session fixture", {}),
        ("pdd_dense_online_qps2", PDD_ONLINE_DENSE,
         "PDD dense online arrivals", {"QPS": "2.0", "NUM_REQUESTS": "16"}),
        ("pdd_moe_online_qps2", PDD_ONLINE_MOE,
         "PDD MoE online arrivals", {"QPS": "2.0", "NUM_REQUESTS": "16"}),
        ("pdd_dense_online_qps8", PDD_ONLINE_DENSE,
         "PDD dense at a higher arrival rate",
         {"QPS": "8.0", "NUM_REQUESTS": "16", "PREFILL_TOKENS": "256"}),
    ]
    return [
        FidelityCase(case_id, "pd_disaggregation", script, purpose, env)
        for case_id, script, purpose, env in rows
    ]


def _pdaf() -> list[FidelityCase]:
    """Sequential attention/FFN disaggregation with KV and M2N transfers."""

    rows = [
        ("pdaf_dense_offline_default", PDAF_OFFLINE_DENSE,
         "the published sequential PD-AF dense defaults", {}),
        ("pdaf_moe_offline_default", PDAF_OFFLINE_MOE,
         "PD-AF MoE with one expert-parallel domain", {}),
        ("pdaf_moe_offline_ep2", PDAF_OFFLINE_MOE_EP,
         "PD-AF MoE with two expert-parallel domains", {}),
        ("pdaf_moe_offline_ep2_many", PDAF_OFFLINE_MOE_EP,
         "PD-AF MoE EP with more requests",
         {"NUM_REQUESTS": "16", "PREFILL_TOKENS": "128", "DECODE_TOKENS": "16"}),
        ("pdaf_dense_offline_cuda_graph", PDAF_OFFLINE_DENSE_CUDA_GRAPH,
         "PD-AF dense with the global CUDA graph contract", {}),
        ("pdaf_moe_offline_cuda_graph", PDAF_OFFLINE_MOE_CUDA_GRAPH,
         "PD-AF MoE with the global CUDA graph contract", {}),
        ("pdaf_dense_online_default", PDAF_ONLINE_DENSE,
         "PD-AF dense online arrivals", {}),
        ("pdaf_moe_online_default", PDAF_ONLINE_MOE,
         "PD-AF MoE online arrivals", {}),
        ("pdaf_moe_online_ep2", PDAF_ONLINE_MOE_EP,
         "PD-AF MoE EP online arrivals", {}),
        ("pdaf_moe_online_cuda_graph", PDAF_ONLINE_MOE_CUDA_GRAPH,
         "PD-AF MoE online with the global CUDA graph contract", {}),
    ]
    return [
        FidelityCase(case_id, "pd_af_disaggregation", script, purpose, env)
        for case_id, script, purpose, env in rows
    ]


def _trained_predictor() -> list[FidelityCase]:
    """Runs with the dummy predictor disabled, fed by checked-in profiling CSVs.

    These exercise dataset loading, training-identity computation, the estimator
    registry, and the persistent predictor cache, which the dummy-mode cases
    never reach.
    """

    rows: list[tuple[str, str, str, dict[str, str], tuple[str, ...]]] = [
        # These two wrappers default DATA_DIR_BASE to an absolute path under
        # the repository root, and the predictor cache key includes the
        # profiling input paths.  A repository-relative base keeps the key
        # identical across two checkouts so that a real change of training
        # identity is visible in the cache file names.
        (
            "trained_dense_csv_smoke",
            PROFILING_SMOKE_DENSE_CSV,
            "checked-in dense CSVs fed straight into the simulator",
            {"DATA_DIR_BASE": "data/profiling"},
            (),
        ),
        (
            "trained_moe_csv_smoke",
            PROFILING_SMOKE_MOE_CSV,
            "checked-in MoE CSVs fed straight into the simulator",
            {"DATA_DIR_BASE": "data/profiling"},
            (),
        ),
        (
            "trained_coloc_dense_h800",
            COLOCATION_OFFLINE_DENSE,
            "co-location dense on h800 profiles without dummy mode",
            {"ENABLE_DUMMY_MODE": "false", "DEVICE": "h800",
             "MODEL_NAME": "llama2_7b_dense_example", "ATTN_TP": "1",
             "NUM_REQUESTS": "4", "PREFILL_TOKENS": "128", "DECODE_TOKENS": "8",
             "DECODE_CUDA_GRAPH_MODE": "none", "MAX_TOKENS_IN_BATCH": "256",
             "LONG_PREFILL_TOKEN_THRESHOLD": "64"},
            ("--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling",),
        ),
        (
            "trained_coloc_dense_h800_cuda_graph",
            COLOCATION_OFFLINE_DENSE,
            "same dense profiles through the kernel-only CUDA graph path",
            {"ENABLE_DUMMY_MODE": "false", "DEVICE": "h800",
             "MODEL_NAME": "llama2_7b_dense_example", "ATTN_TP": "1",
             "NUM_REQUESTS": "4", "PREFILL_TOKENS": "128", "DECODE_TOKENS": "8",
             "DECODE_CUDA_GRAPH_MODE": "full_decode_only",
             "MAX_TOKENS_IN_BATCH": "256", "LONG_PREFILL_TOKEN_THRESHOLD": "64"},
            ("--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling",),
        ),
        (
            "trained_coloc_moe_phi_h800",
            COLOCATION_OFFLINE_MOE,
            "co-location MoE on the tiny Phi profiles without dummy mode",
            {"ENABLE_DUMMY_MODE": "false", "MODEL_NAME": "Phi-tiny-MoE-instruct",
             "ATTN_TP": "1", "MOE_TP": "1", "MOE_EP": "1",
             "MOE_ROUTING_DISTRIBUTION_TYPE": "random",
             "NUM_REQUESTS": "4", "PREFILL_TOKENS": "128", "DECODE_TOKENS": "8",
             "DECODE_CUDA_GRAPH_MODE": "none", "MAX_TOKENS_IN_BATCH": "256",
             "LONG_PREFILL_TOKEN_THRESHOLD": "64"},
            ("--replica_config_device", "h800",
             "--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling"),
        ),
        (
            "trained_coloc_moe_qwen3_h800",
            COLOCATION_OFFLINE_MOE,
            "co-location MoE on the tiny Qwen3 profiles without dummy mode",
            {"ENABLE_DUMMY_MODE": "false", "MODEL_NAME": "Qwen3-30B-A3B-tiny",
             "ATTN_TP": "1", "MOE_TP": "1", "MOE_EP": "1",
             "MOE_ROUTING_DISTRIBUTION_TYPE": "random",
             "NUM_REQUESTS": "4", "PREFILL_TOKENS": "128", "DECODE_TOKENS": "8",
             "DECODE_CUDA_GRAPH_MODE": "none", "MAX_TOKENS_IN_BATCH": "256",
             "LONG_PREFILL_TOKEN_THRESHOLD": "64"},
            ("--replica_config_device", "h800",
             "--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling"),
        ),
    ]
    return [
        FidelityCase(case_id, "trained_predictor", script, purpose, env, extra,
                     uses_trained_predictor=True)
        for case_id, script, purpose, env, extra in rows
    ]


def _dp_placement() -> list[FidelityCase]:
    """Cases whose outcome depends on how requests are placed on DP lanes.

    The rest of the matrix runs one attention DP lane, where lane placement
    cannot vary, so it gives a change to DP placement almost no coverage. These
    cases exist to give the next such change a real blast radius.

    Two properties matter and are covered separately. Placement must not depend
    on how an identical request stream is divided across scheduling calls,
    which needs a run that enters the scheduler many times, so most of these
    are online. And placement across replicas and across lanes interact, which
    needs more than one replica as well as more than one lane.

    Only dense co-location cases appear here, and that is a coverage limit
    worth stating rather than a choice. The prefill role reaches the same
    placement path as the monolithic role, but no shipped recipe can give it
    more than one lane: a dense model in a disaggregated architecture is
    rejected with "Dense models do not support attn data parallelism in
    disaggregated mode", and the MoE wrappers require
    ``ATTN_TP == MOE_TP * MOE_EP`` while the runtime requires
    ``attn_tp * attn_dp == moe_tp * moe_ep``, which have no common solution
    above one lane. Placement changes affecting the prefill role therefore
    have to be validated by unit tests, not by this matrix.
    """

    rows: list[tuple[str, str, str, dict[str, str], tuple[str, ...]]] = [
        (
            "dp_dense_online_lanes2",
            COLOCATION_ONLINE_DENSE,
            "two lanes with arrivals spread over many scheduling calls",
            {"NUM_REQUESTS": "16", "PREFILL_TOKENS": "256", "DECODE_TOKENS": "32",
             "QPS": "2.0", "ATTN_TP": "2", "DECODE_CUDA_GRAPH_MODE": "none"},
            ("--replica_config_attn_dp", "2"),
        ),
        (
            "dp_dense_online_lanes4",
            COLOCATION_ONLINE_DENSE,
            "four lanes, so the lane index wraps more than once",
            {"NUM_REQUESTS": "16", "PREFILL_TOKENS": "256", "DECODE_TOKENS": "32",
             "QPS": "2.0", "ATTN_TP": "4", "DECODE_CUDA_GRAPH_MODE": "none"},
            ("--replica_config_attn_dp", "4"),
        ),
        (
            "dp_dense_offline_lanes2_replicas2",
            COLOCATION_OFFLINE_DENSE,
            "two lanes over two replicas, where the replica and lane terms interact",
            {"NUM_REQUESTS": "16", "PREFILL_TOKENS": "256", "DECODE_TOKENS": "32",
             "NUM_REPLICAS": "2", "ATTN_TP": "2", "DECODE_CUDA_GRAPH_MODE": "none"},
            ("--replica_config_attn_dp", "2"),
        ),
        (
            "dp_dense_online_lanes2_replicas2",
            COLOCATION_ONLINE_DENSE,
            "two lanes over two replicas with arrivals spread across calls",
            {"NUM_REQUESTS": "24", "PREFILL_TOKENS": "256", "DECODE_TOKENS": "16",
             "QPS": "4.0", "NUM_REPLICAS": "2", "ATTN_TP": "2",
             "DECODE_CUDA_GRAPH_MODE": "none"},
            ("--replica_config_attn_dp", "2"),
        ),
    ]
    return [
        FidelityCase(case_id, "dp_placement", script, purpose, env, extra)
        for case_id, script, purpose, env, extra in rows
    ]


def build_cases() -> list[FidelityCase]:
    """Return the full ordered matrix.

    Order is stable so that a partial run selected with ``--start``/``--limit``
    covers the same cases on both sides of a comparison.
    """

    cases: list[FidelityCase] = []
    cases.extend(_trained_predictor())
    cases.extend(_colocation_dense_offline())
    cases.extend(_colocation_moe_offline())
    cases.extend(_colocation_online())
    cases.extend(_colocation_features())
    cases.extend(_pdd())
    cases.extend(_pdaf())
    cases.extend(_dp_placement())

    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise ValueError(f"duplicate fidelity case id: {case.case_id}")
        seen.add(case.case_id)
    return cases


def group_counts(cases: Sequence[FidelityCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.group] = counts.get(case.group, 0) + 1
    return counts
