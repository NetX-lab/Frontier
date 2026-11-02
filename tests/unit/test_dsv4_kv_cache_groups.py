"""CPU contract for vLLM 2cf0a69 DeepSeek-V4 decoder KV groups and planners."""

from collections import Counter

import pytest

from frontier.attention.deepseek_v4 import (
    ATTENTION_CACHE, INDEXER_CACHE, SWA_CACHE, COMPRESSOR_STATE_CACHE,
    INDEX_COMPRESSOR_STATE_CACHE, UNCOMPRESSED_VARIANT, CSA_VARIANT, HCA_VARIANT,
    _approximate_gcd, deepseek_v4_kv_cache_groups, deepseek_v4_tile_schedule_planners,
)


def variants(count):
    return (UNCOMPRESSED_VARIANT, UNCOMPRESSED_VARIANT) + tuple(
        CSA_VARIANT if layer % 2 == 0 else HCA_VARIANT for layer in range(2, count)
    )


@pytest.mark.parametrize("count,values,chunk,swa_count,planners", [
    (43, [21, 43, 21, 20], 22, 2, [0, 1, 2, 3]),
    (6, [2, 6, 2, 2], 2, 3, list(range(6))),
    (7, [3, 7, 3, 2], 3, 3, list(range(7))),
    (8, [3, 8, 3, 3], 3, 3, list(range(8))),
    (9, [4, 9, 4, 3], 4, 3, list(range(8))),
    (10, [4, 10, 4, 4], 4, 3, list(range(8))),
    (11, [5, 11, 5, 4], 6, 2, [0, 1, 2, 3]),
    (12, [5, 12, 5, 5], 6, 2, [0, 1, 2, 3]),
    (13, [6, 13, 6, 5], 7, 2, [0, 1, 2, 3]),
])
def test_worked_layouts(count, values, chunk, swa_count, planners):
    layout = variants(count)
    csa, hca = list(range(2, count, 2)), list(range(3, count, 2))
    assert values == [max(len(csa), len(hca)), count, len(csa), len(hca)]
    assert _approximate_gcd(values, lower_bound=values[0]) == chunk
    groups = deepseek_v4_kv_cache_groups(layout)
    expected = (
        tuple(sorted([(i, ATTENTION_CACHE) for i in csa + hca]
                     + [(i, INDEXER_CACHE) for i in csa])),
        *(tuple((i, SWA_CACHE) for i in range(offset, count, swa_count))
          for offset in range(swa_count)),
        tuple(sorted((i, cache) for i in csa
                     for cache in (COMPRESSOR_STATE_CACHE, INDEX_COMPRESSOR_STATE_CACHE))),
        tuple((i, COMPRESSOR_STATE_CACHE) for i in hca),
    )
    assert groups == expected
    assert len(groups) == 1 + swa_count + 1 + 1
    assert [i for i, planner in enumerate(deepseek_v4_tile_schedule_planners(layout))
            if planner] == planners
    assert all(tuple(sorted(group)) == group for group in groups)


def test_full_groups_match_written_native_inventory_and_role_counts():
    # /data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-canonical-20261025-01/
    # artifacts/nodes/gpu-h200-0082.lgcm.sh.istep.fun/serving/rank0.json,
    # inventory.cache.kv_cache_groups: full MLA, even SWA, odd SWA, CSA/HCA states.
    native = (
        tuple(sorted([(i, ATTENTION_CACHE) for i in range(2, 43)]
                     + [(i, INDEXER_CACHE) for i in range(2, 43, 2)])),
        tuple((i, SWA_CACHE) for i in range(0, 43, 2)),
        tuple((i, SWA_CACHE) for i in range(1, 43, 2)),
        tuple(sorted((i, cache) for i in range(2, 43, 2)
                     for cache in (COMPRESSOR_STATE_CACHE, INDEX_COMPRESSOR_STATE_CACHE))),
        tuple((i, COMPRESSOR_STATE_CACHE) for i in range(3, 43, 2)),
    )
    assert deepseek_v4_kv_cache_groups(variants(43)) == native
    assert Counter(zip(variants(43), deepseek_v4_tile_schedule_planners(variants(43)))) == {
        (UNCOMPRESSED_VARIANT, True): 2, (CSA_VARIANT, True): 1,
        (CSA_VARIANT, False): 20, (HCA_VARIANT, True): 1, (HCA_VARIANT, False): 19,
    }


@pytest.mark.parametrize("values,lower_bound,expected", [
    ([5, 11, 5, 4], 5, 6), ([5, 12, 5, 5], 5, 6),
    ([2, 6, 2, 2], 2, 2), ([2, 3], 8, 8), ([2, 4], None, 2),
    ([2, 4], 0, 2), ([2, 4], -4, 2), ([5], None, 5),
])
def test_approximate_gcd_ties_and_bounds(values, lower_bound, expected):
    assert _approximate_gcd(values, lower_bound=lower_bound) == expected


@pytest.mark.parametrize("values,message", [([], "non-empty"), ([0], "positive"),
                                               ([-1, 2], "positive")])
def test_approximate_gcd_rejects_invalid_values(values, message):
    with pytest.raises(ValueError, match=message):
        _approximate_gcd(values)


@pytest.mark.parametrize("layout,message", [
    ((UNCOMPRESSED_VARIANT, "other", HCA_VARIANT), "Unknown.*other.*layer 1"),
    ((UNCOMPRESSED_VARIANT, HCA_VARIANT), "both CSA and HCA"),
    ((UNCOMPRESSED_VARIANT, CSA_VARIANT), "both CSA and HCA"),
    ((), "both CSA and HCA"),
])
@pytest.mark.parametrize("function", [deepseek_v4_kv_cache_groups, deepseek_v4_tile_schedule_planners])
def test_invalid_layouts_raise(layout, message, function):
    with pytest.raises(ValueError, match=message):
        function(layout)
