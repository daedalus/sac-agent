"""Hypothesis property tests for UtilsSDK utilities."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from sac import UtilsSDK
from sac.core import SearchResult


@given(st.lists(st.integers()))
@settings(max_examples=100)
def test_flatten_preserves_elements(items):
    result = UtilsSDK.flatten([items])
    assert result == items


@given(st.lists(st.lists(st.integers(), max_size=5), max_size=10))
@settings(max_examples=100)
def test_flatten_nested_preserves_count(lists):
    flat = UtilsSDK.flatten(lists)
    expected = sum(1 for sub in lists for _ in sub)
    assert len(flat) == expected


@given(st.lists(st.integers(max_value=100)))
@settings(max_examples=100)
def test_dedupe_by_identity_returns_unique(items):
    result = UtilsSDK.dedupe_by(items, lambda x: x)
    assert len(set(result)) == len(result)
    assert set(result) == set(items)


@given(st.lists(st.tuples(st.integers(), st.text(max_size=10)), min_size=0, max_size=20))
@settings(max_examples=100)
def test_dedupe_by_field_preserves_unique_keys(pairs):
    items = [{"id": k, "val": v} for k, v in pairs]
    result = UtilsSDK.dedupe_by(items, "id")
    seen_ids = set()
    for item in result:
        assert item["id"] not in seen_ids
        seen_ids.add(item["id"])


@given(st.lists(st.integers(min_value=-100, max_value=100), max_size=30))
@settings(max_examples=100)
def test_filter_by_callable_soundness(items):
    threshold = 0
    result = UtilsSDK.filter_by(items, lambda x: x > threshold)
    for v in result:
        assert v > threshold


@given(st.lists(st.dictionaries(st.text(max_size=5), st.integers(), max_size=5), max_size=10))
@settings(max_examples=100)
def test_filter_by_field_consistency(dicts):
    field = "kind"
    tagged = [{"kind": "a" if i % 2 == 0 else "b", **d} for i, d in enumerate(dicts)]
    result_a = UtilsSDK.filter_by(tagged, field, "a")
    result_b = UtilsSDK.filter_by(tagged, field, "b")
    assert len(result_a) + len(result_b) == len(tagged)
    for r in result_a:
        assert r[field] == "a"
    for r in result_b:
        assert r[field] == "b"


@given(st.lists(st.integers(), max_size=20))
@settings(max_examples=50)
def test_empty_or_single_flatten(items):
    if not items:
        assert UtilsSDK.flatten([]) == []
    assert UtilsSDK.flatten([items]) == items


@given(st.from_regex(r"[a-zA-Z0-9_]{1,20}"), st.from_regex(r"[a-zA-Z0-9_ ]{0,50}"))
@settings(max_examples=50)
def test_join_result_fields_dict_contains_both(title, snippet):
    result = {"title": title, "snippet": snippet}
    joined = UtilsSDK.join_result_fields(result)
    assert title in joined
    assert snippet in joined
    assert " | " in joined


@given(st.from_regex(r"[a-zA-Z0-9_ ]{1,20}"))
@settings(max_examples=50)
def test_join_result_fields_search_result(title):
    r = SearchResult(title=title, url="https://x.com", snippet="snippet", domain="x.com")
    joined = UtilsSDK.join_result_fields(r)
    assert title in joined
    assert "snippet" in joined


@given(
    st.lists(st.dictionaries(st.text(max_size=5), st.integers(), max_size=5), max_size=10),
    st.lists(st.text(max_size=10), max_size=3, min_size=1),
)
@settings(max_examples=50)
def test_summarize_coverage_invariants(items, by_fields):
    summary = UtilsSDK.summarize_coverage(items, by_fields)
    for field in by_fields:
        assert f"Coverage by '{field}'" in summary
        distinct_values: set[str] = set()
        for item in items:
            val = item.get(field) if isinstance(item, dict) else None
            distinct_values.add(str(val) if val is not None else "None")
        for val in distinct_values:
            assert val in summary
