"""Tests for Search as Code (SaC)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from sac import (
    AgenticSearchSDK,
    FilesystemSDK,
    LLMSDKClient,
    SaCAgent,
    Sandbox,
    SearchResult,
    SearchSDK,
    UtilsSDK,
)
from sac.core import _extract_domain, _format_items


class TestSearchResult:
    def test_defaults(self):
        r = SearchResult()
        assert r.url == ""
        assert r.title == ""
        assert r.snippet == ""
        assert r.domain == ""

    def test_all_fields(self):
        r = SearchResult(
            url="https://example.com/page",
            title="Example Page",
            snippet="A test page",
            domain="example.com",
        )
        assert r.url == "https://example.com/page"
        assert r.title == "Example Page"
        assert r.snippet == "A test page"
        assert r.domain == "example.com"


class TestExtractDomain:
    def test_normal_url(self):
        assert _extract_domain("https://www.example.com/path") == "www.example.com"

    def test_no_url(self):
        assert _extract_domain("") == ""

    def test_no_scheme(self):
        assert _extract_domain("example.com/path") == ""


class TestFormatItems:
    def test_simple_list(self):
        result = _format_items([{"a": 1}, {"b": 2}])
        assert "a" in result
        assert "b" in result

    def test_truncation(self):
        large = [{"x": "y" * 5000}]
        result = _format_items(large, max_chars=100)
        assert result.endswith("[truncated]")


EXA_SINGLE = """Title: Welcome to Python.org
URL: https://www.python.org/
Published: N/A
Author: N/A
Highlights:
Welcome to Python.org
Python is a programming language.
[...]"""

EXA_MULTI = """Title: Welcome to Python.org
URL: https://www.python.org/
Published: N/A
Author: N/A
Highlights:
Welcome to Python.org
Python is a programming language.

---

Title: Python 3.14.5 Documentation
URL: https://docs.python.org/3/
Published: N/A
Author: N/A
Highlights:
# Python 3.14.5 documentation
Welcome! This is the official documentation."""

EXA_NO_HIGHLIGHTS = """Title: A Simple Page
URL: https://example.com/
Published: N/A
Author: N/A
Highlights:"""

EXA_EXTRA_TEXT = """Title: Some Article
URL: https://example.com/article
Published: 2024-01-01
Author: Test Author
Highlights:
First line of the article.
Second line here.

Extra paragraph not under highlights.
But still relevant."""


class TestSearchSDKParseExaContent:
    def test_single_result(self):
        sdk = SearchSDK()
        results = sdk._parse_exa_content(EXA_SINGLE)
        assert len(results) == 1
        assert results[0].title == "Welcome to Python.org"
        assert results[0].url == "https://www.python.org/"
        assert "Python is a programming language" in results[0].snippet
        assert results[0].domain == "www.python.org"

    def test_multiple_results(self):
        sdk = SearchSDK()
        results = sdk._parse_exa_content(EXA_MULTI)
        assert len(results) == 2
        assert results[0].title == "Welcome to Python.org"
        assert results[1].title == "Python 3.14.5 Documentation"
        assert results[1].url == "https://docs.python.org/3/"
        assert "Welcome! This is the official documentation" in results[1].snippet

    def test_no_highlights(self):
        sdk = SearchSDK()
        results = sdk._parse_exa_content(EXA_NO_HIGHLIGHTS)
        assert len(results) == 1
        assert results[0].snippet == ""

    def test_extra_text_after_highlights(self):
        sdk = SearchSDK()
        results = sdk._parse_exa_content(EXA_EXTRA_TEXT)
        assert len(results) == 1
        assert "First line" in results[0].snippet
        assert "Extra paragraph" in results[0].snippet

    def test_empty_content(self):
        sdk = SearchSDK()
        assert sdk._parse_exa_content("") == []

    def test_no_title_or_url_skipped(self):
        sdk = SearchSDK()
        results = sdk._parse_exa_content("Some random text without structure.")
        assert len(results) == 0


class TestSearchSDKSimulateResults:
    def test_returns_up_to_limit(self):
        sdk = SearchSDK()
        results = sdk._simulate_results("python", limit=3)
        assert len(results) == 3

    def test_each_has_fields(self):
        sdk = SearchSDK()
        results = sdk._simulate_results("test query", limit=2)
        for r in results:
            assert r.url
            assert r.title
            assert r.snippet
            assert r.domain

    def test_empty_query_uses_default_domain(self):
        sdk = SearchSDK()
        results = sdk._simulate_results("", limit=2)
        assert all(r.domain == "example.com" for r in results)


class TestSearchSDKCache:
    def test_cache_hit_returns_same_objects(self):
        sdk = SearchSDK()
        sdk._simulate = True
        r1 = sdk.web("python", limit=2)
        r2 = sdk.web("python", limit=2)
        assert len(r1) == len(r2)
        assert r1[0].url == r2[0].url

    def test_cache_miss_different_query(self):
        sdk = SearchSDK()
        sdk._simulate = True
        r1 = sdk.web("python", limit=2)
        r2 = sdk.web("rust", limit=2)
        assert r1[0].url != r2[0].url
        assert sdk._cache["python:::2"] is r1

    def test_cache_miss_different_limit(self):
        sdk = SearchSDK()
        sdk._simulate = True
        r1 = sdk.web("python", limit=2)
        r2 = sdk.web("python", limit=4)
        assert len(r1) < len(r2)
        assert "python:::2" in sdk._cache
        assert "python:::4" in sdk._cache

    def test_clear_cache(self):
        sdk = SearchSDK()
        sdk._simulate = True
        sdk.web("python", limit=2)
        assert "python:::2" in sdk._cache
        sdk.clear_cache()
        assert sdk._cache == {}

    def test_cache_key_format(self):
        sdk = SearchSDK()
        assert sdk._cache_key("hello world", 5) == "hello world:::5"


class TestFilesystemSDK:
    def test_write_and_read(self, fs_dir):
        fs = FilesystemSDK(fs_dir)
        fs.write("mykey", {"hello": "world"})
        assert fs.read("mykey") == {"hello": "world"}

    def test_read_missing_key(self, fs_dir):
        fs = FilesystemSDK(fs_dir)
        with pytest.raises(KeyError):
            fs.read("nonexistent")

    def test_list_keys(self, fs_dir):
        fs = FilesystemSDK(fs_dir)
        fs.write("alpha", 1)
        fs.write("beta", 2)
        keys = fs.list()
        assert "alpha" in keys
        assert "beta" in keys

    def test_exists(self, fs_dir):
        fs = FilesystemSDK(fs_dir)
        assert not fs.exists("missing")
        fs.write("present", "val")
        assert fs.exists("present")

    def test_safe_key_replaces_special_chars(self):
        safe = FilesystemSDK._safe_key("hello/world:test")
        assert "/" not in safe
        assert ":" not in safe


class TestUtilsSDK:
    def test_dedupe_by_field(self):
        items = [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
            {"id": 1, "name": "c"},
        ]
        result = UtilsSDK.dedupe_by(items, "id")
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    def test_dedupe_by_callable(self):
        items = ["apple", "banana", "avocado"]
        result = UtilsSDK.dedupe_by(items, lambda x: x[0])
        assert len(result) == 2

    def test_filter_by_field(self):
        items = [{"kind": "dog"}, {"kind": "cat"}, {"kind": "dog"}]
        result = UtilsSDK.filter_by(items, "kind", "dog")
        assert len(result) == 2

    def test_filter_by_callable(self):
        items = [1, 2, 3, 4]
        result = UtilsSDK.filter_by(items, lambda x: x > 2)
        assert result == [3, 4]

    def test_summarize_coverage(self):
        items = [
            {"lang": "py", "score": 1},
            {"lang": "js", "score": 2},
            {"lang": "py", "score": 3},
        ]
        summary = UtilsSDK.summarize_coverage(items, ["lang"])
        assert "Coverage by 'lang'" in summary
        assert "py: 2" in summary
        assert "js: 1" in summary

    def test_flatten(self):
        assert UtilsSDK.flatten([[1, 2], [3], [], [4, 5]]) == [1, 2, 3, 4, 5]

    def test_flatten_empty(self):
        assert UtilsSDK.flatten([]) == []

    def test_join_result_fields_dict(self):
        result = {"title": "T", "snippet": "S"}
        assert UtilsSDK.join_result_fields(result) == "T | S"

    def test_join_result_fields_object(self):
        r = SearchResult(title="Hello", snippet="World")
        assert UtilsSDK.join_result_fields(r) == "Hello | World"


class TestLLMSDKClientParseJsonList:
    @pytest.fixture
    def llm(self):
        return LLMSDKClient(api_key="test", base_url="http://localhost:0")

    def test_simple_parse(self, llm):
        raw = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        result = llm._parse_json_list(raw, {"name": str, "age": int})
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[0]["age"] == 30

    def test_missing_field_becomes_none(self, llm):
        raw = '[{"name": "Alice"}]'
        result = llm._parse_json_list(raw, {"name": str, "age": int})
        assert result[0]["age"] is None

    def test_extra_fields_ignored(self, llm):
        raw = '[{"name": "Alice", "extra": "x"}]'
        result = llm._parse_json_list(raw, {"name": str})
        assert "extra" not in result[0]

    def test_json_in_code_fence(self, llm):
        raw = '```json\n[{"a": 1}]\n```'
        result = llm._parse_json_list(raw, {"a": int})
        assert result[0]["a"] == 1

    def test_no_json_array(self, llm):
        assert llm._parse_json_list("just text", {}) == []

    def test_type_conversion_string(self, llm):
        raw = '[{"val": "123"}]'
        result = llm._parse_json_list(raw, {"val": int})
        assert result[0]["val"] == 123

    def test_type_conversion_fails_gracefully(self, llm):
        raw = '[{"val": "not_a_number"}]'
        result = llm._parse_json_list(raw, {"val": int})
        assert result[0]["val"] == "not_a_number"


class TestLLMSDKExtractMany:
    def test_extract_many_empty(self):
        llm = LLMSDKClient(api_key="test", base_url="http://localhost:0")
        assert llm.extract_many([], "test", {"name": str}) == []


class TestSandbox:
    @pytest.fixture
    def sdk(self):
        return AgenticSearchSDK(
            llm_api_key="test",
            llm_base_url="http://localhost:0",
        )

    def test_execute_print(self, sdk):
        sandbox = Sandbox(sdk)
        output = sandbox.execute('print("hello world")')
        assert "hello world" in output

    def test_execute_sdk_access(self, sdk):
        sandbox = Sandbox(sdk)
        output = sandbox.execute("print(sdk.utils.flatten([[1], [2]]))")
        assert "[1, 2]" in output

    def test_execute_error_traceback(self, sdk):
        sandbox = Sandbox(sdk)
        output = sandbox.execute("1/0")
        assert "ERROR" in output
        assert "ZeroDivisionError" in output

    def test_execute_sdk_search_mocked(self, sdk):
        sdk.search._simulate = True
        sandbox = Sandbox(sdk)
        output = sandbox.execute(
            'results = sdk.search.web("test", limit=2)\n'
            "for r in results:\n"
            "    print(r.title, r.url)"
        )
        assert "Result" in output
        assert "example.com" in output


class TestSaCAgent:
    def test_init_requires_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(
                task="test task",
                base_url="http://localhost:0",
                api_key="test",
            )
            assert agent.task == "test task"
            assert agent.max_turns == 15

    def test_fix_code_extracts_from_fence(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '```python\nprint("hello")\n```'
        agent._client.chat.completions.create = MagicMock(return_value=mock_resp)
        fixed = agent._fix_code("bad code", "error", 1)
        assert fixed == 'print("hello")'

    def test_fix_code_no_fence(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = 'print("hello")'
        agent._client.chat.completions.create = MagicMock(return_value=mock_resp)
        fixed = agent._fix_code("bad code", "error", 1)
        assert fixed == 'print("hello")'

    def test_fix_code_empty_response(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = ""
        agent._client.chat.completions.create = MagicMock(return_value=mock_resp)
        fixed = agent._fix_code("bad code", "error", 1)
        assert fixed is None

    def test_parse_response_json(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        result = agent._parse_response('{"turn_type": "synthesis", "answer": "done"}')
        assert result is not None
        assert result["turn_type"] == "synthesis"
        assert result["answer"] == "done"

    def test_parse_response_with_fences(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        raw = '```json\n{"turn_type": "code", "code": "print(1)"}\n```'
        result = agent._parse_response(raw)
        assert result is not None
        assert result["turn_type"] == "code"

    def test_parse_response_invalid(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        assert agent._parse_response("not json at all") is None

    def test_parse_response_braces(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        raw = 'some text {"turn_type": "synthesis", "answer": "ok"} trailing'
        result = agent._parse_response(raw)
        assert result is not None
        assert result["answer"] == "ok"


class TestFormatItemsEdge:
    def test_empty_list(self):
        assert _format_items([]) == "[]"

    def test_custom_type(self):
        class Obj:
            def __init__(self):
                self.x = 1

        result = _format_items([Obj()])
        assert "x" in result
