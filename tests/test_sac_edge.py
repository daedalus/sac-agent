from __future__ import annotations

import os
import sys
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
)
from sac.core import _httpx_client, _log, _proxy_config


class TestCoreEdge:
    def test_proxy_config_default(self):
        config = _proxy_config()
        assert "http" in config
        assert "https" in config

    def test_proxy_config_from_env(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy:8080"}):
            config = _proxy_config()
            assert config["https"] == "http://proxy:8080"

    def test_proxy_config_fallback_http(self):
        with patch.dict(
            os.environ, {"HTTP_PROXY": "http://proxy:9090", "HTTPS_PROXY": ""}
        ):
            config = _proxy_config()
            assert "http://proxy:9090" in config.values()

    def test_search_result_auto_domain(self):
        r = SearchResult(url="https://example.com/path")
        assert r.domain == "example.com"

    def test_search_result_auto_domain_fails_gracefully(self):
        r = SearchResult(url="\x00invalid")
        assert r.domain == ""

    def test_search_result_repr(self):
        r = SearchResult(url="https://x.com", title="X")
        assert "SearchResult" in repr(r)
        assert "x.com" in repr(r)

    def test_httpx_client_no_proxy(self):
        with patch.dict(os.environ, {}, clear=True):
            client = _httpx_client()
            assert client is None

    def test_log_verbose(self):
        from sac.core import VERBOSE

        with patch("sys.stderr"):
            VERBOSE = True
            try:
                _log("test message")
            finally:
                VERBOSE = False

    def test_proxy_config_none(self):
        with patch.dict(os.environ, {}, clear=True):
            config = _proxy_config()
            assert config == {}


class TestSearchSDKEdge:
    def test_web_many_with_dicts(self):
        sdk = SearchSDK()
        sdk._simulate = True
        results = sdk.web_many(
            [{"query": "python"}, {"query": "rust"}],
            limit_per_query=2,
            concurrency=2,
        )
        assert len(results) == 2

    def test_search_one_fallback_to_simulate(self):
        sdk = SearchSDK()
        result = sdk._search_one("test query", 3)
        assert len(result) == 3

    def test_neural_falls_back_to_simulate(self):
        sdk = SearchSDK()
        sdk._simulate = True
        result = sdk.web("hello", 2)
        assert len(result) == 2

    def test_brave_search_http_error_fallback(self, mocker):
        mocker.patch("sac.search.requests.get", side_effect=Exception("HTTP error"))
        sdk = SearchSDK(brave_key="fake_key")
        results = sdk._search_one("test", 2)
        assert len(results) > 0

    def test_exa_search_empty_response_fallback(self, mocker):
        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_resp.raise_for_status.return_value = None
        mocker.patch("sac.search.requests.post", return_value=mock_resp)
        sdk = SearchSDK()
        results = sdk._search_one("test", 2)
        assert len(results) > 0

    def test_exa_data_line_format(self, mocker):
        mock_resp = MagicMock()
        mock_resp.text = 'data: {"result": {"content": [{"text": "Title: T\\nURL: https://x.com\\nHighlights:\\ncontent"}]}}\n'
        mock_resp.raise_for_status.return_value = None
        mocker.patch("sac.search.requests.post", return_value=mock_resp)
        sdk = SearchSDK()
        results = sdk._search_one("test", 2)
        assert len(results) == 1
        assert results[0].title == "T"
        assert results[0].url == "https://x.com"

    def test_exa_bad_json_line_skipped(self, mocker):
        mock_resp = MagicMock()
        mock_resp.text = "data: not json\n"
        mock_resp.raise_for_status.return_value = None
        mocker.patch("sac.search.requests.post", return_value=mock_resp)
        sdk = SearchSDK()
        results = sdk._search_one("test", 2)
        assert len(results) > 0  # fallback to simulate

    def test_web_many_tracks_counts(self):
        sdk = SearchSDK()
        sdk._simulate = True
        sdk.web_many(["a", "b"], limit_per_query=2)
        assert sdk.total_queries == 2
        assert sdk.total_results >= 2


class TestSearchSDKBrave:
    def test_brave_search_parses_results(self, mocker):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "web": {
                "results": [
                    {"url": "https://a.com", "title": "A", "description": "Desc A"},
                    {"url": "https://b.com", "title": "B", "description": "Desc B"},
                ]
            }
        }
        mock_resp.raise_for_status.return_value = None
        mocker.patch("sac.search.requests.get", return_value=mock_resp)
        sdk = SearchSDK(brave_key="key")
        results = sdk._brave_search("test", 2)
        assert len(results) == 2
        assert results[0].title == "A"
        assert results[1].url == "https://b.com"


class TestLLMSDKClientEdge:
    def test_call_llm_uses_reasoning_content(self, mocker):
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg.reasoning_content = "reasoning output"
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]
        client = LLMSDKClient(api_key="test", base_url="http://localhost:0")
        mocker.patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        )
        result = client._call_llm("prompt")
        assert result == "reasoning output"

    def test_call_llm_no_content(self, mocker):
        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.reasoning_content = None
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]
        client = LLMSDKClient(api_key="test", base_url="http://localhost:0")
        mocker.patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        )
        result = client._call_llm("prompt")
        assert result == ""

    def test_synthesize_format(self, mocker):
        client = LLMSDKClient(api_key="test", base_url="http://localhost:0")
        mocker.patch.object(client, "_call_llm", return_value="synthesized")
        result = client.synthesize([{"a": 1}], "summarize")
        assert result == "synthesized"

    def test_plan_format(self, mocker):
        client = LLMSDKClient(api_key="test", base_url="http://localhost:0")
        mocker.patch.object(client, "_call_llm", return_value="search plan")
        result = client.plan("context about X", "find Y")
        assert result == "search plan"

    def test_parse_json_list_not_list_returns_empty(self, llm):
        raw = '{"a": 1}'
        result = llm._parse_json_list(raw, {})
        assert result == []

    def test_parse_json_list_decode_error(self, llm):
        raw = '[{bad json}]'
        result = llm._parse_json_list(raw, {})
        assert result == []

    def test_extract_many_chunks(self, mocker):
        client = LLMSDKClient(api_key="test", base_url="http://localhost:0")
        mock_chunk = mocker.MagicMock()
        mock_chunk.side_effect = lambda items, *a, **kw: [{"id": item["id"]} for item in items]
        mocker.patch.object(client, "_extract_chunk", mock_chunk)
        items = [{"id": i} for i in range(12)]
        result = client.extract_many(items, "extract", {"id": int})
        assert len(result) == 12
        assert mock_chunk.call_count == 2


@pytest.fixture
def llm():
    return LLMSDKClient(api_key="test", base_url="http://localhost:0")


class TestSaCAgentEdge:
    def test_run_returns_synthesis(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(
                task="test",
                base_url="http://localhost:0",
                api_key="test",
                max_turns=1,
            )
        synthesis = '{"turn_type": "synthesis", "answer": "done", "reasoning": "ok"}'
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = synthesis
        agent._client.chat.completions.create = MagicMock(return_value=mock_resp)
        result = agent.run()
        assert result == "done"

    def test_run_code_turn_then_synthesis(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(
                task="test",
                base_url="http://localhost:0",
                api_key="test",
                max_turns=2,
            )
        code_turn = '{"turn_type": "code", "reasoning": "searching", "code": "print(\\"hello\\")"}'
        synthesis = '{"turn_type": "synthesis", "answer": "final", "reasoning": "ok"}'

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = code_turn
        agent._client.chat.completions.create = MagicMock(return_value=mock_resp)

        # After code execution, return synthesis
        mock_synth = MagicMock()
        mock_synth.choices[0].message.content = synthesis

        def side_effect(*args, **kwargs):
            if len(agent._history) > 2:
                return mock_synth
            return mock_resp

        agent._client.chat.completions.create = MagicMock(side_effect=side_effect)
        result = agent.run()
        assert result == "final"

    def test_parse_response_returns_none_on_bad_extract(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        assert agent._parse_response("no braces or json") is None

    def test_fix_code_extracts_from_fence(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '```python\nprint("hello")\n```'
        agent._client.chat.completions.create = MagicMock(return_value=mock_resp)
        fixed = agent._fix_code("bad code", "error", 1)
        assert fixed == 'print("hello")'

    def test_force_synthesis_returns_answer(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        mock_resp = MagicMock()
        mock_resp.choices[
            0
        ].message.content = '{"turn_type": "synthesis", "answer": "forced answer"}'
        agent._client.chat.completions.create = MagicMock(return_value=mock_resp)
        result = agent._force_synthesis()
        assert result == "forced answer"

    def test_record_usage_none(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        agent._record_usage(MagicMock(usage=None))
        assert agent._usage.total == 0

    def test_record_usage_counts(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        resp = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        agent._record_usage(resp)
        assert agent._usage.prompt == 100
        assert agent._usage.completion == 50

    def test_run_parse_failure_with_library(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(
                task="x",
                base_url="http://localhost:0",
                api_key="test",
                with_code_library=True,
            )
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "not json"
        agent._client.chat.completions.create = MagicMock(return_value=mock_resp)
        result = agent.run()
        assert "Error: failed to parse" in result

    def test_empty_code_turn(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test", max_turns=1)
        mock_code = MagicMock()
        mock_code.choices[
            0
        ].message.content = '{"turn_type": "code", "reasoning": "test", "code": ""}'
        agent._client.chat.completions.create = MagicMock(return_value=mock_code)
        result = agent.run()
        assert "turns" in result.lower() or "synthesis" in result.lower()

    def test_call_model_uses_reasoning_content(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test")
        mock_resp = MagicMock()
        msg = MagicMock()
        msg.content = None
        msg.reasoning_content = "reasoning output"
        mock_resp.choices = [MagicMock(message=msg)]
        agent._client.chat.completions.create = MagicMock(return_value=mock_resp)
        result = agent._call_model()
        assert result == "reasoning output"

    def test_force_synthesis_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(task="x", base_url="http://localhost:0", api_key="test", max_turns=1)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = 'not synthesis json'
        agent._client.chat.completions.create = MagicMock(return_value=mock_resp)
        result = agent._force_synthesis()
        assert "Research completed after" in result

    def test_run_context_force_synthesis(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            agent = SaCAgent(
                task="x",
                base_url="http://localhost:0",
                api_key="test",
                max_turns=5,
                context_force_threshold=0.0,
            )
        mock_code = MagicMock()
        mock_code.choices[0].message.content = '{"turn_type": "code", "reasoning": "t", "code": "print(1)"}'
        mock_synth = MagicMock()
        mock_synth.choices[0].message.content = '{"turn_type": "synthesis", "answer": "forced by context"}'
        agent._client.chat.completions.create = MagicMock(side_effect=[mock_code, mock_synth])
        result = agent.run()
        assert "forced by context" in result


class TestUtilsSDKEdge:
    def test_normalize_url_with_scheme(self):
        from sac import UtilsSDK

        assert UtilsSDK.normalize_url("https://Example.COM/Path/") == "https://Example.COM/Path"

    def test_normalize_url_without_scheme(self):
        from sac import UtilsSDK

        assert UtilsSDK.normalize_url("example.com") == "example.com"

    def test_filter_by_regex(self):
        from sac import UtilsSDK

        items = [{"name": "hello"}, {"name": "world"}, {"name": "help"}]
        result = UtilsSDK.filter_by_regex(items, "name", r"^hel")
        assert len(result) == 2
        assert all(r["name"].startswith("hel") for r in result)

    def test_chunk(self):
        from sac import UtilsSDK

        items = list(range(10))
        chunks = UtilsSDK.chunk(items, size=3)
        assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_get_value_with_object(self):
        from sac import UtilsSDK

        class Obj:
            x = 42

        val = UtilsSDK._get_value(Obj(), "x")
        assert val == 42


class TestCLI:
    def test_main_help(self):
        from sac.cli import main

        with patch("sys.argv", ["sac", "--help"]):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0

    def test_main_with_task(self):
        from sac.cli import main

        with patch("sac.cli.SaCAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run.return_value = "answer"
            mock_agent_cls.return_value = mock_agent
            with patch("sys.argv", ["sac", "research task"]):
                main()
                mock_agent_cls.assert_called_once()
                mock_agent.run.assert_called_once()

    def test_main_no_args_interactive(self):
        from sac.cli import interactive, main

        with patch("sac.cli.interactive") as mock_interactive:
            with patch("sys.argv", ["sac"]):
                main()
                mock_interactive.assert_called_once()

    def test_main_with_sandbox_docker(self):
        from sac.cli import main

        with patch("sac.cli.SaCAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run.return_value = "answer"
            mock_agent_cls.return_value = mock_agent
            with patch("sys.argv", ["sac", "--sandbox", "docker", "research"]):
                main()
            _, kwargs = mock_agent_cls.call_args
            assert kwargs["sandbox_backend"] == "docker"

    def test_main_with_sandbox_env_var(self):
        from sac.cli import _execute

        with patch("sac.cli.SaCAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run.return_value = "answer"
            mock_agent_cls.return_value = mock_agent
            with patch.dict(os.environ, {"SANDBOX_BACKEND": "docker"}):
                args = MagicMock()
                args.max_turns = 15
                args.no_cache = False
                args.endpoint = None
                args.api_key = None
                args.model = None
                args.http_proxy = None
                args.https_proxy = None
                args.with_code_library = False
                args.sandbox = None
                args.final_report = None
                args.final_report_format = "md"
                _execute("research", args)
            _, kwargs = mock_agent_cls.call_args
            assert kwargs["sandbox_backend"] == "docker"


class TestFilesystemSDKEdge:
    def test_default_dir_is_tempdir(self):
        fs = FilesystemSDK()
        assert fs.dir.exists()
        assert fs.dir.is_dir()


class TestSandboxEdge:
    def test_execute_search_result_in_namespace(self):
        sdk = AgenticSearchSDK(llm_api_key="test", llm_base_url="http://localhost:0")
        sandbox = Sandbox(sdk)
        output = sandbox.execute(
            'r = SearchResult(url="https://x.com"); print(r.domain)'
        )
        assert "x.com" in output

    def test_docker_backend_execute(self):
        sdk = AgenticSearchSDK(llm_api_key="test", llm_base_url="http://localhost:0")
        result_mock = MagicMock()
        result_mock.text = "hello from docker"
        result_mock.return_code = 0
        session_mock = MagicMock()
        session_mock.run.return_value = result_mock

        clean = Sandbox._docker_session
        Sandbox._docker_session = session_mock
        try:
            sandbox = Sandbox(sdk, backend="docker")
            output = sandbox.execute('print("hello")')
            assert "hello from docker" in output
        finally:
            Sandbox._docker_session = clean

    def test_docker_backend_not_installed(self):
        sdk = AgenticSearchSDK(llm_api_key="test", llm_base_url="http://localhost:0")
        sandbox = Sandbox(sdk, backend="docker")
        Sandbox._docker_session = None
        with patch("builtins.__import__", side_effect=ImportError("no llm-sandbox")):
            with pytest.raises(ImportError, match="llm-sandbox"):
                sandbox.execute("print('x')")

    def test_docker_backend_non_zero_return(self):
        sdk = AgenticSearchSDK(llm_api_key="test", llm_base_url="http://localhost:0")
        result_mock = MagicMock()
        result_mock.text = "something went wrong"
        result_mock.return_code = 1
        session_mock = MagicMock()
        session_mock.run.return_value = result_mock
        clean = Sandbox._docker_session
        Sandbox._docker_session = session_mock
        try:
            sandbox = Sandbox(sdk, backend="docker")
            output = sandbox.execute('print("x")')
            assert "--- ERROR ---" in output
        finally:
            Sandbox._docker_session = clean

    def test_docker_backend_close(self):
        sdk = AgenticSearchSDK(llm_api_key="test", llm_base_url="http://localhost:0")
        session_mock = MagicMock()
        clean = Sandbox._docker_session
        Sandbox._docker_session = session_mock
        try:
            sandbox = Sandbox(sdk, backend="docker")
            sandbox.close()
            assert Sandbox._docker_session is None
            session_mock.__exit__.assert_called_once()
        finally:
            Sandbox._docker_session = clean

    def test_docker_backend_close_idempotent(self):
        sdk = AgenticSearchSDK(llm_api_key="test", llm_base_url="http://localhost:0")
        clean = Sandbox._docker_session
        Sandbox._docker_session = None
        try:
            sandbox = Sandbox(sdk, backend="docker")
            sandbox.close()
        finally:
            Sandbox._docker_session = clean
