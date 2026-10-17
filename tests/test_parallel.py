"""Exercise provider selection through prepared HTTP requests and the agent loop."""

from __future__ import annotations

import json
import os
from concurrent.futures import CancelledError
from unittest.mock import MagicMock

import pytest
import requests

from sac import AgenticSearchSDK, SaCAgent
from sac.cli import _execute, build_parser
from sac.parallel import MCP_PARALLEL_URL, PROTOCOL_VERSION, USER_AGENT
from sac.search import SearchSDK

PAYLOAD = {
    "results": [
        {
            "url": "https://example.org/paper?a=1&b=2#results",
            "title": "Research paper",
            "excerpts": ["First excerpt.", "Second excerpt."],
        },
        {"url": "https://example.net", "title": None, "excerpts": []},
    ]
}


def response(payload, *, status=200, sse=False, headers=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    if sse:
        result.headers["Content-Type"] = "text/event-stream"
        result._content = (
            ": keepalive\n\nevent: message\ndata: " + json.dumps(payload) + "\n\n"
        ).encode()
    else:
        result.headers["Content-Type"] = "application/json"
        result._content = json.dumps(payload).encode()
    return result


@pytest.fixture
def transport(monkeypatch):
    # Isolate environment and saved HTTP credentials, including .netrc.
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(requests.sessions, "get_netrc_auth", lambda *_a, **_kw: None)
    calls = []
    state = {
        "payload": PAYLOAD,
        "sse": False,
        "failure": None,
        "rpc": None,
        "tool_error": False,
    }

    def send(_client, request, **kwargs):
        body = json.loads(request.body)
        calls.append((request, body, kwargs))
        assert request.url == MCP_PARALLEL_URL
        assert request.headers["User-Agent"] == USER_AGENT
        assert request.headers["Accept"] == "application/json, text/event-stream"
        assert "Authorization" not in request.headers
        assert "x-api-key" not in request.headers
        assert kwargs["timeout"] == 30
        if body["method"] == "initialize":
            assert body["params"]["clientInfo"]["name"] == "sac-agent"
            return response(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"protocolVersion": PROTOCOL_VERSION},
                },
                headers={"Mcp-Session-Id": "transport-id"},
            )
        assert request.headers["Mcp-Session-Id"] == "transport-id"
        assert request.headers["MCP-Protocol-Version"] == PROTOCOL_VERSION
        if body["method"] == "notifications/initialized":
            return response({}, status=202)
        assert body["method"] == "tools/call"
        assert body["params"]["name"] == "web_search"
        if state["failure"]:
            raise state["failure"]
        if state["rpc"] is not None:
            return response(state["rpc"])
        return response(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "isError": state["tool_error"],
                    "content": [{"type": "text", "text": json.dumps(state["payload"])}],
                },
            },
            sse=state["sse"],
        )

    monkeypatch.setattr(requests.Session, "send", send)
    return calls, state


@pytest.mark.parametrize("sse", [False, True])
def test_sdk_dispatch_mapping_cache_and_proxy(transport, sse):
    calls, state = transport
    state["sse"] = sse
    sdk = AgenticSearchSDK(
        search_provider="parallel",
        brave_key="keep-existing-key",
        llm_api_key="test",
        http_proxy="http://proxy.invalid:8080",
    )
    assert calls == []  # Construction grants no network access.
    results = sdk.search.web("recent research papers", limit=1)
    assert len(results) == 1
    assert results[0].url == PAYLOAD["results"][0]["url"]
    assert results[0].domain == "example.org"
    assert results[0].snippet == "First excerpt.\nSecond excerpt."
    assert sdk.search.web("recent research papers", limit=1) is results
    assert len(calls) == 3
    assert calls[-1][1]["params"]["arguments"]["search_queries"] == [
        "recent research papers"
    ]
    assert calls[-1][1]["params"]["arguments"]["objective"] == "recent research papers"
    assert calls[-1][2]["proxies"]["https"] == "http://proxy.invalid:8080"


def test_web_many_stable_research_identity(transport):
    calls, _ = transport
    sdk = SearchSDK(search_provider="parallel")
    results = sdk.web_many(
        ["first query", {"query": "second query"}], limit_per_query=2
    )
    assert all(len(group) == 2 for group in results)
    arguments = [
        body["params"]["arguments"]
        for _, body, _ in calls
        if body["method"] == "tools/call"
    ]
    assert len(arguments) == 2
    assert len({arg["session_id"] for arg in arguments}) == 1
    assert results[0][1].title == "https://example.net"


@pytest.mark.parametrize(
    "brave_key,backend", [(None, "_exa_mcp_search"), ("saved-key", "_brave_search")]
)
def test_auto_preserves_existing_choices(mocker, brave_key, backend):
    exa = mocker.patch.object(SearchSDK, "_exa_mcp_search", return_value=[])
    brave = mocker.patch.object(SearchSDK, "_brave_search", return_value=[])
    parallel = mocker.patch("sac.search.parallel_search")
    SearchSDK(brave_key=brave_key).web("query")
    {"_exa_mcp_search": exa, "_brave_search": brave}[backend].assert_called_once()
    parallel.assert_not_called()


def test_explicit_choices_and_disabled_simulation(mocker, transport):
    calls, _ = transport
    exa = mocker.patch.object(SearchSDK, "_exa_mcp_search", return_value=[])
    brave = mocker.patch.object(SearchSDK, "_brave_search", return_value=[])
    SearchSDK(search_provider="exa", brave_key="saved-key").web("query")
    exa.assert_called_once()
    brave.assert_not_called()
    SearchSDK(search_provider="brave", brave_key="saved-key").web("query")
    brave.assert_called_once()
    sdk = SearchSDK(search_provider="parallel")
    sdk._simulate = True
    assert sdk.web("query")
    assert calls == []
    with pytest.raises(ValueError, match="requires"):
        SearchSDK(search_provider="brave")
    with pytest.raises(ValueError, match="search_provider"):
        SearchSDK(search_provider="typo")


@pytest.mark.parametrize(
    "failure",
    [
        requests.Timeout("timeout"),
        requests.ConnectionError("offline"),
        CancelledError(),
        KeyboardInterrupt(),
    ],
)
def test_failure_and_cancellation_close_transport_without_fallback(
    transport, mocker, failure
):
    calls, state = transport
    state["failure"] = failure
    close = mocker.spy(requests.Session, "close")
    fallback = mocker.patch.object(SearchSDK, "_exa_mcp_search")
    sdk = SearchSDK(search_provider="parallel", brave_key="configured")
    with pytest.raises(type(failure)):
        sdk.web("query")
    assert len(calls) == 3
    assert not sdk._cache
    close.assert_called_once()
    fallback.assert_not_called()


@pytest.mark.parametrize(
    "rpc",
    [
        {"id": 2, "error": {"code": -32000, "message": "quota exceeded"}},
        {"id": 99, "result": {}},
    ],
)
def test_rpc_errors_are_not_results(transport, rpc):
    _, state = transport
    state["rpc"] = rpc
    with pytest.raises(ValueError, match="Parallel"):
        SearchSDK(search_provider="parallel").web("query")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": [None]},
        {"results": [{"url": "file:///etc/passwd", "excerpts": []}]},
        {"results": [{"url": "https://example.org", "excerpts": [None]}]},
    ],
)
def test_malformed_results_fail(transport, payload):
    _, state = transport
    state["payload"] = payload
    with pytest.raises(ValueError, match="Parallel"):
        SearchSDK(search_provider="parallel").web("query")


def test_tool_error_and_empty_results(transport):
    _, state = transport
    state["tool_error"] = True
    with pytest.raises(ValueError, match="Parallel search failed"):
        SearchSDK(search_provider="parallel").web("query")
    state["tool_error"] = False
    state["payload"] = {"results": []}
    assert SearchSDK(search_provider="parallel").web("query") == []


def test_zero_limit_never_calls_network(transport):
    calls, _ = transport
    sdk = SearchSDK(search_provider="parallel")
    assert sdk.web("query", limit=0) == []
    with pytest.raises(ValueError, match="nonnegative"):
        sdk.web("query", limit=-1)
    assert calls == []


def test_structured_response_and_multiline_sse(transport):
    _, state = transport
    state["rpc"] = {"id": 2, "result": {"structuredContent": PAYLOAD}}
    assert (
        SearchSDK(search_provider="parallel").web("query")[0].title == "Research paper"
    )

    from sac.parallel import _result

    reply = response({})
    reply.headers["Content-Type"] = "text/event-stream"
    reply._content = (
        b'data: {"jsonrpc":"2.0",\r\ndata: "id":2,"result":{"ok":true}}\r\n\r\n'
    )
    assert _result(reply, 2) == {"ok": True}


@pytest.mark.parametrize("status", [401, 402, 429, 503])
def test_http_failures_do_not_fallback_or_retry(transport, monkeypatch, status):
    calls, _ = transport
    original = requests.Session.send

    def send(client, request, **kwargs):
        if json.loads(request.body)["method"] == "tools/call":
            calls.append((request, {}, kwargs))
            return response({"error": "unavailable"}, status=status)
        return original(client, request, **kwargs)

    monkeypatch.setattr(requests.Session, "send", send)
    sdk = SearchSDK(search_provider="parallel")
    with pytest.raises(requests.HTTPError):
        sdk.web("query")
    assert len(calls) == 3
    assert not sdk._cache


def test_unsupported_protocol_stops_before_search(transport, monkeypatch):
    calls, _ = transport

    def send(_client, _request, **_kwargs):
        calls.append(None)
        return response({"id": 1, "result": {"protocolVersion": "unsupported"}})

    monkeypatch.setattr(requests.Session, "send", send)
    with pytest.raises(ValueError, match="protocol"):
        SearchSDK(search_provider="parallel").web("query")
    assert len(calls) == 1


def test_cli_resolver_to_agent_result_consumption(transport, mocker, tmp_path):
    """Scripted model turns; real CLI wiring, sandbox and search transport mapping."""
    calls, _ = transport
    code = 'results = sdk.search.web("research papers", limit=1)\nfor r in results:\n    sdk.fs.write("source", r.url)\n    print(r.title, r.url, r.snippet)'
    turns = iter(
        [
            {"turn_type": "code", "reasoning": "search", "code": code},
            {
                "turn_type": "synthesis",
                "reasoning": "done",
                "answer": "Research answer",
            },
        ]
    )
    observed = []

    def model(**kwargs):
        observed.append([dict(message) for message in kwargs["messages"]])
        result = MagicMock()
        result.choices[0].message.content = json.dumps(next(turns))
        return result

    original = SaCAgent.__init__

    def initialize(agent, *args, **kwargs):
        original(agent, *args, **kwargs)
        agent._client.chat.completions.create = model
        agent.sdk.fs._dir = tmp_path

    mocker.patch.object(SaCAgent, "__init__", initialize)
    args = build_parser().parse_args(
        ["--search-provider", "parallel", "--max-turns", "3"]
    )
    assert _execute("research", args) == "Research answer"
    assert len(calls) == 3
    feedback = observed[1][-1]["content"]
    assert "https://example.org/paper?a=1&b=2#results" in feedback
    assert "First excerpt." in feedback
