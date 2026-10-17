"""Anonymous Parallel Search MCP transport for the built-in search provider."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

import requests

from sac.core import SearchResult

MCP_PARALLEL_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "sac-agent/0.1.0 (https://github.com/daedalus/sac-agent)"
PROTOCOL_VERSION = "2025-03-26"


def _result(response: requests.Response, request_id: int) -> dict[str, Any]:
    response.raise_for_status()
    if response.headers.get("Content-Type", "").startswith("text/event-stream"):
        messages = []
        for event in response.text.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(
                line[5:].lstrip(" ")
                for line in event.splitlines()
                if line.startswith("data:")
            )
            if data:
                messages.append(json.loads(data))
    else:
        messages = [response.json()]
    for message in messages:
        if not isinstance(message, dict) or message.get("id") != request_id:
            continue
        if "error" in message:
            raise ValueError(f"Parallel MCP error: {message['error']}")
        result = message.get("result")
        if isinstance(result, dict):
            return result
    raise ValueError("Parallel MCP response has no matching result")


def search(
    query: str, limit: int, session_id: str, proxies: dict[str, str]
) -> list[SearchResult]:
    """Search over Streamable HTTP, retaining a stable research-session ID.

    Each invocation owns its transport session so web_many can run concurrently.
    Errors propagate to the caller; they never become fabricated search results.
    """
    if limit < 0:
        raise ValueError("Search limit must be nonnegative")
    if limit == 0:
        return []
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": USER_AGENT,
    }
    with requests.Session() as client:
        response = client.post(
            MCP_PARALLEL_URL,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "sac-agent", "version": "0.1.0"},
                },
            },
            headers=headers,
            proxies=proxies,
            timeout=30,
        )
        initialized = _result(response, 1)
        if initialized.get("protocolVersion") != PROTOCOL_VERSION:
            raise ValueError("Parallel MCP returned an unsupported protocol version")
        headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        if transport_id := response.headers.get("Mcp-Session-Id"):
            headers["Mcp-Session-Id"] = transport_id
        response = client.post(
            MCP_PARALLEL_URL,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=headers,
            proxies=proxies,
            timeout=30,
        )
        response.raise_for_status()
        response = client.post(
            MCP_PARALLEL_URL,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "web_search",
                    "arguments": {
                        "objective": query,
                        "search_queries": [query],
                        "session_id": session_id,
                    },
                },
            },
            headers=headers,
            proxies=proxies,
            timeout=30,
        )
        result = _result(response, 2)
    return _search_results(result, limit)


def _search_results(result: dict[str, Any], limit: int) -> list[SearchResult]:
    if result.get("isError"):
        raise ValueError(f"Parallel search failed: {result.get('content')}")
    payload = result.get("structuredContent")
    if payload is None:
        text = "".join(
            block.get("text", "")
            for block in result.get("content", [])
            if block.get("type") == "text"
        )
        payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Parallel search response has no results list")
    return [_search_result(item) for item in payload["results"][:limit]]


def _search_result(item: dict[str, Any]) -> SearchResult:
    if not isinstance(item, dict):
        raise ValueError("Invalid Parallel search result")
    url = item.get("url")
    excerpts = item.get("excerpts")
    title = item.get("title")
    if (
        not isinstance(url, str)
        or urlsplit(url).scheme not in ("http", "https")
        or not urlsplit(url).netloc
    ):
        raise ValueError("Invalid Parallel search result URL")
    if not isinstance(excerpts, list) or any(
        not isinstance(excerpt, str) for excerpt in excerpts
    ):
        raise ValueError("Invalid Parallel search result excerpts")
    if title is not None and not isinstance(title, str):
        raise ValueError("Invalid Parallel search result title")
    return SearchResult(url=url, title=title or url, snippet="\n".join(excerpts))
