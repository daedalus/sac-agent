from __future__ import annotations

import concurrent.futures
import json
import re
import sys
from typing import Any
from urllib.parse import quote_plus

import requests

from sac.core import SearchResult, _extract_domain, _log, _proxy_config
from sac.retry import FreeUsageLimitError, RateLimitError, TransientError, with_retry

MCP_EXA_URL = "https://mcp.exa.ai/mcp"


class SearchSDK:
    def __init__(
        self,
        brave_key: str | None = None,
        http_proxy: str | None = None,
        https_proxy: str | None = None,
    ) -> None:
        self._brave_key = brave_key
        self._http_proxy = http_proxy
        self._https_proxy = https_proxy
        self._simulate = False
        self.total_queries = 0
        self.total_results = 0
        self._cache: dict[str, list[SearchResult]] = {}

    def _cache_key(self, query: str, limit: int) -> str:
        return f"{query}:::{limit}"

    def clear_cache(self) -> None:
        self._cache.clear()

    def web(self, query: str, limit: int = 8) -> list[SearchResult]:
        return self._search_one(query, limit)

    def web_many(
        self,
        queries: list[str | dict[str, Any]],
        limit_per_query: int = 8,
        concurrency: int = 6,
    ) -> list[list[SearchResult]]:
        normalized = [q["query"] if isinstance(q, dict) else q for q in queries]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(self._search_one, q, limit_per_query) for q in normalized
            ]
            results = [f.result() for f in futures]
        self.total_queries += len(normalized)
        self.total_results += sum(len(r) for r in results)
        return results

    def _search_one(self, query: str, limit: int) -> list[SearchResult]:
        cache_key = self._cache_key(query, limit)
        cached = self._cache.get(cache_key)
        if cached is not None:
            _log(f"Cache hit: {query} (limit={limit})")
            return cached

        if self._simulate:
            results = self._simulate_results(query, limit)
            self._cache[cache_key] = results
            return results
        if self._brave_key:
            try:
                results = self._brave_search(query, limit)
                self._cache[cache_key] = results
                return results
            except Exception as e:
                print(
                    f"[dim red]  Brave error: {e} — fallback to Exa[/]", file=sys.stderr
                )
        try:
            results = self._exa_mcp_search(query, limit)
            self._cache[cache_key] = results
            return results
        except Exception as e:
            _log(f"Exa error: {e}")
            results = self._simulate_results(query, limit)
            self._cache[cache_key] = results
            return results

    def _brave_search(self, query: str, limit: int) -> list[SearchResult]:
        assert self._brave_key is not None

        def _do_request() -> requests.Response:
            try:
                resp = requests.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self._brave_key,
                    },
                    params={"q": query, "count": min(limit, 20)},  # type: ignore[arg-type]
                    proxies=_proxy_config(self._http_proxy, self._https_proxy),
                    timeout=10,
                )
            except requests.Timeout as e:
                raise TransientError(f"Brave timeout: {e}") from e
            except requests.ConnectionError as e:
                raise TransientError(f"Brave connection error: {e}") from e
            if resp.status_code == 402:
                raise FreeUsageLimitError(f"Brave 402: {resp.text[:200]}")
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                raise RateLimitError(
                    f"Brave 429: {resp.text[:200]}",
                    retry_after=float(retry_after) if retry_after else None,
                )
            resp.raise_for_status()
            return resp

        resp = with_retry(_do_request)
        data = resp.json()
        return [
            SearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("description", ""),
            )
            for r in data.get("web", {}).get("results", [])[:limit]
        ]

    def _exa_mcp_search(self, query: str, limit: int) -> list[SearchResult]:
        _log(f"Exa search: query={query!r} limit={limit}")
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "web_search_exa",
                    "arguments": {
                        "query": query,
                        "type": "auto",
                        "numResults": limit,
                        "livecrawl": "fallback",
                    },
                },
            }
        )

        def _do_request() -> requests.Response:
            try:
                resp = requests.post(
                    MCP_EXA_URL,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                    proxies=_proxy_config(self._http_proxy, self._https_proxy),
                    timeout=30,
                )
            except requests.Timeout as e:
                raise TransientError(f"Exa timeout: {e}") from e
            except requests.ConnectionError as e:
                raise TransientError(f"Exa connection error: {e}") from e
            if resp.status_code == 402:
                raise FreeUsageLimitError(f"Exa 402: {resp.text[:200]}")
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                raise RateLimitError(
                    f"Exa 429: {resp.text[:200]}",
                    retry_after=float(retry_after) if retry_after else None,
                )
            resp.raise_for_status()
            return resp

        resp = with_retry(_do_request)

        text = resp.text
        results: list[SearchResult] = []

        for line in text.split("\n"):
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                _log(f"JSON decode error on line: {line[:200]}")
                continue
            content = data.get("result", {}).get("content", [{}])[0].get("text", "")
            if not content:
                continue
            _log(f"Raw Exa content ({len(content)} chars): {content[:300]}...")
            parsed = self._parse_exa_content(content)
            _log(f"Parsed {len(parsed)} results from Exa content")
            results.extend(parsed)

        if results:
            _log(f"Total {len(results)} Exa results, returning first {limit}")
            self.total_results += len(results)
            return results[:limit]
        _log("No Exa results parsed, using simulation")
        return self._simulate_results(query, limit)

    def _parse_exa_content(self, content: str) -> list[SearchResult]:
        _log(f"_parse_exa_content: {len(content)} chars")
        if not content.strip():
            _log("empty content, returning []")
            return []
        results: list[SearchResult] = []
        blocks = re.split(r"\n-{3,}\s*\n", content)
        _log(f"split into {len(blocks)} blocks")
        field_pattern = re.compile(r"^(Title|URL|Highlights|Published|Author|Date|Source|Summary):\s*(.*)", re.IGNORECASE)

        for block in blocks:
            block = block.strip()
            if not block:
                continue
            title = ""
            url = ""
            snippet_lines: list[str] = []
            in_highlights = False
            for line in block.split("\n"):
                line_stripped = line.strip()
                fm = field_pattern.match(line_stripped)
                if fm:
                    field_name = fm.group(1).lower()
                    field_value = fm.group(2).strip()
                    if field_name == "title":
                        title = field_value
                    elif field_name == "url":
                        url = field_value
                    elif field_name == "highlights":
                        in_highlights = True
                        if field_value:
                            snippet_lines.append(field_value)
                    continue
                if in_highlights and line_stripped:
                    snippet_lines.append(line_stripped)

            snippet = " ".join(snippet_lines)
            snippet = re.sub(r"\s*\[\.\.\.\]\s*", " ", snippet).strip()
            if title and url:
                results.append(
                    SearchResult(
                        url=url,
                        title=title,
                        snippet=snippet,
                        domain=_extract_domain(url),
                    )
                )
            else:
                _log(f"  skipped block: title={title!r} url={url!r}")

        if not results:
            _log("  attempting JSON fallback for Exa content")
            try:
                parsed = json.loads(content)
                raw_results = parsed if isinstance(parsed, list) else [parsed]
                for item in raw_results:
                    if isinstance(item, dict):
                        t = item.get("title") or item.get("name", "")
                        u = item.get("url") or item.get("link", "")
                        s = item.get("snippet") or item.get("description", "")
                        if t and u:
                            results.append(
                                SearchResult(url=u, title=t, snippet=s, domain=_extract_domain(u))
                            )
            except json.JSONDecodeError:
                pass

        _log(f"  parsed {len(results)} results total")
        return results

    def _simulate_results(self, query: str, limit: int) -> list[SearchResult]:
        domains = {
            "arxiv": "arxiv.org",
            "github": "github.com",
            "blog": "medium.com",
            "news": "techcrunch.com",
            "paper": "aclanthology.org",
            "default": "example.com",
        }
        results: list[SearchResult] = []
        for i in range(min(limit, 6)):
            topic = query.split()[0].lower() if query else "topic"
            domain = domains.get(topic, domains["default"])
            results.append(
                SearchResult(
                    url=f"https://{domain}/result-{i + 1}-{quote_plus(query[:20])}",
                    title=f"Result {i + 1}: {query.title()[:60]}",
                    snippet=f"This is a simulated result for '{query[:80]}'. "
                    f"Set BRAVE_SEARCH_API_KEY for real results.",
                    domain=domain,
                )
            )
        self.total_results += len(results)
        return results
