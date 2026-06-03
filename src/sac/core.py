from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

VERBOSE = os.environ.get("SAC_VERBOSE", "").lower() in ("1", "true", "yes")


@dataclass
class SearchResult:
    url: str = ""
    title: str = ""
    snippet: str = ""
    domain: str = ""

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            try:
                self.domain = urlparse(self.url).netloc
            except Exception:
                self.domain = ""

    def __repr__(self) -> str:
        return f"SearchResult(domain={self.domain!r}, title={self.title!r})"


def _httpx_client(
    http_proxy: str | None = None, https_proxy: str | None = None
) -> httpx.Client | None:
    proxy_url = (
        https_proxy
        or http_proxy
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
    )
    if proxy_url:
        return httpx.Client(proxy=proxy_url)
    return None


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[sac] {msg}", file=sys.stderr)


def _proxy_config(
    http_proxy: str | None = None, https_proxy: str | None = None
) -> dict[str, str]:
    http = http_proxy or os.environ.get("HTTP_PROXY")
    https = https_proxy or os.environ.get("HTTPS_PROXY") or http
    if not https:
        return {}
    return {"http": https, "https": https}


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or ""


def _format_items(items: list[Any], max_chars: int = 10000) -> str:
    text = json.dumps(items, indent=2, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text
