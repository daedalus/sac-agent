from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from sac.core import _log

DEFAULT_LIMITS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
    "claude-3-sonnet": 200000,
    "claude-4": 200000,
    "claude-4-5": 200000,
    "big-pickle": 128000,
    "default": 8000,
}

CACHE_DIR = Path.home() / ".cache" / "sac-agent"
CACHE_TTL = 6 * 60 * 60


class ModelLimits:
    _cache: dict[str, Any] | None = None
    _last_fetch: float = 0

    @classmethod
    def reset_cache(cls) -> None:
        cls._cache = None
        cls._last_fetch = 0

    @classmethod
    def get_context_limit(cls, model: str, override: int | None = None) -> int:
        if override is not None:
            return override
        return cls._lookup(model)

    @classmethod
    def _lookup(cls, model: str) -> int:
        registry = cls._fetch_registry()
        if registry:
            limit = cls._search_registry(registry, model)
            if limit:
                return limit
        return cls._fallback(model)

    @classmethod
    def _fetch_registry(cls) -> dict[str, Any] | None:
        now = time.time()
        if cls._cache and (now - cls._last_fetch) < CACHE_TTL:
            return cls._cache

        cache_file = CACHE_DIR / "models_registry.json"
        if cache_file.exists() and (now - cache_file.stat().st_mtime) < CACHE_TTL:
            try:
                with open(cache_file) as f:
                    cls._cache = json.load(f)
                    cls._last_fetch = now
                    return cls._cache
            except Exception:
                pass

        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        try:
            resp = requests.get(
                "https://models.dev/api.json",
                timeout=10,
                proxies={"https": proxy} if proxy else None,
            )
            resp.raise_for_status()
            data = resp.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(data, f)
            cls._cache = data
            cls._last_fetch = now
            return data
        except Exception as e:
            _log(f"Failed to fetch models.dev: {e}")
            if cls._cache:
                return cls._cache
            return None

    @classmethod
    def _search_registry(cls, registry: dict[str, Any], model: str) -> int | None:
        model_lower = model.lower()
        for provider_data in registry.values():
            for model_id, model_data in provider_data.get("models", {}).items():
                limit = model_data.get("limit", {})
                ctx = limit.get("context", 0)
                if not ctx:
                    continue
                mid_lower = model_id.lower()
                if (
                    model_lower == mid_lower
                    or model_lower in mid_lower
                    or mid_lower in model_lower
                ):
                    return int(ctx)
        return None

    @classmethod
    def _fallback(cls, model: str) -> int:
        model_lower = model.lower()
        for key, limit in DEFAULT_LIMITS.items():
            if key in model_lower:
                return limit
        return DEFAULT_LIMITS["default"]
