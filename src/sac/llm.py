from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from sac.core import _format_items, _httpx_client

DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_API_KEY = "public"
DEFAULT_MODEL = "big-pickle"


class LLMSDKClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8192,
        max_chars: int = 10000,
        http_proxy: str | None = None,
        https_proxy: str | None = None,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("OPENAI_API_BASE") or DEFAULT_BASE_URL
        )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
        self._model = model
        self._max_tokens = max_tokens
        self._max_chars = max_chars
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            http_client=_httpx_client(http_proxy, https_proxy),
        )

    def synthesize(self, items: list[Any], instruction: str) -> str:
        context = _format_items(items, max_chars=self._max_chars)
        prompt = f"""<context>
{context}
</context>

<instruction>
{instruction}
</instruction>

Please provide a concise synthesis of the above information."""
        return self._call_llm(prompt)

    def plan(self, context: str, goal: str) -> str:
        prompt = f"""<context>
{context}
</context>

<goal>
{goal}
</goal>

Please produce a concrete search plan: what queries to run, what gaps to fill, and in what order."""
        return self._call_llm(prompt)

    def extract_many(
        self,
        items: list[dict[str, Any]],
        instruction: str,
        schema: dict[str, type | str],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        chunk_size = 10
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            records.extend(self._extract_chunk(chunk, instruction, schema))
        return records

    def _extract_chunk(
        self,
        items: list[dict[str, Any]],
        instruction: str,
        schema: dict[str, type | str],
    ) -> list[dict[str, Any]]:
        schema_lines = "\n".join(
            f'  "{k}": <{v.__name__ if isinstance(v, type) else v}>'
            for k, v in schema.items()
        )
        prompt = f"""Extract structured records from the items below.

Schema:
{schema_lines}

Instruction: {instruction}

Items:
{json.dumps(items, indent=2, default=str)}

Return a JSON list of objects matching the schema. Use null for missing fields.
Only return valid JSON, no other text."""
        raw = self._call_llm(prompt, max_tokens=self._max_tokens)
        return self._parse_json_list(raw, schema)

    def _call_llm(self, prompt: str, max_tokens: int | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens or self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            content = msg.reasoning_content
        return content or ""

    def _parse_json_list(
        self, raw: str, schema: dict[str, type | str]
    ) -> list[dict[str, Any]]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
            if not isinstance(data, list):
                return []
            result = []
            for item in data:
                if isinstance(item, dict):
                    row = {}
                    for k, v_type in schema.items():
                        val = item.get(k)
                        if val is not None and isinstance(v_type, type):
                            try:
                                row[k] = v_type(val)
                            except (ValueError, TypeError):
                                row[k] = val
                        else:
                            row[k] = None
                    result.append(row)
            return result
        except json.JSONDecodeError:
            return []
