"""Integration test: full agent.run() loop with mocked LLM and simulated search."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sac import SaCAgent
from sac.sdk import AgenticSearchSDK


class TestAgentIntegration:
    """Full agent loop with mocked LLM and real sandbox + simulated search."""

    @pytest.fixture(autouse=True)
    def _env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            yield

    def _make_agent(self, max_turns: int = 3) -> SaCAgent:
        sdk = AgenticSearchSDK(llm_api_key="test", llm_base_url="http://localhost:0")
        sdk.search._simulate = True
        return SaCAgent(
            task="research test",
            sdk=sdk,
            base_url="http://localhost:0",
            api_key="test",
            max_turns=max_turns,
        )

    def _mock(self, content: str) -> MagicMock:
        resp = MagicMock()
        resp.choices[0].message.content = content
        return resp

    def _llm_chain(self, *contents: str) -> Any:
        """Return a callable that yields mock responses for each content string."""
        mocks = [self._mock(c) for c in contents]

        def factory(**kw):
            return mocks.pop(0)

        return factory

    # ── tests ──────────────────────────────────────────────────

    def test_single_code_then_synthesis(self):
        agent = self._make_agent(max_turns=2)
        code_body = (
            'results = sdk.search.web("test query", limit=3)\n'
            "for r in results:\n"
            '    sdk.fs.write("result:" + r.domain, r.title)\n'
            '    print(r.title, r.url)'
        )
        agent._client.chat.completions.create = self._llm_chain(
            json.dumps({"turn_type": "code", "reasoning": "searching", "code": code_body}),
            json.dumps({"turn_type": "synthesis", "reasoning": "done", "answer": "Found results about test query."}),
        )
        answer = agent.run()
        assert "Found results about test query." in answer
        assert agent.sdk.search.total_results >= 1
        keys = agent.sdk.fs.list()
        assert any("result" in k for k in keys)

    def test_multi_turn_backfill_then_synthesis(self):
        agent = self._make_agent(max_turns=3)
        agent._client.chat.completions.create = self._llm_chain(
            json.dumps({
                "turn_type": "code", "reasoning": "searching",
                "code": (
                    'results = sdk.search.web_many(["python", "rust"], limit_per_query=2)\n'
                    "for lst in results:\n"
                    "    for r in lst:\n"
                    '        sdk.fs.write("page:" + r.domain, r.title)\n'
                    '        print(r.title)'
                ),
            }),
            json.dumps({
                "turn_type": "code", "reasoning": "reading",
                "code": (
                    'keys = sdk.fs.list()\n'
                    'print(f"Persisted {len(keys)} keys")\n'
                    'for k in keys:\n'
                    '    val = sdk.fs.read(k)\n'
                    '    print(k, val)'
                ),
            }),
            json.dumps({"turn_type": "synthesis", "reasoning": "done", "answer": "Covered python and rust thoroughly."}),
        )
        answer = agent.run()
        assert "python and rust" in answer
        assert agent.sdk.search.total_queries >= 2

    def test_code_error_triggers_fix_and_recovers(self):
        agent = self._make_agent(max_turns=3)

        fixer_content = (
            'results = sdk.search.web("fix test", limit=2)\n'
            "for r in results:\n"
            '    print(r.title)'
        )
        fixer_mock = self._mock(
            f"```python\n{fixer_content}\n```"
        )

        chain = [
            json.dumps({"turn_type": "code", "reasoning": "searching", "code": 'print(undefined_var)'}),
            json.dumps({"turn_type": "code", "reasoning": "fixed", "code": fixer_content}),
            json.dumps({"turn_type": "synthesis", "reasoning": "done", "answer": "Recovered from error."}),
        ]

        def side_effect(**kw):
            msgs = kw.get("messages", [])
            last = msgs[-1].get("content", "") if msgs else ""
            if "raised an error" in last:
                return fixer_mock
            return self._mock(chain.pop(0))

        agent._client.chat.completions.create = MagicMock(side_effect=side_effect)
        answer = agent.run()
        assert "Recovered" in answer

    def test_force_synthesis_when_max_turns_exceeded(self):
        agent = self._make_agent(max_turns=1)
        agent._client.chat.completions.create = self._llm_chain(
            json.dumps({"turn_type": "code", "reasoning": "searching", "code": 'print("only turn")'}),
            '{"turn_type": "synthesis", "answer": "forced end"}',
        )
        answer = agent.run()
        assert "forced end" in answer

    def test_synthesis_with_references_section(self):
        agent = self._make_agent(max_turns=2)
        synth_answer = (
            "Found references.\n\n"
            "## References\n"
            "- https://example.com/ref-1\n"
            "- https://example.com/ref-2"
        )
        code_body = (
            'results = sdk.search.web("ref test", limit=2)\n'
            "for r in results:\n"
            '    sdk.fs.write("ref:" + r.domain, r.url)\n'
            '    print(r.url)'
        )
        agent._client.chat.completions.create = self._llm_chain(
            json.dumps({"turn_type": "code", "reasoning": "searching", "code": code_body}),
            json.dumps({"turn_type": "synthesis", "reasoning": "done", "answer": synth_answer}),
        )
        answer = agent.run()
        assert "## References" in answer
        assert "https://" in answer
