from __future__ import annotations

from unittest.mock import patch

import pytest

from sac.models import CACHE_DIR, ModelLimits


class TestModelLimits:
    def test_known_model(self):
        limit = ModelLimits.get_context_limit("gpt-4o")
        assert limit == 128000

    def test_fallback_unknown(self):
        limit = ModelLimits.get_context_limit("nonexistent-model-v42")
        assert limit == 8000

    def test_override_takes_precedence(self):
        limit = ModelLimits.get_context_limit("gpt-4o", override=999)
        assert limit == 999

    def test_substring_matches_defaults(self):
        limit = ModelLimits.get_context_limit("claude-3-5-sonnet-20241022")
        assert limit == 200000

    def test_big_pickle_returns_positive(self):
        limit = ModelLimits.get_context_limit("big-pickle")
        assert limit > 0

    @pytest.fixture
    def _isolated(self, tmp_path):
        with (
            patch.object(ModelLimits, "_cache", None),
            patch.object(ModelLimits, "_last_fetch", 0),
            patch("sac.models.CACHE_DIR", tmp_path / "cache"),
        ):
            yield

    def test_fallback_when_all_sources_unavailable(self, _isolated):
        with patch("sac.models.requests.get", side_effect=Exception("network down")):
            limit = ModelLimits.get_context_limit("gpt-4o")
            assert limit == 128000
