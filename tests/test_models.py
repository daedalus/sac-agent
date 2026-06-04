from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sac.models import CACHE_DIR, ModelLimits


class TestModelLimits:
    @pytest.fixture(autouse=True)
    def _reset(self) -> None:
        ModelLimits.reset_cache()

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
        with patch("sac.models.CACHE_DIR", tmp_path / "cache"):
            yield

    def test_fallback_when_all_sources_unavailable(self, _isolated):
        with patch("sac.models.requests.get", side_effect=Exception("network down")):
            limit = ModelLimits.get_context_limit("gpt-4o")
            assert limit == 128000

    def test_corrupt_file_cache_falls_back(self, _isolated, tmp_path):
        cache_file = tmp_path / "cache" / "models_registry.json"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text("not valid json")
        limit = ModelLimits.get_context_limit("gpt-4o")
        assert limit == 128000

    def test_http_fetch_succeeds(self, _isolated, tmp_path):
        mock_data = {
            "provider": {
                "models": {
                    "custom-model-v1": {"limit": {"context": 64000}},
                }
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        with patch("sac.models.requests.get", return_value=mock_resp):
            limit = ModelLimits.get_context_limit("custom-model-v1")
            assert limit == 64000
            cache_file = tmp_path / "cache" / "models_registry.json"
            assert cache_file.exists()
            cached = json.loads(cache_file.read_text())
            assert cached == mock_data
