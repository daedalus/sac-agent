"""Tests for retry module — exponential backoff for 429/402."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from sac.retry import (
    FreeUsageLimitError,
    MaxRetriesExceeded,
    RateLimitError,
    calculate_delay,
    with_retry,
)


class TestCalculateDelay:
    def test_first_attempt_default(self):
        assert calculate_delay(1) == 2.0

    def test_exponential_backoff(self):
        assert calculate_delay(2) == 4.0
        assert calculate_delay(3) == 8.0
        assert calculate_delay(4) == 16.0

    def test_capped_at_max_delay(self):
        assert calculate_delay(10, max_delay=30.0) == 30.0

    def test_rate_limit_error_honors_retry_after(self):
        err = RateLimitError(retry_after=5.0)
        assert calculate_delay(1, err) == 5.0

    def test_rate_limit_retry_after_still_capped(self):
        err = RateLimitError(retry_after=999.0)
        assert calculate_delay(1, err, max_delay=10.0) == 10.0

    def test_custom_initial_delay(self):
        assert calculate_delay(1, initial_delay=1.0) == 1.0

    def test_custom_backoff_factor(self):
        assert calculate_delay(2, backoff_factor=3.0, initial_delay=1.0) == 3.0


class TestWithRetry:
    def test_success_no_retry(self):
        fn = MagicMock(return_value="ok")
        assert with_retry(fn) == "ok"
        fn.assert_called_once()

    def test_success_after_one_retry(self):
        results = [RateLimitError("first"), "ok"]
        fn = MagicMock(side_effect=results)
        assert with_retry(fn) == "ok"
        assert fn.call_count == 2

    def test_max_retries_exceeded(self):
        fn = MagicMock(side_effect=RateLimitError("always fail"))
        with pytest.raises(MaxRetriesExceeded):
            with_retry(fn, max_retries=3)
        assert fn.call_count == 3

    def test_free_usage_not_retried(self):
        fn = MagicMock(side_effect=FreeUsageLimitError("out of credits"))
        with pytest.raises(FreeUsageLimitError):
            with_retry(fn)
        fn.assert_called_once()

    def test_success_after_multiple_retries(self):
        results = [
            RateLimitError("fail1"),
            RateLimitError("fail2"),
            RateLimitError("fail3"),
            "ok",
        ]
        fn = MagicMock(side_effect=results)
        assert with_retry(fn, max_retries=5) == "ok"
        assert fn.call_count == 4

    def test_rate_limit_with_retry_after_header(self):
        err = RateLimitError("limited", retry_after=0.1)
        fn = MagicMock(side_effect=[err, "ok"])
        assert with_retry(fn, max_retries=2, initial_delay=10.0) == "ok"
        # Should use retry_after=0.1 instead of initial_delay=10.0
        assert fn.call_count == 2


class TestRetryIntegration:
    """Retry applied to real-ish HTTP scenarios via mocking."""

    def test_brave_429_retries_then_succeeds(self, mocker):
        mock_get = mocker.patch("sac.search.requests.get")
        fail_resp = MagicMock(spec=requests.Response)
        fail_resp.status_code = 429
        fail_resp.headers = {"retry-after": "0.01"}
        fail_resp.text = "rate limited"
        fail_resp.raise_for_status.side_effect = requests.HTTPError("429")

        ok_resp = MagicMock(spec=requests.Response)
        ok_resp.status_code = 200
        ok_resp.headers = {}
        ok_resp.json.return_value = {
            "web": {
                "results": [
                    {"url": "https://a.com", "title": "A", "description": "Desc"}
                ]
            }
        }

        mock_get.side_effect = [fail_resp, ok_resp]

        from sac.search import SearchSDK

        sdk = SearchSDK(brave_key="test_key")
        results = sdk._brave_search("test", 1)
        assert len(results) == 1
        assert results[0].title == "A"
        assert mock_get.call_count == 2

    def test_brave_402_not_retried(self, mocker):
        mock_get = mocker.patch("sac.search.requests.get")
        fail_resp = MagicMock(spec=requests.Response)
        fail_resp.status_code = 402
        fail_resp.text = "payment required"
        fail_resp.headers = {}
        mock_get.return_value = fail_resp

        from sac.search import SearchSDK

        sdk = SearchSDK(brave_key="test_key")

        from sac.retry import FreeUsageLimitError

        with pytest.raises(FreeUsageLimitError):
            sdk._brave_search("test", 1)

    def test_brave_429_exhausted(self, mocker):
        mock_get = mocker.patch("sac.search.requests.get")
        fail_resp = MagicMock(spec=requests.Response)
        fail_resp.status_code = 429
        fail_resp.headers = {"retry-after": "1"}
        fail_resp.text = "rate limited"
        fail_resp.raise_for_status.side_effect = requests.HTTPError("429")
        mock_get.return_value = fail_resp

        from sac.search import SearchSDK

        sdk = SearchSDK(brave_key="test_key")

        with pytest.raises(MaxRetriesExceeded):
            sdk._brave_search("test", 1)

    def test_exa_429_retries_then_succeeds(self, mocker):
        mock_post = mocker.patch("sac.search.requests.post")
        fail_resp = MagicMock(spec=requests.Response)
        fail_resp.status_code = 429
        fail_resp.headers = {"retry-after": "0.01"}
        fail_resp.text = "rate limited"
        fail_resp.raise_for_status.side_effect = requests.HTTPError("429")

        ok_resp = MagicMock(spec=requests.Response)
        ok_resp.status_code = 200
        ok_resp.headers = {}
        ok_resp.text = 'data: {"result": {"content": [{"text": "Title: T\\nURL: https://x.com\\nHighlights:\\ncontent"}]}}\n'

        mock_post.side_effect = [fail_resp, ok_resp]

        from sac.search import SearchSDK

        sdk = SearchSDK()
        results = sdk._exa_mcp_search("test", 1)
        assert len(results) == 1
        assert results[0].title == "T"
        assert mock_post.call_count == 2

    def test_exa_402_not_retried(self, mocker):
        mock_post = mocker.patch("sac.search.requests.post")
        fail_resp = MagicMock(spec=requests.Response)
        fail_resp.status_code = 402
        fail_resp.text = "payment required"
        fail_resp.headers = {}
        mock_post.return_value = fail_resp

        from sac.search import SearchSDK

        sdk = SearchSDK()

        with pytest.raises(FreeUsageLimitError):
            sdk._exa_mcp_search("test", 1)
