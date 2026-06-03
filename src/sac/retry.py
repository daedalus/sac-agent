"""Retry with exponential backoff for HTTP 429 (rate limit) and 402 (payment required)."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Base for errors that can be retried."""


class RateLimitError(RetryableError):
    """HTTP 429 — rate limited."""

    def __init__(
        self,
        message: str = "",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class FreeUsageLimitError(RetryableError):
    """HTTP 402 — payment required / free usage exhausted."""


class TransientError(RetryableError):
    """Network-level failure (timeout, connection error, DNS failure) — safe to retry."""


class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts are exhausted."""


DEFAULT_CONFIG = {
    "max_retries": 5,
    "initial_delay": 2.0,
    "backoff_factor": 2.0,
    "max_delay": 120.0,
}


def calculate_delay(
    attempt: int,
    error: RateLimitError | None = None,
    *,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    max_delay: float = 120.0,
) -> float:
    delay = initial_delay * (backoff_factor ** (attempt - 1))
    if isinstance(error, RateLimitError) and error.retry_after is not None:
        delay = min(error.retry_after, max_delay)
    return min(delay, max_delay)


def with_retry(
    func: Any,
    *,
    max_retries: int = 5,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    max_delay: float = 120.0,
) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except FreeUsageLimitError:
            raise
        except RetryableError as e:
            if attempt == max_retries:
                raise MaxRetriesExceeded(
                    f"Request failed after {max_retries} retries"
                ) from e
            delay = calculate_delay(
                attempt,
                e if isinstance(e, RateLimitError) else None,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                max_delay=max_delay,
            )
            logger.info(
                "Retry %d/%d after %.1fs due to: %s",
                attempt, max_retries, delay, e,
            )
            time.sleep(delay)

    raise AssertionError("unreachable")
