"""Tests for cache module — BloomFilter (bytearray) + Cache (SQLite)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sac.cache import (
    SENTINEL,
    BloomFilter,
    Cache,
    _max_entries_from_env,
    _record_cache_hit,
    _record_cache_miss,
    get_cache_metrics_snapshot,
    is_cache_miss,
    reset_cache_metrics,
)


class BloomFilterByteArrayTests(unittest.TestCase):
    """Bloom filter backed by bytearray — no bitarray dependency."""

    def setUp(self):
        self.bf = BloomFilter(1000)

    def test_add_and_query(self):
        self.bf.add("hello")
        self.assertTrue(self.bf.query("hello"))

    def test_query_absent_returns_false(self):
        self.assertFalse(self.bf.query("never_added"))

    def test_multiple_keys(self):
        keys = [f"key{i}" for i in range(100)]
        for k in keys:
            self.bf.add(k)
        for k in keys:
            self.assertTrue(self.bf.query(k), f"missing key: {k}")

    def test_update_returns_true_when_present(self):
        self.bf.add("existing")
        self.assertTrue(self.bf.update("existing"))

    def test_update_returns_false_when_new(self):
        self.assertFalse(self.bf.update("brand_new"))
        self.assertTrue(self.bf.query("brand_new"))

    def test_clear_resets_all_bits(self):
        self.bf.add("hello")
        self.bf.add("world")
        self.bf.clear()
        self.assertFalse(self.bf.query("hello"))
        self.assertFalse(self.bf.query("world"))

    def test_load_factor_zero_on_init(self):
        self.assertEqual(self.bf.load_factor, 0.0)

    def test_load_factor_increases_after_adds(self):
        for i in range(100):
            self.bf.add(f"key{i}")
        self.assertGreater(self.bf.load_factor, 0.0)

    def test_false_positive_rate_within_bounds(self):
        n = 1000
        bf = BloomFilter(n, error_rate=0.01)
        for i in range(n):
            bf.add(f"key{i}")
        false_positives = 0
        trials = n
        for i in range(n, n + trials):
            if bf.query(f"key{i}"):
                false_positives += 1
        rate = false_positives / trials
        self.assertLessEqual(rate, 0.05)

    def test_zero_false_negatives(self):
        n = 500
        for i in range(n):
            self.bf.add(f"test{i}")
        for i in range(n):
            self.assertTrue(self.bf.query(f"test{i}"), f"false negative: test{i}")

    def test_small_capacity_does_not_crash(self):
        tiny = BloomFilter(1)
        tiny.add("a")
        self.assertTrue(tiny.query("a"))

    def test_digest_is_deterministic(self):
        d1 = BloomFilter._digest("hello world")
        d2 = BloomFilter._digest("hello world")
        self.assertEqual(d1, d2)

    def test_different_keys_have_different_digests(self):
        d1 = BloomFilter._digest("alpha")
        d2 = BloomFilter._digest("beta")
        self.assertNotEqual(d1, d2)


class CacheTests(unittest.TestCase):
    """SQLite-backed cache with bloom gate."""

    """SQLite-backed cache with bloom gate."""

    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        self.cache = Cache(self.tmp)
        reset_cache_metrics()

    def tearDown(self):
        self.cache.close()
        os.unlink(self.tmp)

    def test_get_miss_returns_none(self):
        self.assertIsNone(self.cache.get("nonexistent"))

    def test_get_or_sentinel_miss_returns_sentinel(self):
        self.assertIs(self.cache.get_or_sentinel("nonexistent"), SENTINEL)

    def test_set_and_get(self):
        self.cache.set("k1", "value1")
        self.assertEqual(self.cache.get("k1"), "value1")

    def test_overwrite(self):
        self.cache.set("k", "v1")
        self.cache.set("k", "v2")
        self.assertEqual(self.cache.get("k"), "v2")

    def test_complex_value_roundtrip(self):
        data = {"a": [1, 2, 3], "b": "hello", "c": None}
        self.cache.set("complex", data)
        self.assertEqual(self.cache.get("complex"), data)

    def test_list_value_roundtrip(self):
        snippets = [
            {"id": "s1", "content": "void f() {}", "tags": ["memory"]},
            {"id": "s2", "content": "int g() { return 0; }", "tags": []},
        ]
        self.cache.set("snippet:list", snippets)
        self.assertEqual(self.cache.get("snippet:list"), snippets)

    def test_get_or_sentinel_hit_returns_value(self):
        self.cache.set("greeting", "hello")
        self.assertEqual(self.cache.get_or_sentinel("greeting"), "hello")

    def test_is_cache_miss_with_sentinel(self):
        self.assertTrue(is_cache_miss(SENTINEL))

    def test_is_cache_miss_with_value(self):
        self.assertFalse(is_cache_miss("hello"))
        self.assertFalse(is_cache_miss(None))
        self.assertFalse(is_cache_miss(42))

    def test_clear_empties_cache(self):
        self.cache.set("a", 1)
        self.cache.set("b", 2)
        self.cache.clear()
        self.assertIsNone(self.cache.get("a"))
        self.assertIsNone(self.cache.get("b"))

    def test_ttl_expiry(self):
        self.cache.set("ttl:key", "v", ttl_seconds=1)
        self.assertEqual(self.cache.get("ttl:key"), "v")
        self.cache.con.execute(
            "UPDATE cache SET expires_at = ? WHERE key = ?",
            (0.0, self.cache.canonicalize_key("ttl:key")),
        )
        self.cache.con.commit()
        self.assertIsNone(self.cache.get("ttl:key"))
        self.assertIs(self.cache.get_or_sentinel("ttl:key"), SENTINEL)

    def test_prunes_oldest_when_limit_exceeded(self):
        fd, tmp_path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        limited = Cache(tmp_path, max_entries=2)
        limited.set("k1", "v1")
        limited.set("k2", "v2")
        limited.set("k3", "v3")
        self.assertIsNone(limited.get("k1"))
        self.assertEqual(limited.get("k2"), "v2")
        self.assertEqual(limited.get("k3"), "v3")
        limited.close()
        os.unlink(tmp_path)

    def test_key_canonicalization(self):
        self.cache.set("snippet:  file   hash123  ", "value")
        self.assertEqual(self.cache.get("snippet:file hash123"), "value")
        self.assertEqual(self.cache.get("SNIPPET:   file hash123   "), "value")

    def test_bloom_added_on_set_removed_on_clear(self):
        self.cache.set("bloom:key", "v")
        self.assertTrue(
            self.cache._bloom.query(self.cache.canonicalize_key("bloom:key"))
        )
        self.cache.clear()
        self.assertFalse(
            self.cache._bloom.query(self.cache.canonicalize_key("bloom:key"))
        )
        self.assertIsNone(self.cache.get("bloom:key"))

    def test_metrics_record_hits_and_misses(self):
        self.cache.set("metric:key", "value")
        self.cache.get_or_sentinel("metric:key")
        self.cache.get_or_sentinel("metric:unknown")
        snapshot = get_cache_metrics_snapshot()
        self.assertGreaterEqual(snapshot["hits"], 1)
        self.assertGreaterEqual(snapshot["misses"], 1)

    def test_metrics_namespace_tracking(self):
        _record_cache_hit("snippet:abc")
        _record_cache_miss("recon:xyz")
        snapshot = get_cache_metrics_snapshot()
        self.assertIn("snippet", snapshot["hits_by_namespace"])
        self.assertIn("recon", snapshot["misses_by_namespace"])

    def test_reset_metrics_clears_counts(self):
        _record_cache_hit("test:k")
        reset_cache_metrics()
        snapshot = get_cache_metrics_snapshot()
        self.assertEqual(snapshot["hits"], 0)

    def test_max_entries_from_env(self):
        with patch("os.getenv", return_value="1000"):
            self.assertEqual(_max_entries_from_env(), 1000)

    def test_max_entries_from_env_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_max_entries_from_env(), 5000)

    def test_max_entries_from_env_invalid_raises(self):
        with patch("os.getenv", return_value="not_an_int"):
            with self.assertRaises(ValueError):
                _max_entries_from_env()


if __name__ == "__main__":
    unittest.main()
