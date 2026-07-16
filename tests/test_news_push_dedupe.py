import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class NewsPushDedupeTests(unittest.TestCase):
    def setUp(self):
        if hasattr(eth._register_news_push_if_new, "_history"):
            delattr(eth._register_news_push_if_new, "_history")

    def tearDown(self):
        if hasattr(eth._register_news_push_if_new, "_history"):
            delattr(eth._register_news_push_if_new, "_history")

    def test_source_and_small_suffix_variants_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "news_push_dedupe.json"
            with patch.object(eth, "NEWS_PUSH_DEDUPE_PATH", state_path):
                self.assertTrue(eth._register_news_push_if_new(
                    "[Reuters] US jobless claims decline to 208,000, below forecasts", now_ts=1000
                ))
                self.assertFalse(eth._register_news_push_if_new(
                    "[News] US jobless claims decline to 208,000, below forecasts - update", now_ts=1100
                ))

    def test_distinct_headline_is_allowed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "news_push_dedupe.json"
            with patch.object(eth, "NEWS_PUSH_DEDUPE_PATH", state_path):
                self.assertTrue(eth._register_news_push_if_new(
                    "US CPI rises less than expected", now_ts=1000
                ))
                self.assertTrue(eth._register_news_push_if_new(
                    "Ethereum ETF inflows hit a record high", now_ts=1100
                ))

    def test_history_survives_process_cache_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "news_push_dedupe.json"
            with patch.object(eth, "NEWS_PUSH_DEDUPE_PATH", state_path):
                headline = "Bitcoin falls as Treasury yields surge"
                self.assertTrue(eth._register_news_push_if_new(headline, now_ts=1000))
                delattr(eth._register_news_push_if_new, "_history")
                self.assertFalse(eth._register_news_push_if_new(headline, now_ts=1200))


if __name__ == "__main__":
    unittest.main()
