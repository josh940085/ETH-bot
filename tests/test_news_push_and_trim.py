import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import news


class TrimRecentNewsKeysTests(unittest.TestCase):
    def test_keeps_most_recently_inserted_keys(self):
        seen = {}
        for i in range(500):
            seen[f"headline_{i:04d}"] = True

        trimmed = news._trim_recent_news_keys(seen, max_keep=200)

        self.assertEqual(len(trimmed), 200)
        # The 200 most-recently-inserted keys are headline_0300..headline_0499.
        expected = {f"headline_{i:04d}" for i in range(300, 500)}
        self.assertEqual(set(trimmed.keys()), expected)

    def test_reinserting_an_existing_key_does_not_change_its_order(self):
        # Plain dict assignment to an existing key updates the value in
        # place without moving it - this is FIFO-by-first-sighting, which is
        # what process_and_push_news wants: a headline inspected again is a
        # no-op, it should not artificially extend that headline's retention.
        seen = {}
        for i in range(500):
            seen[f"headline_{i:04d}"] = True
        seen["headline_0010"] = True

        trimmed = news._trim_recent_news_keys(seen, max_keep=200)

        self.assertNotIn("headline_0010", trimmed)

    def test_does_not_mutate_input(self):
        seen = {f"headline_{i}": True for i in range(10)}
        original = dict(seen)

        news._trim_recent_news_keys(seen, max_keep=5)

        self.assertEqual(seen, original)

    def test_max_keep_zero_returns_empty(self):
        seen = {"a": True, "b": True}
        self.assertEqual(news._trim_recent_news_keys(seen, max_keep=0), {})


class ProcessAndPushNewsTests(unittest.TestCase):
    """analyze_news_text is called from two places: build_panel_news_items
    (log_result=False, purely for the panel preview, happens for every
    market-relevant headline on every call) and process_and_push_news's own
    push-decision loop (default log_result=True, only for genuinely-new
    headlines). Assertions below key off _post_discord_webhook / the
    positional-only analyze_news_text call to distinguish "was this pushed"
    from "was this merely shown in the panel".
    """

    def setUp(self):
        for attr in ("last_news_set", "startup_news_snapshot_sent"):
            if hasattr(news.process_and_push_news, attr):
                delattr(news.process_and_push_news, attr)
        if hasattr(news._register_news_push_if_new, "_history"):
            delattr(news._register_news_push_if_new, "_history")
        self._tmp = tempfile.TemporaryDirectory()
        self._path_patch = patch.object(
            news, "NEWS_PUSH_DEDUPE_PATH", Path(self._tmp.name) / "news_push_dedupe.json"
        )
        self._path_patch.start()

    def tearDown(self):
        self._path_patch.stop()
        self._tmp.cleanup()
        for attr in ("last_news_set", "startup_news_snapshot_sent"):
            if hasattr(news.process_and_push_news, attr):
                delattr(news.process_and_push_news, attr)
        if hasattr(news._register_news_push_if_new, "_history"):
            delattr(news._register_news_push_if_new, "_history")

    def _skip_startup_snapshot(self):
        news.process_and_push_news.startup_news_snapshot_sent = True
        news.process_and_push_news.last_news_set = {}

    def test_first_call_is_startup_snapshot_and_does_not_push(self):
        with patch.object(news, "_post_discord_webhook") as mock_post:
            panel_items = news.process_and_push_news(
                ["[Reuters] Fed holds interest rates steady"],
                discord_webhook="https://discord.test/webhook",
            )

        mock_post.assert_not_called()
        self.assertTrue(news.process_and_push_news.startup_news_snapshot_sent)
        self.assertEqual(len(panel_items), 1)

    def test_new_relevant_headline_is_analyzed_and_pushed(self):
        self._skip_startup_snapshot()
        with (
            patch.object(
                news,
                "analyze_news_text",
                return_value={"bias": 1, "ai_confidence": 0.9, "confidence": 0.9},
            ) as mock_analyze,
            patch.object(news, "build_news_message", return_value="📊 解讀: 看多\n測試新聞"),
            patch.object(news, "_post_discord_webhook") as mock_post,
        ):
            news.process_and_push_news(
                ["[Reuters] Bitcoin surges past key resistance"],
                discord_webhook="https://discord.test/webhook",
            )

        mock_analyze.assert_any_call("Bitcoin surges past key resistance")
        mock_post.assert_called_once()

    def test_duplicate_headline_is_not_pushed_twice(self):
        self._skip_startup_snapshot()
        headline = ["[Reuters] Bitcoin surges past key resistance"]
        with (
            patch.object(
                news,
                "analyze_news_text",
                return_value={"bias": 1, "ai_confidence": 0.9, "confidence": 0.9},
            ),
            patch.object(news, "build_news_message", return_value="📊 解讀: 看多\n測試新聞"),
            patch.object(news, "_post_discord_webhook") as mock_post,
        ):
            news.process_and_push_news(headline, discord_webhook="https://discord.test/webhook")
            news.process_and_push_news(headline, discord_webhook="https://discord.test/webhook")

        mock_post.assert_called_once()

    def test_low_confidence_headline_is_not_pushed(self):
        self._skip_startup_snapshot()
        with (
            patch.object(
                news,
                "analyze_news_text",
                return_value={"bias": 1, "ai_confidence": 0.1, "confidence": 0.1},
            ),
            patch.object(news, "_post_discord_webhook") as mock_post,
        ):
            news.process_and_push_news(
                ["[Reuters] Bitcoin surges past key resistance"],
                discord_webhook="https://discord.test/webhook",
            )

        mock_post.assert_not_called()

    def test_neutral_bias_headline_is_not_pushed(self):
        self._skip_startup_snapshot()
        with (
            patch.object(
                news,
                "analyze_news_text",
                return_value={"bias": 0, "ai_confidence": 0.9, "confidence": 0.9},
            ),
            patch.object(news, "_post_discord_webhook") as mock_post,
        ):
            news.process_and_push_news(
                ["[Reuters] Bitcoin surges past key resistance"],
                discord_webhook="https://discord.test/webhook",
            )

        mock_post.assert_not_called()

    def test_irrelevant_headline_is_never_pushed(self):
        self._skip_startup_snapshot()
        with patch.object(news, "_post_discord_webhook") as mock_post:
            news.process_and_push_news(
                ["[Reuters] Local weather forecast calls for rain this weekend"],
                discord_webhook="https://discord.test/webhook",
            )

        mock_post.assert_not_called()

    def test_returns_panel_items_for_display(self):
        self._skip_startup_snapshot()
        panel_items = news.process_and_push_news(
            ["[Reuters] Bitcoin surges past key resistance"], discord_webhook=""
        )
        self.assertIsInstance(panel_items, list)


if __name__ == "__main__":
    unittest.main()
