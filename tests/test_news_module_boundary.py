import os
import unittest

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth
import news
import telegram


class NewsModuleBoundaryTests(unittest.TestCase):
    def test_eth_uses_news_module_implementations(self):
        delegated_names = (
            "load_news_model",
            "normalize_news_text",
            "process_and_push_news",
            "refresh_rss_news_cache",
            "translate_news_to_zh",
        )
        for name in delegated_names:
            with self.subTest(name=name):
                self.assertIs(getattr(eth, name), getattr(news, name))

    def test_eth_no_longer_holds_its_own_news_decision_helpers(self):
        # The news-push decision loop (which headlines are new, analyzed,
        # deduped, pushed) lives entirely behind process_and_push_news now -
        # eth.py shouldn't need direct access to these anymore.
        for name in ("analyze_news_text", "build_news_message", "build_panel_news_items"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(eth, name))

    def test_discord_delivery_is_delegated(self):
        # Discord webhook delivery lives in telegram.py (the notification
        # module) rather than news.py, since Discord is used for both news
        # pushes and trade-entry notifications. eth.py no longer needs it
        # directly at all - it's encapsulated inside process_and_push_news.
        self.assertIs(news._post_discord_webhook, telegram._post_discord_webhook)
        self.assertFalse(hasattr(eth, "_post_discord_webhook"))

    def test_host_learning_state_stays_in_trading_core(self):
        self.assertEqual(
            eth.BINANCE_HOST_LEARNING_STATE_PATH.name,
            "btcusdt_binance_host_learning_state.json",
        )
        self.assertEqual(
            eth.BINANCE_HOST_LIVE_LEARNING_STATE_PATH.name,
            "btcusdt_binance_host_live_learning_state.json",
        )


if __name__ == "__main__":
    unittest.main()
