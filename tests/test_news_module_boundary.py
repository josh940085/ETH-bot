import os
import unittest

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth
import news


class NewsModuleBoundaryTests(unittest.TestCase):
    def test_eth_uses_news_module_implementations(self):
        delegated_names = (
            "analyze_news_text",
            "build_news_message",
            "build_panel_news_items",
            "load_news_model",
            "normalize_news_text",
            "refresh_rss_news_cache",
            "translate_news_to_zh",
        )
        for name in delegated_names:
            with self.subTest(name=name):
                self.assertIs(getattr(eth, name), getattr(news, name))

    def test_discord_delivery_is_delegated(self):
        self.assertIs(eth._post_discord_webhook, news._post_discord_webhook)

    def test_host_learning_state_stays_in_trading_core(self):
        self.assertEqual(eth.BINANCE_HOST_LEARNING_STATE_PATH.name, "binance_host_learning_state.json")
        self.assertEqual(eth.BINANCE_HOST_LIVE_LEARNING_STATE_PATH.name, "binance_host_live_learning_state.json")


if __name__ == "__main__":
    unittest.main()
