import os
import unittest

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class NewsRelevanceTests(unittest.TestCase):
    def test_rejects_recent_non_market_pushes(self):
        headlines = [
            "DOJ sentences two former TD Bank employees to prison",
            "Standard Nuclear prices IPO at $15 per share",
            "Standard Nuclear prices IPO at $15 per share on NYSE",
            "AMD: UBS hikes price target ahead of AI event on stronger GPU outlook",
            "Ex-Fed advisor gets over three years in prison for lying about China ties",
            "Form 4 Clear Secure Inc For: 15 July",
            "Warren Buffett initiated Berkshire Hathaway's investment in Alphabet",
            "Whether chipmakers can keep gaining remains uncertain",
            "Bitcoin stalls below $65,600 resistance: Live levels",
            "Ethereum price prediction for this weekend",
        ]
        for headline in headlines:
            with self.subTest(headline=headline):
                self.assertEqual(eth._news_relevance_reason(headline), "")

    def test_keeps_news_that_can_move_global_financial_markets(self):
        expected = {
            "Bitcoin turns lower as soft U.S. inflation data is offset by Iran tensions": "crypto",
            "Ethereum ETF inflows hit a record high": "crypto",
            "US CPI rises less than expected": "macro",
            "Fed signals rates may stay higher as inflation persists": "macro",
            "ECB cuts interest rates as European economy slows": "central_bank",
            "Bank of Japan signals policy tightening as yen weakens": "central_bank",
            "US stock futures fall as Treasury yields surge": "global_equities",
            "Nikkei plunges as global risk-off selling accelerates": "global_equities",
            "Euro falls as bond yields rise across Europe": "rates_fx",
            "Gold surges to record high on safe-haven demand": "commodities",
            "US announces new tariffs on China imports": "trade_policy",
            "Nvidia earnings beat estimates and raises guidance": "mega_cap",
            "Oil rises as Iran threatens Hormuz disruption": "commodities",
            "Russia-Ukraine ceasefire talks collapse amid escalation": "geopolitical",
            "China military drill near Taiwan Strait raises blockade fears": "geopolitical",
            "台股盤中跌逾2000點 AI、權值股重挫": "global_equities",
            "TAIEX plunges 4% as chip stocks lead regional selloff": "global_equities",
            "Brazil's Ibovespa tumbles 4% in broad market selloff": "global_equities",
            "Germany's DAX plunges as European shares retreat": "global_equities",
            "India Sensex plunges 3% as banks lead losses": "global_equities",
            "Saudi Arabia stocks plunge 3% on oil-market fears": "global_equities",
            "South Africa JSE All Share plunges 2.4%": "global_equities",
            "澳股暴跌3% 資源股領跌": "global_equities",
            "韓股大跌4% 外資賣壓沉重": "global_equities",
        }
        for headline, reason in expected.items():
            with self.subTest(headline=headline):
                self.assertEqual(eth._news_relevance_reason(headline), reason)

    def test_major_national_selloffs_override_low_confidence_model(self):
        analysis = eth.analyze_news_text("台股盤中跌逾2000點 AI、權值股重挫", log_result=False)
        self.assertEqual(analysis["bias"], -2)
        self.assertGreaterEqual(analysis["ai_confidence"], 0.82)
        self.assertEqual(analysis["fusion_method"], "major_global_equity_market_move_override")

        for headline in [
            "Japan's Nikkei plunges 3.2% as chip stocks slide",
            "Brazil stocks tumble 4% after fiscal shock",
            "印度股市暴跌3% 銀行股領跌",
            "南非股市大跌2.5%",
        ]:
            with self.subTest(headline=headline):
                analysis = eth.analyze_news_text(headline, log_result=False)
                self.assertEqual(analysis["bias"], -2)
                self.assertGreaterEqual(analysis["ai_confidence"], 0.82)

    def test_small_taiwan_opening_move_does_not_force_push(self):
        bias, confidence = eth._major_equity_market_move_override("台股開盤跌390.9點")
        self.assertEqual((bias, confidence), (0, 0.0))

    def test_small_country_market_move_does_not_force_strong_bias(self):
        bias, confidence = eth._major_equity_market_move_override("Canadian stocks fall 0.3%")
        self.assertEqual((bias, confidence), (0, 0.0))

    def test_every_configured_country_and_index_is_in_global_scope(self):
        for country in eth.GLOBAL_EQUITY_COUNTRY_TERMS:
            headline = f"{country} stocks plunge 3%"
            with self.subTest(country=country):
                self.assertTrue(eth._is_global_equity_market_scope(headline))
                self.assertEqual(eth._major_equity_market_move_override(headline)[0], -2)

        for index_name in eth.GLOBAL_EQUITY_INDEX_TERMS:
            with self.subTest(index=index_name):
                self.assertTrue(eth._is_global_equity_market_scope(index_name))

    def test_news_message_preserves_taiwan_rss_source(self):
        analysis = eth.analyze_news_text("台股盤中跌逾2000點 AI、權值股重挫", log_result=False)
        message = eth.build_news_message(
            "[中央社財經] 台股盤中跌逾2000點 AI、權值股重挫",
            now_time="12:30:00",
            analysis=analysis,
        )
        self.assertIn("來源: 中央社財經", message)

    def test_global_scope_still_rejects_routine_company_noise(self):
        headlines = [
            "Apple appoints new regional sales chief",
            "SmallCap Inc reports quarterly earnings",
            "Tesla analyst raises price target to $500",
            "Local retailer opens its twentieth store",
        ]
        for headline in headlines:
            with self.subTest(headline=headline):
                self.assertEqual(eth._news_relevance_reason(headline), "")

    def test_dedupe_ignores_source_and_exchange_suffix(self):
        first = "[Investing] Standard Nuclear prices IPO at $15 per share"
        second = "[Investing Crypto] Standard Nuclear prices IPO at $15 per share on NYSE"
        self.assertEqual(eth._news_dedupe_key(first), eth._news_dedupe_key(second))


if __name__ == "__main__":
    unittest.main()
