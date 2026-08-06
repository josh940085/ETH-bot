import os
import unittest
from unittest import mock

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import news


class NewsRelevanceTests(unittest.TestCase):
    def test_auto_corrects_obvious_semantic_conflicts(self):
        cases = [
            (
                "An Inflation Double Whammy Awaits Wall Street, Making a Stock Market Crash Likelier",
                1,
                -1,
                "semantic_bearish_conflict",
            ),
            (
                "Bitcoin surges as spot ETF inflows hit a record",
                -1,
                1,
                "semantic_bullish_conflict",
            ),
        ]
        for headline, model_bias, expected_bias, expected_reason in cases:
            with self.subTest(headline=headline):
                with mock.patch.object(
                    news,
                    "predict_news_sentiment_with_confidence",
                    return_value=(model_bias, 0.91),
                ):
                    analysis = news.analyze_news_text(headline, log_result=False)
                self.assertEqual(analysis["bias"], expected_bias)
                self.assertTrue(analysis["correction_applied"])
                self.assertEqual(analysis["correction_reason"], expected_reason)
                self.assertIn("auto_corrected", analysis["tags"])
                message = news.build_news_message(
                    f"[Test] {headline}",
                    now_time="12:30:00",
                    analysis=analysis,
                )
                self.assertIn("🔧 自動修正:", message)

    def test_auto_corrects_bitcoin_whale_accumulation_and_distribution(self):
        cases = [
            (
                "Bitcoin whales accumulate as exchange outflows rise",
                -1,
                1,
                "bitcoin_whale_accumulation_conflict",
            ),
            (
                "BTC whale deposits 5,000 Bitcoin to Binance before selloff",
                1,
                -1,
                "bitcoin_whale_distribution_conflict",
            ),
            (
                "比特幣巨鯨持續累積並轉出交易所",
                -1,
                1,
                "bitcoin_whale_accumulation_conflict",
            ),
            (
                "巨鯨大額比特幣轉入交易所",
                1,
                -1,
                "bitcoin_whale_distribution_conflict",
            ),
        ]
        for headline, model_bias, expected_bias, expected_reason in cases:
            with self.subTest(headline=headline):
                with mock.patch.object(
                    news,
                    "predict_news_sentiment_with_confidence",
                    return_value=(model_bias, 0.91),
                ):
                    analysis = news.analyze_news_text(headline, log_result=False)
                self.assertEqual(analysis["bias"], expected_bias)
                self.assertTrue(analysis["correction_applied"])
                self.assertEqual(analysis["correction_reason"], expected_reason)

    def test_auto_corrects_recent_market_drivers(self):
        cases = [
            (
                "Bitcoin rises as spot Bitcoin ETF inflows extend for a fifth straight day",
                -1,
                1,
                "recent_market_driver_bullish_conflict",
                0,
            ),
            (
                "Bitcoin climbs as Clarity Act progress lifts crypto market sentiment",
                -1,
                1,
                "recent_market_driver_bullish_conflict",
                0,
            ),
            (
                "Bitcoin rises as U.S.-Iran diplomacy improves risk appetite",
                -1,
                1,
                "recent_market_driver_bullish_conflict",
                0,
            ),
            (
                "Bitcoin slips near $63K as spot Bitcoin ETF outflows weaken sentiment",
                1,
                -1,
                "recent_market_driver_bearish_conflict",
                1,
            ),
            (
                "Crypto selloff deepens after $510 million liquidations",
                1,
                -1,
                "recent_market_driver_bearish_conflict",
                1,
            ),
            (
                "Bitcoin falls as chip stock sell-off and Fed uncertainty weigh",
                1,
                -1,
                "recent_market_driver_bearish_conflict",
                1,
            ),
            (
                "Clarity Act delay weighs on Bitcoin sentiment",
                1,
                -1,
                "recent_market_driver_bearish_conflict",
                1,
            ),
            (
                "Oil spikes as Iran conflict escalates and risk assets fall",
                1,
                -1,
                "recent_market_driver_bearish_conflict",
                1,
            ),
        ]
        for headline, model_bias, expected_bias, expected_reason, expected_event_risk in cases:
            with self.subTest(headline=headline):
                with mock.patch.object(
                    news,
                    "predict_news_sentiment_with_confidence",
                    return_value=(model_bias, 0.91),
                ):
                    analysis = news.analyze_news_text(headline, log_result=False)
                self.assertEqual(analysis["bias"], expected_bias)
                self.assertEqual(analysis["event_risk"], expected_event_risk)
                self.assertTrue(analysis["correction_applied"])
                self.assertEqual(analysis["correction_reason"], expected_reason)

    def test_broad_market_direction_overrides_oil_price_move(self):
        cases = [
            (
                "Oil prices rise but US stock futures fall as inflation fears hit markets",
                1,
                -1,
                "global_equity_market_bearish",
                1,
            ),
            (
                "油價上漲但美股期貨下跌 通膨疑慮衝擊大盤",
                1,
                -1,
                "global_equity_market_bearish",
                1,
            ),
            (
                "Stocks Fall as US-Iran Worries Spur Rally in Oil: Markets Wrap",
                1,
                -1,
                "global_equity_market_bearish",
                1,
            ),
            (
                "Oil slips while S&P 500 futures rise ahead of the open",
                -1,
                1,
                "global_equity_market_bullish",
                0,
            ),
        ]
        for headline, model_bias, expected_bias, expected_reason, expected_event_risk in cases:
            with self.subTest(headline=headline):
                with mock.patch.object(
                    news,
                    "predict_news_sentiment_with_confidence",
                    return_value=(model_bias, 0.91),
                ):
                    analysis = news.analyze_news_text(headline, log_result=False)
                self.assertEqual(analysis["bias"], expected_bias)
                self.assertEqual(analysis["event_risk"], expected_event_risk)
                self.assertTrue(analysis["correction_applied"])
                self.assertEqual(analysis["correction_reason"], expected_reason)

    def test_auto_corrects_non_directional_headlines_to_neutral(self):
        headlines = [
            "Here's what happened in crypto today",
            "Over 70% of Gen Z investors hold crypto and only 13% of day traders make money",
        ]
        for headline in headlines:
            with self.subTest(headline=headline):
                with mock.patch.object(
                    news,
                    "predict_news_sentiment_with_confidence",
                    return_value=(1, 0.88),
                ):
                    analysis = news.analyze_news_text(headline, log_result=False)
                self.assertEqual(analysis["bias"], 0)
                self.assertTrue(analysis["correction_applied"])
                self.assertLess(analysis["ai_confidence"], 0.55)

    def test_low_accuracy_safe_mode_neutralizes_unsupported_direction(self):
        with (
            mock.patch.object(
                news,
                "predict_news_sentiment_with_confidence",
                return_value=(1, 0.82),
            ),
            mock.patch.object(
                news,
                "get_prediction_accuracy",
                return_value={"accuracy": 34.47, "total": 850, "correct": 293},
            ),
        ):
            analysis = news.analyze_news_text(
                "Ethereum market outlook remains uncertain this weekend",
                log_result=False,
            )
        self.assertEqual(analysis["bias"], 0)
        self.assertEqual(analysis["correction_reason"], "low_accuracy_neutral_fallback")

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
                self.assertEqual(news._news_relevance_reason(headline), "")

    def test_keeps_news_that_can_move_global_financial_markets(self):
        expected = {
            "Bitcoin turns lower as soft U.S. inflation data is offset by Iran tensions": "crypto",
            "Ethereum ETF inflows hit a record high": "crypto",
            "Bitcoin whales accumulate as exchange balances fall": "crypto",
            "BTC whale deposits 5,000 Bitcoin to Binance before selloff": "crypto",
            "Dormant wallet moves Bitcoin to cold storage": "crypto",
            "巨鯨大額比特幣轉入交易所": "crypto",
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
                self.assertEqual(news._news_relevance_reason(headline), reason)

    def test_major_national_selloffs_override_low_confidence_model(self):
        analysis = news.analyze_news_text("台股盤中跌逾2000點 AI、權值股重挫", log_result=False)
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
                analysis = news.analyze_news_text(headline, log_result=False)
                self.assertEqual(analysis["bias"], -2)
                self.assertGreaterEqual(analysis["ai_confidence"], 0.82)

    def test_small_taiwan_opening_move_does_not_force_push(self):
        bias, confidence = news._major_equity_market_move_override("台股開盤跌390.9點")
        self.assertEqual((bias, confidence), (0, 0.0))

    def test_small_country_market_move_does_not_force_strong_bias(self):
        bias, confidence = news._major_equity_market_move_override("Canadian stocks fall 0.3%")
        self.assertEqual((bias, confidence), (0, 0.0))

    def test_every_configured_country_and_index_is_in_global_scope(self):
        for country in news.GLOBAL_EQUITY_COUNTRY_TERMS:
            headline = f"{country} stocks plunge 3%"
            with self.subTest(country=country):
                self.assertTrue(news._is_global_equity_market_scope(headline))
                self.assertEqual(news._major_equity_market_move_override(headline)[0], -2)

        for index_name in news.GLOBAL_EQUITY_INDEX_TERMS:
            with self.subTest(index=index_name):
                self.assertTrue(news._is_global_equity_market_scope(index_name))

    def test_news_message_preserves_taiwan_rss_source(self):
        analysis = news.analyze_news_text("台股盤中跌逾2000點 AI、權值股重挫", log_result=False)
        message = news.build_news_message(
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
                self.assertEqual(news._news_relevance_reason(headline), "")

    def test_dedupe_ignores_source_and_exchange_suffix(self):
        first = "[Investing] Standard Nuclear prices IPO at $15 per share"
        second = "[Investing Crypto] Standard Nuclear prices IPO at $15 per share on NYSE"
        self.assertEqual(news._news_dedupe_key(first), news._news_dedupe_key(second))


if __name__ == "__main__":
    unittest.main()
