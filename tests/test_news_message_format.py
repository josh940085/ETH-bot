import unittest
from unittest.mock import patch

import news


class NewsMessageFormatTests(unittest.TestCase):
    def _message(self, bias):
        analysis = {
            "bias": bias,
            "sentiment": "偏多" if bias > 0 else ("偏空" if bias < 0 else "中性"),
            "impact": "測試影響",
            "ai_confidence": 0.621,
        }
        with (
            patch.object(news, "translate_news_to_zh", return_value="測試中文新聞"),
            patch.object(news, "get_prediction_accuracy", return_value={"accuracy": 57.5, "correct": 23, "total": 40}),
        ):
            return news.build_news_message("Test headline", now_time="23:34:59", analysis=analysis)

    def test_bullish_news_uses_red_and_news_is_first_line(self):
        message = self._message(1)
        self.assertEqual(message.splitlines()[0], "🔴 新聞(中文): 測試中文新聞")
        self.assertNotIn("🌐 新聞(中文)", message)

    def test_bearish_news_uses_green(self):
        self.assertTrue(self._message(-1).startswith("🟢 新聞(中文):"))

    def test_neutral_news_keeps_yellow(self):
        self.assertTrue(self._message(0).startswith("🟡 新聞(中文):"))


if __name__ == "__main__":
    unittest.main()
