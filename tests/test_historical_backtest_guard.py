import unittest
from unittest import mock

import historical_backtest as hb


def _summaries(return_values, trades):
    return {
        label: {"final_equity": return_values[index], "trades": trades[index]}
        for index, (label, _start, _end, _basename) in enumerate(hb.PERIODS)
    }


class HistoricalBacktestGuardTests(unittest.TestCase):
    def test_accepts_only_when_return_improves_and_counts_hold(self):
        baseline = _summaries([1.01] * 5, [100, 100, 100, 100, 50])
        candidate = _summaries([1.011] * 5, [100, 101, 100, 100, 50])
        result = hb._evaluate_candidate(candidate, baseline)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["return_improved"])
        self.assertEqual(result["trade_delta"], 1)

    def test_rejects_higher_return_when_one_period_trade_count_falls(self):
        baseline = _summaries([1.01] * 5, [100, 100, 100, 100, 50])
        candidate = _summaries([1.02] * 5, [99, 102, 100, 100, 50])
        result = hb._evaluate_candidate(candidate, baseline)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["total_trade_count_preserved"])
        self.assertFalse(result["all_period_trade_counts_preserved"])

    def test_rejects_equal_return_even_when_trade_count_holds(self):
        baseline = _summaries([1.01] * 5, [100, 100, 100, 100, 50])
        candidate = _summaries([1.01] * 5, [101, 100, 100, 100, 50])
        result = hb._evaluate_candidate(candidate, baseline)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["return_improved"])

    def test_openai_review_is_scoped_and_uses_upgraded_model(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "id": "resp_test",
                    "output": [{"content": [{"type": "output_text", "text": "審核完成"}]}],
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                }

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse()

        candidate = _summaries([1.02] * 5, [100, 100, 100, 100, 50])
        baseline = _summaries([1.01] * 5, [100, 100, 100, 100, 50])
        acceptance = hb._evaluate_candidate(candidate, baseline)
        with mock.patch.dict(
            "os.environ",
            {
                "HISTORICAL_BACKTEST_OPENAI_ENABLED": "1",
                "HISTORICAL_BACKTEST_OPENAI_MODEL": "gpt-5.6-terra",
                "HISTORICAL_BACKTEST_OPENAI_REASONING_EFFORT": "medium",
                "OPENAI_API_KEY": "test-key",
            },
            clear=False,
        ):
            result = hb._request_openai_backtest_review(candidate, baseline, acceptance, http_post=fake_post)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["model"], "gpt-5.6-terra")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertFalse(captured["json"]["store"])
        self.assertEqual(captured["json"]["reasoning"]["effort"], "medium")

    def test_openai_review_stays_disabled_outside_historical_opt_in(self):
        with mock.patch.dict("os.environ", {"HISTORICAL_BACKTEST_OPENAI_ENABLED": "0"}, clear=False):
            result = hb._request_openai_backtest_review({}, {}, {})
        self.assertEqual(result["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
