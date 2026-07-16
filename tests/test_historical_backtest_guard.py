import unittest

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


if __name__ == "__main__":
    unittest.main()
