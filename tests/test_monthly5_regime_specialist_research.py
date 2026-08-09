import unittest

import pandas as pd

import monthly5_regime_specialist_research as specialist


class Monthly5RegimeSpecialistResearchTests(unittest.TestCase):
    def _frame(self, count):
        index = pd.date_range("2026-01-01T00:05:00Z", periods=count, freq="5min")
        return pd.DataFrame({"close": [100.0] * count}, index=index)

    def test_range_rsi_position_resets_on_regime_change(self):
        frame = self._frame(5)
        labels = pd.Series(["range", "up"], index=[frame.index[0], frame.index[3]])
        rsi = pd.Series([25.0, 45.0, 55.0], index=frame.index[[0, 2, 4]])
        position, _ = specialist.build_specialist_position(
            frame,
            labels,
            rsi,
            trend_mode="momentum",
            range_mode="rsi30",
        )
        self.assertEqual(position.tolist(), [1.0, 1.0, 1.0, 0.0, 1.0])

    def test_pullback_policy_enters_and_exits_inside_trend(self):
        frame = self._frame(4)
        labels = pd.Series(["up"], index=[frame.index[0]])
        rsi = pd.Series([50.0, 39.0, 50.0, 61.0], index=frame.index)
        position, _ = specialist.build_specialist_position(
            frame,
            labels,
            rsi,
            trend_mode="pullback",
            range_mode="flat",
        )
        self.assertEqual(position.tolist(), [0.0, 1.0, 1.0, 0.0])

    def test_validation_selection_does_not_use_holdout(self):
        base = {
            "months": 2,
            "months_ge_0": 1,
            "min_month_pct": -1.0,
            "max_drawdown_pct": -2.0,
            "avg_month_pct": 1.0,
        }
        rows = [
            {
                "name": "validation",
                "validation": {**base, "months_ge_5": 2},
                "training": {**base, "months_ge_5": 2},
                "holdout_diagnostic": {"months_ge_5": 0},
            },
            {
                "name": "holdout",
                "validation": {**base, "months_ge_5": 1},
                "training": {**base, "months_ge_5": 2},
                "holdout_diagnostic": {"months_ge_5": 100},
            },
        ]
        self.assertEqual(specialist.select_validation_winner(rows)["name"], "validation")


if __name__ == "__main__":
    unittest.main()
