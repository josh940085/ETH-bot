import unittest

import numpy as np
import pandas as pd

import monthly5_regime_selector


class Monthly5RegimeSelectorTests(unittest.TestCase):
    def test_target_regime_uses_last_completed_4h_bar(self):
        states = pd.Series(
            ["up", "range", "down"],
            index=pd.to_datetime(
                ["2026-01-01 20:00:00Z", "2026-01-02 00:00:00Z", "2026-01-02 04:00:00Z"]
            ),
        )
        regimes = monthly5_regime_selector.target_day_regimes(
            ["2026-01-02", "2026-01-03"], states
        )
        self.assertEqual(regimes.tolist(), ["range", "down"])

    def test_4h_classifier_requires_price_and_slope_alignment(self):
        up = np.linspace(100.0, 140.0, 40)
        down = np.linspace(140.0, 80.0, 40)
        frame = pd.DataFrame(
            {"close": np.concatenate([up, down])},
            index=pd.date_range("2026-01-01", periods=80, freq="4h", tz="UTC"),
        )
        labels = monthly5_regime_selector.classify_4h_regimes(frame)
        self.assertEqual(labels.iloc[35], "up")
        self.assertEqual(labels.iloc[-1], "down")

    def test_causal_selector_never_uses_current_return_for_selection(self):
        days = np.array([f"2026-01-{day:02d}" for day in range(1, 9)])
        cache = {
            "R": np.array(
                [
                    [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, -0.9, -0.9],
                    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 4.0, 4.0],
                ]
            ),
            "F": np.zeros((2, 8)),
            "Xday": np.zeros((8, 2)),
            "keys": np.array(["past_winner", "future_winner"]),
            "days": days,
            "fee": 0.0008,
        }
        result = monthly5_regime_selector.run_causal_selector(
            cache,
            np.array(["range"] * 8),
            use_regime=False,
            lookback_days=20,
            nearest_days=20,
            min_regime_days=1,
            warmup_days=4,
        )
        self.assertEqual(result["selected_indices"][6], 0)

    def test_regime_filter_can_select_different_strategies(self):
        count = 12
        regimes = np.array(["up", "down"] * 6)
        returns = np.zeros((2, count))
        returns[0, regimes == "up"] = 0.08
        returns[0, regimes == "down"] = -0.02
        returns[1, regimes == "up"] = -0.02
        returns[1, regimes == "down"] = 0.08
        cache = {
            "R": returns,
            "F": np.zeros_like(returns),
            "Xday": np.zeros((count, 2)),
            "keys": np.array(["bull_strategy", "bear_strategy"]),
            "days": np.array(pd.date_range("2026-01-01", periods=count).strftime("%Y-%m-%d")),
            "fee": 0.0008,
        }
        result = monthly5_regime_selector.run_causal_selector(
            cache,
            regimes,
            use_regime=True,
            lookback_days=20,
            nearest_days=20,
            min_regime_days=2,
            warmup_days=6,
        )
        self.assertEqual(result["selected_indices"][8], 0)
        self.assertEqual(result["selected_indices"][9], 1)

    def test_development_selector_rejects_high_return_tail_risk(self):
        def variant(name, hits, min_month, drawdown):
            return {
                "name": name,
                "regime_development": {
                    "months_ge_5": hits,
                    "months_ge_0": hits,
                    "min_month_pct": min_month,
                    "max_drawdown_pct": drawdown,
                    "avg_flat_time_pct": 30.0,
                },
            }

        safe = variant("safe", 40, -15.0, -45.0)
        risky = variant("risky", 47, -52.0, -63.0)
        winner, growth_winner = monthly5_regime_selector.select_development_variant(
            [safe, risky]
        )
        self.assertEqual(winner["name"], "safe")
        self.assertEqual(growth_winner["name"], "risky")


if __name__ == "__main__":
    unittest.main()
