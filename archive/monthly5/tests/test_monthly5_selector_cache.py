import unittest

import numpy as np
import pandas as pd

import monthly5_selector_cache


class Monthly5SelectorCacheTests(unittest.TestCase):
    def _frame(self, bars=900):
        index = pd.date_range("2025-12-28T00:05:00Z", periods=bars, freq="5min")
        close = 100.0 + np.linspace(0.0, 10.0, bars) + np.sin(np.arange(bars) / 20.0)
        return pd.DataFrame(
            {
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "volume": 1.0,
            },
            index=index,
        )

    def test_signal_is_shifted_one_bar(self):
        frame = self._frame(20)
        signal = monthly5_selector_cache.build_signal(frame, "buy_hold")
        self.assertEqual(signal.iloc[0], 0.0)
        self.assertTrue((signal.iloc[1:] == 1.0).all())

    def test_candidate_library_includes_short_only_regime_strategies(self):
        prefixes = monthly5_selector_cache.strategy_prefixes()
        self.assertIn("ma6_24_sf", prefixes)
        self.assertIn("mom48_sf", prefixes)
        self.assertIn("don24_sf", prefixes)
        self.assertTrue(any("_sf|" in key for key in monthly5_selector_cache.candidate_keys()))

    def test_fee_is_charged_on_position_turnover(self):
        frame = self._frame(3)
        position = pd.Series([0.0, 1.0, 1.0], index=frame.index)
        returns, _ = monthly5_selector_cache.simulate_daily_returns(
            frame,
            position,
            leverage=1,
            round_trip_fee=0.0008,
        )
        first_return = frame["close"].iloc[1] / frame["close"].iloc[0] - 1.0
        second_return = frame["close"].iloc[2] / frame["close"].iloc[1] - 1.0
        expected = (1.0 + first_return - 0.0004) * (1.0 + second_return) - 1.0
        self.assertAlmostEqual(float(returns.iloc[0]), expected, places=10)

    def test_recursive_prefix_is_stable(self):
        frame = self._frame()
        report = monthly5_selector_cache.verify_recursive_stability(
            frame,
            cutoffs=("2025-12-29", "2025-12-30"),
        )
        self.assertTrue(report["prefix_stable"])
        self.assertTrue(report["recursive_stable"])

    def test_cache_has_explicit_schema_and_leverage_cap(self):
        frame = self._frame()
        cache = monthly5_selector_cache.build_cache(
            frame,
            start_day="2025-12-29",
            end_day="2025-12-30",
        )
        self.assertEqual(int(cache["schema_version"][0]), 1)
        self.assertEqual(str(cache["feature_schema"][0]), monthly5_selector_cache.FEATURE_SCHEMA)
        self.assertEqual(cache["R"].shape[0], len(cache["keys"]))
        self.assertTrue(all("|lev" in key for key in cache["keys"]))
        self.assertTrue(all(int(key.split("|lev", 1)[1].split("|", 1)[0]) <= 5 for key in cache["keys"]))

    def test_generated_cache_verification_is_embedded_by_main_contract(self):
        cache = {"R": np.zeros((1, 1))}
        verification = {"prefix_stable": True, "recursive_stable": True}
        verified = monthly5_selector_cache.embed_verification(cache, verification)
        self.assertTrue(bool(verified["prefix_stable"][0]))
        self.assertTrue(bool(verified["recursive_stable"][0]))


if __name__ == "__main__":
    unittest.main()
