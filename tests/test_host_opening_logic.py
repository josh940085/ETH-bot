import os
import unittest

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class HostOpeningLogicTests(unittest.TestCase):
    def test_pressure_absorption_long_for_stealth_rise(self):
        result = eth._score_host_opening_logic(
            price=65146.0,
            timeframe_kline_view={
                "high_tf_score": 0.5,
                "mid_tf_score": 0.5,
                "low_tf_score": 0.45,
            },
            range_pos=0.82,
            htf=1,
            mid_trend=1,
            breakout=0,
            regime="bull_trend",
            volume_spike=True,
            buy_pressure=True,
            sell_pressure=False,
            sweep_high=False,
            sweep_low=False,
            support_hits=0,
            resistance_hits=1,
            repeated_support_tests=0,
            repeated_resistance_tests=8,
            repeated_test_pressure=0.95,
            macro_bias=1.5,
        )

        self.assertEqual(result["direction"], "long")
        self.assertEqual(result["mode"], "pressure_absorption_long")
        self.assertIn("壓力連測陰漲吸籌，買壓量能同向", result["reasons"])

    def test_pressure_tests_without_buy_pressure_still_waits_rejection(self):
        result = eth._score_host_opening_logic(
            price=65146.0,
            timeframe_kline_view={
                "high_tf_score": 0.5,
                "mid_tf_score": 0.5,
                "low_tf_score": 0.1,
            },
            range_pos=0.82,
            htf=1,
            mid_trend=1,
            breakout=0,
            regime="bull_trend",
            volume_spike=True,
            buy_pressure=False,
            sell_pressure=False,
            sweep_high=False,
            sweep_low=False,
            support_hits=0,
            resistance_hits=1,
            repeated_support_tests=0,
            repeated_resistance_tests=8,
            repeated_test_pressure=0.95,
            macro_bias=1.5,
        )

        self.assertNotEqual(result["direction"], "long")
        self.assertNotEqual(result["mode"], "pressure_absorption_long")


if __name__ == "__main__":
    unittest.main()
