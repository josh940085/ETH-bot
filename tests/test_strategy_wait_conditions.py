import os
import unittest
from unittest.mock import patch

import eth


class StrategyWaitConditionsTests(unittest.TestCase):
    def test_stop_distance_explains_current_and_required_values(self):
        decision = {
            "final": "觀望（停損距離過近）",
            "risk_rate": 0.0018,
            "multitimeframe_bull_reclaim": {
                "enabled": True,
                "applied": False,
                "diagnostics": {"required_price": 1943.73},
            },
        }
        with patch.dict(os.environ, {"TRADE_MIN_ENTRY_RISK_RATE": "0.003"}, clear=False):
            result = eth._build_strategy_wait_conditions(
                decision,
                1930.76,
                "waiting",
                "觀望（停損距離過近）",
            )

        self.assertEqual(result[0]["key"], "stop_distance")
        self.assertEqual(result[0]["current"], "0.180%")
        self.assertEqual(result[0]["target"], "至少 0.300%")
        self.assertEqual(result[1]["key"], "bull_reclaim_price")
        self.assertEqual(result[1]["target"], "站上 1943.73")

    def test_pending_confirmation_shows_hold_duration_and_price_validation(self):
        with patch.dict(os.environ, {"TRADE_ENTRY_CONFIRM_MIN_WAIT_SEC": "15"}, clear=False):
            result = eth._build_strategy_wait_conditions(
                {},
                1930.76,
                "pending_confirmation",
                "等待進場確認",
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["key"], "entry_confirmation")
        self.assertEqual(result[0]["target"], "維持 15 秒")
        self.assertIn("Binance Mark Price", result[0]["detail"])


if __name__ == "__main__":
    unittest.main()
