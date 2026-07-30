import os
import unittest
from unittest.mock import patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class TpNetProfitFloorTests(unittest.TestCase):
    def test_long_tp_is_raised_above_costs_and_minimum_net_profit(self):
        with (
            patch.object(eth, "_estimate_trade_cost_rate_est", return_value=0.0014),
            patch.dict(eth.os.environ, {"TRADE_TP_MIN_NET_PROFIT_RATE": "0.0005"}),
        ):
            tp, required_rate = eth._ensure_minimum_net_profit_tp(
                "long",
                100_000.0,
                100_100.0,
                hold_hours=6,
            )

        self.assertAlmostEqual(required_rate, 0.0019)
        self.assertEqual(tp, 100_190.0)

    def test_short_tp_is_lowered_beyond_costs_and_minimum_net_profit(self):
        with (
            patch.object(eth, "_estimate_trade_cost_rate_est", return_value=0.0014),
            patch.dict(eth.os.environ, {"TRADE_TP_MIN_NET_PROFIT_RATE": "0.0005"}),
        ):
            tp, required_rate = eth._ensure_minimum_net_profit_tp(
                "short",
                100_000.0,
                99_900.0,
                hold_hours=6,
            )

        self.assertAlmostEqual(required_rate, 0.0019)
        self.assertEqual(tp, 99_810.0)

    def test_existing_profitable_tp_is_not_moved_closer(self):
        with patch.object(eth, "_estimate_trade_cost_rate_est", return_value=0.0014):
            long_tp, _ = eth._ensure_minimum_net_profit_tp("long", 100_000.0, 102_000.0)
            short_tp, _ = eth._ensure_minimum_net_profit_tp("short", 100_000.0, 98_000.0)

        self.assertEqual(long_tp, 102_000.0)
        self.assertEqual(short_tp, 98_000.0)


if __name__ == "__main__":
    unittest.main()
