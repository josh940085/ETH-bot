import os
import time
import unittest
from unittest.mock import patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class MfeProfitLockTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = dict(eth.active_trade)
        eth.active_trade.update(
            {
                "open": True,
                "direction": "long",
                "entry": 100.0,
                "avg_entry": 100.0,
                "tp": 103.0,
                "sl": 98.0,
                "open_time": 1000.0,
                "peak_favorable_price": 100.0,
                "peak_favorable_rate": 0.0,
                "profit_lock_active": False,
                "trade_source": "signal",
                "entry_host_confidence": 0.60,
            }
        )

    def tearDown(self):
        eth.active_trade.clear()
        eth.active_trade.update(self.snapshot)

    def test_locks_profit_after_peak_giveback_and_hold_time(self):
        with (
            patch.object(eth, "sync_position_panel"),
            patch.object(eth, "send_telegram"),
            patch.object(eth, "_get_follow_mode_enabled", return_value=False),
            patch.object(eth, "_is_daily_min_position", return_value=False),
            patch.object(eth, "_has_scaling_opposing_pressure", return_value=True),
        ):
            locked = eth.maybe_lock_profit_after_reversal(
                100.65,
                favorable_price=101.20,
                atr=0.20,
                now_ts=1000.0 + 4 * 3600.0,
            )
        self.assertTrue(locked)
        self.assertTrue(eth.active_trade["profit_lock_active"])
        self.assertGreater(eth.active_trade["sl"], 100.0)
        self.assertEqual(eth.active_trade["tp"], 103.0)

    def test_does_not_lock_without_reversal_or_before_hold_time(self):
        with (
            patch.object(eth, "sync_position_panel"),
            patch.object(eth, "send_telegram"),
            patch.object(eth, "_get_follow_mode_enabled", return_value=False),
            patch.object(eth, "_is_daily_min_position", return_value=False),
            patch.object(eth, "_has_scaling_opposing_pressure", return_value=False),
        ):
            self.assertFalse(
                eth.maybe_lock_profit_after_reversal(
                    100.90,
                    favorable_price=101.20,
                    atr=0.20,
                    now_ts=1000.0 + 2 * 3600.0,
                )
            )
            self.assertFalse(
                eth.maybe_lock_profit_after_reversal(
                    101.00,
                    favorable_price=101.20,
                    atr=0.20,
                    now_ts=1000.0 + 4 * 3600.0,
                )
            )


if __name__ == "__main__":
    unittest.main()
