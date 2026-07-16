import os
import unittest
from unittest.mock import patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class StrategyExecutionSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.panel_state = dict(eth.POSITION_PANEL_STATE)
        self.active_trade = dict(eth.active_trade)

    def tearDown(self):
        eth.POSITION_PANEL_STATE.clear()
        eth.POSITION_PANEL_STATE.update(self.panel_state)
        eth.active_trade.clear()
        eth.active_trade.update(self.active_trade)

    def test_validated_mark_price_is_used_for_strategy(self):
        price = eth._validated_strategy_mark_price(
            {"price": 1871.80},
            1871.20,
            reference_price=1871.40,
        )
        self.assertEqual(price, 1871.80)

    def test_strategy_rejects_mark_price_outlier(self):
        with self.assertRaisesRegex(RuntimeError, "交叉驗證失敗"):
            eth._validated_strategy_mark_price(
                {"price": 1900.0},
                1871.20,
                reference_price=1871.40,
            )

    @patch("eth.time.time", return_value=2000.0)
    def test_only_actual_open_sets_long_or_short_signal(self, _time):
        eth.active_trade["direction"] = "long"
        eth.POSITION_PANEL_STATE["binance_mark_price_ts"] = 1999
        decision = {"final": "做多", "score": 0.7, "ai_prob": 0.72, "regime": "bull_trend"}

        eth._update_panel_execution_snapshot(
            decision, 1871.8, "pending_confirmation", reason="等待確認", actual_open=False
        )
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_signal"], "wait")
        self.assertFalse(eth.POSITION_PANEL_STATE["strategy_actual_open"])

        eth._update_panel_execution_snapshot(
            decision, 1871.8, "opened", reason="Binance 實際開單成功", actual_open=True
        )
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_signal"], "long")
        self.assertTrue(eth.POSITION_PANEL_STATE["strategy_actual_open"])
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_evaluated_ts"], 2000)


if __name__ == "__main__":
    unittest.main()
