import os
import unittest
from unittest.mock import call, patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


def _protection_order(order_type):
    return {
        "type": order_type,
        "side": "BUY",
        "positionSide": "SHORT",
        "closePosition": True,
    }


class BinanceProtectionOrderTests(unittest.TestCase):
    def test_install_submits_and_verifies_both_protection_orders(self):
        confirmed = [
            _protection_order("TAKE_PROFIT_MARKET"),
            _protection_order("STOP_MARKET"),
        ]
        with (
            patch.object(eth, "_list_binance_protection_orders", side_effect=[[], confirmed]),
            patch.object(eth, "_submit_binance_protection_order") as submit,
        ):
            ok, message = eth._install_and_verify_binance_protection(
                "BUY",
                "SHORT",
                True,
                0.012,
                1875.0,
                1925.0,
                attempts=1,
            )

        self.assertTrue(ok)
        self.assertIn("均已由 Binance 確認", message)
        self.assertEqual(
            submit.call_args_list,
            [
                call("BUY", "SHORT", True, 0.012, "TAKE_PROFIT_MARKET", 1875.0),
                call("BUY", "SHORT", True, 0.012, "STOP_MARKET", 1925.0),
            ],
        )

    def test_tp_failure_does_not_skip_sl_submission(self):
        stop_only = [_protection_order("STOP_MARKET")]

        def submit_order(*args):
            if args[4] == "TAKE_PROFIT_MARKET":
                raise RuntimeError("tp rejected")
            return {"algoId": 2}

        with (
            patch.object(eth, "_list_binance_protection_orders", side_effect=[[], stop_only]),
            patch.object(eth, "_submit_binance_protection_order", side_effect=submit_order) as submit,
        ):
            ok, message = eth._install_and_verify_binance_protection(
                "BUY",
                "SHORT",
                True,
                0.012,
                1875.0,
                1925.0,
                attempts=1,
            )

        self.assertFalse(ok)
        self.assertIn("未確認 TP", message)
        self.assertEqual(submit.call_count, 2)
        self.assertEqual(submit.call_args_list[1].args[4], "STOP_MARKET")

    def test_open_is_blocked_before_market_order_when_tp_sl_invalid(self):
        with (
            patch.object(eth, "_get_follow_mode_enabled", return_value=True),
            patch.object(eth, "_is_real_copy_enabled", return_value=True),
            patch.object(eth, "_binance_futures_signed_request") as request,
            patch.object(eth, "WS_PRICE", 1900.0),
        ):
            ok, message = eth.execute_copy_trade_open(
                "short",
                0.05,
                tp=None,
                sl=1925.0,
            )

        self.assertFalse(ok)
        self.assertIn("缺少有效 TP/SL", message)
        request.assert_not_called()

    def test_emergency_close_uses_reduce_only_market_order(self):
        with (
            patch.object(eth, "_binance_futures_signed_request", return_value={"orderId": 7}) as request,
            patch.object(eth, "_cancel_existing_binance_protection_orders"),
        ):
            ok, message = eth._emergency_close_unprotected_position(
                "short",
                "SHORT",
                False,
                0.012,
            )

        self.assertTrue(ok)
        self.assertIn("立即市價平倉", message)
        params = request.call_args.args[2]
        self.assertEqual(params["side"], "BUY")
        self.assertEqual(params["type"], "MARKET")
        self.assertEqual(params["reduceOnly"], "true")


if __name__ == "__main__":
    unittest.main()
