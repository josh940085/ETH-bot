import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import verify_monthly5_account


class VerifyMonthly5AccountTests(unittest.TestCase):
    def test_open_positions_filters_symbol_and_zero_amounts(self):
        rows = [
            {"symbol": "ETHUSDT", "positionAmt": "0"},
            {"symbol": "BTCUSDT", "positionAmt": "0.005", "entryPrice": "64000", "markPrice": "64100", "leverage": "4"},
            {"symbol": "SOLUSDT", "positionAmt": "1"},
        ]

        positions = verify_monthly5_account._open_positions(rows, "BTCUSDT")

        self.assertEqual(
            positions,
            [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.005",
                    "entryPrice": "64000",
                    "markPrice": "64100",
                    "leverage": "4",
                }
            ],
        )

    def test_account_passes_when_binance_symbol_is_flat(self):
        eth_module = SimpleNamespace(
            COPY_TRADE_SYMBOL="BTCUSDT",
            _binance_futures_signed_get=lambda path, params: [
                {"symbol": "BTCUSDT", "positionAmt": "0"},
            ],
        )

        ok, payload = verify_monthly5_account._verify_account_flat(eth_module)

        self.assertTrue(ok)
        self.assertEqual(payload["symbol"], "BTCUSDT")
        self.assertEqual(payload["open_count"], 0)

    def test_main_fails_when_symbol_has_open_position(self):
        eth_module = SimpleNamespace(
            COPY_TRADE_SYMBOL="BTCUSDT",
            _binance_futures_signed_get=lambda path, params: [
                {"symbol": "BTCUSDT", "positionAmt": "-0.002", "entryPrice": "64100", "markPrice": "64000", "leverage": "3"},
            ],
        )

        with (
            patch.object(verify_monthly5_account, "_load_eth_module", return_value=eth_module),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            code = verify_monthly5_account.main([])

        self.assertEqual(code, 1)
        self.assertIn("FAIL monthly5_account", stdout.getvalue())
        self.assertIn('"open_count":1', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
