import sys
import types
import unittest
from unittest.mock import Mock, patch

import panel_realtime_server


class PanelMarketKlineSourceTests(unittest.TestCase):
    def _fake_eth(self, source="kraken"):
        def fetch(symbol, interval, **kwargs):
            self.assertEqual(kwargs.get("source_preference"), "kraken_first")
            self.assertIs(kwargs.get("allow_binance_fallback"), False)
            return [[0, "1", "2", "0.5", "1.5", "10"]], source

        return types.SimpleNamespace(
            KLINE_INTERVAL_MS={"4h": 14_400_000},
            _fetch_market_kline_rows=fetch,
        )

    def test_panel_explicitly_disables_binance_fallback(self):
        with patch.dict(sys.modules, {"eth": self._fake_eth()}):
            rows, source = panel_realtime_server._fetch_market_klines_sync("ETHUSDT", "4h", 5)

        self.assertEqual(source, "kraken")
        self.assertEqual(len(rows), 1)

    def test_panel_rejects_binance_source(self):
        with patch.dict(sys.modules, {"eth": self._fake_eth("binance_futures")}):
            with self.assertRaisesRegex(RuntimeError, "must not use Binance"):
                panel_realtime_server._fetch_market_klines_sync("ETHUSDT", "4h", 5)

    @patch("panel_realtime_server.requests.get")
    def test_current_price_uses_binance_mark_price_endpoint(self, get):
        response = Mock()
        response.json.return_value = {"markPrice": "1874.62"}
        get.return_value = response

        price = panel_realtime_server._fetch_binance_mark_price_sync("ETHUSDT")

        self.assertEqual(price, 1874.62)
        get.assert_called_once_with(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": "ETHUSDT"},
            timeout=6,
        )
        response.raise_for_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
