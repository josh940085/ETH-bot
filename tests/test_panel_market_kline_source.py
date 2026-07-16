import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import panel_realtime_server


class PanelMarketKlineSourceTests(unittest.TestCase):
    def test_local_preview_can_read_public_market_data_without_viewer_token(self):
        request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))

        with patch.object(panel_realtime_server, "_viewer_authorized_http", return_value=False):
            self.assertTrue(panel_realtime_server._market_data_authorized_http(request))

    def test_remote_market_data_still_requires_viewer_authorization(self):
        request = SimpleNamespace(client=SimpleNamespace(host="203.0.113.10"))

        with patch.object(panel_realtime_server, "_viewer_authorized_http", return_value=False):
            self.assertFalse(panel_realtime_server._market_data_authorized_http(request))

    def test_direct_file_origin_is_allowed_for_local_preview(self):
        with patch.dict(panel_realtime_server.os.environ, {"POSITION_PANEL_ALLOWED_ORIGINS": "https://example.com"}):
            self.assertIn("null", panel_realtime_server._load_origins())

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
    @patch("panel_realtime_server.time.time", return_value=2000.0)
    def test_current_price_cross_checks_binance_payloads(self, _time, get):
        mark_response = Mock()
        mark_response.json.return_value = {
            "symbol": "ETHUSDT", "markPrice": "1874.62", "indexPrice": "1874.50", "time": 1_999_500,
        }
        ticker_response = Mock()
        ticker_response.json.return_value = {"symbol": "ETHUSDT", "price": "1874.70", "time": 1_999_600}
        get.side_effect = [mark_response, ticker_response]

        snapshot = panel_realtime_server._fetch_binance_mark_price_sync("ETHUSDT")

        self.assertEqual(snapshot["price"], 1874.62)
        self.assertEqual(snapshot["symbol"], "ETHUSDT")
        self.assertLess(snapshot["max_deviation_rate"], 0.01)
        self.assertEqual(get.call_count, 2)
        mark_response.raise_for_status.assert_called_once_with()
        ticker_response.raise_for_status.assert_called_once_with()

    @patch("panel_realtime_server.requests.get")
    @patch("panel_realtime_server.time.time", return_value=2000.0)
    def test_current_price_rejects_stale_payload(self, _time, get):
        mark_response = Mock()
        mark_response.json.return_value = {
            "symbol": "ETHUSDT", "markPrice": "1874.62", "indexPrice": "1874.50", "time": 1_900_000,
        }
        ticker_response = Mock()
        ticker_response.json.return_value = {"symbol": "ETHUSDT", "price": "1874.70", "time": 1_900_000}
        get.side_effect = [mark_response, ticker_response]

        with self.assertRaisesRegex(RuntimeError, "stale"):
            panel_realtime_server._fetch_binance_mark_price_sync("ETHUSDT")

    @patch("panel_realtime_server.requests.get")
    @patch("panel_realtime_server.time.time", return_value=2000.0)
    def test_current_price_rejects_symbol_mismatch(self, _time, get):
        mark_response = Mock()
        mark_response.json.return_value = {
            "symbol": "BTCUSDT", "markPrice": "1874.62", "indexPrice": "1874.50", "time": 1_999_500,
        }
        ticker_response = Mock()
        ticker_response.json.return_value = {"symbol": "ETHUSDT", "price": "1874.70", "time": 1_999_600}
        get.side_effect = [mark_response, ticker_response]

        with self.assertRaisesRegex(RuntimeError, "symbol mismatch"):
            panel_realtime_server._fetch_binance_mark_price_sync("ETHUSDT")

    @patch("panel_realtime_server.requests.get")
    @patch("panel_realtime_server.time.time", return_value=2000.0)
    def test_current_price_rejects_cross_check_outlier(self, _time, get):
        mark_response = Mock()
        mark_response.json.return_value = {
            "symbol": "ETHUSDT", "markPrice": "2100", "indexPrice": "1874.50", "time": 1_999_500,
        }
        ticker_response = Mock()
        ticker_response.json.return_value = {"symbol": "ETHUSDT", "price": "1874.70", "time": 1_999_600}
        get.side_effect = [mark_response, ticker_response]

        with self.assertRaisesRegex(RuntimeError, "cross-check failed"):
            panel_realtime_server._fetch_binance_mark_price_sync("ETHUSDT")


if __name__ == "__main__":
    unittest.main()
