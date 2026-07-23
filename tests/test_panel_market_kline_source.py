import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import panel_realtime_server


class PanelMarketKlineSourceTests(unittest.TestCase):
    def test_live_one_minute_cache_is_shorter_than_regular_kline_cache(self):
        self.assertEqual(panel_realtime_server.MARKET_LIVE_DATA_TTL_SEC, 3.0)
        self.assertLess(
            panel_realtime_server.MARKET_LIVE_DATA_TTL_SEC,
            panel_realtime_server.MARKET_DATA_TTL_SEC,
        )

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

    def _fake_coinbase_eth(self):
        def fetch(symbol, interval, **kwargs):
            return [[1_980_000, "1871", "1872", "1870", "1871.5", "10"]], "coinbase"

        return types.SimpleNamespace(
            KLINE_INTERVAL_MS={"1m": 60_000},
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

    @patch("panel_realtime_server.time.time", return_value=2000.0)
    @patch("panel_realtime_server.requests.get")
    def test_live_one_minute_candle_uses_coinbase_ticker(self, get, _time):
        response = Mock()
        response.json.return_value = {"price": "1871.8"}
        get.return_value = response

        with patch.dict(sys.modules, {"eth": self._fake_coinbase_eth()}):
            rows, source = panel_realtime_server._fetch_market_klines_sync("ETHUSDT", "1m", 2)

        self.assertEqual(source, "coinbase_live")
        self.assertEqual(rows[-1]["close"], 1871.8)
        self.assertEqual(rows[-1]["high"], 1872.0)
        response.raise_for_status.assert_called_once_with()

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

    @patch("panel_realtime_server.time.time", return_value=2000.0)
    def test_recent_valid_mark_price_can_cover_brief_refresh_failure(self, _time):
        cached = {
            "ts": 1998.0,
            "payload": {
                "validated": True,
                "price": 1874.62,
                "exchange_ts": 1999.0,
            },
        }

        payload = panel_realtime_server._usable_cached_market_price(cached, 2000.0)

        self.assertEqual(payload["price"], 1874.62)

    def test_stale_mark_price_cannot_cover_refresh_failure(self):
        cached = {
            "ts": 1900.0,
            "payload": {
                "validated": True,
                "price": 1874.62,
                "exchange_ts": 1900.0,
            },
        }

        self.assertIsNone(panel_realtime_server._usable_cached_market_price(cached, 2000.0))

    def test_mark_price_failures_use_bounded_exponential_backoff(self):
        cache_key = "mark_price:TESTUSDT"
        panel_realtime_server._clear_market_price_failures(cache_key)
        try:
            with patch.object(panel_realtime_server, "MARKET_PRICE_MAX_COOLDOWN_SEC", 30.0):
                delays = [
                    panel_realtime_server._mark_market_price_failure(cache_key, 2000.0)
                    for _ in range(6)
                ]
            self.assertEqual(delays[:4], [2.0, 4.0, 8.0, 16.0])
            self.assertEqual(delays[4:], [30.0, 30.0])
            self.assertEqual(
                panel_realtime_server.MARKET_PRICE_FAILURE_UNTIL[cache_key],
                2030.0,
            )
        finally:
            panel_realtime_server._clear_market_price_failures(cache_key)

    def test_mark_price_backoff_stays_inside_validated_cache_horizon(self):
        cache_key = "mark_price:TESTUSDT"
        panel_realtime_server._clear_market_price_failures(cache_key)
        try:
            with patch.object(panel_realtime_server, "MARKET_PRICE_MAX_COOLDOWN_SEC", 5.0):
                delays = [
                    panel_realtime_server._mark_market_price_failure(cache_key, 2000.0)
                    for _ in range(6)
                ]

            self.assertEqual(delays, [2.0, 4.0, 5.0, 5.0, 5.0, 5.0])
            self.assertEqual(
                panel_realtime_server.MARKET_PRICE_FAILURE_UNTIL[cache_key],
                2005.0,
            )
        finally:
            panel_realtime_server._clear_market_price_failures(cache_key)

    def test_success_clears_mark_price_failure_backoff(self):
        cache_key = "mark_price:TESTUSDT"
        panel_realtime_server._mark_market_price_failure(cache_key, 2000.0)

        panel_realtime_server._clear_market_price_failures(cache_key)

        self.assertNotIn(cache_key, panel_realtime_server.MARKET_PRICE_FAILURE_UNTIL)
        self.assertNotIn(cache_key, panel_realtime_server.MARKET_PRICE_FAILURE_COUNT)


if __name__ == "__main__":
    unittest.main()
